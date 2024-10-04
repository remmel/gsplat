"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import pytest
import torch
from typing_extensions import Literal, assert_never

from gsplat._helper import load_test_data

device = torch.device("cuda:0")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    torch.manual_seed(42)

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_ray_tetra_intersection():
    from gsplat.cuda._torch_impl import _ray_tetra_intersection
    from gsplat.cuda._wrapper import ray_tetra_intersection

    torch.manual_seed(42)

    N = 100000
    rays_o = torch.randn(N, 3, device=device)
    rays_d = torch.randn(N, 3, device=device)
    rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
    vertices = torch.randn(N, 4, 3, device=device)
    vertices.requires_grad = True

    entry_face_ids, exit_face_ids, t_entrys, t_exits = ray_tetra_intersection(
        rays_o, rays_d, vertices
    )
    _entry_face_ids, _exit_face_ids, _t_entrys, _t_exits, hit = _ray_tetra_intersection(
        rays_o, rays_d, vertices
    )

    torch.testing.assert_close(entry_face_ids, _entry_face_ids)
    torch.testing.assert_close(exit_face_ids, _exit_face_ids)

    # if intersection happens, check if the t values are correct
    isect_ids = torch.where((entry_face_ids >= 0) & (exit_face_ids >= 0))[0]
    torch.testing.assert_close(
        t_entrys[isect_ids], _t_entrys[isect_ids], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        t_exits[isect_ids], _t_exits[isect_ids], atol=2e-5, rtol=1e-6
    )

    # if intersection happens, check if the gradient of vertices are correct
    v_t_entrys = torch.randn_like(t_entrys)
    v_t_exits = torch.randn_like(t_exits)
    v_vertices = torch.autograd.grad(
        (
            t_entrys[isect_ids] * v_t_entrys[isect_ids]
            + t_exits[isect_ids] * v_t_exits[isect_ids]
        ).sum(),
        vertices,
        retain_graph=True,
        allow_unused=True,
    )[0]
    _v_vertices = torch.autograd.grad(
        (
            _t_entrys[isect_ids] * v_t_entrys[isect_ids]
            + _t_exits[isect_ids] * v_t_exits[isect_ids]
        ).sum(),
        vertices,
        retain_graph=True,
        allow_unused=True,
    )[0]
    torch.testing.assert_close(v_vertices, _v_vertices, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3])
def test_rasterize_to_pixels(test_data, channels: int):
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    C = len(Ks)
    colors = torch.randn(C, len(means), channels, device=device)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)

    covars, precis = quat_scale_to_covar_preci(quats, scales, compute_preci=True, triu=True)

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    opacities = opacities.repeat(C, 1)

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    enable_culling = True
    tquats = torch.rand_like(quats)
    tscales = torch.mean(scales, dim=1)

    from gsplat.cuda._torch_impl import _quat_scale_to_matrix

    tvertices = torch.tensor(
        [
            [math.sqrt(8 / 9), 0, -1 / 3],
            [-math.sqrt(2 / 9), math.sqrt(2 / 3), -1 / 3],
            [-math.sqrt(2 / 9), -math.sqrt(2 / 3), -1 / 3],
            [0, 0, 1],
        ],
        device=means.device,
        dtype=means.dtype,
    )  # [4, 3]
    rotmats = _quat_scale_to_matrix(tquats, tscales[:, None])  # [N, 3, 3]
    tvertices = torch.einsum("nij,kj->nki", rotmats, tvertices)  # [N, 4, 3]
    tvertices = tvertices + means[:, None, :]  # [N, 4, 3]

    # forward
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        enable_culling=enable_culling,
        camtoworlds=torch.linalg.inv(viewmats),
        Ks=Ks,
        means3d=means,
        precis=precis,
        tvertices=tvertices,
    )
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        enable_culling=enable_culling,
        camtoworlds=torch.linalg.inv(viewmats),
        Ks=Ks,
        means3d=means,
        precis=precis,
        tvertices=tvertices,
    )
    torch.testing.assert_close(render_colors, _render_colors)
    torch.testing.assert_close(render_alphas, _render_alphas)
