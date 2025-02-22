import time

import torch
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
import argparse
from typing import Tuple

from gsplat.rendering import _rasterization, rasterization


def load_ply(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load a PLY file and return the Gaussian parameters."""
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']

    # Extract the parameters
    means = torch.tensor(np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1), dtype=torch.float32)
    scales = torch.tensor(np.stack([vertex[f'scale_{i}'] for i in range(3)], axis=1), dtype=torch.float32)
    quats = torch.tensor(np.stack([vertex[f'rot_{i}'] for i in range(4)], axis=1), dtype=torch.float32)
    opacities = torch.tensor(vertex['opacity'], dtype=torch.float32)
    colors = torch.tensor(np.stack([vertex[f'f_dc_{i}'] for i in range(3)], axis=1), dtype=torch.float32) # [N, 3]

    nb_rest = len([prop.name for prop in vertex.properties if prop.name.startswith('f_rest_')])
    sh_degree = 0
    if nb_rest :
        colors_rest = torch.tensor(np.stack([vertex[f'f_rest_{i}'] for i in range(nb_rest)], axis=1), dtype=torch.float32)
        K = nb_rest // 3 + 1 #total_coeffs_per_channel
        colors = torch.cat([
            colors.reshape(-1, 1, 3),
            colors_rest.reshape(-1, K-1, 3)
        ], dim=1) # [N, K, 3]
        sh_degree = int(np.sqrt(K) - 1)

    return means, scales, quats, opacities, colors, sh_degree


def get_projection_matrix(width: int, height: int) -> torch.Tensor:
    """Create camera intrinsic matrix from the projection matrix."""
    proj = np.array([
        [2., 0., 0., 0.],
        [0., -2., 0., 0.],
        [0., 0., 1.0010010010010009, 1.],
        [0., 0., -0.20020020020020018, 0.]
    ])

    # Extract focal lengths and principal points from projection matrix
    fx = proj[0, 0] * width / 2
    fy = -proj[1, 1] * height / 2
    cx = width / 2
    cy = height / 2

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)

    print(f"Camera {width}x{height} f={fx:.1f}x{fy:.1f} c={cx:.1f}x{cy:.1f}")

    return K


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Render a PLY file using _rasterization')
    parser.add_argument('--ply_path', default='results/garden/ply/point_cloud_999.ply', help='Path to the PLY file')
    parser.add_argument('--width', type=int, default=1000, help='Image width') #646 420
    parser.add_argument('--height', type=int, default=1000, help='Image height')
    parser.add_argument('--output', type=str, default='render.png', help='Output image path')
    args = parser.parse_args()

    # Load the PLY file
    print("Loading PLY file...")
    means, scales, quats, opacities, colors, sh_degree = load_ply(args.ply_path)

    scales = torch.exp(scales)  # [N, 3]
    opacities = torch.sigmoid(opacities)  # [N,]

    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    means = means.to(device)
    scales = scales.to(device)
    quats = quats.to(device)
    opacities = opacities.to(device)
    colors = colors.to(device)

    # Set up view matrix
    view_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.1, 3, 1]
    ], dtype=torch.float32, device=device).T
    # Get camera intrinsics from projection matrix
    K = get_projection_matrix(args.width, args.height).to(device)

    # Add batch dimension
    viewmats = view_matrix.unsqueeze(0)
    Ks = K.unsqueeze(0)


    def wrapped_rasterization():
        return rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=args.width,
            height=args.height,
            sh_degree=sh_degree
        )

    # Render
    print("Rendering...")
    # warmup
    _ = wrapped_rasterization()

    times = []
    renders = None

    for i in range(10):
        time_rstz = time.time()
        renders, alphas, info = wrapped_rasterization()
        times.append(time.time() - time_rstz)
    print(f"Rendering ~{sum(times)/len(times):.3f}s - [{min(times):.3f}s, {max(times):.3f}s]")

    # Save the render
    img = renders[0].cpu().numpy()  # Remove batch dimension
    img = np.clip(img, 0, 1)
    plt.imsave(args.output, img)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()