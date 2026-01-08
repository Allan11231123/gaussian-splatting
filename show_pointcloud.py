#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import open3d as o3d
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and visualize a .ply point cloud file."
    )
    parser.add_argument(
        "--ply_file",
        required=True,
        type=str,
        help="Path to the .ply point cloud file."
    )
    parser.add_argument(
        "--show_normals",
        action='store_true',
        help="Whether to show point normals if available."
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="If > 0, downsample the point cloud with a voxel grid of this size."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(args.ply_file)
    if pcd.is_empty():
        print(f"Failed to load or empty point cloud: {args.ply_file}")
        sys.exit(1)
    print(f"Loaded point cloud: {args.ply_file}")
    print(f"Points count: {len(pcd.points)}")

    # Optionally downsample
    if args.voxel_size > 0:
        print(f"Downsampling with voxel size = {args.voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"Downsampled point count: {len(pcd.points)}")

    # Estimate normals if not present
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,
                max_nn=30
            )
        )
    # Set pointcloud color to single value that is easy to review
    colors = np.zeros_like(pcd.points)
    colors[:,0] = 1
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Viewer",
        point_show_normal=args.show_normals,
        mesh_show_back_face=False,
        mesh_show_wireframe=False,
    )

if __name__ == "__main__":
    main()
