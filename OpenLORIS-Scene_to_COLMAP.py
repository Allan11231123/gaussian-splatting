#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenLORIS-Scene Single-Frame -> COLMAP binary.

Purpose:
  To prepare data for 3DGS training using ONLY ONE timestamp.
  Geometry is strictly initialized from the depth map of that specific frame.
  No multi-view constraints are used (avoids temporal misalignment artifacts).

Outputs:
<out_dir>/
  images/
    <timestamp>.png
  sparse/0/
    cameras.bin
    images.bin (Contains only 1 image)
    points3D.bin (Dense point cloud unprojected from depth)
"""

import os, sys
import argparse
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# ----------------- COLMAP Binary Writers -----------------

def write_cameras_binary(cameras: Dict[int, dict], out_path: Path):
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, cam in cameras.items():
            # PINHOLE = 1
            f.write(struct.pack("<iiQQ", int(cam_id), 1, int(cam["width"]), int(cam["height"])))
            f.write(struct.pack("<" + "d"*4, *cam["params"]))

def write_images_binary(images: Dict[int, dict], out_path: Path):
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, img in images.items():
            q = img["qvec"]
            t = img["tvec"]
            f.write(struct.pack("<i", int(img_id)))
            f.write(struct.pack("<dddd", q[0], q[1], q[2], q[3]))
            f.write(struct.pack("<ddd", t[0], t[1], t[2]))
            f.write(struct.pack("<i", int(img["camera_id"])))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0)) # No 2D points

def write_points3d_binary(points: Dict[int, dict], out_path: Path):
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, P in points.items():
            f.write(struct.pack("<Q", int(pid)))
            f.write(struct.pack("<ddd", P["xyz"][0], P["xyz"][1], P["xyz"][2]))
            f.write(struct.pack("<BBB", P["rgb"][0], P["rgb"][1], P["rgb"][2]))
            f.write(struct.pack("<d", 0.01)) # error
            f.write(struct.pack("<Q", 0))    # track length

# ----------------- Utils -----------------

def write_debug_ply(points: Dict[int, dict], out_path: Path):
    """
    Writes the point cloud to a .ply file for visualization in MeshLab/CloudCompare.
    """
    print(f"Writing debug PLY to {out_path}...")
    with open(out_path, "wb") as f:
        # PLY Header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        
        # PLY Body
        for pid, P in points.items():
            xyz = P["xyz"]
            rgb = P["rgb"]
            # pack: float x3, uchar x3
            f.write(struct.pack("<fffBBB", 
                                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                                int(rgb[0]), int(rgb[1]), int(rgb[2])))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_tum_poses(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load TUM poses: timestamp tx ty tz qx qy qz qw"""
    data = []
    if not path.exists():
        print(f"Warning: Pose file {path} not found. Using identity pose.")
        return np.array([0.0]), np.array([[0,0,0,0,0,0,1]])

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 8: continue
            data.append([float(x) for x in parts])
    data = np.array(data)
    # Sort by timestamp
    idx = np.argsort(data[:, 0])
    return data[idx, 0], data[idx, 1:]

def find_nearest_depth(query_ts: float, depth_dir: Path):
    all_depth_images = list(sorted(depth_dir.glob("*.png")))
    timestamps = [float(image.stem) for image in all_depth_images]
    idx = np.searchsorted(timestamps, query_ts)
    candidates = []
    if idx < len(timestamps): candidates.append(idx)
    if idx > 0: candidates.append(idx - 1)
    
    best_idx = -1
    best_dt = float('inf')
    
    for i in candidates:
        dt = abs(timestamps[i] - query_ts)
        if dt < best_dt:
            best_dt = dt
            best_idx = i
            
    if best_idx != -1:
        nearest_timestamp = timestamps[best_idx]
        print(f"Nearest timestamp: {nearest_timestamp}")
        return all_depth_images[best_idx]
    return None

def get_transform_matrix(pose_data) -> np.ndarray:
    """TUM pose -> 4x4 Matrix"""
    tx, ty, tz, qx, qy, qz, qw = pose_data
    T = np.eye(4)
    r = R.from_quat([qx, qy, qz, qw])
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T

# ----------------- Core Logic -----------------

def process_single_frame(args):
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    color_dir = input_dir / "color"
    depth_dir = input_dir / "aligned_depth"
    
    # 1. Select the specific image
    all_images = sorted(list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg")))
    if args.sample_index >= len(all_images):
        raise ValueError(f"Index {args.sample_index} out of range (Total {len(all_images)})")
    
    target_img_path = all_images[args.sample_index]
    print(f"Processing Single Frame: {target_img_path.name}")

    # Try to parse timestamp from filename
    ts = float(target_img_path.stem)

    # # 2. Load Pose (Optional but good for world alignment)
    # gt_path = input_dir / "groundtruth.txt"
    # if not gt_path.exists(): gt_path = input_dir / "odometry.txt"
    
    # ts_poses, poses_data = load_tum_poses(gt_path)
    # pose_data = find_nearest_pose(ts, ts_poses, poses_data)
    
    # if pose_data is None:
    #     print("Warning: No matching pose found. Using Identity at Origin.")
    #     # tx, ty, tz, qx, qy, qz, qw
    #     pose_data = [0, 0, 0, 0, 0, 0, 1]

    # Coordinate Transforms
    # OpenLORIS Body (ROS): X-fwd, Y-left, Z-up
    # # COLMAP/OpenCV Cam: X-right, Y-down, Z-fwd
    # # We need R_body_to_cam
    # R_b2c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    # T_body_to_cam = np.eye(4); T_body_to_cam[:3, :3] = R_b2c

    # directly use the transform matrix from `trans_matrix.yaml`
    # and ignore the world coordinate (use base_link instead)
    T_camera_to_world = np.array([[9.9792816252667338e-03, 6.5348103708624539e-03, 9.9992885256485176e-01, 2.2648368490900000e-01,],
                                     [-9.9982014658446139e-01, 1.6192923276330706e-02, 9.8723715283343672e-03, -5.1141940356500000e-02,],
                                     [-1.6127257115523985e-02, -9.9984753112121250e-01, 6.6952288046080444e-03, 9.1600000000000004e-01,],
                                     [0., 0., 0., 1.]])
    # T_world_to_camera (COLMAP view matrix)
    T_cw = np.linalg.inv(T_camera_to_world)
    
    # q = R.from_matrix(T_cw[:3, :3]).as_quat() # x,y,z,w
    q = Quaternion(matrix=T_cw[:3, :3])
    # q_colmap = [q[3], q[0], q[1], q[2]] # w,x,y,z
    q_colmap = [q.w, q.x, q.y, q.z]
    t_colmap = T_cw[:3, 3]

    # 3. Load Image & Intrinsics
    im_pil = Image.open(target_img_path)
    W, H = im_pil.size
    
    # Prepare Output Directories
    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse" / "0"
    ensure_dir(images_dir); ensure_dir(sparse_dir)
    
    # Copy Image
    shutil.copy2(target_img_path, images_dir / target_img_path.name)

    # 4. Generate Points3D from Depth
    print("Unprojecting depth map to 3D points...")
    target_depth_path = find_nearest_depth(ts, depth_dir)
    print(f"Nearest depth image: {target_depth_path}")
    depth_path = target_depth_path
    # depth_path = depth_dir / target_img_path.name # Assuming same name # not correct, need fixing
    # if not depth_path.exists():
    #     # OpenLORIS sometimes has different naming or suffix for depth, try .png
    #     depth_path = depth_dir / (target_img_path.stem + ".png")
    
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    d_img = Image.open(depth_path)
    depth_raw = np.array(d_img)
    if depth_raw.dtype != np.uint16 and depth_raw.dtype != np.int32:
        print(f"Warning: Depth image dtype is {depth_raw.dtype}, expected uint16/int32.")
    depth = depth_raw.astype(np.float32) / 1000.0 # mm to meters
    
    # Resize depth if not matching color
    if depth.shape != (H, W):
        depth = np.array(d_img.resize((W, H), Image.NEAREST), dtype=np.float32) / 1000.0

    # Back-projection
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    
    # Create meshgrid
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    
    # Flatten
    us = us.flatten(); vs = vs.flatten(); zs = depth.flatten()
    colors = np.array(im_pil).reshape(-1, 3)

    # Filter invalid depth
    valid_mask = (zs > 0.1) & (zs < args.max_depth)
    
    # Subsample (Stride) to avoid too many points
    stride = args.point_stride
    indices = np.arange(len(us))
    # Combine valid mask and stride
    selection = indices[valid_mask][::stride]
    
    us = us[selection]
    vs = vs[selection]
    zs = zs[selection]
    colors = colors[selection]
    
    # Vectorized Unprojection (Camera Frame)
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    
    # Points in Camera Frame (N, 3)
    pts_cam = np.stack([xs, ys, zs], axis=1)
    
    # Transform to World Frame
    # P_world = R_wc * P_cam + t_wc
    R_wc = T_camera_to_world[:3, :3]
    t_wc = T_camera_to_world[:3, 3]
    
    pts_world = (pts_cam @ R_wc.T) + t_wc

    # 5. Build Dictionaries
    cameras = {
        1: {
            "width": W, "height": H,
            "params": [fx, fy, cx, cy]
        }
    }
    
    images = {
        1: {
            "qvec": q_colmap,
            "tvec": t_colmap,
            "camera_id": 1,
            "name": target_img_path.name
        }
    }
    
    points = {}
    for i in range(len(pts_world)):
        points[i+1] = {
            "xyz": pts_world[i],
            "rgb": colors[i]
        }
        
    print(f"Generated {len(points)} 3D points from depth map.")
    
    # 6. Write Binary
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3d_binary(points, sparse_dir / "points3D.bin")
    
    # 7. Write debug .ply
    write_debug_ply(points, sparse_dir / "debug_points.ply")
    
    print("\nDone.")
    print(f"Output: {out_dir}")
    print("Structure ready for single-frame initialization.")

def main():
    parser = argparse.ArgumentParser(description="OpenLORIS Single-Frame to COLMAP")
    parser.add_argument("--input-dir", required=True, help="Path to scene folder (e.g. market1-1)")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the image to use (0-based)")
    
    # Camera Params (D435i default)
    parser.add_argument("--fx", type=float, default=611.45098876953125)
    parser.add_argument("--fy", type=float, default=611.48571777343750)
    parser.add_argument("--cx", type=float, default=433.20397949218750)
    parser.add_argument("--cy", type=float, default=249.47302246093750)
    
    # Point Cloud Params
    parser.add_argument("--max-depth", type=float, default=5.0, help="Max depth to use (meters)")
    parser.add_argument("--point-stride", type=int, default=5, help="Pixel stride for point generation (higher = fewer points)")

    args = parser.parse_args()
    
    print("Input Arguments:")
    print(f"\tInput dir: {args.input_dir}")
    print(f"\tOutput dir: {args.out_dir}")
    print(f"\t\tsample index: {args.sample_index}\n")
    print(f"\tfx: {args.fx}, fy: {args.fy}, cx: {args.cx}, cy: {args.cy}\n")
    print(f"\tmax-depth: {args.max_depth}")
    print(f"\tpoint-stride: {args.point_stride}\n")
    
    print("Processing...")
    process_single_frame(args)

if __name__ == "__main__":
    main()