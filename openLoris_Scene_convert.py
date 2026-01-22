import os, sys
import argparse
import open3d as o3d
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert depth/RGB information from OpenLORIS-Scene to COLMAP-strutured 3DGS input (For specified timestamp)."
    )
    parser.add_argument(
        "--depth_file",
        required=True,
        type=str,
        help="Path to the depth file (Recommended: use the aligned depth information)."
    )
    parser.add_argument(
        "--image_file",
        required=True,
        type=str,
        help="Path to the RGB image."
    )
    parser.add_argument(
        "--output_ply",
        required=True,
        type=str,
        help="Path to store the converted point cloud content."
    )
    return parser.parse_args()

def generate_point_cloud(rgb_path, depth_path, output_ply_path):
    color_raw = cv2.imread(rgb_path)
    color_raw = cv2.cvtColor(color_raw,cv2.COLOR_BGR2RGB)
    print(f"Color image:\n{color_raw.shape[:2]}")
    
    depth_raw = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
    print(f"Depth info:\n{depth_raw.shape[:2]}")
    
    color_image = o3d.geometry.Image(color_raw)
    depth_image = o3d.geometry.Image(depth_raw)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1000.0,
        depth_trunc=65.0,
        convert_rgb_to_intensity=False,
    )
    # Intrinsic params are from sensors.yaml
    # Use the params from camera(d400_color_optical_frame),
    # since we are using the aligned depth image.
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=848,height=480,
        fx=611.45098876953125,fy=611.48571777343750,
        cx=433.20397949218750,cy=249.47302246093750,
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,intrinsics
    )
    
    colors = np.asarray(pcd.colors)
    if len(colors)>0:
        print("----- Conducting color verifying -----")
        indices = np.random.choice(len(colors),5)
        for idx in indices:
            r, g, b = colors[idx]
            print(f"point {idx}: R - {r:.2f}, G - {g:.2f}, B - {b:.2f}")
        avg_r = np.mean(colors[:,0])
        avg_g = np.mean(colors[:,1])
        avg_b = np.mean(colors[:,2])
        print("Average color:")
        print(f"\tR : {avg_r}")
        print(f"\tG : {avg_g}")
        print(f"\tB : {avg_b}")
    
    # Transform pcd from camera frame to base_link frame
    # use transform matrix from trans_matrix.yaml
    #   parent frame: base_link
    #   chile frame: d400_color_optical_frame
    T_base_camera = np.array([
        [9.9792816252667338e-03, 6.5348103708624539e-03, 9.9992885256485176e-01, 2.2648368490900000e-01],
        [-9.9982014658446139e-01, 1.6192923276330706e-02, 9.8723715283343672e-03, -5.1141940356500000e-02],
        [-1.6127257115523985e-02, -9.9984753112121250e-01, 6.6952288046080444e-03, 9.1600000000000004e-01],
        [0., 0., 0., 1.]
    ])
    pcd.transform(T_base_camera)
    # Visualize before write out
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [pcd,axes],
        window_name="Point Cloud Viewer",
        mesh_show_back_face=False,
        mesh_show_wireframe=False,
    )
    o3d.io.write_point_cloud(output_ply_path,pcd)
    print(f"Pointcloud has been save to {output_ply_path}\n\tNumber of points: {len(pcd.points)}")

def main():
    args = parse_args()
    generate_point_cloud(args.image_file,args.depth_file,args.output_ply)
    sys.exit(0)
    
    
if __name__=='__main__':
    main()
