import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import open3d as o3d
import json
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
flow3r_dir = os.path.join(current_dir, 'point_cloud')
if flow3r_dir not in sys.path:
    sys.path.append(flow3r_dir)

from point_cloud_flow3r import flow3r, correct_input

def setup_aligner(flow3r_output):
    # freeze the constants
    local_points = torch.tensor(flow3r_output['local_points'], dtype=torch.float32, requires_grad=False)
    flow = torch.tensor(flow3r_output['flow'], dtype=torch.float32, requires_grad=False)
    conf = torch.tensor(flow3r_output['conf'], dtype=torch.float32, requires_grad=False)
    
    # extract local colors for export later
    local_colors = np.array(flow3r_output.get('local_colors', []))
    
    # assume (N, 4, 4)
    poses = torch.tensor(flow3r_output['camera_poses'], dtype=torch.float32)
    
    # split the 4x4 matrix into a 3x3 rotation matrix and a 3D translation vector
    rotations_3x3 = poses[:, :3, :3] # (N, 3, 3)
    translations = poses[:, :3, 3] # (N, 3)
    
    # convert 3x3 rotation matrices into a safer 6D continuous format
    rotations_6d = matrix_to_rotation_6d(rotations_3x3) # (N, 6)
    
    opt_rotations = nn.Parameter(rotations_6d)
    opt_translations = nn.Parameter(translations)
    
    return local_points, flow, conf, opt_rotations, opt_translations, local_colors

def optimizer(local_points, flow_indices, conf, opt_rotations, opt_translations, num_iterations=500):
    optimizer = optim.Adam([opt_rotations, opt_translations], lr=0.01)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # 6D representation back to 3x3 rotation matrices
        rotations_3x3 = rotation_6d_to_matrix(opt_rotations)

        # local points to global space: Global = (Rotation * Local) + Translation
        global_points = torch.bmm(local_points, rotations_3x3.transpose(1, 2)) + opt_translations.unsqueeze(1)

        # retrieve the 3D coords for points across different frames
        points_A, points_B = get_points(global_points, flow_indices) 

        # loss = euclidean distance
        squared_distances = torch.sum((points_A - points_B) ** 2, dim=-1)
        
        # give more focus to distances with higher condidense for the loss
        weighted_distances = squared_distances * conf
        
        # sum the error across all matched points to get a single loss
        loss = torch.sum(weighted_distances)

        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss = {loss.item():.4f}")

    return opt_rotations, opt_translations

def fuse_and_export_to_3dgs(local_points, local_colors, opt_rotations, opt_translations, filenames, output_dir, camera_angle_x=0.85):
    with torch.no_grad():
        final_rotations = rotation_6d_to_matrix(opt_rotations)
        # put points in correct global space
        global_points = torch.bmm(local_points, final_rotations.transpose(1, 2)) + opt_translations.unsqueeze(1)
    
    # detach and cast to 64-bit float for open3d
    global_points_np = global_points.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    
    # normalize colors to 0.0 - 1.0 and cast to 64-bit float
    colors_np = (local_colors.reshape(-1, 3) / 255.0).astype(np.float64) 

    # load open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np) 

    # remove points with more than 2.0 standrd deviation
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned_pcd = pcd.select_by_index(ind)

    o3d.io.write_point_cloud(f"{output_dir}/points3D.ply", cleaned_pcd)

    # save json
    frames = []
    final_rotations_np = final_rotations.detach().cpu().numpy()
    final_translations_np = opt_translations.detach().cpu().numpy()

    for i in range(len(filenames)):
        # rebuild the 4x4 transformation matrix for 3DGS
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = final_rotations_np[i]
        transform_matrix[:3, 3] = final_translations_np[i]
        
        frames.append({
            "file_path": filenames[i],
            "transform_matrix": transform_matrix.tolist()
        })
        
    # include camera intrinsics in the root of the json
    with open(f"{output_dir}/transforms.json", "w") as f:
        json.dump({
            "camera_angle_x": camera_angle_x,
            "frames": frames
        }, f, indent=4)

def get_matrix_for_frame(frame):
    pass

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_flow3r"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"

    input = correct_input(video, FFMPEG_PATH)
    output = flow3r(input)