import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
flow3r_dir = os.path.join(parent_dir, 'point_cloud')

if flow3r_dir not in sys.path:
    sys.path.append(flow3r_dir)

from point_cloud_flow3r import flow3r, correct_input

import torch.nn.functional as F

def get_points(global_points, flow):
    """
    Extracts the specific pairs of matching 3D coordinates across different frames
    
    flow is a tensor of shape (M, 4) where M is the total matches:
    Column 0: The index of Frame A
    Column 1: The index of the Point in Frame A
    Column 2: The index of Frame B
    Column 3: The index of the Point in Frame B
    """
    # Extract the indices as integers (long) so PyTorch can use them to look up array positions
    frame_A_indices = flow[:, 0].long()
    point_A_indices = flow[:, 1].long()
    
    frame_B_indices = flow[:, 2].long()
    point_B_indices = flow[:, 3].long()
    
    # Pluck the exact 3D coordinates out of the global_points tensor
    # points_A and points_B will both output as shape (M, 3)
    points_A = global_points[frame_A_indices, point_A_indices]
    points_B = global_points[frame_B_indices, point_B_indices]
    
    return points_A, points_B

def matrix_to_rotation_6d(matrix: torch.Tensor):
    """
    Converts rotation matrices into 6D rotation representatio
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rotation_6d_to_matrix(d6: torch.Tensor):
    """
    Converts 6D rotation representation by Gram schmidt orthogonalization into 3x3 rotation matrix
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def safe_tensor(data):
    if data is None:
        return None
    
    if isinstance(data, torch.Tensor):
        return data.detach().clone()
    
    return torch.tensor(data, dtype=torch.float32)

def setup_aligner(flow3r_output, background_mask):
    # background_mask shape: (N_frames, N_points) — 1=background, 0=foreground
    # comes from your SAM 2.1 pipeline, flattened to match point cloud layout
    
    local_points = safe_tensor(flow3r_output.get('local_points'))
    flow = safe_tensor(flow3r_output.get('flow'))
    poses = safe_tensor(flow3r_output.get('camera_poses'))

    device = local_points.device
    flow = flow.to(device)
    poses = poses.to(device)
    background_mask = background_mask.to(device)

    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
        
    if local_points.dim() == 5 and local_points.shape[0] == 1:
        N = local_points.shape[1]
        local_points = local_points.squeeze(0).reshape(N, -1, 3)

    # Build per-match confidence:
    # A match is "background" only if BOTH endpoints are background
    frame_A = flow[:, 0].long()
    point_A = flow[:, 1].long()
    frame_B = flow[:, 2].long()
    point_B = flow[:, 3].long()

    conf_A = background_mask[frame_A, point_A]  # shape (M,)
    conf_B = background_mask[frame_B, point_B]  # shape (M,)
    conf = conf_A * conf_B  # 1 only if both points are background, else 0
    
    rotations_3x3 = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    rotations_6d = matrix_to_rotation_6d(rotations_3x3)
    opt_rotations = nn.Parameter(rotations_6d)
    opt_translations = nn.Parameter(translations)
    
    return local_points, flow, conf, opt_rotations, opt_translations

def generate_matches(world_points, distance_threshold=0.5):
    device = world_points.device
    N = world_points.shape[1]
    
    all_matches = []
    
    for i in range(N - 1):
        frame_A = i
        frame_B = i + 1
        
        pts_A = world_points[0, frame_A].reshape(-1, 3).cpu().numpy()
        pts_B = world_points[0, frame_B].reshape(-1, 3).cpu().numpy()
        
        tree_B = cKDTree(pts_B)
        
        # find the 1 closest neighbor within the threhold 
        distances, indices = tree_B.query(pts_A, k=1, distance_upper_bound=distance_threshold)
        
        # filter out  points that had no match within the threshold
        valid_mask = distances != float('inf')
        
        # image indices for the successful matches
        matched_A = np.where(valid_mask)[0]
        matched_B = indices[matched_A]
        
        for a_idx, b_idx in zip(matched_A, matched_B):
            all_matches.append([frame_A, a_idx, frame_B, b_idx])
            
    print(f"Generated {len(all_matches)} total matches.")
    
    if len(all_matches) == 0:
        print("Warning: No matches found. Adjust the distance threshold.")
        all_matches.append([0, 0, 1, 0])
        
    return torch.tensor(all_matches, dtype=torch.float32, device=device)

def optimizer(local_points, flow_indices, conf, opt_rotations, opt_translations, num_iterations=500):
    optimizer = optim.Adam([opt_rotations, opt_translations], lr=0.01)
    huber = torch.nn.HuberLoss(reduction='none', delta=0.1)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        rotations_3x3 = rotation_6d_to_matrix(opt_rotations)
        global_points = torch.bmm(local_points, rotations_3x3.transpose(1, 2)) + opt_translations.unsqueeze(1)
        points_A, points_B = get_points(global_points, flow_indices)

        # huber per element → sum over xyz → shape (M,)
        per_match_loss = huber(points_A, points_B).sum(dim=-1)
        
        # multiply by conf: foreground matches get 0, background get full loss
        weighted_loss = (per_match_loss * conf).sum()

        weighted_loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            n_active = conf.sum().item()
            print(f"Iteration {iteration}, Loss = {weighted_loss.item():.4f}, Active matches = {n_active:.0f}")

    return opt_rotations, opt_translations

def fuse_and_export_to_3dgs(local_points, video_tensor, opt_rotations, opt_translations, filenames, output_dir, camera_angle_x=0.85):
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        final_rotations = rotation_6d_to_matrix(opt_rotations)
        # put points in correct global space
        global_points = torch.bmm(local_points, final_rotations.transpose(1, 2)) + opt_translations.unsqueeze(1)
    
    # for 4dgs isolate only first frame
    frame_points = global_points[0].detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    
    colors = video_tensor[0, 0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8) 

    # uncomment it only to view it in meshlab
    # frame_points[:, 2] = -frame_points[:, 2]

    combined_data = np.hstack((frame_points, colors))
    
    # meshlab .ply compatability
    ply_path = os.path.join(output_dir, "points3D.ply")
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(frame_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        np.savetxt(f, combined_data, fmt='%.5f %.5f %.5f %d %d %d')

    # export camera poses for all frames
    frames = []
    final_rotations_np = final_rotations.detach().cpu().numpy()
    final_translations_np = opt_translations.detach().cpu().numpy()

    for i in range(len(filenames)):
        transform_matrix = np.eye(4)
        c2w = np.eye(4)
        c2w[:3, :3] = final_rotations_np[i]
        c2w[:3, 3] = final_translations_np[i]
        w2c = np.linalg.inv(c2w)  # SOM expects world-to-camera
        transform_matrix = w2c
        
        frames.append({
            "file_path": filenames[i],
            "transform_matrix": transform_matrix.tolist()
        })
        
    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({
            "camera_angle_x": camera_angle_x,
            "frames": frames
        }, f, indent=4)

def export_depth_maps(local_points, opt_rotations, opt_translations, output_dir):
    # the goal is to populate these 2 folders
    os.makedirs(os.path.join(output_dir, "depth", "1x"), exist_ok=True) # full resolution
    os.makedirs(os.path.join(output_dir, "depth", "2x"), exist_ok=True) # half resolution

    # we need (H*W, 1)

    points = local_points.squeeze(0)
    N_frames, H, W, _ = points.shape

    with torch.no_grad():
        final_rotations = rotation_6d_to_matrix(opt_rotations)
        # transfer points to global space to match the aligner result
        points_flat = points.reshape(N_frames, -1, 3)
        global_points = torch.bmm(points_flat, final_rotations.transpose(1, 2)) + opt_translations.unsqueeze(1)
        global_points = global_points.reshape(N_frames, H, W, 3)

    for i in range(N_frames):
        depth_1x = global_points[i, :, :, 2].detach().cpu().numpy()
        depth_1x = np.clip(depth_1x, 0, None).astype(np.float32) # remove negatives
        depth_1x = depth_1x[:, :, np.newaxis]  # (H, W, 1)

        filename = f"0_{i:05d}.npy"
        np.save(os.path.join(output_dir, "depth", "1x", filename), depth_1x)

        # 2x = half resolution
        depth_2x = depth_1x.reshape(H//2, 2, W//2, 2, 1).mean(axis=(1, 3))
        np.save(os.path.join(output_dir, "depth", "2x", filename), depth_2x)


if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = r"videos\bike_aligner"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"
    
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    input_tensor = correct_input(video, FFMPEG_PATH, frames=7)
    
    filenames = []
    for i in range(input_tensor.shape[1]): # Loop through the number of frames
        filename = f"frame_{i:04d}.png"
        filepath = os.path.join(OUTPUT_NAME, filename)
        
        frame_tensor = input_tensor[0, i].permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_tensor * 255.0).astype(np.uint8)
        
        from PIL import Image
        Image.fromarray(frame_np).save(filepath)
        
        filenames.append(filename)

    output = flow3r(input_tensor)
    
    if output.get('flow') is None:
        # We pass output['points'] which contains the predicted world points
        output['flow'] = generate_matches(output['points'])

    original_local_points = output.get('local_points')

    background_mask = torch.load("background_mask.pt")
    local_points, flow, conf, opt_rotations, opt_translations = setup_aligner(output, background_mask)
    opt_rotations, opt_translations = optimizer(local_points, flow, conf, opt_rotations, opt_translations, num_iterations=500)
    
    fuse_and_export_to_3dgs(local_points, input_tensor, opt_rotations, opt_translations, filenames, OUTPUT_NAME, camera_angle_x=0.85)
    export_depth_maps(original_local_points, opt_rotations, opt_translations, OUTPUT_NAME)

# python lifting\stereo\global_aligner.py