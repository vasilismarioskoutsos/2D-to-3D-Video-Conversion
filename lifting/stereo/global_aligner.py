import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
import json
import os
import sys
import torchvision

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
    Column 0: index of frame A
    Column 1: index of the point in frame A
    Column 2: index of frame B
    Column 3: index of the point in frame B
    """
    # extract indices as integers so PyTorch can use them to look up array positions
    frame_A_indices = flow[:, 0].long()
    point_A_indices = flow[:, 1].long()
    
    frame_B_indices = flow[:, 2].long()
    point_B_indices = flow[:, 3].long()
    
    # get 3D coordinates out of the global_points tensor
    # points_A and points_B (M, 3)
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

def compute_metric_scale(flow3r_local_points, reference_frame_idx=0, reference_metric_depth=None):
    """
    Computes a scalar to convert flow3r's output into metric units.
    Pass in the metric depth map for one reference frame
    Returns the median over valid pixels.
    
    If reference_metric_depth is none we return 1.0 and assume flow3r is already metric
    """
    if reference_metric_depth is None:
        print("No reference metric depth")
        return 1.0
    
    # local_points (1, N_frames, H, W, 3) or (N_frames, H, W, 3)
    pts = flow3r_local_points
    if pts.dim() == 5:
        pts = pts.squeeze(0)
    
    flow3r_depth = pts[reference_frame_idx, :, :, 2].detach().cpu().numpy()
    metric_depth = reference_metric_depth
    
    # only use pixels where both depths are positive and finite
    valid = (flow3r_depth > 0.01) & (metric_depth > 0.01) & np.isfinite(flow3r_depth) & np.isfinite(metric_depth)
    if valid.sum() == 0:
        print("WARNING: no valid pixels for scale alignment. Returning 1.0.")
        return 1.0
    
    ratio = np.median(metric_depth[valid] / flow3r_depth[valid])
    print(f"Computed metric scale ratio: {ratio:.4f}")
    return float(ratio)

def setup_aligner(flow3r_output, background_mask, metric_scale=1.0):
    # background mask shape (N_frames, N_points)
    
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

    # apply metric scale to local_points
    local_points = local_points * metric_scale

    # a match is background only if both endpoints are background
    frame_A = flow[:, 0].long()
    point_A = flow[:, 1].long()
    frame_B = flow[:, 2].long()
    point_B = flow[:, 3].long()

    conf_A = background_mask[frame_A, point_A]  # shape (M,)
    conf_B = background_mask[frame_B, point_B]  # shape (M,)
    conf = conf_A * conf_B  # 1 only if both points are background, else 0
    
    rotations_3x3 = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    # also scale translation
    # units as local_points, so they need the same scale factor applied
    translations = translations * metric_scale

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
    huber = torch.nn.HuberLoss(reduction='none', delta=0.1) # !! delta

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        rotations_3x3 = rotation_6d_to_matrix(opt_rotations)
        global_points = torch.bmm(local_points, rotations_3x3.transpose(1, 2)) + opt_translations.unsqueeze(1)
        points_A, points_B = get_points(global_points, flow_indices)

        per_match_loss = huber(points_A, points_B).sum(dim=-1)
        
        # multiply by conf so foreground matches get 0, background get full loss
        # normalize by number of active matches so the loss magnitude doesnt depend on how many static vs dynamic pixels happen to be in the scene
        n_active = conf.sum().clamp(min=1.0)
        weighted_loss = (per_match_loss * conf).sum() / n_active

        weighted_loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss = {weighted_loss.item():.6f}, Active matches = {n_active.item():.0f}")

    return opt_rotations, opt_translations

def fuse_and_export_to_3dgs(local_points, video_tensor, opt_rotations, opt_translations, filenames, output_dir, fx, fy, cx, cy, image_width, image_height):
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

    # compute camera angle from focal length and image width
    camera_angle_x = 2.0 * np.arctan2(image_width / 2.0, fx)

    for i in range(len(filenames)):
        c2w = np.eye(4)
        c2w[:3, :3] = final_rotations_np[i]
        c2w[:3, 3] = final_translations_np[i]
        w2c = np.linalg.inv(c2w)  # SOM expects world-to-camera
        transform_matrix = w2c
        
        frames.append({
            "file_path": filenames[i],
            "transform_matrix": transform_matrix.tolist()
        })
        
    # write full intrinsics
    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({
            "camera_angle_x": float(camera_angle_x),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w": int(image_width),
            "h": int(image_height),
            "frames": frames
        }, f, indent=4)

def export_depth_maps(local_points, opt_rotations, opt_translations, output_dir, metric_scale=1.0):
    os.makedirs(os.path.join(output_dir, "aligned_depth_anything", "1x"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "aligned_depth_anything", "2x"), exist_ok=True)

    points = local_points.squeeze(0) # (N, H, W, 3) already in camera space
    N_frames, H, W, _ = points.shape

    for i in range(N_frames):
        # apply metricscale so saved depths are in the same units as the rest of the pipeline
        depth_1x = points[i, :, :, 2].detach().cpu().numpy() * metric_scale
        
        depth_1x = np.where(np.isfinite(depth_1x) & (depth_1x > 0), depth_1x, 0.0).astype(np.float32)

        filename = f"frame_{i:04d}.npy"
        np.save(os.path.join(output_dir, "aligned_depth_anything", "1x", filename), depth_1x)

        # som uses for its low resolution loss 2x 2x downsample
        depth_2x = depth_1x.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))
        np.save(os.path.join(output_dir, "aligned_depth_anything", "2x", filename), depth_2x)

def rotmat_to_qvec(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

def export_droid_recon(opt_rotations, opt_translations, output_dir, H, W, fx, fy, cx, cy):
    final_rotations = rotation_6d_to_matrix(opt_rotations).detach().cpu().numpy()
    final_translations = opt_translations.detach().cpu().numpy()

    N = len(final_rotations)
    traj_c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        c2w = np.eye(4)
        c2w[:3, :3] = final_rotations[i]
        c2w[:3, 3] = final_translations[i]
        traj_c2w[i] = c2w

    recon = {
        "traj_c2w": traj_c2w,
        "img_shape": (H, W),
        "intrinsics": (fx, fy, cx, cy),
        "tstamps": np.arange(N, dtype=np.float32)
    }
    np.save(os.path.join(output_dir, "droid_recon.npy"), recon)
    print(f"Exported droid_recon.npy with {N} frames, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

def get_intrinsics_from_flow3r(flow3r_output, image_width, image_height):
    """
    Recovers intrinsics from flow3r's local_points by inverting the pinhole projection
    For each pixel (u, v) with cameraspace point (X, Y, Z):
        u = fx * X/Z + cx
        v = fy * Y/Z + cy
    """
    K = flow3r_output.get('intrinsics', None)
    print(K)
    
    # recover intrinsics from local_points geometry
    local_points = flow3r_output.get('local_points')
    if local_points is None:
        print("WARNING: flow3r has no local_points either. Using 60-degree FoV fallback.")
        fx = (image_width / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
        return fx, fx, image_width / 2.0, image_height / 2.0
    
    # use the first frame for intrinsics estimation
    pts = local_points
    if pts.dim() == 5:
        pts = pts.squeeze(0)
    pts_frame0 = pts[0].detach().cpu().numpy() # (H, W, 3)
    
    H, W, _ = pts_frame0.shape
    cx = W / 2.0
    cy = H / 2.0
    
    # build pixel grid
    us, vs = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    u_offsets = (us - cx).astype(np.float32) # (H, W)
    v_offsets = (vs - cy).astype(np.float32)
    
    X = pts_frame0[..., 0]
    Y = pts_frame0[..., 1]
    Z = pts_frame0[..., 2]
    
    # only use pixels with Z > 0 and not near image center
    valid = (Z > 0.01) & np.isfinite(Z) & (np.abs(X) > 1e-6) & (np.abs(Y) > 1e-6)
    valid &= np.abs(u_offsets) > 10 # exclude near principal point pixels because numerically unstable
    valid &= np.abs(v_offsets) > 10
    
    if valid.sum() < 100:
        print("WARNING: not enough valid pixels for intrinsics recovery. Using 60-degree FoV fallback.")
        fx = (image_width / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
        return fx, fx, cx, cy
    
    fx_estimates = u_offsets[valid] * Z[valid] / X[valid]
    fy_estimates = v_offsets[valid] * Z[valid] / Y[valid]
    
    # use median for robustness against outlier pixels
    fx = float(np.median(fx_estimates))
    fy = float(np.median(fy_estimates))
    
    print(f"Recovered intrinsics from local_points geometry: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    return fx, fy, cx, cy

def estimate_poses_from_matches(local_points, flow, background_mask, metric_scale=1.0):
    """
    Initialize camera poses by finding the rigid transformation that best aligns background points between consecutive frames. 
    Bypasses Flow3r's pose estimates
    """
    pts = local_points.squeeze(0) if local_points.dim() == 5 else local_points
    pts_flat = pts.reshape(pts.shape[0], -1, 3) * metric_scale
    
    N = pts_flat.shape[0]
    device = pts_flat.device

    # move auxiliary tensors to the same device as pts_flat
    flow = flow.to(device)
    background_mask = background_mask.to(device)
    
    # set frame 0 pose to identity
    poses = [torch.eye(4, device=device, dtype=pts_flat.dtype)]
    
    flow_np = flow.cpu().numpy() if flow.is_cuda else flow.numpy()
    
    for i in range(1, N):
        # find matches between frame (i-1) and frame i, restricted to background
        mask = (flow_np[:, 0] == i-1) & (flow_np[:, 2] == i)
        if mask.sum() < 10:
            # not enough matches, copy previous pose
            poses.append(poses[-1].clone())
            continue
        
        sub = flow[mask]
        pa_idx = sub[:, 1].long()
        pb_idx = sub[:, 3].long()
        
        # restrict to background matches
        bg_a = background_mask[i-1, pa_idx]
        bg_b = background_mask[i, pb_idx]
        bg_match = (bg_a * bg_b) > 0.5
        
        if bg_match.sum() < 10:
            poses.append(poses[-1].clone())
            continue
        
        pa = pts_flat[i-1, pa_idx[bg_match]] # (M, 3) in frame i-1 camera space
        pb = pts_flat[i, pb_idx[bg_match]] # (M, 3) in frame i camera space
        
        # apply previous c2w to put pa into world space
        prev_c2w = poses[-1]
        pa_world = (prev_c2w[:3, :3] @ pa.T).T + prev_c2w[:3, 3]
        
        # find R, t so that pb in frame i camera space) maps to pa world space
        centroid_a = pa_world.mean(dim=0)
        centroid_b = pb.mean(dim=0)
        
        a_centered = pa_world - centroid_a
        b_centered = pb - centroid_b
        
        H = b_centered.T @ a_centered # (3, 3)
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        # ensure proper rotation
        if torch.linalg.det(R) < 0:
            Vt[-1, :] = -Vt[-1, :]
            R = Vt.T @ U.T
        
        t = centroid_a - R @ centroid_b
        
        c2w = torch.eye(4, device=device, dtype=pts_flat.dtype)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        poses.append(c2w)
    
    return torch.stack(poses, dim=0)

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = r"videos\bike_4dgs"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"
    NUM_FRAMES = 24
    n_stride = 2 # every nth frame
    
    os.makedirs(OUTPUT_NAME, exist_ok=True)
    input_tensor = correct_input(video, FFMPEG_PATH, frames=NUM_FRAMES * n_stride)
    input_tensor = input_tensor[:, ::n_stride] # stride dim 1 time
    input_tensor = input_tensor[:, :NUM_FRAMES]
    images_dir = os.path.join(OUTPUT_NAME, "images/1x")
    os.makedirs(images_dir, exist_ok=True)
    filenames = []
    for i in range(input_tensor.shape[1]): # Loop through the number of frames
        filename = f"frame_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        
        frame_tensor = input_tensor[0, i].permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_tensor * 255.0).astype(np.uint8)
        
        from PIL import Image
        Image.fromarray(frame_np).save(filepath)
        
        filenames.append(filename)

    output = flow3r(input_tensor)
    
    if output.get('flow') is None:
        output['flow'] = generate_matches(output['points'])

    original_local_points = output.get('local_points')

    image_width = 518
    image_height = 518

    # load UniDepth outputs
    unidepth_depth_path = r"C:\vasilis\2D-to-3D-Video-Conversion\unidepth_frame0.npy"
    unidepth_intrinsics_path = r"C:\vasilis\2D-to-3D-Video-Conversion\unidepth_intrinsics.npy"

    if os.path.exists(unidepth_depth_path):
        reference_metric_depth = np.load(unidepth_depth_path)
        print(f"Loaded UniDepth metric depth, shape={reference_metric_depth.shape}, "
            f"median={np.median(reference_metric_depth):.3f}m")
    else:
        reference_metric_depth = None
        print(f"No {unidepth_depth_path} found; using metric_scale=1.0")

    if os.path.exists(unidepth_intrinsics_path):
        K = np.load(unidepth_intrinsics_path)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        print(f"Loaded UniDepth intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    else:
        fx, fy, cx, cy = get_intrinsics_from_flow3r(output, image_width, image_height)

    metric_scale = 1.0

    background_mask = torch.load("background_mask.pt")
    #local_points, flow, conf, opt_rotations, opt_translations = setup_aligner(output, background_mask, metric_scale=metric_scale)
    init_poses = estimate_poses_from_matches(
        safe_tensor(output.get('local_points')),
        safe_tensor(output.get('flow')),
        background_mask,
        metric_scale=metric_scale
    )

    # then run setup_aligner but override poses with these
    output_with_better_poses = dict(output)
    output_with_better_poses['camera_poses'] = init_poses

    local_points, flow, conf, opt_rotations, opt_translations = setup_aligner(
        output_with_better_poses, background_mask, metric_scale=metric_scale
    )
    opt_rotations, opt_translations = optimizer(local_points, flow, conf, opt_rotations, opt_translations, num_iterations=500)
    
    fuse_and_export_to_3dgs(local_points, input_tensor, opt_rotations, opt_translations, filenames, OUTPUT_NAME, fx=fx, fy=fy, cx=cx, cy=cy, image_width=image_width, image_height=image_height)
    export_depth_maps(original_local_points, opt_rotations, opt_translations, OUTPUT_NAME, metric_scale=metric_scale)
    export_droid_recon(opt_rotations, opt_translations, OUTPUT_NAME, H=image_height, W=image_width, fx=fx, fy=fy, cx=cx, cy=cy)

# python lifting\stereo\global_aligner.py