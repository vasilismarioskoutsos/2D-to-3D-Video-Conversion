import numpy as np
import torch

# === Camera trajectory ===
recon = np.load(r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\droid_recon.npy", 
                allow_pickle=True).item()
traj = recon['traj_c2w']
print("=== CAMERA TRAJECTORY ===")
print(f"Num cameras: {len(traj)}")
print(f"Intrinsics: {recon['intrinsics']}")
print(f"Image shape: {recon['img_shape']}")
print(f"First camera position: {traj[0, :3, 3]}")
print(f"Last camera position:  {traj[-1, :3, 3]}")

# otal path length and net displacement
distances = []
for i in range(len(traj)-1):
    dist = np.linalg.norm(traj[i+1, :3, 3] - traj[i, :3, 3])
    distances.append(dist)
total_path = sum(distances)
net_disp = np.linalg.norm(traj[-1, :3, 3] - traj[0, :3, 3])
print(f"Total path length: {total_path:.4f}")
print(f"Net displacement (start to end): {net_disp:.4f}")
print(f"Mean per-frame motion: {np.mean(distances):.4f}")
print(f"Median per-frame motion: {np.median(distances):.4f}")
print(f"Max per-frame motion: {np.max(distances):.4f}")
print(f"Min per-frame motion: {np.min(distances):.4f}")

# check that rotations are valid rotation matrices
print("\n=== ROTATION SANITY ===")
for i in [0, len(traj)//2, len(traj)-1]:
    R = traj[i, :3, :3]
    det = np.linalg.det(R)
    orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
    print(f"Frame {i}: det(R)={det:.6f} (should be 1.0), "
          f"||R R^T - I||={orthogonality_error:.6f} (should be ~0)")

# === Depth checks across multiple frames ===
print("\n=== DEPTH CHECKS ===")
import os
depth_dir = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\aligned_depth_anything\1x"
for frame_idx in [0, len(traj)//2, len(traj)-1]:
    path = os.path.join(depth_dir, f"frame_{frame_idx:04d}.npy")
    if not os.path.exists(path):
        print(f"  frame_{frame_idx:04d}.npy: MISSING")
        continue
    d = np.load(path)
    valid = d > 0
    nonzero_pct = 100.0 * valid.sum() / d.size
    mean_valid = d[valid].mean() if valid.any() else 0.0
    print(f"  frame_{frame_idx:04d}: shape={d.shape}, "
        f"min={d.min():.4f}, max={d.max():.4f}, "
        f"mean(valid)={mean_valid:.4f}, "
        f"nonzero={nonzero_pct:.1f}%")

# === Cross-check intrinsics consistency ===
print("\n=== INTRINSICS CROSS-CHECK ===")
import json
transforms_path = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\transforms.json"
with open(transforms_path) as f:
    transforms = json.load(f)
print(f"transforms.json: fx={transforms.get('fl_x')}, fy={transforms.get('fl_y')}, "
      f"cx={transforms.get('cx')}, cy={transforms.get('cy')}")
print(f"droid_recon.npy: {recon['intrinsics']}")
print(f"They should match.")

# python lifting\stereo\check_aligner.py