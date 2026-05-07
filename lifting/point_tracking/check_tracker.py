import numpy as np
import os, glob

tracks_dir = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\bootstapir\1x"
files = sorted(glob.glob(os.path.join(tracks_dir, "*.npy")))
print(f"Total track files: {len(files)} (expected 576)")

# Check a sample file
sample = np.load(files[0])
print(f"\nSample file: {os.path.basename(files[0])}")
print(f"Shape: {sample.shape}")
print(f"Columns: x, y, occ_logit, dist_logit")
print(f"x range: {sample[:, 0].min():.1f} to {sample[:, 0].max():.1f} (should be ~0-518)")
print(f"y range: {sample[:, 1].min():.1f} to {sample[:, 1].max():.1f} (should be ~0-518)")
print(f"occ_logit range: {sample[:, 2].min():.1f} to {sample[:, 2].max():.1f}")
print(f"dist_logit range: {sample[:, 3].min():.1f} to {sample[:, 3].max():.1f}")

# Verify N is constant within each query frame (REQUIRED by SoM)
print("\n=== N CONSTANCY CHECK ===")
all_consistent = True
for q in range(24):
    files_for_q = sorted(glob.glob(os.path.join(tracks_dir, f"frame_{q:04d}_frame_*.npy")))
    sizes = [np.load(f).shape[0] for f in files_for_q]
    if len(set(sizes)) > 1:
        print(f"  Query {q}: INCONSISTENT - sizes range from {min(sizes)} to {max(sizes)}")
        all_consistent = False
    elif len(files_for_q) != 24:
        print(f"  Query {q}: WRONG TARGET COUNT - {len(files_for_q)} (expected 24)")
        all_consistent = False
if all_consistent:
    print("All query frames have constant N across all 24 targets. SoM-compliant.")

# Verify self-pair files exist (frame_X_frame_X.npy)
print("\n=== SELF-PAIR CHECK ===")
for q in range(24):
    self_pair_path = os.path.join(tracks_dir, f"frame_{q:04d}_frame_{q:04d}.npy")
    if not os.path.exists(self_pair_path):
        print(f"  MISSING: frame_{q:04d}_frame_{q:04d}.npy")
print("Self-pairs verified.")