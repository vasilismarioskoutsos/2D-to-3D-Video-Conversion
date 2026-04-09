import numpy as np
import glob
import os

# SET YOUR SCENE NAME HERE
SCENE = "bike_3d_result"
depth_dir = r'C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_depths'
# We build the exact path the script wants
output_dir = os.path.join(r'C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_metric', SCENE)

os.makedirs(output_dir, exist_ok=True)
files = glob.glob(os.path.join(depth_dir, "*.npy"))

print(f"Converting {len(files)} files to .npz format...")
for f in files:
    depth_array = np.load(f)
    # The script demands a .npz with these exact keys
    save_path = os.path.join(output_dir, os.path.basename(f).replace('.npy', '.npz'))
    np.savez(save_path, depth=depth_array, fov=60.0)

print(f"Done! Files are ready in: {output_dir}")