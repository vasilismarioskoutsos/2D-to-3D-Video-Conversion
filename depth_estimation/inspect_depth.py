import numpy as np
import os
import imageio.v3 as iio

DEPTH_DIR = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\aligned_depth_anything\1x"
MASKS_DIR = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\masks\1x"

print(f"{'Frame':>5} | {'rider_med_depth':>15} | {'bg_med_depth':>15} | {'rider/bg ratio':>15}")
print("-" * 60)
for i in range(24):
    depth = np.load(os.path.join(DEPTH_DIR, f"frame_{i:04d}.npy"))
    mask = iio.imread(os.path.join(MASKS_DIR, f"frame_{i:04d}.png"))
    if mask.ndim == 3: mask = mask[..., 0]
    fg = mask > 127
    
    if fg.sum() == 0:
        print(f"{i:>5} | {'no foreground':>15}")
        continue
    
    rider_d = np.median(depth[fg])
    bg_d = np.median(depth[~fg & (depth > 0)])
    ratio = rider_d / bg_d if bg_d > 0 else 0
    
    print(f"{i:>5} | {rider_d:>15.3f} | {bg_d:>15.3f} |{ratio:>15.3f}")