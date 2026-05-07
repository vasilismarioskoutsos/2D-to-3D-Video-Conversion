import os
import cv2
import numpy as np
import imageio.v3 as iio

TRACKS_DIR = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\bootstapir\1x"
IMAGES_DIR = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\images\1x"
MASKS_DIR = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\masks\1x"
OUTPUT_PATH = "som_tracks_visualization.mp4"
NUM_FRAMES = 24
QUERY_FRAME = 12 # which query frame's tracks to visualize across all targets

# load original video frames
images = []
for i in range(NUM_FRAMES):
    img = iio.imread(os.path.join(IMAGES_DIR, f"frame_{i:04d}.png"))
    images.append(img)

# load fg masks so we can color foreground tracks differently from background)
masks = []
for i in range(NUM_FRAMES):
    m = iio.imread(os.path.join(MASKS_DIR, f"frame_{i:04d}.png"))
    if m.ndim == 3:
        m = m[..., 0]
    masks.append(m > 127)

# load query points
query_path = os.path.join(TRACKS_DIR, f"frame_{QUERY_FRAME:04d}_frame_{QUERY_FRAME:04d}.npy")
query_array = np.load(query_path) # (N, 4)
query_xy = query_array[:, :2]
N = len(query_xy)

# determine which query points are foreground vs background based on mask at QUERY_FRAME
fg_query = masks[QUERY_FRAME][query_xy[:, 1].astype(int), query_xy[:, 0].astype(int)]

# load all target tracks for this query
all_tracks = []
all_vis = [] # (N,) visibility booleans
for t in range(NUM_FRAMES):
    path = os.path.join(TRACKS_DIR, f"frame_{QUERY_FRAME:04d}_frame_{t:04d}.npy")
    arr = np.load(path)
    all_tracks.append(arr[:, :2])

    # occlusion logit < 0 means visible
    all_vis.append(arr[:, 2] < 0)

# render each frame with the tracks overlaid
H, W = images[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, 12.0, (W, H))

# keep a trail of past positions so we can see motion
trail_length = 5

for t in range(NUM_FRAMES):
    frame = images[t].copy()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # draw tracks at current frame
    for i in range(N):
        x, y = all_tracks[t][i]
        x_int, y_int = int(x), int(y)
        if not (0 <= x_int < W and 0 <= y_int < H):
            continue
        
        if not all_vis[t][i]:
            continue # don't draw occluded tracks
        
        color = (0, 0, 255) if fg_query[i] else (0, 255, 0)
        cv2.circle(frame_bgr, (x_int, y_int), radius=2, color=color, thickness=-1)
        
        # draw a trail showing where the point was in previous frames
        for trail_t in range(max(0, t - trail_length), t):
            if all_vis[trail_t][i]:
                tx, ty = all_tracks[trail_t][i]
                tx_int, ty_int = int(tx), int(ty)
                if 0 <= tx_int < W and 0 <= ty_int < H:
                    # fade older trail points
                    alpha = (trail_t - (t - trail_length)) / trail_length
                    fade_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(frame_bgr, (tx_int, ty_int), radius=1, color=fade_color, thickness=-1)
    
    # add frame number for reference
    cv2.putText(frame_bgr, f"Frame {t} (query frame {QUERY_FRAME})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"Red=fg ({int(fg_query.sum())}) Green=bg ({int((~fg_query).sum())})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    out_video.write(frame_bgr)

out_video.release()
print(f"Saved track visualization to {OUTPUT_PATH}")