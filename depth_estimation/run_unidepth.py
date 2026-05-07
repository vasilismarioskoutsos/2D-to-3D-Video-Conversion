import os
import sys
import numpy as np
import torch
import imageio.v3 as iio
import subprocess

# add UniDepth to path
unidepth_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UniDepth")
if unidepth_dir not in sys.path:
    sys.path.insert(0, unidepth_dir)

from unidepth.models import UniDepthV2

# config — defaults match your pipeline layout
VIDEO_PATH = r"videos\bike_cut.mp4"
IMAGES_DIR = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\images\1x"
DEPTH_OUT_1X = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\aligned_depth_anything\1x"
DEPTH_OUT_2X = r"C:\vasilis\2D-to-3D-Video-Conversion\videos\bike_4dgs\aligned_depth_anything\2x"
SINGLE_FRAME_DEPTH_OUT = r"C:\vasilis\2D-to-3D-Video-Conversion\unidepth_frame0.npy"
SINGLE_FRAME_INTRINSICS_OUT = r"C:\vasilis\2D-to-3D-Video-Conversion\unidepth_intrinsics.npy"
FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
NUM_FRAMES = 24
TARGET_RES = 518  # square resolution matching the rest of the pipeline

# mode: "all" runs on every saved frame and writes to aligned_depth_anything/1x and 2x.
# "single" extracts frame 0 from the source video and writes the legacy
# unidepth_frame0.npy + unidepth_intrinsics.npy used by global_aligner.
MODE = "all"


def load_model(device):
    print("Loading UniDepthV2...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model


def infer_depth(model, img_uint8_hw3, device):
    # img_uint8_hw3: (H, W, 3) uint8 numpy array
    if img_uint8_hw3.ndim == 2:
        img_uint8_hw3 = np.stack([img_uint8_hw3] * 3, axis=-1)
    elif img_uint8_hw3.shape[2] == 4:
        img_uint8_hw3 = img_uint8_hw3[..., :3]
    
    rgb_tensor = torch.from_numpy(img_uint8_hw3.copy()).permute(2, 0, 1).float().to(device)
    
    with torch.no_grad():
        result = model.infer(rgb_tensor)
    
    depth = result["depth"].squeeze().cpu().numpy().astype(np.float32)
    intrinsics = result["intrinsics"].squeeze().cpu().numpy().astype(np.float32)
    return depth, intrinsics


def extract_frame_from_video(video_path, ffmpeg_path, target_res):
    # uses ffmpeg to pull frame 0 at the target square resolution with letterbox padding
    # to match what the rest of the pipeline expects
    print(f"Extracting frame 0 from {video_path} at {target_res}x{target_res}...")
    cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vf", f"scale={target_res}:{target_res}:force_original_aspect_ratio=decrease,"
               f"pad={target_res}:{target_res}:(ow-iw)/2:(oh-ih)/2",
        "-vframes", "1",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(target_res, target_res, 3)
    print(f"Frame 0 shape: {frame.shape}")
    return frame


def save_depth_pair(depth, out_1x_dir, out_2x_dir, frame_idx):
    # writes 1x .npy and a 2x mean-pooled .npy to match SoM's data layout
    os.makedirs(out_1x_dir, exist_ok=True)
    os.makedirs(out_2x_dir, exist_ok=True)
    
    out_1x_path = os.path.join(out_1x_dir, f"frame_{frame_idx:04d}.npy")
    np.save(out_1x_path, depth)
    
    H, W = depth.shape
    depth_2x = depth.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    out_2x_path = os.path.join(out_2x_dir, f"frame_{frame_idx:04d}.npy")
    np.save(out_2x_path, depth_2x)


def run_all_frames(model, device):
    # processes every saved frame in IMAGES_DIR and writes depth maps in SoM's expected layout
    print(f"Running on all {NUM_FRAMES} frames in {IMAGES_DIR}")
    
    for i in range(NUM_FRAMES):
        img_path = os.path.join(IMAGES_DIR, f"frame_{i:04d}.png")
        if not os.path.exists(img_path):
            print(f"Frame {i:2d}: image not found at {img_path}, skipping")
            continue
        
        img = iio.imread(img_path)
        depth, _ = infer_depth(model, img, device)
        save_depth_pair(depth, DEPTH_OUT_1X, DEPTH_OUT_2X, i)
        
        print(f"Frame {i:2d}: depth median = {np.median(depth):.3f}m, "
              f"range [{depth.min():.2f}, {depth.max():.2f}]")
    
    print(f"\nDone. Wrote {NUM_FRAMES} depth maps to {DEPTH_OUT_1X}")


def run_single_frame(model, device):
    # legacy mode: pulls frame 0 directly from the source video, saves to flat .npy files
    # used by global_aligner.py for metric scale calibration and intrinsics
    frame = extract_frame_from_video(VIDEO_PATH, FFMPEG_PATH, TARGET_RES)
    
    print("Running UniDepth inference...")
    depth, intrinsics = infer_depth(model, frame, device)
    
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.3f} to {depth.max():.3f} meters")
    print(f"Depth median: {np.median(depth):.3f} meters")
    print("Intrinsics:")
    print(intrinsics)
    print(f"  fx = {intrinsics[0, 0]:.2f}")
    print(f"  fy = {intrinsics[1, 1]:.2f}")
    print(f"  cx = {intrinsics[0, 2]:.2f}")
    print(f"  cy = {intrinsics[1, 2]:.2f}")
    
    np.save(SINGLE_FRAME_DEPTH_OUT, depth)
    np.save(SINGLE_FRAME_INTRINSICS_OUT, intrinsics)
    print(f"Saved depth to {SINGLE_FRAME_DEPTH_OUT}")
    print(f"Saved intrinsics to {SINGLE_FRAME_INTRINSICS_OUT}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    
    if MODE == "all":
        run_all_frames(model, device)
    elif MODE == "single":
        run_single_frame(model, device)
    else:
        raise ValueError(f"Unknown MODE: {MODE}. Use 'all' or 'single'.")

# python depth_estimation\run_unidepth.py