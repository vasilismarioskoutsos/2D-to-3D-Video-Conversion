import os
import sys
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import imageio.v3 as iio

cotracker_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cotracker3", "co-tracker"
)
if cotracker_root not in sys.path:
    sys.path.insert(0, cotracker_root)

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

TAPIR_VISIBLE_LOGIT = -10.0
TAPIR_OCCLUDED_LOGIT = 10.0

# tracks beyond this many frames from the query are marked occluded because the are unreliable
MAX_TRACK_DISTANCE = 16

def run_cotracker(checkpoint_path, video_tensor, FRAME_SIZE):
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker = CoTrackerPredictor(checkpoint=checkpoint_path)
    cotracker = cotracker.to(device)
    cotracker.eval()
    print("CoTracker loaded successfully")

    # move video to same device as the model so chunked inference doesn't fail with input typeand weight type
    video_tensor = video_tensor.to(device)

    total_frames = video_tensor.shape[1]
    start_frame = 0

    total_tracks = []
    total_visibility = []
    next_queries = None # we will use this variable to pass points between chunks

    for end_frame in range(FRAME_SIZE, total_frames + FRAME_SIZE, FRAME_SIZE):
        if end_frame > total_frames:
            end_frame = total_frames
        video_chunk = video_tensor[:, start_frame:end_frame, :, :, :]

        with torch.no_grad():
            if next_queries is None: # for the first chunk only init coTracker
                pred_tracks, pred_visibility = cotracker(
                    video=video_chunk, 
                    grid_size=10,
                    grid_query_frame=0
                )
            else:
                pred_tracks, pred_visibility = cotracker(
                    video=video_chunk, 
                    queries=next_queries
                )
        
        total_tracks.append(pred_tracks)
        total_visibility.append(pred_visibility)

        # get the coords from the last frame of this chunk
        last_coords = pred_tracks[:, -1, :, :]
        
        # cotracker queries need to be [time, x, y] 
        time_tensor = torch.zeros_like(last_coords[:, :, :1]) 
        next_queries = torch.cat([time_tensor, last_coords], dim=2)

        start_frame = end_frame

    return total_tracks, total_visibility

def visualise_cotracker(combined_tracks, combined_visibility, video_tensor, OUTPUT_NAME):
    # set up coTracker visualizer
    vis = Visualizer(
        save_dir=r"videos", 
        pad_value=120, 
        linewidth=2
    )

    vis.visualize(
        video=video_tensor,
        tracks=combined_tracks,
        visibility=combined_visibility,
        filename=OUTPUT_NAME
    )

def load_video_for_som(video_path, target_h=518, target_w=518, max_frames=None, stride=1):
    video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    if stride > 1:
        video_frames = video_frames[::stride]
    if max_frames is not None:
        video_frames = video_frames[:max_frames]
        
    video_tensor = video_frames.permute(0, 3, 1, 2).float()
    video_tensor = F.interpolate(video_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor

def get_source_dimensions(video_path):
    # peek at one frame to learn the original aspect ratio
    video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    return int(video_frames.shape[2]), int(video_frames.shape[1]) # (width, height)

def compute_valid_content_mask(target_h, target_w, source_w, source_h):
    scale = min(target_w / source_w, target_h / source_h)
    content_w = int(source_w * scale)
    content_h = int(source_h * scale)
    pad_x = (target_w - content_w) // 2
    pad_y = (target_h - content_h) // 2
    
    valid = np.zeros((target_h, target_w), dtype=bool)
    valid[pad_y:pad_y + content_h, pad_x:pad_x + content_w] = True
    return valid

def load_foreground_masks(masks_dir, num_frames, target_h=518, target_w=518):
    masks = []
    for i in range(num_frames):
        path = os.path.join(masks_dir, f"frame_{i:04d}.png")
        if not os.path.exists(path):
            masks.append(np.zeros((target_h, target_w), dtype=bool))
            continue
        m = iio.imread(path)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append(m > 127)
    return masks

def sample_query_points(fg_mask, num_fg_points=200, num_bg_points=100, target_h=518, target_w=518, total_n=300, valid_mask=None):
    # Som requires query points at unique integer pixel coordinates (the trainer rounds
    
    if valid_mask is None:
        valid_mask = np.ones((target_h, target_w), dtype=bool)
    
    fg_yx = np.argwhere(fg_mask & valid_mask)
    
    collected = set()
    
    # foreground points
    if len(fg_yx) > 0:
        n_fg_target = min(num_fg_points, len(fg_yx))
        sel = np.random.choice(len(fg_yx), n_fg_target, replace=False)
        for y, x in fg_yx[sel]:
            collected.add((int(x), int(y)))
            if len(collected) >= total_n:
                break
    
    # background grid points to fill remaining slots
    if len(collected) < total_n:
        n_bg_needed = total_n - len(collected)
        grid_step = max(1, int(np.sqrt(target_h * target_w / (n_bg_needed * 4))))
        ys = np.arange(grid_step // 2, target_h, grid_step)
        xs = np.arange(grid_step // 2, target_w, grid_step)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        bg_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # exclude bg grid points outside the valid region or inside the foreground
        in_valid = valid_mask[bg_pts[:, 1].clip(0, target_h-1), bg_pts[:, 0].clip(0, target_w-1)]
        bg_pts = bg_pts[in_valid]
        if len(fg_yx) > 0:
            bg_in_fg = fg_mask[bg_pts[:, 1].clip(0, target_h-1), bg_pts[:, 0].clip(0, target_w-1)]
            bg_pts = bg_pts[~bg_in_fg]
        
        np.random.shuffle(bg_pts)
        
        for x, y in bg_pts:
            if (int(x), int(y)) not in collected:
                collected.add((int(x), int(y)))
                if len(collected) >= total_n:
                    break
    
    # must land in valid region and outside fg
    attempts = 0
    while len(collected) < total_n and attempts < total_n * 100:
        rx = int(np.random.randint(1, target_w - 1))
        ry = int(np.random.randint(1, target_h - 1))
        if (rx, ry) not in collected and valid_mask[ry, rx]:
            if len(fg_yx) == 0 or not fg_mask[ry, rx]:
                collected.add((rx, ry))
        attempts += 1
    
    while len(collected) < total_n:
        rx = int(np.random.randint(1, target_w - 1))
        ry = int(np.random.randint(1, target_h - 1))
        if (rx, ry) not in collected and valid_mask[ry, rx]:
            collected.add((rx, ry))
    
    result = np.array(list(collected), dtype=np.float32)
    assert len(result) == total_n, f"Expected {total_n} unique points, got {len(result)}"
    return result

def run_cotracker_single_pass(cotracker, video_tensor, query_xy, query_frame_idx, device):
    N = query_xy.shape[0]
    t_col = np.full((N, 1), query_frame_idx, dtype=np.float32)
    queries = np.concatenate([t_col, query_xy], axis=1)
    queries_t = torch.from_numpy(queries).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(
            video=video_tensor.to(device),
            queries=queries_t,
        )
    
    tracks = pred_tracks[0].cpu().numpy()
    visibility = pred_visibility[0].cpu().numpy()
    return tracks, visibility

def cotracker_to_tapir_format(tracks_t_n_2, visibility_t_n, query_frame_idx=None, max_track_distance=MAX_TRACK_DISTANCE, fg_masks=None, query_xy=None):
    T, N, _ = tracks_t_n_2.shape
    out = []
    
    # determine which query points started on foreground
    started_on_fg = None
    if query_xy is not None and fg_masks is not None and query_frame_idx is not None:
        q_mask = fg_masks[query_frame_idx]
        qx = query_xy[:, 0].astype(int).clip(0, q_mask.shape[1] - 1)
        qy = query_xy[:, 1].astype(int).clip(0, q_mask.shape[0] - 1)
        started_on_fg = q_mask[qy, qx]
    
    for t in range(T):
        track_xy = tracks_t_n_2[t]
        vis = visibility_t_n[t].copy()
        
        if query_frame_idx is not None and abs(t - query_frame_idx) > max_track_distance:
            vis[:] = False
        
        # if a track started on foreground but the target position isn't on foreground anymore, mark occluded
        if started_on_fg is not None and fg_masks is not None and t < len(fg_masks):
            target_mask = fg_masks[t]
            tx = track_xy[:, 0].astype(int).clip(0, target_mask.shape[1] - 1)
            ty = track_xy[:, 1].astype(int).clip(0, target_mask.shape[0] - 1)
            on_target_fg = target_mask[ty, tx]
            # for tracks that started on fg, require they stay on fg
            drifted_off_fg = started_on_fg & ~on_target_fg
            vis = vis & ~drifted_off_fg
        
        occs = np.where(vis, TAPIR_VISIBLE_LOGIT, TAPIR_OCCLUDED_LOGIT).astype(np.float32)
        dists = np.where(vis, TAPIR_VISIBLE_LOGIT, TAPIR_OCCLUDED_LOGIT).astype(np.float32)
        
        per_target = np.stack([track_xy[:, 0], track_xy[:, 1], occs, dists], axis=1).astype(np.float32)
        out.append(per_target)
    return out

def make_self_pair_array(query_xy):
    N = query_xy.shape[0]
    arr = np.zeros((N, 4), dtype=np.float32)
    arr[:, 0] = query_xy[:, 0]
    arr[:, 1] = query_xy[:, 1]
    arr[:, 2] = TAPIR_VISIBLE_LOGIT
    arr[:, 3] = TAPIR_VISIBLE_LOGIT
    return arr

def export_tracks_for_som(checkpoint_path, video_path, masks_dir, output_dir, target_h=518, target_w=518, num_fg_points=200, num_bg_points=100, max_frames=None, stride=1, max_track_distance=MAX_TRACK_DISTANCE):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tracks_dir = os.path.join(output_dir, "bootstapir", "1x")
    os.makedirs(tracks_dir, exist_ok=True)
    
    video_tensor_for_som = load_video_for_som(
        video_path, target_h=target_h, target_w=target_w, max_frames=max_frames, stride=stride
    )
    num_frames = video_tensor_for_som.shape[1]
    print(f"Loaded {num_frames} frames at {target_h}x{target_w} for SoM export (stride={stride})")
    
    # letterbox aware mask so query points only land on real video content
    source_w, source_h = get_source_dimensions(video_path)
    valid_mask = compute_valid_content_mask(target_h, target_w, source_w, source_h)
    valid_pct = 100.0 * valid_mask.sum() / (target_h * target_w)
    print(f"Source: {source_w}x{source_h}. Valid content region: {valid_mask.sum()} pixels "
          f"({valid_pct:.1f}% of {target_h}x{target_w})")
    
    fg_masks = load_foreground_masks(masks_dir, num_frames, target_h=target_h, target_w=target_w)
    
    frame_names = [f"frame_{i:04d}" for i in range(num_frames)]
    
    cotracker = CoTrackerPredictor(checkpoint=checkpoint_path)
    cotracker = cotracker.to(device)
    cotracker.eval()
    print("CoTracker3 loaded for SoM export")
    
    for q_idx in range(num_frames):
        query_xy = sample_query_points(
            fg_masks[q_idx],
            num_fg_points=num_fg_points,
            num_bg_points=num_bg_points,
            target_h=target_h,
            target_w=target_w,
            total_n=300,
            valid_mask=valid_mask,
        )
        
        if len(query_xy) == 0:
            query_xy = np.array([[target_w / 2, target_h / 2]], dtype=np.float32)
            print(f"No query points for frame {q_idx}")
        
        tracks_tn2, vis_tn = run_cotracker_single_pass(
            cotracker, video_tensor_for_som, query_xy, q_idx, device
        )
        
        per_target = cotracker_to_tapir_format(
            tracks_tn2, vis_tn,
            query_frame_idx=q_idx,
            max_track_distance=max_track_distance,
            fg_masks=fg_masks,
            query_xy=query_xy,
        )
                
        q_name = frame_names[q_idx]
        for t_idx, target_array in enumerate(per_target):
            t_name = frame_names[t_idx]
            out_path = os.path.join(tracks_dir, f"{q_name}_{t_name}.npy")
            
            # replace cotracker output with query coords
            if t_idx == q_idx:
                target_array = make_self_pair_array(query_xy)
            
            np.save(out_path, target_array)
        
        print(f"[{q_idx + 1}/{num_frames}] Saved {num_frames} target files for query frame {q_idx} (N={len(query_xy)})")
    
    print(f"Done. Wrote {num_frames * num_frames} track files to {tracks_dir}")

if __name__ == "__main__":
    checkpoint_path = r"lifting\point_tracking\cotracker3\co-tracker\checkpoints\scaled_offline.pth"
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_tracker"
    NUM_FRAMES = 24

    n_stride = 2
    video_frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')
    video_frames = video_frames[::n_stride][:NUM_FRAMES]

    video_tensor = video_frames.permute(0, 3, 1, 2).float()
    video_tensor = F.interpolate(video_tensor, size=(384, 512), mode='bilinear', align_corners=False)
    video_tensor = video_tensor.unsqueeze(0)

    FRAME_SIZE = 20
    total_tracks, total_visibility = run_cotracker(checkpoint_path, video_tensor, FRAME_SIZE)

    combined_tracks = torch.cat(total_tracks, dim=1)
    combined_visibility = torch.cat(total_visibility, dim=1)

    tracks_np = combined_tracks.cpu().numpy()
    visibility_np = combined_visibility.cpu().numpy()
    np.savez_compressed("tracking_data.npz", tracks=tracks_np, visibility=visibility_np)
    print("Data saved")

    visualise_cotracker(combined_tracks, combined_visibility, video_tensor, OUTPUT_NAME)

    SOM_OUTPUT_DIR = r"videos\bike_4dgs"
    masks_dir = os.path.join(SOM_OUTPUT_DIR, "masks", "1x")
    
    export_tracks_for_som(
        checkpoint_path=checkpoint_path,
        video_path=video,
        masks_dir=masks_dir,
        output_dir=SOM_OUTPUT_DIR,
        target_h=518,
        target_w=518,
        num_fg_points=200,
        num_bg_points=100,
        max_frames=NUM_FRAMES,
        stride=n_stride,
        max_track_distance=16,
    )