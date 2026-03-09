import torch
import torchvision
import numpy as np
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
import torch.nn.functional as F

def run_cotracker(checkpoint_path, video_tensor, FRAME_SIZE):
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker = CoTrackerPredictor(checkpoint=checkpoint_path)
    cotracker = cotracker.to(device)
    cotracker.eval()

    print("CoTracker loaded successfully")

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
                    grid_size=10, # instead of using x,y coords, generate an evenly spaced 10x10 grid of points over the image and track
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

if __name__ == "__main__":
    checkpoint_path = r"lifting\point_tracking\cotracker3\scaled_offline.pth"
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_tracker"

    video_frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')
    # shape should be [Batch, Frames, Channels, Height, width]
    video_tensor = video_frames.permute(0, 3, 1, 2).float()
    # resize video
    video_tensor = F.interpolate(video_tensor, size=(384, 512), mode='bilinear', align_corners=False)
    # add batch dimension
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