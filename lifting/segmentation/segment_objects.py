import sys
import os
import torch
import numpy as np
import subprocess as sp
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from helpers import get_dimensions
from lifting.point_tracking.find_moving_pixels import run_waft, get_source_and_dest_fast, get_source_and_dest_waft, lo_ransac, get_final_points, group_points_to_objects, center_of_cluster
from lifting.segmentation.helpers import show_mask, show_points, show_box

current_dir = os.path.dirname(os.path.abspath(__file__))
sam2_dir = os.path.join(current_dir, 'segment-anything-2') 
sys.path.append(sam2_dir)

from sam2.build_sam import build_sam2_video_predictor

def get_frames_for_sam(video, FFMPEG_PATH, width, height):
    command = [FFMPEG_PATH,
               '-f', 'mp4',
               '-i', video,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo',
               'pipe:1']
    
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

    frames = []
    while True:
        data = pipe.stdout.read(height * width * 3)

        if len(data) != (height * width  * 3):
            break

        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame)

    pipe.stdout.close()
    pipe.wait()

    return frames

def sam2_1(video, checkpoint, model_cfg, points, frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=video, frames=frames)

    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
    ann_frame_idx = 0  # the frame index we interact with
    prompts = {}
    for point in points:
        pos_click = np.array([point], dtype=np.float32)
        # for labels, 1 means positive click and 0 means negative click
        labels = np.array([1], np.int32)
        prompts[ann_obj_id] = pos_click, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=pos_click,
            labels=labels,
        )

        ann_obj_id += 1

    # run propagation throughout the video and collect the results in a dict
    video_segments = {} # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments

def gen_segmentation_video(video_segments, frames, OUTPUT_NAME):
    # render the segmentation results every few frames
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_NAME + ".mp4", fourcc, 30.0, (width, height))

    # color for mask
    mask_color = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.5 # 50% transparency

    for frame_idx in range(len(frames)):
        current_frame = frames[frame_idx].copy() 

        # check if we have masks for this specific frame
        if frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask_bool = np.squeeze(out_mask) # in case of extra dimensions
                
                # apply the color only where the mask is true
                current_frame[mask_bool] = (current_frame[mask_bool] * (1 - alpha) + mask_color * alpha).astype(np.uint8)

        bgr_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        
        # write the painted frame to the final video
        out_video.write(bgr_frame)

    out_video.release()

# for shape of motion pipeline
def build_background_mask(video_segments, num_frames, height, width):
    # start with everything as background
    background_mask = np.ones((num_frames, height, width), dtype=np.float32)
    
    for frame_idx, objects in video_segments.items():
        for obj_id, mask in objects.items():
            # mask shape is (1, H, W), squeeze to (H, W)
            mask_2d = np.squeeze(mask)
            # wherever SAM says foreground, set to 0
            background_mask[frame_idx][mask_2d] = 0.0
    
    # flatten H*W for point cloud indexing
    background_mask = background_mask.reshape(num_frames, -1)
    
    return torch.tensor(background_mask, dtype=torch.float32)

def save_segments_npz(video_segments, filename="sam2_results.npz"):
    # creates keys like "frame0_obj0", "frame0_obj1", ...
    flat_dict = {f"{frame}_{obj}": mask 
                 for frame, objs in video_segments.items() 
                 for obj, mask in objs.items()}
    np.savez_compressed(filename, **flat_dict)
    print(f"Saved segments to {filename}")

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_sam"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"

    checkpoint = r"lifting/segmentation/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = r"configs/sam2.1/sam2.1_hiera_t.yaml"

    checkpoint_path = r"lifting\point_tracking\waft\tar-c-t-kitti.pth"
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_waft"
    RANSAC_ERROR_THRESHOLD = 3.0

    video_frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')
    flow_list, frame1, frame2 = run_waft(checkpoint_path, video_frames)
    #draw_image_overlay(flow_list, frame1)
    flow_tensor = flow_list[-1]
    flow_data = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    #source, dest = get_source_and_dest_fast(frame1, frame2)
    source, dest = get_source_and_dest_waft(flow_data)
    H, inliers, outliers = lo_ransac(source, dest, RANSAC_ERROR_THRESHOLD)
    if H is not None:
        final_points = get_final_points(source, outliers)
        point_labels = group_points_to_objects(final_points) # each index has a 2d array of points
        objects = center_of_cluster(point_labels)

    width, height = get_dimensions(video, FFPROBE_PATH)
    frames = get_frames_for_sam(video, FFMPEG_PATH, width, height)

    # points have to be [[x1, y1], [x2, y2], ...]
    results = sam2_1(video, checkpoint, model_cfg, objects, frames=frames)
    background_mask = build_background_mask(results, len(frames), height, width)
    torch.save(background_mask, "background_mask.pt")
    save_segments_npz(results)
    #gen_segmentation_video(results, frames, OUTPUT_NAME)

# C:/vasilis/2D-to-3D-Video-Conversion/.venv/Scripts/python.exe -m lifting.segmentation.segment_objects