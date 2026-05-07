import sys
import os
import torch
import numpy as np
import subprocess as sp
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from helpers import get_dimensions
from lifting.point_tracking.find_moving_pixels import run_waft, get_source_and_dest_waft, lo_ransac, get_final_points, group_points_to_objects, box_cluster
from lifting.segmentation.helpers import show_mask, show_points, show_box

current_dir = os.path.dirname(os.path.abspath(__file__))
sam2_dir = os.path.join(current_dir, 'segment-anything-2') 
sys.path.append(sam2_dir)

from sam2.build_sam import build_sam2_video_predictor

def make_strided_video(input_video, output_video, n_stride, num_frames, ffmpeg_path):
    # creates a video containing only every nth frame from the input
    cmd = [
        ffmpeg_path,
        "-y",  # overwrite if exists
        "-i", input_video,
        "-vf", f"select='not(mod(n\\,{n_stride}))',setpts=N/FRAME_RATE/TB",
        "-vframes", str(num_frames),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video,
    ]
    sp.run(cmd, check=True, capture_output=True)
    print(f"Wrote strided video: {output_video}")

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

    ann_obj_id = 0 # unique id to each object we interact with
    ann_frame_idx = 0 # frame index we interact with
    prompts = {}

    for object in points:
        if isinstance(object, np.ndarray) and object.shape == (4,):
            # if it's a bounding box [x_min, y_min, x_max, y_max]
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=object,
            )
        else:
            # if it's a point [x, y]
            pos_click = np.array([object], dtype=np.float32)
            labels = np.array([1], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=pos_click,
                labels=labels,
            )
        ann_obj_id += 1

    # run propagation through the video and collect the results in a dict
    video_segments = {} # segmentation results
    num_target_frames = len(frames)  # cap
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx >= num_target_frames:
            break
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
                
                # apply the color where the mask is true
                current_frame[mask_bool] = (current_frame[mask_bool] * (1 - alpha) + mask_color * alpha).astype(np.uint8)

        bgr_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        
        # write the painted frame to the final video
        out_video.write(bgr_frame)

    out_video.release()

# for shape of motion pipeline
def build_background_mask(video_segments, num_frames, height, width, target_size=518):
    background_mask = np.ones((num_frames, target_size, target_size), dtype=np.float32)
    
    for frame_idx, objects in video_segments.items():
        for obj_id, mask in objects.items():
            mask_2d = np.squeeze(mask)
            # resize mask to target_size
            mask_resized = cv2.resize(
                mask_2d.astype(np.uint8), 
                (target_size, target_size), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            background_mask[frame_idx][mask_resized] = 0.0
    
    background_mask = background_mask.reshape(num_frames, -1)
    return torch.tensor(background_mask, dtype=torch.float32)

def save_segments_npz(video_segments, filename="sam2_results.npz"):
    # creates keys like "frame0_obj0", "frame0_obj1", ...
    flat_dict = {f"{frame}_{obj}": mask 
                 for frame, objs in video_segments.items() 
                 for obj, mask in objs.items()}
    np.savez_compressed(filename, **flat_dict)
    print(f"Saved segments to {filename}")

def save_masks_as_png(video_segments, num_frames, height, width, output_dir, target_size=518):
    mask_dir = os.path.join(output_dir, "masks", "1x")
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_frames):
        mask = np.zeros((height, width), dtype=np.uint8)

        if i in video_segments:
            for obj_id, obj_mask in video_segments[i].items():
                mask_2d = np.squeeze(obj_mask)
                mask[mask_2d] = 255

        # resize to match flow3r resolution
        mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        filename = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(mask_dir, filename), mask_resized)

    print(f"Saved {num_frames} mask PNGs to {mask_dir}")

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_sam"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"

    checkpoint = r"lifting/segmentation/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = r"configs/sam2.1/sam2.1_hiera_b+.yaml"

    checkpoint_path = r"lifting\point_tracking\waft\tar-c-t-kitti.pth"
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = r"videos\bike_4dgs"
    RANSAC_ERROR_THRESHOLD = 3.0
    NUM_FRAMES = 24
    n_stride = 2 # every nth frame
    STEP = 2

    video_frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')
    video_frames = video_frames[::n_stride][:NUM_FRAMES]
    flow_list, frame1, frame2 = run_waft(checkpoint_path, video_frames)
    #draw_image_overlay(flow_list, frame1)
    flow_tensor = flow_list[-1]
    flow_data = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    #source, dest = get_source_and_dest_fast(frame1, frame2)
    source, dest = get_source_and_dest_waft(flow_data, step=STEP)
    H, inliers, outliers = lo_ransac(source, dest, RANSAC_ERROR_THRESHOLD)
    if H is not None:
        final_points = get_final_points(source, outliers)
        point_labels = group_points_to_objects(final_points, step=STEP) # each index has a 2d array of points
        #objects = center_of_cluster(point_labels)
        objects = box_cluster(point_labels)

    width, height = get_dimensions(video, FFPROBE_PATH)
    frames = get_frames_for_sam(video, FFMPEG_PATH, width, height)
    frames = frames[::n_stride][:NUM_FRAMES]

    strided_video_path = r"videos\bike_cut_strided.mp4"
    make_strided_video(video, strided_video_path, n_stride, NUM_FRAMES, FFMPEG_PATH)

    # points have to be [[x1, y1], [x2, y2], ...]
    results = sam2_1(strided_video_path, checkpoint, model_cfg, objects, frames=frames)
    background_mask = build_background_mask(results, len(frames), height, width)
    torch.save(background_mask, "background_mask.pt")
    save_segments_npz(results)
    save_masks_as_png(results, len(frames), height, width, OUTPUT_NAME)

    #gen_segmentation_video(results, frames, OUTPUT_NAME)

# C:/vasilis/2D-to-3D-Video-Conversion/.venv/Scripts/python.exe -m lifting.segmentation.segment_objects