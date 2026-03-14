import os
import numpy as np
import cv2
import sys
from PIL import Image
from video_processing import load_segments_npz

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from helpers import get_dimensions

propainter_dir = os.path.join(current_dir, 'ProPainter')
if propainter_dir not in sys.path:
    sys.path.append(propainter_dir)

from inference_propainter import run_propainter, run_propainter_waft

def video_to_pil(video, video_segments, FFPROBE_PATH):
    
    width, height = get_dimensions(video, FFPROBE_PATH)
    
    frames = []
    dilated_frames = []
    for frame_idx in sorted(video_segments.keys()):

        mask_array = np.zeros((height, width), dtype=np.uint8)

        for out_obj_id, out_mask in video_segments[frame_idx].items():

            if out_mask is not None:
                mask_bool = np.squeeze(out_mask)
                mask_array[mask_bool] = 255

        pil_frame = Image.fromarray(mask_array, mode='L') # mode is for grayscale conversion

        # expand the pixels for propainter
        kernel = np.ones((5, 5), np.uint8)
        dilated_frame = cv2.dilate(mask_array, kernel, iterations=2)

        pil_dilated_frame = Image.fromarray(dilated_frame, mode='L')

        frames.append(pil_frame)
        dilated_frames.append(pil_dilated_frame)

    return frames, dilated_frames

if __name__ == "__main__":
    video = r"videos\bike_reverse_padding.mp4"
    OUTPUT_FOLDER = r"videos"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"
    SAM_NPZ = r"sam2_results_padding.npz"

    video_segments = load_segments_npz(SAM_NPZ)
    flow_masks, masks_dilated = video_to_pil(video, video_segments, FFPROBE_PATH)
    run_propainter_waft(video, flow_masks, masks_dilated, OUTPUT_FOLDER, neighbor_length=5, subvideo_length=18)

# C:/proj/2d_to_3d/venv/Scripts/python.exe c:/proj/2d_to_3d/lifting/inpainting/lama_inpainting.py