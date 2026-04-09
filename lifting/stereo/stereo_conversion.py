import os
import sys
import torch
import torchvision
import torch.nn.functional as F
import subprocess as sp
import numpy as np
import cv2
from intrinsic_matrix import get_disparity_from_flow3r_points

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from helpers import get_dimensions, get_video_fps

flow3r_dir = os.path.abspath(os.path.join(current_dir, '..'))
if flow3r_dir not in sys.path:
    sys.path.append(flow3r_dir)

from point_cloud.point_cloud_flow3r import flow3r, correct_input

def left_right_eyes(image, disparity_map):
    _, _, H, W = image.shape

    half_shift = disparity_map / 2.0

    # convert pixel shifts into -1, 1
    normalized_shift = half_shift / (W / 2.0)

    # blank base coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing='ij'
    )
    
    # make the (1, H, W, 2) tensor required by grid_sample
    base_grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0)

    # left eye shifts right
    left_grid = base_grid.clone()
    left_grid[..., 0] += normalized_shift.squeeze() # apply shift only to x axis

    # right eye shifts left
    right_grid = base_grid.clone()
    right_grid[..., 0] -= normalized_shift.squeeze() # apply shift only to x axis

    # backward wrapping
    left_eye = F.grid_sample(
        image, 
        left_grid, 
        mode='bilinear',
        padding_mode='border', # if a shift pulls from offscreen, stretch the edge pixel
        align_corners=True
    )

    right_eye = F.grid_sample(
        image, 
        right_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )

    return left_eye, right_eye

def upscale_disparity(disparity_map, target_width, target_height):
    scale_factor = target_width / float(disparity_map.shape[-1])

    if disparity_map.ndim == 2:
        disparity_map = disparity_map.unsqueeze(0).unsqueeze(0)

    # stretch the grid
    upscaled_map = F.interpolate(
        disparity_map, 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=False
    )

    final_disparity = upscaled_map * scale_factor

    return final_disparity

def frames_to_tensor(video, FFMPEG_PATH, width, height):
    command = [FFMPEG_PATH,
                '-f', 'mp4',
                '-i', video,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                'pipe:1']
    
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    data = pipe.stdout.read()
    np_frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, height, width, 3)
    tensor_frames = torch.from_numpy(np_frames).float() / 255.0
    tensor_frames = tensor_frames.permute(0, 3, 1, 2)

    return tensor_frames


def tensor_to_cv2(tensor):
    img_array = tensor.squeeze(0).cpu().numpy()
    
    # (H, W, C)
    img_array = np.transpose(img_array, (1, 2, 0))
    
    img_array = (img_array * 255.0).clip(0, 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_cv2

if __name__ == "__main__":
    video = r"videos\bike_cut_4sec.mp4"
    OUTPUT_VIDEO = r"videos\bike_stereo.mp4"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"
    FPS = get_video_fps(video, FFPROBE_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_tensor, _, _ = torchvision.io.read_video(video, pts_unit='sec')
    frames = video_tensor.shape[0]
    flow3r_input = correct_input(video, FFMPEG_PATH, frames)
    points = flow3r(flow3r_input)

    target_width, target_height = get_dimensions(video, FFPROBE_PATH)

    video_tensors = frames_to_tensor(video, FFMPEG_PATH, target_width, target_height)

    # height and width are from the flow3r rescaling process, we upscale the resolution in upscale_disparity
    height = points.shape[2] 
    width = points.shape[3]

    sbs_width = target_width * 2
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (sbs_width, target_height))

    for i in range(frames):
        disparity_map = get_disparity_from_flow3r_points(points, i, width, height, intensity_divisor=30.0)
        final_disparity_map = upscale_disparity(disparity_map, target_width, target_height)

        frame_tensor = video_tensors[i].unsqueeze(0).to(final_disparity_map.device)
        left_eye_tensor, right_eye_tensor = left_right_eyes(frame_tensor, final_disparity_map)

        left_img = tensor_to_cv2(left_eye_tensor)
        right_img = tensor_to_cv2(right_eye_tensor)

        # check if the 2 views are different
        if i == 0:
            # calculate the pixel difference
            difference_map = cv2.absdiff(left_img, right_img)
            
            score = np.sum(difference_map)
            if score == 0:
                print("IMAGES ARE IDENTICAL")
            else:
                print(f"Images are different. Shift Score: {score}")

        sbs_frame = np.concatenate((left_img, right_img), axis=1)

        out.write(sbs_frame)
        print(f"Rendered frame {i+1}/{frames}")

    out.release()

# C:/vasilis/2D-to-3D-Video-Conversion/.venv/Scripts/python.exe c:/vasilis/2D-to-3D-Video-Conversion/lifting/stereo/stereo_conversion.py