import os
import sys
import torch
import numpy as np
import subprocess as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
flow3r_dir = os.path.join(current_dir, r'lifting/point_cloud/flow3r')
sys.path.append(flow3r_dir)

from models.flow3r import Flow3r

def flow3r(video_tensor):
    # init
    model = Flow3r(pos_type='rope100', decoder_size='large')

    state_dict = torch.load("checkpoints/flow3r.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    model.cuda().eval()

    with torch.no_grad():
        output = model(video_tensor.cuda())

    # results
    world_points = output['points'] # shape (B, N, H, W, 3)

    return world_points

def correct_input(video, FFMPEG_PATH, W, H):
    command = [FFMPEG_PATH,
            '-f', 'mp4',
            '-i', video,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            '-vf', f'scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2', # preserve aspect ratio without distorting the image
            'pipe:1']
    
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    data = pipe.stdout.read()

    array = np.frombuffer(data, dtype=np.float32)
    tensor = torch.from_numpy(array).reshape(1, 10, 3, 518, 518)

    return tensor

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_sam"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"