import numpy as np
import subprocess as sp
from helpers import get_video_fps

def raw_frame_to_grayscale_image(frame):
    min_val = np.min(frame)
    max_val = np.max(frame)

    # prevent division by zero if the frame is completely blank
    if max_val == min_val:
        print("All black frame")
        return np.zeros_like(frame, dtype=np.uint8) 
    
    normalized = (frame - min_val) / (max_val - min_val) # min max normalization
    pixels = (normalized * 255.0).astype(np.uint8)
    
    return pixels

def pixels_to_video(video, FFMPEG_PATH):
    command = [FFMPEG_PATH,
                '-y', 
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{W}x{H}',
                '-pix_fmt', 'gray',
                '-r', str(FPS),
                '-i', '-', 
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', # standard MP4 format
                OUTPUT_MP4]

    pipe = sp.Popen(command, stdin=sp.PIPE)

    video.reshape(-1, H, W)
    for frame in video:
        gray_frame = raw_frame_to_grayscale_image(frame)
        pipe.stdin.write(gray_frame.tobytes())

    pipe.stdin.close()
    pipe.wait() # wait for FFmpeg to finish encoding

if __name__ == "__main__":
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"
    ORIGINAL_VIDEO = r"videos\bike.mp4"
    RAW_FILE = r"C:\proj\2d_to_3d\videos\bike_depth.raw"
    OUTPUT_MP4 = r"C:\proj\2d_to_3d\videos\bike_depth.mp4"
    FPS = get_video_fps(ORIGINAL_VIDEO, FFPROBE_PATH)
    H, W = 518, 518 # must match dimensions from model

    depth_video = np.memmap(RAW_FILE, dtype=np.float32, mode='r').reshape(-1, H, W) 
    total_frames = depth_video.shape[0]

    pixels_to_video(depth_video, FFMPEG_PATH)