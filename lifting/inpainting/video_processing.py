import os
import subprocess as sp
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from helpers import get_dimensions, get_video_fps

def get_first_n_frames(video, FFMPEG_PATH, padding_num, width, height):
    command = [FFMPEG_PATH,
                '-i', video,
                '-vframes', str(padding_num),
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', 
                '-f', 'image2pipe',
                'pipe:1']
    
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    
    frame_size = width * height * 3
    frames = []
    
    while True:
        frame_bytes = pipe.stdout.read(frame_size)
        if not frame_bytes or len(frame_bytes) < frame_size:
            break
        frames.append(frame_bytes)
    
    pipe.stdout.close()
    pipe.wait()

    return frames

def get_last_n_frames(video, FFMPEG_PATH, FFPROBE_PATH, n, width, height):
    command = [FFPROBE_PATH, 
                '-v', 'error', 
                '-select_streams', 'v:0', 
                '-count_packets', '-show_entries', 'stream=nb_read_packets', 
                '-of', 'csv=p=0', video]
                 
    total_frames = int(sp.check_output(command).decode('utf-8').strip())
    start_frame = max(0, total_frames - n)

    command = [FFMPEG_PATH,
               '-i', video,
               '-vf', f"select='gte(n\,{start_frame})'",
               '-vsync', '0',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo',
               '-f', 'image2pipe',
               'pipe:1']
               
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    
    frame_size = width * height * 3
    frames = []
    
    while True:
        frame_bytes = pipe.stdout.read(frame_size)
        if not frame_bytes or len(frame_bytes) < frame_size:
            break
        frames.append(frame_bytes)
    
    pipe.stdout.close()
    pipe.wait()
        
    return frames

def add_padding_frames(video, first_frames, last_frames, FFMPEG_PATH, output_path, width, height, fps):
    # reverse the frames in memory
    reverse_first = first_frames[::-1]
    reverse_last = last_frames[::-1]

    n_first = len(reverse_first)
    n_last = len(reverse_last)

    # split the pipe input in 2 identical streams
    # trim the first stream to grab the front padding
    # trim the second stream to grab the back padding
    # concat them to video
    filter_complex = (
        f"[1:v]split=2[in_front][in_back];"
        f"[in_front]trim=start_frame=0:end_frame={n_first},setpts=PTS-STARTPTS[front];"
        f"[in_back]trim=start_frame={n_first}:end_frame={n_first + n_last},setpts=PTS-STARTPTS[back];"
        f"[front][0:v][back]concat=n=3:v=1:a=0[outv]"
    )

    command = [
        FFMPEG_PATH,
        '-y',
        '-i', video,           
        '-f', 'rawvideo', 
        '-vcodec', 'rawvideo',
        '-s', f"{width}x{height}", 
        '-pix_fmt', 'rgb24', 
        '-r', str(fps), 
        '-i', '-',
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        output_path
    ]

    pipe = sp.Popen(command, stdin=sp.PIPE)

    try:
        for frame in reverse_first:
            pipe.stdin.write(frame)
        for frame in reverse_last:
            pipe.stdin.write(frame)
    except BrokenPipeError:
        print("FFmpeg closed the pipe early")

    pipe.stdin.close()
    pipe.wait()

def load_segments_npz(filename="sam2_results.npz"):
    data = np.load(filename)
    video_segments = {}
    for key in data.files:
        frame_str, obj_str = key.split('_')
        frame, obj = int(frame_str), int(obj_str)
        
        if frame not in video_segments:
            video_segments[frame] = {}
        video_segments[frame][obj] = data[key]
        
    return video_segments

def save_segments_npz(video_segments, filename="sam2_results.npz"):
    # creates keys like "frame0_obj0", "frame0_obj1", ...
    flat_dict = {f"{frame}_{obj}": mask 
                 for frame, objs in video_segments.items() 
                 for obj, mask in objs.items()}
    np.savez_compressed(filename, **flat_dict)
    print(f"Saved segments to {filename}")

def add_padding_sam_npz(padding_num, filename="sam2_results.npz"):
    video_segments = load_segments_npz(filename)

    # find the total number of frames
    max_frame = max(video_segments.keys())
    total_original_frames = max_frame + 1
    
    new_video_segments = {}
    new_frame_idx = 0
    
    # front padding
    for i in range(padding_num - 1, -1, -1):
        if i in video_segments:
            new_video_segments[new_frame_idx] = video_segments[i]
        new_frame_idx += 1
        
    # original frmaes
    for i in range(total_original_frames):
        if i in video_segments:
            new_video_segments[new_frame_idx] = video_segments[i]
        new_frame_idx += 1
        
    # back padding
    start_last = total_original_frames - padding_num
    for i in range(max_frame, start_last - 1, -1):
        if i in video_segments:
            new_video_segments[new_frame_idx] = video_segments[i]
        new_frame_idx += 1
        
    # Save the re-indexed segments to a new file
    save_segments_npz(new_video_segments, filename="sam2_results_padding.npz")
    

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_VIDEO = r"videos\bike_reverse_padding.mp4"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"
    PADDING = 30

    #width, height = get_dimensions(video, FFPROBE_PATH)
    #fps = get_video_fps(video, FFPROBE_PATH)
    #first_frames = get_first_n_frames(video, FFMPEG_PATH, PADDING, width, height)
    #last_frames = get_last_n_frames(video, FFMPEG_PATH, FFPROBE_PATH, PADDING, width, height)
    #add_padding_frames(video, first_frames, last_frames, FFMPEG_PATH, OUTPUT_VIDEO, width, height, fps)

    add_padding_sam_npz(PADDING)

# C:/vasilis/2D-to-3D-Video-Conversion/.venv/Scripts/python.exe c:/vasilis/2D-to-3D-Video-Conversion/lifting/inpainting/video_processing.py
# C:/proj/2d_to_3d/venv/Scripts/python.exe c:/proj/2d_to_3d/lifting/inpainting/video_processing.py
