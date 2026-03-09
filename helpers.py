import numpy as np
import subprocess as sp

def get_video_fps(video_path, ffprobe_path):
    command = [
        ffprobe_path,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1', # format the output as raw text
        video_path
    ]
    
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate()
    
    # output will be a fraction string
    fps_string = out.decode('utf-8').strip()
    
    # split the fraction and calculate the actual value
    numerator, denominator = fps_string.split('/')
    fps = float(numerator) / float(denominator)
    
    return fps

def get_dimensions(video_path, ffprobe_path):
    command = [
        ffprobe_path,
        '-v', 'error',
        '-select_streams', 'v:0', # select the first video stream
        '-show_entries', 'stream=width,height', # ask only for width and height
        '-of', 'csv=s=x:p=0', # output as raw values separated by an x
        video_path
    ]
    
    result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    
    # strip whitespace
    output = result.stdout.strip()
    
    if output:
        # split into two variables and convert them to integers
        width_str, height_str = output.split('x')
        
        width = int(width_str)
        height = int(height_str)
        
        return width, height
    else:
        print(f"Failed to read video: {result.stderr}")
        return None, None