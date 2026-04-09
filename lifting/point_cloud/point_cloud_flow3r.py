import os
import sys
import torch
import numpy as np
import subprocess as sp
current_dir = os.path.dirname(os.path.abspath(__file__))
flow3r_dir = os.path.join(current_dir, 'flow3r')
if flow3r_dir not in sys.path:
    sys.path.append(flow3r_dir)

from flow3r.models.flow3r import Flow3r

def flow3r(video_tensor):
    # init
    model = Flow3r(pos_type='rope100', decoder_size='large')

    state_dict = torch.load(r"C:\vasilis\2D-to-3D-Video-Conversion\lifting\point_cloud\flow3r\flow3r\checkpoints\flow3r.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.cuda().eval()

    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    with torch.no_grad():
        output = model(video_tensor.cuda())
        print("AVAILABLE KEYS FROM MODEL:", output.keys())

    # results
    world_points = output['points'] # shape (B, N, H, W, 3)

    #return world_points
    return output

def correct_input(video, FFMPEG_PATH, frames):
    target_size = 518
    
    command = [FFMPEG_PATH,
            '-f', 'mp4',
            '-i', video,
            '-vframes', str(frames), # Ensure we only grab exactly 10 frames
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            '-vf', f'scale={target_size}:{target_size}:force_original_aspect_ratio=decrease,pad={target_size}:{target_size}:(ow-iw)/2:(oh-ih)/2', 
            'pipe:1']
    
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    data = pipe.stdout.read()

    array = np.frombuffer(data, dtype=np.uint8)
    
    # (frames, height, width, channels)
    array = array.reshape(frames, target_size, target_size, 3)

    # convert to tensor and normalize
    tensor = torch.from_numpy(array.copy()).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    
    # add the batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor

def export_to_ply(world_points, output_path="output.ply", frame_idx=0):    
    # isolate one frame
    frame_points = world_points[0, frame_idx] # (518, 518, 3)
    flat_points = frame_points.reshape(-1, 3).cpu().numpy()
    
    # ASCIi .ply file
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(flat_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        np.savetxt(f, flat_points, fmt='%.5f %.5f %.5f')

def export_to_ply_blender(world_points, video_tensor, output_path="output.ply", frame_idx=0):    
    # isolate 3D points
    frame_points = world_points[0, frame_idx].reshape(-1, 3).cpu().numpy()
    
    # isolate RGB colors
    colors = video_tensor[0, frame_idx].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors = colors * 255.0 
    
    # fake alpha
    alphas = np.full((colors.shape[0], 1), 255.0)
    
    combined_data = np.hstack((frame_points, colors, alphas))
        
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(frame_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")

        np.savetxt(f, combined_data, fmt='%.5f %.5f %.5f %d %d %d %d')

def export_to_ply_meshlab(world_points, video_tensor, output_path="output.ply", frame_idx=0):    
    frame_points = world_points[0, frame_idx].reshape(-1, 3).cpu().numpy()
    
    # invert the z axis to fix the reversed depth
    frame_points[:, 2] = -frame_points[:, 2]
    
    colors = video_tensor[0, frame_idx].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors = colors * 255.0 
    
    combined_data = np.hstack((frame_points, colors))
        
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(frame_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        np.savetxt(f, combined_data, fmt='%.5f %.5f %.5f %d %d %d')
                
if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = "bike_sam"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"

    input = correct_input(video, FFMPEG_PATH, frames=7)
    #points = flow3r(input)
    output = flow3r(input)
    #export_to_ply_meshlab(points, input)
    #local_points = output['local_points']
    #print('Z min:', local_points[..., 2].min().item())
    #print('Z max:', local_points[..., 2].max().item())
    print('local_points shape:', output['local_points'].shape)

                

# C:/vasilis/2D-to-3D-Video-Conversion/.venv/Scripts/python.exe c:/vasilis/2D-to-3D-Video-Conversion/lifting/point_cloud/point_cloud_flow3r.py
