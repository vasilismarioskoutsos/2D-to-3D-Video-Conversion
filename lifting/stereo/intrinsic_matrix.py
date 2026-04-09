import torch

def get_disparity_from_flow3r_points(points, frame_num, width, height, intensity_divisor=30.0):
    cx = width // 2

    X = points[0, frame_num, :, :, 0]
    Z = points[0, frame_num, :, :, 2]

    # average depth of the scene
    average_Z = torch.median(Z).item()

    tc = average_Z / intensity_divisor # baseline (interaxial distance) = the physical space between the left camera and the right camera

    # 2D grid of u coords
    u_grid = torch.arange(width, device=points.device).unsqueeze(0).expand(height, width)

    # mask to prevent division by zero
    valid_mask = torch.abs(X) > 1e-5

    fx_guesses = torch.abs((u_grid[valid_mask] - cx) * Z[valid_mask] / X[valid_mask])
    fx_median = torch.median(fx_guesses).item()

    # safe version of Z where no depth can ever drop below 0.1
    Z_safe = torch.clamp(Z, min=0.1)

    d_array = (tc * fx_median) / Z_safe

    return d_array

if __name__ == "__main__":
    video = r"videos\bike_cut.mp4"
    OUTPUT_VIDEO = r"videos\bike_reverse_padding.mp4"
    FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"
    FFPROBE_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffprobe.exe"

    #height = points.shape[2] 
    #width = points.shape[3]