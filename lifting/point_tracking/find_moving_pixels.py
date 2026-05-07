import cv2
import torch
import torchvision
import numpy as np
import poselib
import os
from lifting.point_tracking.waft.waft_utils import InferenceWrapper
from lifting.point_tracking.waft.waft_utils import flow_to_image
from lifting.point_tracking.waft.model import ViTWarpV8

class Args:
    dav2_backbone = 'vits' # installed from DepthAnythingV2
    network_backbone = 'vits'
    iters = 12 # default
    var_max = 1.0
    var_min = -1.0

def run_waft(checkpoint_path, video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Args()
    model = ViTWarpV8(args)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    wrapper = InferenceWrapper(model, scale=0, train_size=(384, 512), pad_to_train_size=True, tiling=True)

    frame1 = video[0].permute(2, 0, 1).unsqueeze(0).float()
    frame2 = video[1].permute(2, 0, 1).unsqueeze(0).float()

    frame1, frame2 = frame1.to(device), frame2.to(device)
    
    with torch.no_grad():
        output = wrapper.calc_flow(frame1, frame2)

    flow_list = output['flow'] # contains 1 tensor for every iteration of waft

    return flow_list, frame1, frame2

def draw_image_overlay(flow_list, frame1):
    for i, flow_tensor in enumerate(flow_list):
        # remove batch dimension = (2, H, W)
        flow_data = flow_tensor[0]
        
        # permute for flow_to_image
        flow_data = flow_data.permute(1, 2, 0).detach().cpu().numpy()
        
        # convert to RGB image
        flow_image = flow_to_image(flow_data, convert_to_bgr=True)
        
        # convert original image to BGR
        frame1_np = frame1[0].permute(1, 2, 0).cpu().numpy()
        image_bgr = cv2.cvtColor((frame1_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # overlay to the original image
        overlay = cv2.addWeighted(image_bgr, 0.5, flow_image, 0.5, 0)

        cv2.imwrite("overlay_waft.png", overlay)

def get_source_and_dest_waft(flow_data, step=2):
    h, w = flow_data.shape[:2]

    # generate the source points
    y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(np.float64)
    
    # u = horizontal movement, v = vertical movement
    u = flow_data[y.astype(int), x.astype(int), 0].astype(np.float64)
    v = flow_data[y.astype(int), x.astype(int), 1].astype(np.float64)

    # get [x, y] pairs
    source = np.vstack((x, y)).T
    
    dest = np.vstack((x + u, y + v)).T

    return source, dest

def get_source_and_dest_fast(frame1, frame2):
    # convert to bgr
    frame1_np = frame1[0].permute(1, 2, 0).cpu().numpy()
    frame1_bgr = cv2.cvtColor((frame1_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    frame2_np = frame2[0].permute(1, 2, 0).cpu().numpy()
    frame2_bgr = cv2.cvtColor((frame2_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # opencv tracking requires grayscale images
    gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)

    # find up to 1000 distinct corners in frame 1 (source)
    source_points = cv2.goodFeaturesToTrack(
        gray1, 
        maxCorners=1000, 
        qualityLevel=0.01, 
        minDistance=10
    )

    if source_points is None:
        print("Could not find any features to track")
        return None, None

    # track the same corners into frame 2 (dest)
    dest_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, 
        gray2, 
        source_points, 
        None
    )

    # filter any points that the tracker lost
    # status array contains 1 if the point was found, 0 if lost
    good_source = source_points[status == 1]
    good_dest = dest_points[status == 1]

    # format the arrays for poselib
    source = good_source.reshape(-1, 2).astype(np.float64)
    dest = good_dest.reshape(-1, 2).astype(np.float64)

    return source, dest

def lo_ransac(source, dest, error):
    source_points = np.array(source, dtype=np.float64).reshape(-1, 2)
    dest_points = np.array(dest, dtype=np.float64).reshape(-1, 2)

    options = poselib.RansacOptions()
    options['max_reproj_error'] = error # if a pixel moves differently than the rest of the background by more than 3 pixels, ignore it
    
    H, info = poselib.estimate_homography(source_points, dest_points, options)

    if H is not None:
        inliers = np.array(info['inliers'], dtype=bool)
        outliers = ~inliers
        
        print("\n--- Lo-Ransac Estimation ---")
        print("Homography Matrix (H):")
        print(H)
        print(f"Inliers: {np.sum(inliers)}")
        print(f"Outliers: {np.sum(outliers)}")
        
        return H, inliers, outliers
    else:
        print("Could not find a valid background homography")
        return None, None, None

def get_final_points(source, outliers):
    return source[outliers]

def draw_final_points(points, frame):
    frame_np = frame[0].permute(1, 2, 0).cpu().numpy()
    image = cv2.cvtColor((frame_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)
    
    cv2.imwrite("final_points.png", image)

def group_points_to_objects(points, step=2):
    # connected_components needs the points to be 1 pixel apart
    scaled_x = (points[:, 0] // step).astype(int)
    scaled_y = (points[:, 1] // step).astype(int)

    # find size of grid
    max_x = scaled_x.max() + 1
    max_y = scaled_y.max() + 1

    grid = np.zeros((max_y, max_x), dtype=np.uint8)

    grid[scaled_y, scaled_x] = 255 # foreground
    num_objects, point_labels = cv2.connectedComponents(grid, connectivity=8)

    objects_points = []
    for label in range(1, num_objects):
        # find which of the original points land on this labels footprint
        mask = point_labels[scaled_y, scaled_x] == label
        total_points = points[mask]
        
        if len(total_points) > 0:
            objects_points.append(total_points)

    return objects_points

def center_of_cluster(point_labels):
    objects = []
    for object in point_labels:
        mean_center = np.mean(object, axis=0)

        # find the distance from the mean to all actual points
        distances = np.linalg.norm(object - mean_center, axis=1)
        
        # pick the point with the smallest distance
        best_point = object[np.argmin(distances)]
        objects.append(best_point)

    return objects

def box_cluster(point_labels, min_points=1):
    objects = []
    for obj in point_labels:
        if len(obj) < min_points:
            continue  # skip noise clusters
        x_min, y_min = obj.min(axis=0)
        x_max, y_max = obj.max(axis=0)
        objects.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
    return objects

if __name__ == "__main__":
    checkpoint_path = r"lifting\point_tracking\waft\tar-c-t-kitti.pth"
    video = r"videos\bike_cut.mp4"
    OUTPUT_NAME = r"videos\bike_4dgs"
    RANSAC_ERROR_THRESHOLD = 3.0
    NUM_FRAMES = 24
    n_stride = 2 # every nth frame

    video_frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')
    video_frames = video_frames[::n_stride][:NUM_FRAMES]

    detection_frames = torch.stack([video_frames[0], video_frames[20]])
    flow_list, frame1, frame2 = run_waft(checkpoint_path, detection_frames)
    #draw_image_overlay(flow_list, frame1)
    flow_tensor = flow_list[-1]
    flow_data = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    #source, dest = get_source_and_dest_fast(frame1, frame2)
    source, dest = get_source_and_dest_waft(flow_data, step=5)
    H, inliers, outliers = lo_ransac(source, dest, RANSAC_ERROR_THRESHOLD)
    if H is not None:
        final_points = get_final_points(source, outliers)
        #draw_final_points(final_points, frame1)
        point_labels = group_points_to_objects(final_points) # each index has a 2d array of points
        objects = center_of_cluster(point_labels)
        draw_final_points(objects, frame1)

# C:/proj/2d_to_3d/venv/Scripts/python.exe -m lifting.point_tracking.find_moving_pixels