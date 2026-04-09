# fix_images_cvd.py
import sys
sys.path.insert(0, r'C:\vasilis\2D-to-3D-Video-Conversion\4DGS\Instant4D')
import numpy as np
import cv2
import json
import os

# Load the CVD output (has better depths + original images)
d = np.load('outputs/bike_3d_result_sgd_cvd_hr.npz')
images = d['images']  # (97, 232, 416, 3)

# Save images
img_dir = 'example/bike_3d_result/images'
os.makedirs(img_dir, exist_ok=True)
for i, img in enumerate(images):
    path = f'{img_dir}/{i+1:05d}.png'
    cv2.imwrite(path, img[:, :, ::-1])
print(f'Saved {len(images)} images')

# Fix JSON paths
for split in ['train', 'test']:
    json_path = f'example/bike_3d_result/transforms_{split}.json'
    with open(json_path) as f:
        data = json.load(f)
    for frame in data['frames']:
        fname = frame['file_path'].split('/')[-1]
        frame['file_path'] = f'images/{fname}'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Fixed {json_path}')