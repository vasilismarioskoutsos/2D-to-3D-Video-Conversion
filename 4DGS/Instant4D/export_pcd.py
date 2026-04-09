import sys
sys.path.insert(0, r'C:\vasilis\2D-to-3D-Video-Conversion\4DGS\Instant4D')
import numpy as np
import struct

d = np.load('example/bike_3d_result/filtered_cvd.npz')
xyz = d['xyz']
rgb = (d['rgb'] * 255).clip(0, 255).astype('uint8')

print(f"Points: {len(xyz)}")
print(f"XYZ range: x={xyz[:,0].min():.2f} to {xyz[:,0].max():.2f}")

with open('example/bike_3d_result/scene_pointcloud.ply', 'wb') as f:
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    f.write(header.encode())
    for i in range(len(xyz)):
        f.write(struct.pack('fff', xyz[i,0], xyz[i,1], xyz[i,2]))
        f.write(struct.pack('BBB', rgb[i,0], rgb[i,1], rgb[i,2]))

print("Saved scene_pointcloud.ply")