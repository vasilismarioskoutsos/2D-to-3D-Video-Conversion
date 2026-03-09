import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy import interpolate

def generate_gaussian(size, sigma):
    """
    Generate a 2D Gaussian pattern based on the distance to the center.

    Args:
        size (tuple): (height, width) of the output pattern.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: 2D Gaussian pattern.
    """
    height, width = size
    # Create a coordinate grid
    y = torch.arange(0, height).view(-1, 1) / height
    x = torch.arange(0, width).view(1, -1) / width
    
    # Compute the center coordinates
    center_y, center_x = 0.5, 0.5
    
    # Compute the squared distance from each point to the center
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    
    # Apply the Gaussian function
    gaussian = torch.exp(-distance_squared / (2 * sigma ** 2))
    return gaussian

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val += val
        self.count += n
        self.avg = self.val / self.count

class InferenceWrapper(object):
    def __init__(self, model, scale=0, train_size=None, pad_to_train_size=False, tiling=False):
        self.model = model
        self.train_size = train_size
        self.scale = scale
        self.pad_to_train_size = pad_to_train_size
        self.tiling = tiling

    def inference_padding(self, image):
        h, w = self.train_size
        H, W = image.shape[2:]
        pad_h = max(h - H, 0)
        pad_w = max(w - W, 0)
        padded_image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)
        return padded_image, pad_h, pad_w

    def patch_inference(self, image1, image2, patches, tile_h, tile_w):
        output = None
        n, _, h, w = image1.shape
        valid = torch.zeros((n, h, w), device=image1.device)
        for h_ij, w_ij in patches:
            hl, hr = h_ij, h_ij + tile_h
            wl, wr = w_ij, w_ij + tile_w
            weight = generate_gaussian((hr - hl, wr - wl), 1).to(image1.device).unsqueeze(0)
            image1_ij = image1[:, :, hl: hr, wl: wr]
            image2_ij = image2[:, :, hl: hr, wl: wr]
            output_ij = self.model(image1_ij, image2_ij)
            valid[:, hl: hr, wl: wr] += weight
            if output is None:
                output = {}
                for key in output_ij.keys():
                    if 'flow' in key:
                        output[key] = [torch.zeros((n, 2, h, w), device=image1.device) for _ in range(len(output_ij[key]))]
                    elif 'info' in key:
                        output[key] = [torch.zeros((n, 4, h, w), device=image1.device) for _ in range(len(output_ij[key]))]
                    else:
                        output[key] = [torch.zeros((n, 2, h, w), device=image1.device) for _ in range(len(output_ij[key]))]

            for i in range(len(output_ij['flow'])):
                for key in output.keys():
                    output[key][i][:, :, hl: hr, wl: wr] += weight * output_ij[key][i]
        
        for i in range(len(output['flow'])):
            for key in output.keys():
                output[key][i] /= valid.unsqueeze(1)
        
        return output

    def forward_flow(self, image1, image2):
        H, W = image1.shape[2:]
        if self.pad_to_train_size:
            image1, inf_pad_h, inf_pad_w = self.inference_padding(image1)
            image2, inf_pad_h, inf_pad_w = self.inference_padding(image2)
        else:
            inf_pad_h, inf_pad_w = 0, 0

        if self.tiling and self.pad_to_train_size:
            h, w = image1.shape[2:]
            tile_h, tile_w = self.train_size
            step_h, step_w = tile_h // 4 * 3, tile_w // 4 * 3
            patches = []
            for i in range(0, h, step_h):
                for j in range(0, w, step_w):
                    h_ij = max(min(i, h - tile_h), 0)
                    w_ij = max(min(j, w - tile_w), 0)
                    patches.append((h_ij, w_ij))

            # remove duplicates
            patches = list(set(patches))
        else:
            h, w = image1.shape[2:]
            tile_h, tile_w = h, w
            patches = [(0, 0)]

        output = self.patch_inference(image1, image2, patches, tile_h, tile_w)

        for i in range(len(output['flow'])):
            for key in output.keys():
                output[key][i] = output[key][i][:, :, inf_pad_h // 2: inf_pad_h // 2 + H, inf_pad_w // 2: inf_pad_w // 2 + W]

        return output
    
    def calc_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self.scale, mode='bilinear', align_corners=True)
        img2 = F.interpolate(image2, scale_factor=2 ** self.scale, mode='bilinear', align_corners=True)
        H, W = img1.shape[2:]
        output = self.forward_flow(img1, img2)
        for i in range(len(output['flow'])):
            for key in output.keys():
                if 'flow' in key:
                    output[key][i] = F.interpolate(output[key][i], scale_factor=0.5 ** self.scale, mode='bilinear', align_corners=True) * (0.5 ** self.scale)
                else:
                    output[key][i] = F.interpolate(output[key][i], scale_factor=0.5 ** self.scale, mode='bilinear', align_corners=True)
        return output

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def load_ckpt(model, path):
    """ Load checkpoint """
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

def resize_data(img1, img2, flow, factor=1.0):
    _, _, h, w = img1.shape
    h = int(h * factor)
    w = int(w * factor)
    img1 = F.interpolate(img1, (h, w), mode='area')
    img2 = F.interpolate(img2, (h, w), mode='area')
    flow = F.interpolate(flow, (h, w), mode='area') * factor
    return img1, img2, flow

class Padder:
    """ Pads images such that dimensions are divisible by factor """
    def __init__(self, dims, mode='sintel', factor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht + 8) // factor) + 1) * factor - self.ht
        pad_wd = (((self.wd + 8) // factor) + 1) * factor - self.wd
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, x):
        return F.pad(x, self._pad, mode='constant', value=0)

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def transform(T, p):
    assert T.shape == (4,4)
    return np.einsum('H W j, i j -> H W i', p, T[:3,:3]) + T[:3, 3]

def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum('H W, H W j, i j -> H W i', depth1, img_1_coords, np.linalg.inv(K1))
    rel_pose = pose2 @ np.linalg.inv(pose1)
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum('H W j, i j -> H W i', cam2_coords, K2))

def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data['T0'], data['T1'], data['K0'], data['K1'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0
    H, W = depth1.shape
    coords1 = reproject(depth1, data['T1'], data['T0'], data['K1'], data['K0'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0
    return flow_01, flow_10

def check_cycle_consistency(flow_01, flow_10):
    H, W = flow_01.shape[:2]
    new_coords = flow_01 + np.stack(
        np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1
    )
    flow_reprojected = cv2.remap(
        flow_10, new_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR
    )
    cycle = flow_reprojected + flow_01
    cycle = np.linalg.norm(cycle, axis=-1)
    mask = (cycle < 0.1 * min(H, W)).astype(np.float32)
    return mask