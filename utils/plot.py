import numpy as np
import meshplot as mp
import torch

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.vstack([[fx, 0., px, 0.],
                      [0., fy, py, 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])

def plot_3d_pointcloud(pixels, depth, img_shape, edges=None, point_size=0.1, zNear=0.85, zFar=4.0, fov_deg=49.1):

    indices = torch.tensor(pixels.T, dtype=torch.int64)
    z_3d = torch.tensor(depth)
    
    depth = 2.0 * zNear * zFar / (zFar + zNear - z_3d * (zFar - zNear))
    
    height, width = img_shape
    K = intrinsic_from_fov(height, width, fov_deg)
    
    cam_coords = []
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Loop through each pixel
    for i, (v, u) in enumerate(indices):
            # Apply equation in fig 3
            x = -(v - v0) * depth[i] / fx
            y =  (u - u0) * depth[i] / fy
            z = -depth[i]
            cam_coords.append([y.item(), x.item(), z.item()])
    cam_coords_tensor = torch.FloatTensor(cam_coords)

    if edges is not None:
        mp.plot(cam_coords_tensor.numpy(), edges, c=depth.numpy(), shading={"point_size": point_size,"colormap": "plasma_r", "line_color": "red"})
    else:
        mp.plot(cam_coords_tensor.numpy(), c=depth.numpy(), shading={"point_size": point_size,"colormap": "plasma_r"})