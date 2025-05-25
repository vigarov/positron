import torch

def lab2rgb(lab):
    L, a, b = torch.split(lab, 1, dim=-1)
    
    # LAB to XYZ
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0
    
    xyz_condition = torch.tensor(0.008856, device=lab.device, dtype=lab.dtype)
    
    x_cube = x ** 3
    x_linear = (x - 16.0/116.0) / 7.787
    x = torch.where(x_cube > xyz_condition, x_cube, x_linear) * 0.95047
    
    y_cube = y ** 3
    y_linear = (y - 16.0/116.0) / 7.787
    y = torch.where(y_cube > xyz_condition, y_cube, y_linear)
    
    z_cube = z ** 3
    z_linear = (z - 16.0/116.0) / 7.787
    z = torch.where(z_cube > xyz_condition, z_cube, z_linear) * 1.08883
    
    xyz = torch.cat([x, y, z], dim=-1)
    rgb_from_xyz = torch.tensor([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ], device=lab.device, dtype=lab.dtype)
    
    orig_shape = xyz.shape[:-1]
    xyz_reshaped = xyz.reshape(-1, 3)
    
    rgb = torch.matmul(xyz_reshaped, rgb_from_xyz.T)
    rgb = rgb.reshape(*orig_shape, 3)
    
    r, g, b = torch.split(rgb, 1, dim=-1)
    rgb_condition = torch.tensor(0.0031308, device=lab.device, dtype=lab.dtype)
    
    r_gamma = 1.055 * (r ** (1.0/2.4)) - 0.055
    r_linear = 12.92 * r
    r = torch.where(r > rgb_condition, r_gamma, r_linear)
    
    g_gamma = 1.055 * (g ** (1.0/2.4)) - 0.055
    g_linear = 12.92 * g
    g = torch.where(g > rgb_condition, g_gamma, g_linear)
    
    b_gamma = 1.055 * (b ** (1.0/2.4)) - 0.055
    b_linear = 12.92 * b
    b = torch.where(b > rgb_condition, b_gamma, b_linear)
    
    r = torch.clamp(r, 0.0, 1.0) * 255.0
    g = torch.clamp(g, 0.0, 1.0) * 255.0
    b = torch.clamp(b, 0.0, 1.0) * 255.0
    
    rgb = torch.cat([r, g, b], dim=-1)
    return rgb


def rgb2lab(rgb):
    r, g, b = torch.split(rgb / 255.0, 1, dim=-1)
    rgb_condition = torch.tensor(0.04045, device=rgb.device, dtype=rgb.dtype)
    
    r_gamma = ((r + 0.055) / 1.055) ** 2.4
    r_linear = r / 12.92
    r = torch.where(r > rgb_condition, r_gamma, r_linear)
    
    g_gamma = ((g + 0.055) / 1.055) ** 2.4
    g_linear = g / 12.92
    g = torch.where(g > rgb_condition, g_gamma, g_linear)
    
    b_gamma = ((b + 0.055) / 1.055) ** 2.4
    b_linear = b / 12.92
    b = torch.where(b > rgb_condition, b_gamma, b_linear)
    
    rgb = torch.cat([r, g, b], dim=-1)
    xyz_from_rgb = torch.tensor([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ], device=rgb.device, dtype=rgb.dtype)
    
    orig_shape = rgb.shape[:-1]
    rgb_reshaped = rgb.reshape(-1, 3)
    
    xyz = torch.matmul(rgb_reshaped, xyz_from_rgb.T)
    xyz = xyz.reshape(*orig_shape, 3)
    
    x, y, z = torch.split(xyz, 1, dim=-1)
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883
    
    xyz_condition = torch.tensor(0.008856, device=rgb.device, dtype=rgb.dtype)
    
    x_cbrt = x ** (1.0/3.0)
    x_linear = (7.787 * x) + 16.0/116.0
    x = torch.where(x > xyz_condition, x_cbrt, x_linear)
    
    y_cbrt = y ** (1.0/3.0)
    y_linear = (7.787 * y) + 16.0/116.0
    y = torch.where(y > xyz_condition, y_cbrt, y_linear)
    
    z_cbrt = z ** (1.0/3.0)
    z_linear = (7.787 * z) + 16.0/116.0
    z = torch.where(z > xyz_condition, z_cbrt, z_linear)
    
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    
    lab = torch.cat([L, a, b], dim=-1)
    return lab
