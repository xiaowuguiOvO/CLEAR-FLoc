from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import *


def localize(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # probablility is -l1norm
    prob_vol = torch.stack(
        [
            -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H,W,O)
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive

    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
    
    if pred_y.numel() == 0:
        # Handle case where no max is found (e.g., NaNs)
        pred = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        # Take the first occurrence if multiple maxima exist
        pred_y = pred_y[0:1]
        pred_x = pred_x[0:1]
        orn = orientations[pred_y, pred_x]
        # from orientation indices to radians
        orn = orn / orn_slice * 2 * torch.pi
        pred = torch.cat((pred_x.float(), pred_y.float(), orn.float()))
    
    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )
def localize_unloc(
    desdf: torch.tensor,     # (H, W, O), counter clockwise
    d_hat: torch.tensor,     # (V,) - 预测深度
    b_hat: torch.tensor,     # (V,) - 预测不确定性
    orn_slice: int = 36,          # 方向切片数 (e.g., 36)
    return_np: bool = True
) -> tuple:
    """
    模仿原版 localize 函数的结构 (滑动窗口), 但使用 UnLoc (Eq. 1) 的
    不确定性感知公式来替换 L1-norm 和 lambda。

    Input:
        desdf: (H, W, O) - 预计算的地面真值 (GT) 深度
        d_hat: (V,) - 模型预测的深度
        b_hat: (V,) - 模型预测的不确定性
        orn_slice: (int) - 方向的数量 O (必须与 desdf.shape[2] 匹配)
        return_np: (bool) - 是否返回 numpy 数组
    Output:
        prob_vol: 概率体积 (H, W, O)
        prob_dist: 概率分布 (H, W)
        orientations: 最佳方向索引 (H, W)
        pred: (3, ) 预测姿态 [x, y, theta]
    """

    # --- 1. 模仿原版的设置 ---
    
    # 翻转 d_hat 和 b_hat，使其与 desdf 的逆时针方向匹配
    # (这模仿了原版中的 torch.flip(rays, [0]))
    d_hat = torch.flip(d_hat, [0])
    b_hat = torch.flip(b_hat, [0])

    O = desdf.shape[2]  # 方向数, e.g., 36
    V = d_hat.shape[0]  # 射线数, e.g., 11

    # 扩展 d_hat 和 b_hat 以便在循环中广播
    # (V,) -> (1, 1, V)
    d_hat = d_hat.reshape((1, 1, -1))
    b_hat = b_hat.reshape((1, 1, -1))

    # 模仿原版的循环填充
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")
    # pad_desdf shape is (H, W, O + V), e.g., (112, 67, 47)

    eps = 1e-6 # 确保数值稳定性

    # --- 2. 模仿原版的循环，但替换核心数学 ---
    
    log_prob_list = []
    
    # 在 O (e.g., 36) 个方向上循环
    for i in range(O):
        # 获取 (H, W, V) 的地面真值深度切片
        # e.g., (112, 67, 11)
        d_gt_slice = pad_desdf[:, :, i : i + V]
        
        # --- UnLoc 公式 (Eq. 1) ---
        # 在这里替换了 -torch.norm(..., p=1.0)
        
        # |d_gt - d_hat|
        abs_error = torch.abs(d_gt_slice - d_hat) # (H, W, V)
        
        # -log(2b)
        log_b_term = -torch.log(2 * b_hat + eps) # (1, 1, V)
        
        # -|error| / b
        error_term = -abs_error / (b_hat + eps) # (H, W, V)
        
        # log(P_j) = log_b_term + error_term
        log_likelihood_per_ray = log_b_term + error_term # (H, W, V)
        
        # log(P_total) = Σ log(P_j)
        # 沿射线维度 (V) 求和
        log_prob_sum = torch.sum(log_likelihood_per_ray, dim=2) # (H, W)
        # --- UnLoc 公式结束 ---
        
        log_prob_list.append(log_prob_sum)

    # 模仿原版：将 (H, W) 的列表堆叠成 (H, W, O)
    prob_vol_log = torch.stack(log_prob_list, dim=2)

    # 模仿原版：从 log 空间转换回概率空间
    # (原版中是 torch.exp(prob_vol / lambd))
    prob_vol = torch.exp(prob_vol_log)

    # --- 3. 模仿原版的 Maxpooling 和 Pred ---
    
    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    # (添加 [0] 来处理
    pred_y_all, pred_x_all = torch.where(prob_dist == prob_dist.max())
    if pred_y_all.shape[0] == 0:
        # 如果所有概率都为 0 (下溢)，则返回一个默认值
        H, W, _ = desdf.shape
        pred_y, pred_x, orn_idx = torch.tensor(H//2), torch.tensor(W//2), torch.tensor(0)
    else:
        pred_y = pred_y_all[0] # 取第一个匹配
        pred_x = pred_x_all[0] # 取第一个匹配
        orn_idx = orientations[pred_y, pred_x]

    # from orientation indices to radians
    orn_rad = (orn_idx.float() / orn_slice) * 2 * torch.pi
    
    # 组合 pred 向量 [x, y, theta]
    pred = torch.stack([
        pred_x.to(orn_rad.dtype), 
        pred_y.to(orn_rad.dtype), 
        orn_rad
    ])

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )
        
def get_ray_from_depth(d, V=11, dv=10, a0=None, F_W=3 / 8):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    
    # Handle NaNs from linear interpolation (out of bounds)
    if np.isnan(interp_d).any():
        interp_d_nearest = griddata(np.arange(W).reshape(-1, 1), d, w, method="nearest")
        mask = np.isnan(interp_d)
        interp_d[mask] = interp_d_nearest[mask]
        # Final safety check
        interp_d = np.nan_to_num(interp_d, nan=0.0)

    rays = interp_d / np.cos(angles)

    return rays

def get_ray_from_depth_unloc(d_in, b_in, V=11, dv=10, a0=None, F_W=3 / 8):
    """
    "模仿" get_ray_from_depth 的非均匀采样，但同时处理 d 和 b。
    
    1. 计算非均匀的采样坐标 w (V-dim)
    2. 使用 w 将 d_in 和 b_in (W_in-dim) 插值到 V-dim
    3. 将插值后的 (d, b) 传播到射线空间 (r, b_r)
    
    Input:
        d_in: (W_in,) 1d depths from model (e.g., 40-dim)
        b_in: (W_in,) 1d uncertainties from model (e.g., 40-dim)
        V: number of output rays (e.g., 11)
    Output:
        r_out: (V,) 插值后的射线长度 (r_hat)
        b_r_out: (V,) 插值后并传播的射线不确定性 (b_r_hat)
    """
    W_in = d_in.shape[0] # e.g., 40
    
    # 1. (同原函数) 计算 11 个射线的角度
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi
    
    # 2. (同原函数) 计算 11 个非均匀的采样坐标 'w'
    if a0 is None:
        w = np.tan(angles) * W_in * F_W + (W_in - 1) / 2
    else:
        w = np.tan(angles) * W_in * F_W + a0
        
    # 3. 插值：使用相同的 'w' 坐标来插值 d 和 b
    x_in = np.arange(W_in).reshape(-1, 1) # 输入坐标 (0..39)
    
    d_interp = griddata(x_in, d_in, w, method="linear")
    b_interp = griddata(x_in, b_in, w, method="linear")

    # 处理因插值（例如 w 超出 0-39 范围）导致的 NaN
    d_interp = np.nan_to_num(d_interp, nan=np.nanmean(d_interp))
    b_interp = np.nan_to_num(b_interp, nan=np.nanmean(b_interp))

    # 4. (同原函数) 从深度传播到射线长度
    cos_angles = np.cos(angles)
    
    # 避免 90 度角时除以零
    cos_angles[np.abs(cos_angles) < 1e-6] = 1e-6

    # r = d / cos(a)
    r_out = d_interp / cos_angles
    
    # 传播不确定性: b_r = b / |cos(a)|
    b_r_out = b_interp / np.abs(cos_angles)
    
    return r_out, b_r_out

def transit(
    prob_vol,
    transition,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Input:
        prob_vol: torch.tensor(H, W, O), probability volume before the transition
        transition: ego motion
        sig_o: stddev of rotation
        sig_x: stddev in x translation
        sig_w: stddev in y translation
        tsize: translational filter size
        rsize: rotational filter size
        resolution: resolution of the grid [m/pixel]
    """
    H, W, O = list(prob_vol.shape)
    # construction O filters
    filters_trans, filter_rot = get_filters(
        transition,
        O,
        sig_o=sig_o,
        sig_x=sig_x,
        sig_y=sig_y,
        tsize=tsize,
        rsize=rsize,
        resolution=resolution,
    )  # (O, 5, 5), (5,)

    # set grouped 2d convolution, O as channels
    prob_vol = prob_vol.permute((2, 0, 1))  # (O, H, W)

    # convolve with the translational filters
    # NOTE: make sure the filter is convolved correctly need to flip
    prob_vol = F.conv2d(
        prob_vol,
        weight=filters_trans.unsqueeze(1).flip([-2, -1]),
        bias=None,
        groups=O,
        padding="same",
    )  # (O, H, W)

    # convolve with rotational filters
    # reshape as batch
    prob_vol = prob_vol.permute((1, 2, 0))  # (H, W, O)
    prob_vol = prob_vol.reshape((H * W, 1, O))  # (HxW, 1, O)
    prob_vol = F.pad(
        prob_vol, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
    )
    prob_vol = F.conv1d(
        prob_vol, weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
    )  # TODO (HxW, 1, O)

    # reshape
    prob_vol = prob_vol.reshape([H, W, O])  # (H, W, O)
    # normalize
    prob_vol = prob_vol / prob_vol.sum()

    return prob_vol


def get_filters(
    transition,
    O=36,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Return O different filters according to the ego-motion
    Input:
        transition: torch.tensor (3,), ego motion
    Output:
        filters_trans: torch.tensor (O, 5, 5)
                    each filter is (fH, fW)
        filters_rot: torch.tensor (5)
    """
    # NOTE: be careful about the orienation order, what is the orientation of the first layer?

    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # calculate center of the gaussian for 36 orientations
    # center for orientation stays the same
    center_o = transition[-1]
    # center_x and center_y depends on the orientation, in total O different, rotate
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=transition.device)
        / O
        * 2
        * torch.pi
    )  # (O,)
    c_th = torch.cos(orns).reshape((O, 1, 1))  # (O, 1, 1)
    s_th = torch.sin(orns).reshape((O, 1, 1))  # (O, 1, 1)
    center_x = transition[0] * c_th - transition[1] * s_th  # (O, 1, 1)
    center_y = transition[0] * s_th + transition[1] * c_th  # (O, 1, 1)

    # add uncertainty
    filters_trans = torch.exp(
        -((grid_x - center_x) ** 2) / (sig_x**2) - (grid_y - center_y) ** 2 / (sig_y**2)
    )  # (O, 5, 5)
    # normalize
    filters_trans = filters_trans / filters_trans.sum(-1).sum(-1).reshape((O, 1, 1))

    # rotation filter
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=transition.device)
        / O
        * 2
        * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o**2))  # (5)

    return filters_trans, filter_rot
