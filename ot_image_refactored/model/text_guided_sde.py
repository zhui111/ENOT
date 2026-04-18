import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.silu(input)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.dim = dim
        self.scale = scale
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        input = input * self.scale + 1
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class TextGuidedSDE(nn.Module):
    def __init__(self, shift_model, epsilon, n_steps,
                 time_dim, n_last_steps_without_noise,
                 use_positional_encoding=True, use_gradient_checkpoint=False,
                 predict_shift=True, image_input=True): # 注意：这里 image_input 默认改为 True
        super().__init__()
        self.shift_model = shift_model
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_last_steps_without_noise = n_last_steps_without_noise
        self.use_positional_encoding = use_positional_encoding
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.times = np.linspace(0, 1, n_steps+1).tolist()
        self.predict_shift = predict_shift
        self.image_input = image_input
        
        self.time = nn.Sequential(
            TimeEmbedding(time_dim, scale=n_steps),
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
    
    def forward(self, x0, context=None, return_trajectory=True):            
        t0 = 0.0
        trajectory = [x0]
        times = [t0]
        shifts = []

        x, t = x0, t0

        for i, t_next in enumerate(self.times[1:]):

            if i >= len(self.times[1:]) - self.n_last_steps_without_noise:
                # 透传 context
                x, shift = self._step(x, t, t_next - t, context=context, add_noise=False)
            else:
                # 透传 context
                x, shift = self._step(x, t, t_next - t, context=context, add_noise=True)

            t = t_next
            if return_trajectory:
                trajectory.append(x)
                times.append(t)
                shifts.append(shift)
                    
        if not return_trajectory:
            trajectory.append(x)
            times.append(t)
            shifts.append(shift)
            
        trajectory = torch.stack(trajectory, dim=1)
        times = torch.tensor(times)[None, :].repeat(trajectory.shape[0], 1).to(x0.device)
        shifts = torch.stack(shifts, dim=1)
        
        return trajectory, times, shifts
    
    def _step(self, x, t, delta_t, context=None, add_noise=True):
        if self.predict_shift:
            shift_dt = self._get_shift(x, t, context) # 透传 context
            shifted_x = x + shift_dt
            shift = shift_dt / (torch.tensor(delta_t).to(x.device))
        else:
            shifted_x = self._get_shift(x, t, context) # 透传 context
            shift = (shifted_x - x) / (torch.tensor(delta_t).to(x.device))
            
        noise = self._sample_noise(x, delta_t)
        
        if add_noise:
            return shifted_x + noise, shift
        
        return shifted_x, shift

    def _get_shift(self, x, t, context=None):
        batch_size = x.shape[0]
        # --- 时间步编码 (保持原样) ---
        if self.use_positional_encoding:
            t = torch.tensor(t).repeat(batch_size).to(x.device)
            t = self.time(t)
            if self.image_input:
                t = t[:, :, None, None]
        else:
            t = torch.tensor(t).repeat(batch_size)[:, None].to(x.device)
            if self.image_input:
                t = t[:, None, None, None]
        
        # --- 调用网络获取速度场 ---
        if self.use_gradient_checkpoint:
            if context is not None:
                return torch.utils.checkpoint.checkpoint(self.shift_model, x, t, context, use_reentrant=False)
            return torch.utils.checkpoint.checkpoint(self.shift_model, x, t, use_reentrant=False)
        
        # 对于图像输入 (我们目前处于 Latent 隐空间，也是 [B, C, H, W] 格式)
        if self.image_input:
            if context is not None:
                return self.shift_model(x, t, context=context) # 触发带有前置 Attention 的网络
            return self.shift_model(x, t)                      # 兼容无文本引导的模式
            
        x = torch.cat((x, t), dim=1)
        if context is not None:
            return self.shift_model(x, context=context)
        return self.shift_model(x)
        
    def _sample_noise(self, x, delta_t):
        noise = math.sqrt(self.epsilon) * math.sqrt(delta_t) * torch.randn_like(x)
        return noise
            
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def integrate(values, times):
    """用于在损失函数中计算 \int ||v_t||^2 dt"""
    deltas = times[0, 1:] - times[0, :-1]
    if values.device.type == "cuda":
        deltas = deltas.cuda()
    return (values * deltas[None, :]).sum(dim=1)

