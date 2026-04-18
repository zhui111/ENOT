import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from configs.config import Config

class LatentVAE(nn.Module):
    def __init__(self, model_id=None):
        super().__init__()
        self.model_id = model_id if model_id else Config.VAE_PATH
        
        print(f"[VAE] 正在从本地加载权重: {self.model_id}")
        
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_id, 
                local_files_only=True
            )
        except Exception as e:
            print(f"❌ 加载 VAE 失败，请检查路径是否正确: {self.model_id}")
            print(f"错误信息: {e}")
            raise

        self.vae.to(Config.DEVICE).eval()
        self.vae.requires_grad_(False) # 冻结
        
        self.scaling_factor = 0.18215 

    @torch.no_grad()
    def encode(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """像素空间 -> 隐空间"""
        x_pixel = x_pixel.clamp(-1.0, 1.0).to(dtype=torch.float32, device=Config.DEVICE)
        latent_dist = self.vae.encode(x_pixel).latent_dist
        z = latent_dist.mode()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z_latent: torch.Tensor) -> torch.Tensor:
        """隐空间 -> 像素空间"""
        z_latent = (z_latent / self.scaling_factor).to(dtype=torch.float32, device=Config.DEVICE)
        x_pixel = self.vae.decode(z_latent).sample
        return x_pixel.clamp(-1.0, 1.0)