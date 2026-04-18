import os
import torch
import torch.nn as nn
import clip
from contextlib import nullcontext
from configs.config import Config

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        
        # 直接拿到带有 /root/ 的文件绝对路径
        weight_path = model_path if model_path else getattr(Config, "CLIP_CACHE_DIR", "/root/autodl-tmp/clip/RemoteCLIP-ViT-L-14.pt")
        
        # 加入物理文件检查，防止静默报错
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"❌ 致命错误：找不到文件 {weight_path}！请检查 AutoDL 的数据盘路径。")
            
        print(f"[CLIP] 正在物理强制加载本地权重文件: {weight_path}")
        
        try:
            # 第一个参数直接传入 weight_path 变量！
            self.model, self.preprocess = clip.load(
                weight_path, 
                device=Config.DEVICE, 
                jit=False 
            )
        except Exception as e:
            print(f"❌ 物理加载本地权重失败: {e}")
            raise
            
        self.model.eval()
        self.model.requires_grad_(False) # 彻底冻结 CLIP，不参与梯度更新，节省显存

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """对特征进行 L2 归一化，稳定交叉注意力的点积运算"""
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        if not isinstance(texts, (list, tuple)) or len(texts) == 0:
            raise ValueError("输入必须是非空字符串列表。")
            
        # 1. 文本分词，截断至最大长度 77 (CLIP 标准长度)
        tokens = clip.tokenize(texts, truncate=True).to(Config.DEVICE)
        
        # 2. 提取 CLIP 的 Transformer 文本模块内部特征
        m = self.model
        
        # 开启混合精度以加速推理并节省显存 (CLIP 默认在 CUDA 上使用 fp16 效果最好)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if Config.DEVICE == "cuda" else nullcontext()
        with autocast_context:
            # 获取初始 Embedding 并加上位置编码
            x = m.token_embedding(tokens).float()          # [B, 77, d_model]
            x = x + m.positional_embedding.float()
            
            # Transformer 期待的输入维度是 [SeqLen, Batch, d_model]
            x = x.permute(1, 0, 2)
            x = m.transformer(x)
            x = x.permute(1, 0, 2)                         # 换回 [B, 77, d_model]
            # 通过最后的 LayerNorm 和 投影层对齐维度
            x = m.ln_final(x).float()
            x = x @ m.text_projection.float()              # [B, 77, D_out]
        # 3. 对每个 Token 的特征进行 L2 归一化
        return self._l2_normalize(x)
import torch.nn.functional as F



class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        weight_path = model_path if model_path else getattr(Config, "CLIP_CACHE_DIR", "/root/autodl-tmp/clip/RemoteCLIP-ViT-L-14.pt")
        # 同样加载模型，但不参与训练
        self.model, _ = clip.load(weight_path, device=Config.DEVICE, jit=False)
        self.model.eval()
        self.model.requires_grad_(False)
        
        # CLIP 的标准归一化参数
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    # 没有使用这两个
    # @torch.no_grad()
    # def get_text_features(self, texts: list[str]) -> torch.Tensor:
    #     """提取文本的全局特征 [B, 768]"""
    #     tokens = clip.tokenize(texts, truncate=True).to(Config.DEVICE)
    #     features = self.model.encode_text(tokens)
    #     return features / (features.norm(dim=-1, keepdim=True) + 1e-8)

    # def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
    #     """提取图像的全局特征 [B, 768]。注意：输入 images 需为 [-1, 1] 范围的张量"""
    #     # 1. 缩放到 CLIP 需要的 224x224
    #     images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
    #     # 2. 从 [-1, 1] 转换到 [0, 1]
    #     images = (images + 1.0) / 2.0
    #     # 3. CLIP 标准化
    #     images = (images - self.mean) / self.std
    #     # 4. 提取特征  
    #     features = self.model.encode_image(images)
    #     return features / (features.norm(dim=-1, keepdim=True) + 1e-8)