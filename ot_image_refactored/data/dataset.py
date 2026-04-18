import os
import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from configs.config import Config

class PairEditDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = True):
        super().__init__()
        self.is_train = is_train
        
        # 读取数据
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV 不存在：{csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # 预处理：缩放、转Tensor、归一化到 [-1, 1] (配合扩散/VAE模型标准)
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        src_path = str(row[Config.CSV_COL_SRC])
        edt_path = str(row[Config.CSV_COL_EDT])
        text = str(row[Config.CSV_COL_TEXT])

        # 1. 读取原图和目标图
        x_src = self._load_image(src_path)
        y_tgt = self._load_image(edt_path)

        # 2. 遥感专属数据增强 (仅训练时)
        if self.is_train:
            # 随机旋转 90/180/270
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                x_src = torch.rot90(x_src, k, dims=[-2, -1])
                y_tgt = torch.rot90(y_tgt, k, dims=[-2, -1])

            # 随机翻转
            if random.random() < 0.5:
                x_src = torch.flip(x_src, dims=[-1])
                y_tgt = torch.flip(y_tgt, dims=[-1])
            if random.random() < 0.5:
                x_src = torch.flip(x_src, dims=[-2])
                y_tgt = torch.flip(y_tgt, dims=[-2])
            
            # 轻微颜色抖动 (保持光谱特性，整体乘系数)
            if random.random() < 0.3:
                factor = 1.0 + (random.random() - 0.5) * 0.2
                x_src = (x_src * factor).clamp(-1.0, 1.0)
                y_tgt = (y_tgt * factor).clamp(-1.0, 1.0)

        return {
            "x_src": x_src,
            "y_tgt": y_tgt,
            "text": text
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        xs = torch.stack([b["x_src"] for b in batch], dim=0) # [B, 3, H, W]
        ys = torch.stack([b["y_tgt"] for b in batch], dim=0) # [B, 3, H, W]
        texts = [b["text"] for b in batch]                   # List[str]
        return {
            "x_src": xs,
            "y_tgt": ys,
            "text": texts
        }