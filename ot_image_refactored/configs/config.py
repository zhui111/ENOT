import os
import torch
import random
import numpy as np

class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_ROOT = "/root/autodl-tmp"
    CSV_TRAIN = "/root/autodl-tmp/levir_mci_train.csv"
    VAE_PATH = os.getenv("SD_VAE", "/root/autodl-tmp/sd-vae-ft-mse")
    CLIP_CACHE_DIR = os.getenv("CLIP_CACHE", "/root/autodl-tmp/clip/RemoteCLIP-ViT-L-14.pt")
    
    IMG_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    
    # CSV 列名映射
    CSV_COL_SRC = "src"
    CSV_COL_EDT = "edt"
    CSV_COL_TEXT = "text"

    
    VAE_SCALE_FACTOR = 8           # VAE 降维倍数 (256 -> 32)
    LATENT_CHANNELS = 4            # VAE 输出通道
    LATENT_SIZE = IMG_SIZE // VAE_SCALE_FACTOR # 32
    TEXT_EMBED_DIM = 768           # CLIP 输出维度
    NUM_ATTN_HEADS = 4             # 前置交叉注意力的头数
    TIME_DIM = 128                # 前置交叉注意力的时间维度

    
    UNET_BASE_FACTOR = 32
    SDE_N_STEPS = 10               # 对应你原版的 integration steps
    EPSILON = 0.02                 # SDE 噪声强度
    DRIFT_CLIP = 0.25              # 速度场截断
    N_LAST_STEPS_WO_NOISE = 1      # 最后几步去噪
    PREDICT_SHIFT = True           # 是否预测速度场

    
    EPOCHS = 100
    LR_G = 3e-5
    LR_D = 3e-5
    SAVE_FREQ = 20
    RESUME_PATH = "/root/autodl-tmp/V30_TextGuided_ENOT/ckpt/ENOT_ep0100.pt"
    # Loss 权重 (融合你的设置)
    LAMBDA_SUP = 0.10              # Target 监督权重 (MSE/L1)
    ADV_WEIGHT = 1.0               # 判别器对抗损失权重
    OT_REG_WEIGHT = 1.0            # 动能正则化权重 (NORM_SQ_SCALE)
    DIR_WEIGHT = 2.0

    RUN_NAME = "enot_text_guided"
    OUT_DIR = f"./runs/{RUN_NAME}"
    SAMPLES_DIR = f"{OUT_DIR}/samples"
    CKPT_DIR = f"{OUT_DIR}/ckpt"
    @classmethod
    def setup(cls):
        """初始化随机种子并创建文件夹"""
        # 设种子
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        
        # 建文件夹
        os.makedirs(cls.SAMPLES_DIR, exist_ok=True)
        os.makedirs(cls.CKPT_DIR, exist_ok=True)

# 自动执行 Setup
Config.setup()