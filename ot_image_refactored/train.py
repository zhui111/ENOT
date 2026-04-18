import os
import argparse
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

from configs.config import Config
from data.dataset import PairEditDataset
from model.vae import LatentVAE
from model.text_encoder import CLIPTextEncoder          
from model.cunet import TextGuidedCUNet
from model.text_guided_sde import TextGuidedSDE, integrate
from model.discriminator import ResNet_D


def parse_args():
    parser = argparse.ArgumentParser(description="Text-Guided ENOT Training")
    parser.add_argument("--run-name",          type=str,   required=True)
    parser.add_argument("--out-dir",           type=str,   required=True)
    parser.add_argument("--csv",               type=str,   required=True)
    parser.add_argument("--csv-val",           type=str,   required=True)
    parser.add_argument("--device",            type=str,   default="cuda")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--img-size",          type=int,   default=256)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--unet-base",         type=int,   default=32)
    parser.add_argument("--time-dim",          type=int,   default=128)
    parser.add_argument("--latent-ch",         type=int,   default=4)
    parser.add_argument("--epsilon",           type=float, default=0.05)
    parser.add_argument("--n-last-wo-noise",   type=int,   default=1)
    parser.add_argument("--lambda-sup",        type=float, default=0.1)
    parser.add_argument("--norm-sq-scale",     type=float, default=1.0)
    parser.add_argument("--enot-adv-weight",   type=float, default=1.0)
    parser.add_argument("--dir-weight",        type=float, default=0.5)   # ← 新增，从0.5开始
    parser.add_argument("--viz-interval",      type=int,   default=500)
    parser.add_argument("--save-every-epochs", type=int,   default=5)
    parser.add_argument("--viz-nrow",          type=int,   default=4)
    parser.add_argument("--resume",            type=str,   default=None)
    return parser.parse_args()


def get_epsilon(epoch, total_epochs, max_eps):
    warmup_ratio  = 0.2
    warmup_epochs = int(total_epochs * warmup_ratio)
    if epoch <= warmup_epochs:
        return max_eps * (epoch / max(1, warmup_epochs))
    return max_eps


def draw_and_save_grid(x_src, y_tgt, x_gen, text_list, save_path, writer, tag, step):
    n_viz = x_src.shape[0]
    fig, axes = plt.subplots(3, n_viz, figsize=(n_viz * 3.5, 10))
    if n_viz == 1:
        axes = axes[:, None]
    for i in range(n_viz):
        src_img = ((x_src[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        tgt_img = ((y_tgt[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        gen_img = ((x_gen[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(src_img); axes[0, i].axis("off"); axes[0, i].set_title("Source",  fontsize=12)
        axes[1, i].imshow(tgt_img); axes[1, i].axis("off"); axes[1, i].set_title("Target",  fontsize=12)
        axes[2, i].imshow(gen_img); axes[2, i].axis("off")
        axes[2, i].set_title("\n".join(textwrap.wrap(text_list[i], width=28)), fontsize=10, color="red")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if writer:
        writer.add_figure(tag, fig, step)
    plt.close(fig)


def main():
    args = parse_args()

    # ── Config 同步 ──────────────────────────────────────────
    Config.SEED              = args.seed
    Config.DEVICE            = args.device
    Config.DATA_ROOT         = args.out_dir
    Config.CSV_TRAIN         = args.csv
    Config.IMG_SIZE          = args.img_size
    Config.BATCH_SIZE        = args.batch_size
    Config.EPOCHS            = args.epochs
    Config.LR_G              = args.lr
    Config.LR_D              = args.lr
    Config.UNET_BASE_FACTOR  = args.unet_base
    Config.TIME_DIM          = args.time_dim
    Config.LATENT_CHANNELS   = args.latent_ch
    Config.EPSILON           = args.epsilon
    Config.N_LAST_STEPS_WO_NOISE = args.n_last_wo_noise
    Config.LAMBDA_SUP        = args.lambda_sup
    Config.OT_REG_WEIGHT     = args.norm_sq_scale
    Config.ADV_WEIGHT        = args.enot_adv_weight
    Config.DIR_WEIGHT        = args.dir_weight

    
    Config.LATENT_SIZE = Config.IMG_SIZE // Config.VAE_SCALE_FACTOR  # 256//8 = 32

    # INTEGRAL_SCALE 归一化系数，防止 OT 损失随隐变量维度放大
    INTEGRAL_SCALE = 1.0 / (
        Config.LATENT_CHANNELS * Config.LATENT_SIZE * Config.LATENT_SIZE
    )  # 1/(4*32*32) = 1/4096

    # ── 目录 ────────────────────────────────────────────────
    run_out_dir       = os.path.join(args.out_dir, args.run_name)
    samples_train_dir = os.path.join(run_out_dir, "samples", "train")
    samples_val_dir   = os.path.join(run_out_dir, "samples", "val")
    ckpt_dir          = os.path.join(run_out_dir, "ckpt")
    tb_log_dir        = os.path.join(run_out_dir, "tb_logs")
    for d in [samples_train_dir, samples_val_dir, ckpt_dir, tb_log_dir]:
        os.makedirs(d, exist_ok=True)
    Config.CKPT_DIR = ckpt_dir

    writer = SummaryWriter(log_dir=tb_log_dir)

    # ── 数据 ────────────────────────────────────────────────
    train_dataset = PairEditDataset(args.csv,     is_train=True)
    val_dataset   = PairEditDataset(args.csv_val, is_train=False)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  num_workers=4,
                               collate_fn=PairEditDataset.collate_fn)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size,
                               shuffle=False, num_workers=4,
                               collate_fn=PairEditDataset.collate_fn)

    # ── 模型 ────────────────────────────────────────────────
    device     = torch.device(Config.DEVICE)
    vae        = LatentVAE().to(device)
    clip_model = CLIPTextEncoder().to(device)

    net_G = TextGuidedCUNet(
        in_channels    = Config.LATENT_CHANNELS,
        n_classes      = Config.LATENT_CHANNELS,
        z_channels     = Config.TIME_DIM,
        text_dim       = 768,
        base_factor    = Config.UNET_BASE_FACTOR,
        latent_size    = Config.LATENT_SIZE,       # ← 传入，供 dir_proj 计算输入维度
    ).to(device)

    sde = TextGuidedSDE(
        shift_model               = net_G,
        epsilon                   = Config.EPSILON,
        n_steps                   = 10,
        time_dim                  = Config.TIME_DIM,
        n_last_steps_without_noise= Config.N_LAST_STEPS_WO_NOISE,
        predict_shift             = True,
        use_gradient_checkpoint   = True,
    ).to(device)

    net_D = ResNet_D(
        size = Config.LATENT_SIZE,
        nc   = Config.LATENT_CHANNELS,
    ).to(device)

    opt_G = optim.Adam(net_G.parameters(), lr=Config.LR_G, betas=(0.5, 0.9))
    opt_D = optim.Adam(net_D.parameters(), lr=Config.LR_D, betas=(0.5, 0.9))

    # ── 断点续训 ─────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"🔍 正在从断点恢复训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        net_G.load_state_dict(ckpt["net_G"])
        net_D.load_state_dict(ckpt["net_D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        global_step = (start_epoch - 1) * len(train_loader)
        print(f"✅ 成功加载！将从第 {start_epoch} 轮继续训练。")

    print(f"🚀 Started Training: {args.run_name} | Logs: {tb_log_dir}")
    log_interval = 50

    # ════════════════════════════════════════════════════════
    #                       训练主循环
    # ════════════════════════════════════════════════════════
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        net_G.train()
        net_D.train()

        current_eps = get_epsilon(epoch, Config.EPOCHS, Config.EPSILON)
        sde.set_epsilon(current_eps)
        writer.add_scalar("HyperParams/Epsilon", current_eps, epoch)
        last_log_time = time.time()

        for step, batch in enumerate(train_loader):
            global_step += 1

            # ── 读取 batch ───────────────────────────────────
            x_src_pixel = batch["x_src"].to(device)
            y_tgt_pixel = batch["y_tgt"].to(device)
            text_list   = batch["text"]

            # ── 判断哪些样本有真实变化 ───────────────────────
            # 数据集里 no_change 文本只有一种固定形式，精确匹配即可
            is_changed_list = [
                0.0 if "there is no difference" in t.lower() else 1.0
                for t in text_list
            ]
            is_changed = torch.tensor(is_changed_list, device=device)  # [B]

            # ── 编码（冻结模型，no_grad）─────────────────────
            with torch.no_grad():
                x0      = vae.encode(x_src_pixel)          # [B, 4, 32, 32]
                y1      = vae.encode(y_tgt_pixel)          # [B, 4, 32, 32]
                context = clip_model(text_list).float()    # [B, 77, 768]

            # ════════════════════════════════════════════════
            # 判别器（D）回合
            # ════════════════════════════════════════════════
            opt_D.zero_grad()
            with torch.no_grad():
                trajectory_d, _, _ = sde(x0, context=context)
                x1_fake = trajectory_d[:, -1]              # [B, 4, 32, 32]

            d_real = net_D(y1)
            d_fake = net_D(x1_fake)
            loss_D_base = (d_fake - d_real).mean()
            reg_D       = (d_real ** 2 + d_fake ** 2).mean()
            loss_D      = loss_D_base + reg_D * 0.001
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(net_D.parameters(), max_norm=1.0)
            opt_D.step()

            # ════════════════════════════════════════════════
            # 生成器（G）回合
            # ════════════════════════════════════════════════
            opt_G.zero_grad()
            x0_g = x0.detach().requires_grad_(True)
            trajectory, times, shifts = sde(x0_g, context=context)
            x1_gen = trajectory[:, -1]                     # [B, 4, 32, 32]

            # 对抗项
            loss_adv = -net_D(x1_gen).mean()

            # OT 动能正则项（归一化到每维度）
            norm     = torch.norm(shifts.flatten(start_dim=2), p=2, dim=-1) ** 2
            integral = (INTEGRAL_SCALE * integrate(norm, times)).mean()
            loss_ot  = integral

            # 像素级监督项
            loss_target = F.l1_loss(x1_gen, y1)

            # ── 方向一致性损失（隐空间，梯度完整）───────────
            # 隐空间位移向量：x1_gen 有梯度，x0 detach
            dir_latent      = (x1_gen - x0.detach())           # [B, 4, 32, 32]
            dir_latent_flat = dir_latent.flatten(start_dim=1)  # [B, 4096]

            # 文本变化语义向量：context 本身就是变化描述
            # [B, 77, 768] → mean → [B, 768]，detach 因为 CLIP 已冻结
            context_pooled = context.mean(dim=1).detach()      # [B, 768]
            context_pooled = F.normalize(context_pooled, dim=-1)

            # 投影：隐空间方向 → 文本语义空间（dir_proj 可训练）
            # 梯度链：loss_dir → dir_proj → dir_latent_flat → x1_gen → net_G ✅
            dir_latent_proj = net_G.dir_proj(dir_latent_flat)  # [B, 768]
            dir_latent_proj = F.normalize(dir_latent_proj, dim=-1)

            cos_sim = F.cosine_similarity(dir_latent_proj, context_pooled, dim=-1)  # [B]

            # 有变化样本：方向对齐损失（用 sum/n 避免被 no_change 样本稀释）
            n_changed  = is_changed.sum().clamp(min=1.0)
            loss_dir   = ((1.0 - cos_sim) * is_changed).sum() / n_changed

            # no_change 样本：惩罚位移幅度（让网络对无变化图像尽量不动）
            magnitude    = dir_latent_flat.norm(dim=-1)         # [B]
            n_no_change  = (1.0 - is_changed).sum().clamp(min=1.0)
            loss_no_change = (magnitude * (1.0 - is_changed)).sum() / n_no_change
            # ─────────────────────────────────────────────────

            loss_G = (Config.OT_REG_WEIGHT       * loss_ot        +
                      Config.ADV_WEIGHT           * loss_adv       +
                      Config.LAMBDA_SUP           * loss_target    +
                      Config.DIR_WEIGHT           * loss_dir       +
                      Config.DIR_WEIGHT * 0.3     * loss_no_change)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
            opt_G.step()

            # ── TensorBoard ──────────────────────────────────
            writer.add_scalar("Loss/Discriminator",       loss_D.item(),         global_step)
            writer.add_scalar("Loss_D_parts/loss_D_base", loss_D_base.item(),    global_step)
            writer.add_scalar("Loss_D_parts/reg_D",       reg_D.item(),          global_step)
            writer.add_scalar("Loss/Generator_Total",     loss_G.item(),         global_step)
            writer.add_scalar("Loss_G_parts/Adversarial", loss_adv.item(),       global_step)
            writer.add_scalar("Loss_G_parts/L1_Target",   loss_target.item(),    global_step)
            writer.add_scalar("Loss_G_parts/OT_Energy",   loss_ot.item(),        global_step)
            writer.add_scalar("Loss_G_parts/L_Dir",       loss_dir.item(),       global_step)
            writer.add_scalar("Loss_G_parts/L_NoChange",  loss_no_change.item(), global_step)
            writer.add_scalar("Metrics/magnitude_mean",   magnitude.mean().item(), global_step)

            # ← 修复 P0-1：变量作用域问题，统一在这里计算再写入
            if is_changed.sum() > 0:
                cos_changed = cos_sim[is_changed > 0.5].mean().item()
                writer.add_scalar("Metrics/cos_sim_changed", cos_changed, global_step)
            if (1.0 - is_changed).sum() > 0:
                cos_no_change = cos_sim[is_changed < 0.5].mean().item()
                writer.add_scalar("Metrics/cos_sim_no_change", cos_no_change, global_step)

            # ── 控制台日志 ───────────────────────────────────
            if (step + 1) % log_interval == 0 or (step + 1) == len(train_loader):
                elapsed     = time.time() - last_log_time
                steps_taken = (step + 1) % log_interval or log_interval
                s_per_it    = elapsed / steps_taken
                progress    = (step + 1) / len(train_loader)
                bar         = '█' * int(20 * progress) + '-' * (20 - int(20 * progress))
                print(
                    f"Epoch [{epoch}/{Config.EPOCHS}] (eps={current_eps:.4f}) "
                    f"[{step+1:03d}/{len(train_loader)}] |{bar}| "
                    f"G={loss_G.item():.4f} D={loss_D.item():.4f} "
                    f"dir={loss_dir.item():.4f} nc={loss_no_change.item():.4f} "
                    f"[{s_per_it:.2f}s/it]",
                    flush=True
                )
                last_log_time = time.time()

            # ── 可视化 ───────────────────────────────────────
            if global_step % args.viz_interval == 0:
                sde.eval()
                with torch.no_grad():
                    n_viz = min(args.viz_nrow, x_src_pixel.shape[0])
                    x_gen_pixel = vae.decode(x1_gen[:n_viz].detach())
                    epoch_train_dir = os.path.join(samples_train_dir, f"epoch_{epoch:03d}")
                    os.makedirs(epoch_train_dir, exist_ok=True)
                    draw_and_save_grid(
                        x_src_pixel[:n_viz], y_tgt_pixel[:n_viz], x_gen_pixel,
                        text_list[:n_viz],
                        os.path.join(epoch_train_dir, f"step_{global_step:06d}.png"),
                        writer, "Visuals/Train_Batch", global_step
                    )
                sde.train()

        # ── Epoch 结束 ───────────────────────────────────────
        evaluate_and_save(epoch, val_loader, vae, clip_model, sde,
                          device, samples_val_dir, writer, args.viz_nrow)
        save_checkpoint(epoch, net_G, net_D, opt_G, opt_D,
                        current_eps, args.save_every_epochs)

    writer.close()
    print("🎉 Finished!")


def evaluate_and_save(epoch, loader, vae, clip_model, sde,
                      device, save_dir, writer, nrow):
    sde.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        n_viz       = min(nrow, batch["x_src"].shape[0])
        x_src_pixel = batch["x_src"][:n_viz].to(device)
        y_tgt_pixel = batch["y_tgt"][:n_viz].to(device)
        text_list   = batch["text"][:n_viz]
        x0          = vae.encode(x_src_pixel)
        context     = clip_model(text_list).float()
        trajectory, _, _ = sde(x0, context=context)
        x_gen_pixel = vae.decode(trajectory[:, -1])

    epoch_val_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_val_dir, exist_ok=True)
    draw_and_save_grid(
        x_src_pixel, y_tgt_pixel, x_gen_pixel, text_list,
        os.path.join(epoch_val_dir, f"val_ep{epoch:03d}.png"),
        writer, "Visuals/Validation", epoch
    )
    sde.train()


def save_checkpoint(epoch, net_G, net_D, opt_G, opt_D, current_eps, save_freq):
    ckpt = {
        "epoch":  epoch,
        "net_G":  net_G.state_dict(),
        "net_D":  net_D.state_dict(),
        "opt_G":  opt_G.state_dict(),
        "opt_D":  opt_D.state_dict(),
        "eps_t":  float(current_eps),
    }
    torch.save(ckpt, os.path.join(Config.CKPT_DIR, "ENOT_latest.pt"))
    if epoch % save_freq == 0 or epoch == Config.EPOCHS:
        path = os.path.join(Config.CKPT_DIR, f"ENOT_ep{epoch:04d}.pt")
        torch.save(ckpt, path)
        print(f"💾 Saved: {path}")


if __name__ == "__main__":
    main()