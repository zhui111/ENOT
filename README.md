# Text-Guided Latent ENOT for Remote-Sensing Image Editing

This repository contains a PyTorch implementation of a **text-guided latent-space image editing framework** based on an Entropic Neural Optimal Transport (ENOT)-style stochastic flow. The code is designed for paired remote-sensing image editing data, where each training example contains:

1. a source image,
2. a target or edited image,
3. a natural-language description of the desired change.

The model learns to transport the source image latent representation toward the target image latent representation while using text features as semantic guidance. Instead of directly editing high-resolution RGB pixels, the project uses a frozen Stable Diffusion VAE to map images into a compact latent space and trains a text-conditioned continuous transformation model in that latent space.

---

## 1. Project Summary

The central objective of this work is to model **semantic image transformation** as a text-conditioned transport process:

\[
    z_0 = \mathrm{VAE}_{enc}(x_{src}), \qquad
    z_1 = \mathrm{VAE}_{enc}(y_{tgt})
\]

where `x_src` is the source image, `y_tgt` is the edited target image, and the transport from `z_0` to `z_1` is guided by a text instruction. The learned generator predicts latent shifts over a discretized stochastic differential equation (SDE) trajectory:

\[
    z_{t+\Delta t} = z_t + v_\theta(z_t, t, c_{text}) + \sqrt{\epsilon \Delta t}\,\xi
\]

where:

- `v_theta` is the text-guided CUNet shift model,
- `c_text` is the CLIP/RemoteCLIP token-level text representation,
- `epsilon` controls the injected stochastic noise,
- the final steps can be run without noise for cleaner generation.

The training objective combines:

- an adversarial ENOT-style discriminator objective in latent space,
- an optimal-transport kinetic-energy regularizer,
- an L1 supervision term between generated and target latents.

---

## 2. Repository Structure

```text
.
├── train.py                     # Main training entry point
├── configs/
│   └── config.py                # Global configuration and default paths
├── data/
│   └── dataset.py               # Paired image-edit dataset and augmentations
└── model/
    ├── vae.py                   # Frozen Stable Diffusion VAE wrapper
    ├── text_encoder.py          # Frozen CLIP/RemoteCLIP text encoder
    ├── cunet.py                 # Text-conditioned CUNet generator
    ├── text_guided_sde.py       # Discretized text-guided SDE transport process
    └── discriminator.py         # Spectral-normalized ResNet discriminator
```

---

## 3. Main Components

### 3.1 Dataset: `PairEditDataset`

Defined in [`data/dataset.py`](data/dataset.py), `PairEditDataset` reads a CSV file containing paired source/target image paths and a text instruction.

Expected CSV columns are configured in [`configs/config.py`](configs/config.py):

| Column | Meaning | Default config name |
| --- | --- | --- |
| `src` | path to the source image | `Config.CSV_COL_SRC` |
| `edt` | path to the edited/target image | `Config.CSV_COL_EDT` |
| `text` | text description of the edit | `Config.CSV_COL_TEXT` |

Example CSV:

```csv
src,edt,text
/path/to/source_001.png,/path/to/target_001.png,"add a new building in the upper-left region"
/path/to/source_002.png,/path/to/target_002.png,"remove the road segment near the center"
/path/to/source_003.png,/path/to/target_003.png,"there is no difference"
```

The dataset performs the following preprocessing:

- resizes both source and target images to `Config.IMG_SIZE`, default `256 x 256`,
- converts images to RGB,
- converts images to tensors,
- normalizes pixel values to `[-1, 1]`, which matches the VAE input convention.

During training, paired augmentations are applied consistently to both source and target images:

- random 90/180/270 degree rotation,
- random horizontal flip,
- random vertical flip,
- mild brightness-like multiplicative color jitter.

These augmentations are useful for remote-sensing imagery because spatial orientation and flips often preserve semantic validity.

---

### 3.2 Frozen VAE Latent Space

Defined in [`model/vae.py`](model/vae.py), `LatentVAE` wraps `diffusers.AutoencoderKL` and loads a local Stable Diffusion VAE checkpoint.

The default VAE path is controlled by:

```python
Config.VAE_PATH = os.getenv("SD_VAE", "/root/autodl-tmp/sd-vae-ft-mse")
```

The VAE is frozen during training:

- no gradients are computed through the VAE,
- the generator and discriminator are trained only in latent space,
- the latent scaling factor is fixed to `0.18215`, following the Stable Diffusion latent convention.

For the default image size of `256 x 256`, the VAE scale factor is `8`, so RGB images are encoded as:

```text
[B, 3, 256, 256]  ->  [B, 4, 32, 32]
```

This makes training significantly more memory-efficient than direct pixel-space transport.

---

### 3.3 Frozen CLIP / RemoteCLIP Text Encoder

Defined in [`model/text_encoder.py`](model/text_encoder.py), `CLIPTextEncoder` loads a local CLIP-compatible model file, intended here for RemoteCLIP-style text encoding.

The default path is controlled by:

```python
Config.CLIP_CACHE_DIR = os.getenv(
    "CLIP_CACHE",
    "/root/autodl-tmp/clip/RemoteCLIP-ViT-L-14.pt"
)
```

The text encoder:

- tokenizes each text instruction using CLIP tokenization,
- keeps token-level Transformer features instead of only one global sentence vector,
- outputs features with shape approximately `[B, 77, 768]`,
- is fully frozen during training.

Token-level text features are important because the generator uses cross-attention layers to align spatial latent features with the semantic instruction.

The training loop also applies a simple text-dropout strategy:

```python
text_list_cfg = [t if random.random() > 0.1 else "" for t in text_list]
```

This means about 10% of training samples receive an empty text condition, which can help the model remain robust when textual guidance is weak or absent.

---

### 3.4 Text-Guided CUNet Generator

Defined in [`model/cunet.py`](model/cunet.py), `TextGuidedCUNet` is the main trainable generator. It predicts the latent shift field used by the SDE.

The architecture is based on a CUNet-style U-Net with conditional normalization and text injection.

Key design choices:

#### 3.4.1 Early Text Injection

`TextEarlyInjectionBlock` injects text into the latent input before the main U-Net encoder. It uses cross-attention where:

- image latent patches act as queries,
- text token features act as keys and values.

This gives the model early access to the semantic instruction before the image features are downsampled.

#### 3.4.2 Multi-Scale Cross Attention

`CrossAttentionBlock` is inserted at several U-Net stages:

- deep encoder feature level,
- bottleneck feature level,
- early decoder feature level.

This allows text conditioning to influence both high-level semantic features and reconstruction-stage features.

#### 3.4.3 Time and Text Conditioning

The SDE time embedding is injected through conditional instance normalization (`CondINorm`). The model also projects global text information into the time-conditioning channel:

```python
text_global = self.text_bottleneck_proj(context.mean(dim=1))[:, :, None, None]
t = t + text_global
```

This lets the text instruction modify the time-dependent dynamics of the transport process.

---

### 3.5 Text-Guided SDE Transport

Defined in [`model/text_guided_sde.py`](model/text_guided_sde.py), `TextGuidedSDE` performs a fixed-step discretized stochastic transport process.

By default, the training script uses:

```python
n_steps = 10
predict_shift = True
use_gradient_checkpoint = True
```

At each step, the generator predicts a shift. Noise is added for most steps, while the last `n_last_steps_without_noise` steps can be deterministic. This design keeps the process exploratory during most of the trajectory but encourages a cleaner final latent.

The SDE returns:

```python
trajectory, times, shifts = sde(x0, context=context)
```

where:

- `trajectory` has shape `[B, n_steps + 1, C, H, W]`,
- `times` stores the discretized time points,
- `shifts` stores the predicted velocity/shift terms used for OT regularization.

The helper function `integrate(values, times)` computes the discretized time integral used in the kinetic-energy loss.

---

### 3.6 Latent-Space Discriminator

Defined in [`model/discriminator.py`](model/discriminator.py), `ResNet_D` is a spectral-normalized ResNet discriminator operating directly on VAE latents.

Instead of classifying RGB images, it receives latent tensors such as `[B, 4, 32, 32]`. This is consistent with the ENOT-style latent transport formulation and avoids the additional cost of decoding images during every discriminator update.

The discriminator uses:

- spectral normalization,
- residual blocks,
- average-pooling downsampling,
- a final linear head producing a scalar score.

---

## 4. Training Objective

The training loop is implemented in [`train.py`](train.py). For each batch, the code performs one discriminator update followed by one generator update.

### 4.1 Discriminator Loss

The discriminator compares real target latents `y1` and generated final latents `x1_fake`:

```python
d_real = net_D(y1)
d_fake = net_D(x1_fake)
loss_D_base = (d_fake - d_real).mean()
reg_D = (d_real ** 2 + d_fake ** 2).mean()
loss_D = loss_D_base + reg_D * 0.001
```

The regularization term prevents discriminator scores from growing without bound.

---

### 4.2 Generator Loss

The generator loss combines three active terms:

```python
loss_G = (
    Config.OT_REG_WEIGHT * loss_ot +
    Config.ADV_WEIGHT    * loss_adv +
    Config.LAMBDA_SUP    * loss_target
)
```

#### Adversarial Term

```python
loss_adv = -net_D(x1_gen).mean()
```

This encourages generated latents to receive high discriminator scores.

#### OT Energy Term

```python
norm = torch.norm(shifts.flatten(start_dim=2), p=2, dim=-1) ** 2
integral = (INTEGRAL_SCALE * integrate(norm, times)).mean()
loss_ot = integral
```

This penalizes excessive transport energy and encourages a smoother, more optimal path from source latent to target latent.

The code normalizes the integral by the latent dimensionality:

```python
INTEGRAL_SCALE = 1.0 / (latent_channels * latent_size * latent_size)
```

For the default setting, this is:

```text
1 / (4 * 32 * 32) = 1 / 4096
```

#### Latent Supervision Term

```python
loss_target = F.l1_loss(x1_gen, y1)
```

This directly supervises the final generated latent against the target latent.

---

## 5. Training Outputs

For a run named `my_experiment` with `--out-dir ./runs`, outputs are written to:

```text
runs/
└── my_experiment/
    ├── ckpt/
    │   ├── ENOT_latest.pt
    │   ├── ENOT_ep0005.pt
    │   ├── ENOT_ep0010.pt
    │   └── ...
    ├── samples/
    │   ├── train/
    │   │   └── epoch_XXX/
    │   │       └── step_XXXXXX.png
    │   └── val/
    │       └── epoch_XXX/
    │           └── val_epXXX.png
    └── tb_logs/
        └── TensorBoard event files
```

Visualization grids contain three rows:

1. source image,
2. target image,
3. generated image with the text instruction shown as the title.

TensorBoard logs include:

- discriminator loss,
- generator total loss,
- adversarial loss,
- latent L1 target loss,
- OT energy loss,
- epsilon schedule,
- training and validation image grids.

---

## 6. Installation

This project does not currently include a `requirements.txt`, but the code imports the following major packages:

```text
torch
torchvision
tensorboard
numpy
pandas
Pillow
matplotlib
diffusers
clip
```

A typical environment can be prepared with commands similar to the following:

```bash
conda create -n text-guided-enot python=3.10 -y
conda activate text-guided-enot

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard numpy pandas pillow matplotlib diffusers
```

For CLIP, install the CLIP package compatible with your RemoteCLIP checkpoint. For example, if using OpenAI CLIP:

```bash
pip install git+https://github.com/openai/CLIP.git
```

If your RemoteCLIP model requires a different loader or package version, install the package recommended by the checkpoint provider.

---

## 7. Required Pretrained Assets

Before training, make sure the following local assets exist.

### 7.1 Stable Diffusion VAE

The code expects a local `diffusers.AutoencoderKL` directory, for example:

```text
/root/autodl-tmp/sd-vae-ft-mse
```

You can override it with:

```bash
export SD_VAE=/path/to/sd-vae-ft-mse
```

On Windows PowerShell:

```powershell
$env:SD_VAE = "D:\path\to\sd-vae-ft-mse"
```

### 7.2 CLIP / RemoteCLIP Weights

The code expects a local CLIP-compatible weight file, for example:

```text
/root/autodl-tmp/clip/RemoteCLIP-ViT-L-14.pt
```

You can override it with:

```bash
export CLIP_CACHE=/path/to/RemoteCLIP-ViT-L-14.pt
```

On Windows PowerShell:

```powershell
$env:CLIP_CACHE = "D:\path\to\RemoteCLIP-ViT-L-14.pt"
```

---

## 8. Running Training

Basic command:

```bash
python train.py \
  --run-name enot_text_guided_v1 \
  --out-dir ./runs \
  --csv /path/to/train.csv \
  --csv-val /path/to/val.csv \
  --device cuda \
  --img-size 256 \
  --batch-size 4 \
  --epochs 100 \
  --lr 1e-4
```

Example using the default AutoDL-style data layout:

```bash
python train.py \
  --run-name V30_TextGuided_ENOT \
  --out-dir /root/autodl-tmp/runs \
  --csv /root/autodl-tmp/levir_mci_train.csv \
  --csv-val /root/autodl-tmp/levir_mci_val.csv \
  --device cuda \
  --batch-size 4 \
  --epochs 100 \
  --epsilon 0.05 \
  --lambda-sup 0.1 \
  --norm-sq-scale 1.0 \
  --enot-adv-weight 1.0
```

Windows PowerShell example:

```powershell
python train.py `
  --run-name enot_text_guided_v1 `
  --out-dir .\runs `
  --csv D:\data\train.csv `
  --csv-val D:\data\val.csv `
  --device cuda `
  --img-size 256 `
  --batch-size 4 `
  --epochs 100 `
  --lr 1e-4
```

---

## 9. Command-Line Arguments

The main script exposes the following arguments:

| Argument | Default | Description |
| --- | ---: | --- |
| `--run-name` | required | Name of the experiment run. |
| `--out-dir` | required | Root output directory for checkpoints, samples, and TensorBoard logs. |
| `--csv` | required | Training CSV path. |
| `--csv-val` | required | Validation CSV path. |
| `--device` | `cuda` | Device string, e.g. `cuda` or `cpu`. |
| `--seed` | `42` | Random seed. |
| `--img-size` | `256` | Input image size after resizing. |
| `--batch-size` | `4` | Training and validation batch size. |
| `--epochs` | `100` | Number of training epochs. |
| `--lr` | `1e-4` | Learning rate for both generator and discriminator. |
| `--unet-base` | `32` | Base channel multiplier for the CUNet. |
| `--time-dim` | `128` | Time embedding dimension. |
| `--latent-ch` | `4` | VAE latent channel count. |
| `--epsilon` | `0.05` | Maximum SDE noise strength. |
| `--n-last-wo-noise` | `1` | Number of final SDE steps without noise. |
| `--lambda-sup` | `0.1` | Weight for latent L1 target supervision. |
| `--norm-sq-scale` | `1.0` | Weight for OT kinetic-energy regularization. |
| `--enot-adv-weight` | `1.0` | Weight for adversarial generator loss. |
| `--dir-weight` | `0.5` | Parsed direction-loss weight; currently related direction-loss code is commented out. |
| `--viz-interval` | `500` | Save training visualization every N global steps. |
| `--save-every-epochs` | `5` | Save numbered checkpoints every N epochs. |
| `--viz-nrow` | `4` | Number of samples shown in visualization grids. |
| `--resume` | `None` | Path to a checkpoint for resuming training. |

---

## 10. Resuming Training

To resume from a saved checkpoint:

```bash
python train.py \
  --run-name enot_text_guided_v1 \
  --out-dir ./runs \
  --csv /path/to/train.csv \
  --csv-val /path/to/val.csv \
  --resume ./runs/enot_text_guided_v1/ckpt/ENOT_latest.pt
```

The checkpoint contains:

- generator parameters,
- discriminator parameters,
- generator optimizer state,
- discriminator optimizer state,
- epoch index,
- current epsilon value.

The script resumes from `checkpoint_epoch + 1`.

---

## 11. Monitoring with TensorBoard

After training starts, open TensorBoard with:

```bash
tensorboard --logdir ./runs/enot_text_guided_v1/tb_logs
```

If using the default output pattern where multiple experiments are placed under `./runs`, you can also run:

```bash
tensorboard --logdir ./runs
```

Useful panels include:

- `Loss/Discriminator`,
- `Loss/Generator_Total`,
- `Loss_G_parts/Adversarial`,
- `Loss_G_parts/L1_Target`,
- `Loss_G_parts/OT_Energy`,
- `HyperParams/Epsilon`,
- `Visuals/Train_Batch`,
- `Visuals/Validation`.

---

## 12. Epsilon Warmup

The training script warms up the SDE noise level during the first 20% of epochs:

```python
def get_epsilon(epoch, total_epochs, max_eps):
    warmup_ratio = 0.2
    warmup_epochs = int(total_epochs * warmup_ratio)
    if epoch <= warmup_epochs:
        return max_eps * (epoch / max(1, warmup_epochs))
    return max_eps
```

This prevents the generator from being exposed to the full stochasticity at the beginning of training, which can improve stability.

---

## 13. Reproducibility Notes

The configuration file sets random seeds for:

- Python `random`,
- NumPy,
- PyTorch CPU,
- PyTorch CUDA.

However, exact reproducibility may still depend on:

- CUDA version,
- cuDNN settings,
- GPU model,
- dataloader worker scheduling,
- nondeterministic PyTorch kernels,
- stochastic text dropout during training.

For stricter reproducibility, consider additionally setting deterministic PyTorch behavior and fixing dataloader worker seeds.

---

## 14. Current Implementation Notes

The current codebase is compact and research-oriented. A few implementation details are worth noting before running large experiments:

1. **Local pretrained files are required.**  
   The VAE and CLIP/RemoteCLIP weights are loaded with local paths. If the paths are wrong, training will stop immediately with a file-loading error.

2. **The active generator objective uses three losses.**  
   Directional loss and no-change-specific metrics appear in commented code, but they are not currently active in `loss_G`.

3. **`--dir-weight` is parsed but not currently active.**  
   The command-line argument is synchronized into `Config.DIR_WEIGHT`, but the related direction-loss terms are commented out in the training loop.

4. **Check constructor compatibility before training.**  
   The training script passes `latent_size=Config.LATENT_SIZE` when constructing `TextGuidedCUNet`. In the current `model/cunet.py`, the `TextGuidedCUNet` constructor is defined with `in_channels`, `n_classes`, `z_channels`, `text_dim`, and `base_factor`. If this mismatch is present in your local copy, either remove the `latent_size` keyword in `train.py` or update the constructor to accept it before launching training.

---

## 15. Suggested Experiment Workflow

A practical workflow for using this repository is:

1. Prepare paired source/target image data.
2. Write train and validation CSV files with `src`, `edt`, and `text` columns.
3. Download or place the Stable Diffusion VAE locally.
4. Download or place the CLIP/RemoteCLIP text encoder weights locally.
5. Set `SD_VAE` and `CLIP_CACHE` environment variables if the defaults do not match your machine.
6. Run a short smoke test with a small batch size and a few epochs.
7. Inspect generated validation grids in `samples/val`.
8. Inspect TensorBoard losses and visualizations.
9. Increase the number of epochs and tune loss weights.
10. Compare checkpoints using the validation image grids.

Recommended first smoke test:

```bash
python train.py \
  --run-name smoke_test \
  --out-dir ./runs \
  --csv /path/to/train.csv \
  --csv-val /path/to/val.csv \
  --device cuda \
  --batch-size 1 \
  --epochs 2 \
  --viz-interval 10 \
  --save-every-epochs 1
```

---

## 16. Possible Future Extensions

This codebase can be extended in several directions:

- add an explicit inference script for editing new source images using user-provided text,
- add quantitative evaluation metrics such as LPIPS, SSIM, PSNR, CLIPScore, or remote-sensing task-specific metrics,
- activate and refine directional semantic consistency loss,
- add no-change preservation loss for samples whose text indicates no edit,
- compare token-level text conditioning with global sentence-level conditioning,
- add classifier-free guidance during inference,
- add mixed-precision training with `torch.cuda.amp`,
- export a reproducible `requirements.txt` or `environment.yml`,
- support distributed training for larger datasets,
- add checkpoint-based validation over the full validation set instead of one visualization batch.

---

## 17. Citation / Acknowledgement Placeholder

If this repository is used in a paper, report, or thesis, consider citing the related foundations behind the implementation:

- optimal transport and entropic optimal transport,
- neural optimal transport / ENOT-style learning,
- stochastic differential equation based generative modeling,
- Stable Diffusion VAE latent representations,
- CLIP or RemoteCLIP text-image representation learning,
- U-Net architectures with cross-attention conditioning.

A project-specific citation can be added here once the corresponding manuscript or technical report is finalized.

---

## 18. Short Description

This project implements a text-guided latent optimal-transport model for paired remote-sensing image editing. It encodes images with a frozen Stable Diffusion VAE, encodes edit instructions with a frozen CLIP/RemoteCLIP text encoder, learns a text-conditioned CUNet shift model inside a discretized SDE, and trains the transport process using adversarial, optimal-transport energy, and latent supervision losses.
