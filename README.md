# PaleVision

PaleVision models opsin spectral responses with learnable evolutionary visual perception layers (EVPL) and trains semantic segmentation on hyperspectral imagery.

## Structure

```
configs/       experiment configs
dataloader/    dataset loaders
models/        EVPL, MiT backbone, SegFormer head
util/          training helpers
datasets/      put datasets here (not shipped)
work_dir/      logs and checkpoints (not shipped)
```

## Install

```bash
conda create -n palevision python=3.10 -y
conda activate palevision
pip install -r requirements.txt
```

Install a CUDA build of PyTorch that matches your driver, then install `mmcv` for your torch/cuda combo if needed.

## Data layout (Mars example)

```
datasets/Mars/MSL/train/{img,hsi,label}
datasets/Mars/MSL/test/{img,hsi,label}
```

`hsi` files are HDF5 with a `cube` dataset of shape `[31, H, W]`.

## Train (Mars)

```bash
torchrun --nproc_per_node=1 mars_exp.py \
  --config configs/mars.yaml \
  --workdir work_dir/mars \
  --train_path datasets/Mars/MSL/train \
  --val_path datasets/Mars/MSL/test \
  --encoder_type mit_b0
```

Multi-GPU:

```bash
torchrun --nproc_per_node=2 mars_exp.py \
  --config configs/mars.yaml \
  --workdir work_dir/mars \
  --train_path datasets/Mars/MSL/train \
  --val_path datasets/Mars/MSL/test
```

Smoke check (few steps, small resolution):

```bash
torchrun --nproc_per_node=1 mars_exp.py \
  --config configs/mars.yaml \
  --workdir work_dir/mars_smoke \
  --train_path datasets/Mars/MSL/train \
  --val_path datasets/Mars/MSL/test \
  --batch_size 1 \
  --input_size 128 \
  --max_steps 3 \
  --skip_val \
  --num_workers 2
```

Other entry points: `modid_exp.py`, `mhsi_exp.py`, `lib_hsi_exp_v2.py`, `minneapple_exp.py`.

## Notes

- No pretrained weights are included.
- Checkpoints written under `work_dir/` are ignored by git.
