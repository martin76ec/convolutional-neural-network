# Lab 1 — CNNs (Tarea Computacional 1)

## Project Overview
Deep Learning lab assignment implementing and comparing 3 approaches on a torchvision dataset:
1. CNN from scratch (baseline)
2. Pre-trained ResNet18 as feature extractor (frozen backbone)
3. Pre-trained ResNet18 fine-tuning (partial unfreeze of layer4)

## Dataset
- **torchvision Flowers102** (102 flower categories, ~8k images)
- NOT CIFAR-10/Fashion-MNIST/MNIST (assignment requirement)
- Data augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter (3 techniques)
- Image size: 224x224, ImageNet normalization

## Assignment Requirements
- At least 2 data augmentation techniques (we have 3)
- Early stopping during training (patience=5, monitor val_loss)
- Train loss vs val loss plots for each approach + overfitting commentary
- Final comparison of 3 approaches (accuracy/loss/curves/training time)
- Pre-trained model must NOT be MobileNet (we use ResNet18)

## Architecture
- **src/dataset.py** — Flowers102 dataloaders with augmentation, supports `--subset N` for quick testing
- **src/cnn.py** — Custom CNN: 4 ConvBlocks (Conv2d + BatchNorm + MaxPool + Dropout), GAP, 2 FC layers
- **src/pretrained.py** — ResNet18 feature extractor (frozen) + fine-tuning (unfreeze layer4)
- **src/lightning.py** — LightningWrapper with train/val loss + accuracy logging, weight decay
- **src/trainer.py** — train_model() with EarlyStopping, ModelCheckpoint, CSVLogger, GPU detection log
- **src/plotting.py** — reads CSV logs, generates loss_comparison.png + accuracy_comparison.png
- **main.py** — runs all 3 approaches, supports `--step` for individual runs, `--epochs`, `--subset`

## Commands (Makefile)
- `make train` — full pipeline (all 3 approaches + plots + comparison)
- `make cnn` — only CNN from scratch
- `make feature_extractor` — only ResNet18 feature extractor
- `make fine_tuning` — only ResNet18 fine-tuning
- `make plots` — generate plots from existing logs
- `make dataset` — download dataset only
- `make clean` — remove logs/checkpoints/plots
- All accept `EPOCHS=N` (default 50)

## Environment Setup
- **Local (macOS, MPS):** `uv sync` — uses default PyPI wheels
- **Remote (H200 GPU, CUDA 12.8):** `uv sync` — pyproject.toml has `[[tool.uv.index]]` for PyTorch CUDA 12.8 wheels
- Python 3.12+, managed with uv
- Dependencies: torch, torchvision, lightning, torchmetrics, matplotlib, scipy, datasets, optuna

## Key Decisions
- Dataset: Flowers102 (small enough to train quickly, colorful images good for augmentation)
- Pre-trained model: ResNet18 (not MobileNet, per assignment)
- Fine-tuning: unfreeze from `layer4` (last residual block)
- Hyperparams: CNN lr=1e-3, FE lr=1e-3, FT lr=1e-4 (lower for fine-tuning)
- Regularization: Dropout(0.25) in CNN, Dropout(0.5) in classifier heads, weight_decay=1e-4
- Early stopping: patience=5 on val_loss

## Assignment Source
https://pointy-whistle-142.notion.site/Tarea-Computacional-1-CNNs-2f605737435780499812c54a56b366e2