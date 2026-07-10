import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.lightning import LightningWrapper


def log_hardware():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        msg = f"USING CUDA — {gpu_name} ({gpu_mem:.1f} GB)"
    elif torch.backends.mps.is_available():
        msg = "USING Apple MPS (Metal)"
    else:
        msg = "NO GPU DETECTED — using CPU"

    print("\n" + "=" * 60)
    print(f"  >>>  {msg}  <<<")
    print("=" * 60 + "\n")


def train_model(model, train_loader, val_loader, lr=1e-3, weight_decay=1e-4, max_epochs=50, model_name="model"):
    lightning_model = LightningWrapper(model, lr=lr, weight_decay=weight_decay)

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=f"checkpoints/{model_name}")
    logger = CSVLogger("logs", name=model_name)

    log_hardware()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        enable_checkpointing=True,
    )

    start = time.time()
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - start

    best_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
    val_acc = trainer.callback_metrics.get("val_acc", 0.0)

    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "val_acc": float(val_acc),
        "train_time": elapsed,
        "log_dir": logger.log_dir,
        "trainer": trainer,
        "model": lightning_model,
    }