from torch import nn
import torch
import torch.nn.functional as F
import lightning.pytorch as LP
from torchmetrics import Accuracy


class LightningWrapper(LP.LightningModule):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_acc = Accuracy(task="multiclass", num_classes=model.num_classes if hasattr(model, "num_classes") else 102)
        self.val_acc = Accuracy(task="multiclass", num_classes=model.num_classes if hasattr(model, "num_classes") else 102)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)