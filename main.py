import optuna
from src.cnn import CNN
from src.dataset import get_dataloaders
from src.lightning import LightningWrapper
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl


def objective(trial):
    train_loader, val_loader = get_dataloaders()

    num_layers_hyperparam = trial.suggest_int("num_layers", 2, 4)
    base_channels_hyperparam = trial.suggest_categorical("base_channels", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = CNN(
        num_layers_hyperparam, base_channels=base_channels_hyperparam, num_classes=1000
    )
    lightning = LightningWrapper(model, lr)

    pruner_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        callbacks=[pruner_callback],
        enable_checkpointing=False,
        logger=True,
    )

    trainer.fit(lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer.callback_metrics["val_loss"].item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
