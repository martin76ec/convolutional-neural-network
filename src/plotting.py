import os
import csv
import glob
import matplotlib.pyplot as plt


def read_csv_log(log_dir):
    csv_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        candidates = glob.glob(os.path.join(log_dir, "version_*", "metrics.csv"))
        # highest numeric version (lexicographic sort would put version_9 after version_10)
        candidates.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).rsplit("_", 1)[-1]))
        if candidates:
            csv_path = candidates[-1]

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("train_loss") and row.get("epoch") is not None:
                train_loss.append((int(float(row["epoch"])), float(row["train_loss"])))
            if row.get("val_loss") and row.get("epoch") is not None:
                val_loss.append((int(float(row["epoch"])), float(row["val_loss"])))
            if row.get("train_acc") and row.get("epoch") is not None:
                train_acc.append((int(float(row["epoch"])), float(row["train_acc"])))
            if row.get("val_acc") and row.get("epoch") is not None:
                val_acc.append((int(float(row["epoch"])), float(row["val_acc"])))

    def aggregate(rows):
        if not rows:
            return [], []
        by_epoch = {}
        for epoch, val in rows:
            by_epoch.setdefault(epoch, []).append(val)
        epochs = sorted(by_epoch.keys())
        return epochs, [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]

    return {
        "train_loss": aggregate(train_loss),
        "val_loss": aggregate(val_loss),
        "train_acc": aggregate(train_acc),
        "val_acc": aggregate(val_acc),
    }


def plot_losses(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, data) in zip(axes, results.items()):
        tl_x, tl_y = data["train_loss"]
        vl_x, vl_y = data["val_loss"]
        ax.plot(tl_x, tl_y, label="Train Loss", marker="o")
        ax.plot(vl_x, vl_y, label="Val Loss", marker="s")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss plots saved to {output_path}")


def plot_accuracies(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, data) in zip(axes, results.items()):
        ta_x, ta_y = data["train_acc"]
        va_x, va_y = data["val_acc"]
        ax.plot(ta_x, ta_y, label="Train Acc", marker="o")
        ax.plot(va_x, va_y, label="Val Acc", marker="s")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Accuracy plots saved to {output_path}")