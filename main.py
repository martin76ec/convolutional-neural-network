import argparse
from src.dataset import get_dataloaders
from src.cnn import CNN
from src.pretrained import get_feature_extractor, get_fine_tuned
from src.trainer import train_model
from src.plotting import read_csv_log, plot_losses, plot_accuracies

NUM_CLASSES = 102


def get_loaders(subset):
    subset_size = subset if subset > 0 else None
    return get_dataloaders(subset_size=subset_size)


def run_cnn(train_loader, val_loader, epochs):
    print("=" * 60)
    print("1/3 — CNN from scratch")
    print("=" * 60)
    model = CNN(num_classes=NUM_CLASSES, base_channels=32, dropout=0.25)
    return train_model(model, train_loader, val_loader, lr=1e-3, max_epochs=epochs, model_name="cnn_scratch")


def run_feature_extractor(train_loader, val_loader, epochs):
    print("\n" + "=" * 60)
    print("2/3 — Pre-trained ResNet18 (feature extractor)")
    print("=" * 60)
    model = get_feature_extractor(num_classes=NUM_CLASSES)
    return train_model(model, train_loader, val_loader, lr=1e-3, max_epochs=epochs, model_name="resnet_feature_extractor")


def run_fine_tuning(train_loader, val_loader, epochs):
    print("\n" + "=" * 60)
    print("3/3 — Pre-trained ResNet18 (fine-tuning)")
    print("=" * 60)
    model = get_fine_tuned(num_classes=NUM_CLASSES, unfreeze_from="layer4")
    return train_model(model, train_loader, val_loader, lr=1e-4, max_epochs=epochs, model_name="resnet_fine_tuning")


def run_plots(results):
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    all_results = {
        "CNN from scratch": read_csv_log(results["cnn_scratch"]["log_dir"]),
        "ResNet18 Feature Extractor": read_csv_log(results["resnet_feature_extractor"]["log_dir"]),
        "ResNet18 Fine-tuning": read_csv_log(results["resnet_fine_tuning"]["log_dir"]),
    }
    plot_losses(all_results, "loss_comparison.png")
    plot_accuracies(all_results, "accuracy_comparison.png")


def print_comparison(results):
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Approach':<30} {'Best Val Loss':>14} {'Val Acc':>10} {'Time (s)':>10}")
    print("-" * 66)
    for key in ["cnn_scratch", "resnet_feature_extractor", "resnet_fine_tuning"]:
        r = results[key]
        print(f"{r['model_name']:<30} {r['best_val_loss']:>14.4f} {r['val_acc']:>10.4f} {r['train_time']:>10.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, help="Use only N training samples (0 = full dataset)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--step", choices=["cnn", "feature_extractor", "fine_tuning", "plots", "all"], default="all")
    args = parser.parse_args()

    results = {}

    if args.step in ("cnn", "all"):
        train_loader, val_loader, _ = get_loaders(args.subset)
        results["cnn_scratch"] = run_cnn(train_loader, val_loader, args.epochs)

    if args.step in ("feature_extractor", "all"):
        train_loader, val_loader, _ = get_loaders(args.subset)
        results["resnet_feature_extractor"] = run_feature_extractor(train_loader, val_loader, args.epochs)

    if args.step in ("fine_tuning", "all"):
        train_loader, val_loader, _ = get_loaders(args.subset)
        results["resnet_fine_tuning"] = run_fine_tuning(train_loader, val_loader, args.epochs)

    if args.step in ("plots", "all"):
        if len(results) < 3:
            results = {
                "cnn_scratch": {"log_dir": "logs/cnn_scratch", "model_name": "cnn_scratch", "best_val_loss": 0, "val_acc": 0, "train_time": 0},
                "resnet_feature_extractor": {"log_dir": "logs/resnet_feature_extractor", "model_name": "resnet_feature_extractor", "best_val_loss": 0, "val_acc": 0, "train_time": 0},
                "resnet_fine_tuning": {"log_dir": "logs/resnet_fine_tuning", "model_name": "resnet_fine_tuning", "best_val_loss": 0, "val_acc": 0, "train_time": 0},
            }
        run_plots(results)

    if args.step == "all":
        print_comparison(results)


if __name__ == "__main__":
    main()