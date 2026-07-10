import argparse
from src.dataset import get_dataloaders
from src.cnn import CNN
from src.pretrained import get_feature_extractor, get_fine_tuned
from src.trainer import train_model
from src.plotting import read_csv_log, plot_losses, plot_accuracies

NUM_CLASSES = 102


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, help="Use only N training samples (0 = full dataset)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    subset_size = args.subset if args.subset > 0 else None
    train_loader, val_loader, test_loader = get_dataloaders(subset_size=subset_size)

    print("=" * 60)
    print("1/3 — CNN from scratch")
    print("=" * 60)
    cnn_model = CNN(num_classes=NUM_CLASSES, base_channels=32, dropout=0.25)
    cnn_results = train_model(cnn_model, train_loader, val_loader, lr=1e-3, max_epochs=args.epochs, model_name="cnn_scratch")

    print("\n" + "=" * 60)
    print("2/3 — Pre-trained ResNet18 (feature extractor)")
    print("=" * 60)
    fe_model = get_feature_extractor(num_classes=NUM_CLASSES)
    fe_results = train_model(fe_model, train_loader, val_loader, lr=1e-3, max_epochs=args.epochs, model_name="resnet_feature_extractor")

    print("\n" + "=" * 60)
    print("3/3 — Pre-trained ResNet18 (fine-tuning)")
    print("=" * 60)
    ft_model = get_fine_tuned(num_classes=NUM_CLASSES, unfreeze_from="layer4")
    ft_results = train_model(ft_model, train_loader, val_loader, lr=1e-4, max_epochs=args.epochs, model_name="resnet_fine_tuning")

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    all_results = {
        "CNN from scratch": read_csv_log(cnn_results["log_dir"]),
        "ResNet18 Feature Extractor": read_csv_log(fe_results["log_dir"]),
        "ResNet18 Fine-tuning": read_csv_log(ft_results["log_dir"]),
    }
    plot_losses(all_results, "loss_comparison.png")
    plot_accuracies(all_results, "accuracy_comparison.png")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Approach':<30} {'Best Val Loss':>14} {'Val Acc':>10} {'Time (s)':>10}")
    print("-" * 66)
    for r in [cnn_results, fe_results, ft_results]:
        print(f"{r['model_name']:<30} {r['best_val_loss']:>14.4f} {r['val_acc']:>10.4f} {r['train_time']:>10.1f}")


if __name__ == "__main__":
    main()