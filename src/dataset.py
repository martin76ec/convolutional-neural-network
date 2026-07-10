from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = "./data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def _make_subset(dataset, n):
    if n is None or n <= 0:
        return dataset
    from torch.utils.data import Subset
    import random
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return Subset(dataset, indices)


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, subset_size: int | None = None):
    train_set = datasets.Flowers102(root=DATA_DIR, split="train", download=True, transform=train_transform)
    val_set = datasets.Flowers102(root=DATA_DIR, split="val", download=True, transform=eval_transform)
    test_set = datasets.Flowers102(root=DATA_DIR, split="test", download=True, transform=eval_transform)

    if subset_size is not None:
        train_set = _make_subset(train_set, subset_size)
        val_set = _make_subset(val_set, subset_size // 2)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader