from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataloaders():
    imagenet = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)

    imagenet_tiny = imagenet.take(10)
    imagenet_tiny = list(imagenet_tiny)

    train_subset = imagenet_tiny[:10]
    val_subset = imagenet_tiny[10:]

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def collate_fn(batch):
        images = [transform(item['image'].convert("RGB")) for item in batch]
        labels = [item['label'] for item in batch]
        return torch.stack(images), torch.tensor(labels)

    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader
