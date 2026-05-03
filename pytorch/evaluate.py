import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PatchDataset
from model import MitosisCNN


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    testset = PatchDataset(args.test_dir, augment=False)
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MitosisCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"loss={total_loss / total:.4f}")
    print(f"accuracy={100 * correct / total:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", default="mitosis_cnn.pt")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    evaluate(args)
