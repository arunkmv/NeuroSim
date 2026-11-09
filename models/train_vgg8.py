"""Train the NeuroSim VGG8 on CIFAR-10 in full precision.

The saved checkpoint is a plain state_dict whose keys match VGG.layers / VGG.classifier,
so it loads directly with vgg8(pretrained="vgg8_cifar10.pth").
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from models.vgg import vgg8

# Standard CIFAR-10 statistics. Use the SAME normalization as NeuroSim's dataset.py,
# otherwise inference accuracy will drop. Check get_cifar10() and copy its values here.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=5, help="linear LR warmup epochs")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="vgg8_cifar10.pth")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def loaders(args):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = torchvision.datasets.CIFAR10(args.data_path, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(args.data_path, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=500, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader


def init_weights(model):
    # The model has no BatchNorm, so a proper He init matters for stable training.
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = loaders(args)
    model = vgg8().to(device)
    init_weights(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.wd, nesterov=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    def lr_at(step):
        # linear warmup, then cosine decay to 0
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * progress))

    best_acc, step = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        t0, running_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
            step += 1

        test_acc = evaluate(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out)

        print(f"epoch {epoch + 1:3d}/{args.epochs}  lr {lr_at(step - 1):.4f}  "
              f"loss {running_loss / seen:.4f}  train {100.0 * correct / seen:.2f}%  "
              f"test {test_acc:.2f}%  best {best_acc:.2f}%  ({time.time() - t0:.1f}s)",
              flush=True)

    print(f"Done. Best test accuracy {best_acc:.2f}% -> saved to {args.out}")


if __name__ == "__main__":
    main()
