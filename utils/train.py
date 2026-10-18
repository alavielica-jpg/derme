"""Train the skin-condition classifier (PyTorch transfer learning).

Run from the repo root (never from inside utils/):
    python -m utils.train

Quick check that everything runs end to end (1 epoch per phase):
    python -m utils.train --epochs-head 1 --epochs-ft 1

The epoch with the lowest validation loss is saved to config.MODEL_PATH.
"""
import argparse
import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from utils import config
from utils.preprocess import get_eval_transform, get_train_transform

UNFREEZE_BLOCKS = 4  # how many of MobileNetV2's 19 feature blocks to fine-tune
PATIENCE = 3         # stop a phase after this many epochs without improvement
NUM_WORKERS = 2      # background image loading; set to 0 if Windows complains


def make_datasets():
    train_ds = datasets.ImageFolder(
        config.PROC_DIR / "train", transform=get_train_transform(config.IMG_SIZE))
    val_ds = datasets.ImageFolder(
        config.PROC_DIR / "val", transform=get_eval_transform(config.IMG_SIZE))
    if train_ds.classes != val_ds.classes:
        raise SystemExit(
            f"train and val have different classes:\n  {train_ds.classes}\n  {val_ds.classes}")
    return train_ds, val_ds


def class_weights(train_ds):
    #Rarer classes get bigger weights so the loss doesn't ignore them.
    counts = torch.bincount(
        torch.tensor(train_ds.targets), minlength=len(train_ds.classes)).float()
    return counts.sum() / (len(counts) * counts)


def build_model(n_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(model.last_channel, n_classes))
    return model


def freeze(model, n_unfrozen_blocks):
    #Freeze the whole body, then unfreeze its last n blocks. Head is always trainable.
    for p in model.features.parameters():
        p.requires_grad = False
    if n_unfrozen_blocks > 0:
        for block in model.features[-n_unfrozen_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_train_mode(model, n_unfrozen_blocks):
    model.train()
    first_trainable = len(model.features) - n_unfrozen_blocks
    for i, block in enumerate(model.features):
        if i < first_trainable:
            block.eval()


def run_epoch(model, loader, criterion, device, optimizer=None):
    #One pass over the data. Trains if an optimizer is given, otherwise only measures.
    training = optimizer is not None
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)
    return total_loss / seen, correct / seen


def train_phase(name, model, train_loader, val_loader, criterion, optimizer,
                epochs, n_unfrozen, device, best):
    print(f"\n=== {name} ===")
    waited = 0
    phase_best = float("inf")  # patience is counted within this phase only
    for epoch in range(1, epochs + 1):
        start = time.time()
        set_train_mode(model, n_unfrozen)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        model.eval()
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        note = ""
        if va_loss < best["loss"]:  # best across both phases: this is what gets saved
            best.update(loss=va_loss, acc=va_acc, state=copy.deepcopy(model.state_dict()))
            note = "  <- best overall"
        if va_loss < phase_best:    # best within this phase: this drives early stopping
            phase_best = va_loss
            waited = 0
        else:
            waited += 1

        print(f"epoch {epoch:2d}/{epochs}  "
              f"train loss {tr_loss:.3f} acc {tr_acc:.1%}  |  "
              f"val loss {va_loss:.3f} acc {va_acc:.1%}  "
              f"({time.time() - start:.0f}s){note}")

        if waited >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping this phase early.")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])  # continue from the best epoch, not the last


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=config.HEAD_EPOCHS)
    ap.add_argument("--epochs-ft", type=int, default=config.FINE_TUNE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random weights instead of ImageNet (only for smoke tests)")
    a = ap.parse_args()

    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_ds, val_ds = make_datasets()
    classes = train_ds.classes
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(classes))
    print("Classes and training images:")
    for name, n in zip(classes, counts.tolist()):
        print(f"  {name:20s} {n}")

    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device))
    model = build_model(len(classes), pretrained=not a.no_pretrained).to(device)
    best = {"loss": float("inf"), "acc": 0.0, "state": None}

    # Phase 1: only the new head learns
    freeze(model, 0)
    optimizer = torch.optim.Adam(trainable_params(model), lr=config.HEAD_LR)
    train_phase("Phase 1: head only", model, train_loader, val_loader, criterion,
                optimizer, a.epochs_head, 0, device, best)

    # Phase 2: last blocks of the body also learn, gently
    freeze(model, UNFREEZE_BLOCKS)
    optimizer = torch.optim.Adam(trainable_params(model), lr=config.FINE_TUNE_LR)
    train_phase("Phase 2: fine-tune", model, train_loader, val_loader, criterion,
                optimizer, a.epochs_ft, UNFREEZE_BLOCKS, device, best)

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in best["state"].items()},
        "classes": classes,
        "img_size": tuple(config.IMG_SIZE),
        "arch": "mobilenet_v2",
    }, config.MODEL_PATH)
    print(f"\nSaved best model (val loss {best['loss']:.3f}, val acc {best['acc']:.1%}) "
          f"to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()