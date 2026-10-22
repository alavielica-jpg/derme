"""Evaluate the trained model on the held-out test photos.

Run from the repo root (never from inside utils/):
    python -m utils.evaluate 

Prints:  overall and per-class results, a "just guess the biggest class" baseline,
         accuracy by skin-tone band, and the confidence threshold. The threshold is
         chosen on the VALIDATION photos and then only checked on the test photos.

Writes:  reports/confusion_matrix.png, reports/failures.csv, models/threshold.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from utils import config
from utils.preprocess import get_eval_transform
from utils.train import build_model

MIN_ACCEPTED = 10  # a threshold must keep at least this many photos to be trusted


def load_model(device):
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], ckpt["img_size"]


def get_probs(model, loader, device):
    #Probability of each class for every test image, as a numpy array.
    chunks = []
    with torch.no_grad():
        for images, _ in loader:
            chunks.append(torch.softmax(model(images.to(device)), dim=1).cpu())
    return torch.cat(chunks).numpy()


def tone_band(fst_label):
    #'FST4' -> 'medium (FST 3-4)'. Bands are coarse because the test set is small.
    match = re.search(r"\d", fst_label or "")
    if not match:
        return "unknown"
    n = int(match.group())
    if n <= 2:
        return "light (FST 1-2)"
    return "medium (FST 3-4)" if n <= 4 else "dark (FST 5-6)"


def skin_tone_report(paths, y_true, y_pred):
    manifest_path = config.PROC_DIR / "manifest.csv"
    if not manifest_path.exists():
        print("\nNo manifest.csv found, skipping the skin-tone check.")
        return
    band_of = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            band_of[row["file"]] = tone_band(row["fitzpatrick"])

    hits = {}
    for path, t, p in zip(paths, y_true, y_pred):
        key = Path(path).relative_to(config.PROC_DIR).as_posix()
        hits.setdefault(band_of.get(key, "unknown"), []).append(t == p)

    print("\nAccuracy by skin-tone band (dermatologist Fitzpatrick label):")
    for band in sorted(hits):
        n = len(hits[band])
        print(f"  {band:20s} n={n:3d}  accuracy {np.mean(hits[band]):.1%}")
    print("  (small groups swing a lot: one photo can move a band by several points)")


def plot_confusion(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def choose_threshold(probs, y_true, target_acc):
    """Lowest confidence cut-off whose kept photos reach target_acc (validation data).

    If none does, fall back to the cut-off that keeps the more confident half of the
    photos, so the app does not answer 'inconclusive' for everything.
    """
    y_pred, conf = probs.argmax(1), probs.max(1)
    print("\nConfidence threshold, chosen on validation photos:")
    print("threshold  kept  coverage  accuracy_on_kept")
    chosen = None
    for t in np.arange(0.30, 0.96, 0.05):
        mask = conf >= t
        kept = int(mask.sum())
        acc = (y_pred[mask] == y_true[mask]).mean() if kept else float("nan")
        print(f"{t:8.2f}  {kept:4d}  {mask.mean():8.1%}  {acc:15.1%}")
        if chosen is None and kept >= MIN_ACCEPTED and acc >= target_acc:
            chosen = float(round(t, 2))
    if chosen is None:
        chosen = float(np.floor(np.median(conf) * 100) / 100)
        print(f"No threshold with >= {MIN_ACCEPTED} photos reached {target_acc:.0%}. "
              f"Fallback: {chosen} keeps the more confident half.")
    return chosen


def analyze(probs, y_true, paths, classes, target_acc, val_probs, val_true):
    """All the reporting. Takes plain numpy arrays so it doesn't depend on the model."""
    n_classes = len(classes)
    labels = list(range(n_classes))
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Overall + per class, with the baseline to compare against
    counts = np.bincount(y_true, minlength=n_classes)
    biggest = int(counts.argmax())
    print(f"Test photos: {len(y_true)}")
    print(f"Accuracy: {(y_pred == y_true).mean():.1%}   "
          f"(baseline: always guessing '{classes[biggest]}' = {counts.max() / len(y_true):.1%})\n")
    print(classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = cm.diagonal() / cm.sum(1).clip(min=1)
    print("Per-class accuracy (share of each condition's photos found):")
    for name, n, acc in zip(classes, counts, per_class):
        print(f"  {name:20s} {acc:6.1%}   ({n} test photos)")
    print(f"Average of the per-class accuracies: {per_class.mean():.1%}")
    plot_confusion(cm, classes)

    skin_tone_report(paths, y_true, y_pred)

    # Embarrassing failures: wrong AND confident
    wrong = np.where(y_pred != y_true)[0]
    wrong = wrong[np.argsort(-conf[wrong])]
    with open(config.REPORT_DIR / "failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "predicted", "confidence"])
        for i in wrong:
            w.writerow([paths[i], classes[y_true[i]], classes[y_pred[i]], f"{conf[i]:.3f}"])
    print(f"\n{len(wrong)} errors. Most confident mistakes first: reports/failures.csv")

    # "Inconclusive" threshold: chosen on validation photos, then checked on test
    chosen = choose_threshold(val_probs, val_true, target_acc)
    mask = conf >= chosen
    kept = int(mask.sum())
    acc_kept = (y_pred[mask] == y_true[mask]).mean() if kept else float("nan")
    print(f"\nOn the test photos, threshold {chosen} keeps {kept} of {len(conf)} "
          f"({mask.mean():.0%}) with accuracy {acc_kept:.1%} on those.")
    config.THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.THRESHOLD_PATH.write_text(json.dumps({"threshold": chosen}))
    print("Saved confidence threshold:", chosen)


def main(target_acc):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, img_size = load_model(device)

    def load_split(split):
        ds = datasets.ImageFolder(config.PROC_DIR / split,
                                  transform=get_eval_transform(img_size))
        if ds.classes != classes:
            raise SystemExit(
                f"{split} folders {ds.classes} don't match the model's classes {classes}. "
                "Rebuild the data with the same classes you trained on.")
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
        return ds, get_probs(model, loader, device)

    val_ds, val_probs = load_split("val")
    test_ds, probs = load_split("test")
    analyze(probs, np.array(test_ds.targets), [p for p, _ in test_ds.samples], classes,
            target_acc, val_probs, np.array(val_ds.targets))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-acc", type=float, default=0.85)
    main(ap.parse_args().target_acc)