import argparse
import ast
import csv
import json
import random
import shutil
from collections import Counter

from utils import config

SCIN_DIR = config.RAW_DIR / "scin"
META_PATH = SCIN_DIR / "metadata.json"
IMG_DIR = SCIN_DIR / "images"


def top_label(record):
    raw = record.get("weighted_skin_condition_label")
    if not raw:
        return None, 0.0
    weights = ast.literal_eval(raw)  
    if not weights:
        return None, 0.0
    best = max(weights.values())
    winners = [k for k, v in weights.items() if v == best]
    if len(winners) > 1: 
        return None, 0.0
    return winners[0], best


def survey(records, min_weight):
    counts, strong = Counter(), Counter()
    for r in records:
        label, w = top_label(r)
        if label is None:
            continue
        counts[label] += 1
        if w >= min_weight:
            strong[label] += 1

    print(f"{'condition':40s} {'cases':>6s} {'w>=' + str(min_weight):>8s}")
    for label, n in counts.most_common(40):
        print(f"{label:40s} {n:6d} {strong[label]:8d}")
    print(f"\n{len(records)} records, {sum(counts.values())} with a clear top label")

    grade = Counter(r.get("dermatologist_gradable_for_skin_condition_1") for r in records)
    print("\nImage-quality field (dermatologist_gradable_for_skin_condition_1):")
    for value, n in grade.most_common():
        print(f"  {value}: {n}")


def build(records, classes, min_weight, val_frac, test_frac):
    random.seed(config.SEED)
    wanted = {c.lower(): c for c in classes}
    by_class = {c: [] for c in classes}

    for r in records:
        label, w = top_label(r)
        if label is None or w < min_weight:
            continue
        if label.lower() in wanted:
            by_class[wanted[label.lower()]].append(r)

    empty = [c for c, cases in by_class.items() if not cases]
    if empty:
        raise SystemExit(
            f"No cases found for: {', '.join(empty)}. "
            "Check spelling with --survey (names must match exactly, ignoring case)."
        )

    if config.PROC_DIR.exists():
        print(f"Removing old {config.PROC_DIR} (it is rebuilt from raw data)")
        shutil.rmtree(config.PROC_DIR)

    manifest, missing = [], 0
    print(f"\n{'class':28s} {'cases':>6s} {'train':>6s} {'val':>5s} {'test':>5s}  (images)")
    for cls, cases in by_class.items():
        folder = cls.lower().replace(" ", "_")
        random.shuffle(cases)
        n = len(cases)
        n_test, n_val = int(n * test_frac), int(n * val_frac)
        splits = {
            "test": cases[:n_test],
            "val": cases[n_test:n_test + n_val],
            "train": cases[n_test + n_val:],
        }
        img_counts = {}
        for split, split_cases in splits.items():
            out = config.PROC_DIR / split / folder
            out.mkdir(parents=True, exist_ok=True)
            img_counts[split] = 0
            for r in split_cases:
                for fname in r.get("files") or []:
                    src = IMG_DIR / fname
                    if not src.exists():
                        missing += 1
                        continue
                    shutil.copy2(src, out / fname)
                    img_counts[split] += 1
                    manifest.append({
                        "file": f"{split}/{folder}/{fname}",
                        "split": split,
                        "label": folder,
                        "case_id": r["case_id"],
                        "fitzpatrick": r.get("dermatologist_fitzpatrick_skin_type_label_1"),
                        "monk_us": r.get("monk_skin_tone_label_us"),
                    })
        print(f"{folder:28s} {n:6d} {img_counts['train']:6d} {img_counts['val']:5d} "
              f"{img_counts['test']:5d}")

    with open(config.PROC_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nWrote {len(manifest)} images and manifest.csv to {config.PROC_DIR}")
    if missing:
        print(f"Warning: {missing} files listed in metadata were not found in {IMG_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true", help="count conditions and exit")
    ap.add_argument("--classes", nargs="+", help="condition names to include")
    ap.add_argument("--min-weight", type=float, default=0.5,
                    help="minimum vote share for a case's top label")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    a = ap.parse_args()

    with open(META_PATH, encoding="utf-8") as f:
        records = json.load(f)

    if a.survey:
        survey(records, a.min_weight)
    elif a.classes:
        build(records, a.classes, a.min_weight, a.val, a.test)
    else:
        ap.error("use --survey first, then --classes <names>")


if __name__ == "__main__":
    main()