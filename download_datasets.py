"""
Download SCIN and Fitzpatrick17k into data/raw/.

    pip install datasets pillow requests pandas

Run in this order:
    python download_datasets.py inspect   # look at columns first (small download)
    python download_datasets.py scin      # export SCIN images + metadata
    python download_datasets.py fitz      # fetch Fitzpatrick17k images from their URLs

If huggingface.co is unreachable from your network, try a mirror first:
    Windows (PowerShell):  $env:HF_ENDPOINT="https://hf-mirror.com"
    Linux/macOS:           export HF_ENDPOINT=https://hf-mirror.com
"""
import json
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image

RAW = Path("data/raw")

SCIN_REPO = "google/scin"
FITZ_REPO = "spycoder/fitzpatrick"

FITZ_LABEL_COL = "label"
FITZ_URL_COL = "url"
FITZ_ID_COL = "md5hash"
FITZ_KEEP = ["acne", "eczema", "psoriasis"]  


def load(repo):
    ds = load_dataset(repo)
    return ds["train"] if "train" in ds else ds[list(ds.keys())[0]]


def short(v):
    if isinstance(v, Image.Image):
        return f"<PIL image {v.size}>"
    if isinstance(v, list):
        return [short(x) for x in v[:3]]
    return v


def inspect():
    for repo in (SCIN_REPO, FITZ_REPO):
        print("=" * 70, "\n", repo)
        try:
            ds = load(repo)
        except Exception as e:
            print("FAILED to load:", e)
            continue
        print(ds)
        print("\nColumns:", ds.column_names)
        print("\nFirst row:")
        for k, v in ds[0].items():
            print(f"  {k}: {short(v)}")


def export_scin():
    ds = load(SCIN_REPO)
    out = RAW / "scin"
    (out / "images").mkdir(parents=True, exist_ok=True)

    meta = []
    for i, row in enumerate(ds):
        rec, files = {}, []
        for k, v in row.items():
            items = v if isinstance(v, list) else [v]
            if items and all(isinstance(x, Image.Image) for x in items):
                for j, im in enumerate(items):
                    fn = f"{i:05d}_{j}.jpg"
                    im.convert("RGB").save(out / "images" / fn, quality=95)
                    files.append(fn)
            else:
                rec[k] = v
        rec["files"] = files
        meta.append(rec)
        if i % 500 == 0:
            print(f"SCIN: {i}/{len(ds)}")

    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, default=str)
    print(f"Done. {len(meta)} cases saved to {out}")


def export_fitz():
    ds = load(FITZ_REPO)
    for col in (FITZ_LABEL_COL, FITZ_URL_COL):
        if col not in ds.column_names:
            sys.exit(f"Column '{col}' not found. Columns are: {ds.column_names}")

    out = RAW / "fitzpatrick17k"
    (out / "images").mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (research project)"}

    kept = failed = 0
    log, bad = [], []
    for i, row in enumerate(ds):
        label = str(row[FITZ_LABEL_COL]).lower()
        if not any(k in label for k in FITZ_KEEP):
            continue
        name = row.get(FITZ_ID_COL) or f"{i:06d}"
        path = out / "images" / f"{name}.jpg"

        if not path.exists():
            ok = False
            for _ in range(2):
                try:
                    r = requests.get(row[FITZ_URL_COL], headers=headers, timeout=20)
                    r.raise_for_status()
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(r.content)
                    Image.open(tmp).convert("RGB").save(path, quality=95)
                    tmp.unlink()
                    ok = True
                    break
                except Exception:
                    time.sleep(1)
            if not ok:
                failed += 1
                bad.append(row[FITZ_URL_COL])
                continue
            time.sleep(0.2)  # be polite to the source sites

        kept += 1
        log.append({k: v for k, v in row.items() if not isinstance(v, Image.Image)} | {"file": path.name})
        if kept % 100 == 0:
            print(f"Fitzpatrick: {kept} saved, {failed} failed")

    with open(out / "metadata.json", "w") as f:
        json.dump(log, f, default=str)
    (out / "failed_urls.txt").write_text("\n".join(bad))
    print(f"Done. {kept} saved, {failed} failed (see failed_urls.txt)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": inspect, "scin": export_scin, "fitz": export_fitz}[cmd]()