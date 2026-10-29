# Derme: a skin screening tool

Derme gives a first, non-diagnostic look at a skin concern from a single photo.
You upload a photo (or take one with your camera), Derme checks that the photo is
good enough to read, and a neural network offers an *observation* about which of
five conditions the photo most resembles.

**Derme is a screening tool, not a diagnosis.** It is often wrong (see
[Honest limits](#honest-limits)) and it never replaces a dermatologist.

Built by Elika Alavi as a computer vision course project.

---

## How to run it (about 10 minutes)

You need **Python 3.10 or newer** and an internet connection for the first install.
The trained model is already included in the repository, so no training is needed.

**Windows (Command Prompt or PowerShell)**

```
git clone https://github.com/alavielica-jpg/derme.git
cd derme
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```
git clone https://github.com/alavielica-jpg/derme.git
cd derme
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If you received a ZIP file instead, unzip it, open a terminal in the unzipped
folder, and start from the `python -m venv .venv` line.

The browser opens at `http://localhost:8501`. If it does not, open that address
yourself.

Notes:
- The install downloads PyTorch (CPU version), so it takes a few minutes the first
  time. No graphics card is needed.
- The first photo takes a few seconds while the model loads. After that it is fast.
- The local database file (`screener.db`) is created automatically the first time you run the app.

---

## Using the app

The app has four pages, reached from the quiet menu at the top.

1. **Welcome**: press *Begin*.
2. **Screen**: enter a name or ID for the client, then upload a photo or use the
   camera. You get either a retake request or an observation.
3. **History**: the earlier screenings for that client.
4. **About and limits**: what Derme recognizes, how the quality check works, its
   accuracy, and what is stored.

Good photos to try: a clear, close, evenly lit photo of a skin area in daylight.
To see the quality check in action, try a very dark or blurry photo; Derme will ask
for a retake instead of guessing.

---

## How it works

```
photo -> OpenCV quality gate -> preprocessing -> MobileNetV2 -> confidence check -> result + database log
```

1. **Quality gate (OpenCV).** Before any prediction, the photo is checked for blur
   (variance of the Laplacian), lighting and exposure, resolution, and how much of
   the image looks like skin. If it fails, the user is asked to retake the photo and
   no prediction is made. The thresholds were tuned on example photos.
2. **Preprocessing (PyTorch / torchvision).** The photo is resized and normalized
   the same way the network was trained.
3. **Classifier (MobileNetV2, transfer learning).** A MobileNetV2 network pretrained
   on ImageNet was adapted to five skin conditions: first only the new classification
   head was trained, then the last blocks were fine-tuned. Training used class
   weights (softened with a square root), label smoothing, weight decay, dropout, and
   data augmentation (zoom crops, flips, rotation, brightness and contrast changes,
   small blur, hidden patches).
4. **Confidence check.** A confidence threshold, chosen on validation photos, decides
   whether Derme shows an observation or says it could not read the photo clearly.
5. **Storage (SQLite).** Only results are saved: labels, scores, quality measures,
   and file names. **The photos themselves are never stored.**

**Conditions recognized:** eczema, urticaria, insect bite, folliculitis, psoriasis.

**Data:** the SCIN skin condition image dataset (downloaded from Hugging Face),
filtered to dermatologist-labelled cases of these five conditions. Photos were split
by *case*, so photos of the same case never appear in both training and test data.

---

## Honest limits

Measured on 247 held-out test photos:

| Measure | Result |
|---|---|
| Overall accuracy | 44.5% (guessing "eczema" every time gives 43.7%) |
| Average of per-class accuracies | 37.1% (20% would be chance) |
| Accuracy on the more confident half of photos | 58.4% |

What this means:
- Derme is wrong more often than it is right. Several of these conditions look alike
  in everyday photos, especially eczema, urticaria, and insect bites.
- The dataset is small and made of ordinary photos, not clinical images.
- Photos from the same case tend to succeed or fail together, so the test results are
  less certain than the photo count suggests.
- Skin-tone groups were checked, but the darker-skin group has only 21 test photos,
  so no reliable conclusion about bias can be drawn.
- For these reasons the app uses the word "observation", marks low-confidence results
  as unclear, and always advises seeing a dermatologist.

The confusion matrix is in `reports/confusion_matrix.png`.

---

## Project layout

```
app.py                  Streamlit app (all four pages)
download_datasets.py    Downloads the SCIN dataset (only needed for retraining)
utils/                  Quality gate, preprocessing, training, evaluation, prediction,
                        and the SQLite layer for clients and screening history
models/                 Trained model (skin_model.pt) and confidence threshold (threshold.json)
reports/                Confusion matrix of the final model
notebooks/              Early exploration of the data and the quality-gate thresholds
tests/                  Automated tests
.streamlit/             Theme settings for the app
requirements.txt        Python packages
```

The training data is not included in the repository (it is large and downloaded
separately).

---

## Retraining (optional)

Not needed to run the app. The steps, in order:

```
python download_datasets.py scin
python -m utils.organize_data --help        # shows how to choose classes and the label-confidence cutoff
python -m utils.train
python -m utils.evaluate
```

Training takes about 30 minutes on a laptop CPU. `utils.train` saves its result as
`models/skin_model_candidate.pt` so the current model is never overwritten; copy it to
`models/skin_model.pt` to use it.

---

## Possible next steps

A larger and cleaner dataset, including photos from more skin tones; a chat feature
that explains results; and short general-care information for each condition.