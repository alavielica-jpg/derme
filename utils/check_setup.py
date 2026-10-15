import sys
import importlib
from importlib import metadata

# (pip name, import name)
REQUIRED = [
    ("streamlit", "streamlit"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("python-dotenv", "dotenv"),
    ("pytest", "pytest"),
    ("notebook", "notebook"),
    ("ipykernel", "ipykernel"),
    ("nbstripout", "nbstripout"),
]

# Only one of these is needed
FRAMEWORKS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("tensorflow", "tensorflow"),
]


def check(pip_name, import_name):
    try:
        importlib.import_module(import_name)
        try:
            version = metadata.version(pip_name)
        except metadata.PackageNotFoundError:
            version = "unknown version"
        print(f"  OK       {pip_name:<15} {version}")
        return True
    except Exception as e:
        print(f"  MISSING  {pip_name:<15} ({type(e).__name__}: {e})")
        return False


print(f"Python {sys.version.split()[0]}")
print(f"Running from: {sys.executable}\n")

if ".venv" not in sys.executable:
    print("WARNING: this doesn't look like your .venv. Activate it first.\n")

print("Required packages:")
results = [check(p, i) for p, i in REQUIRED]

print("\nModel framework (need torch + torchvision OR tensorflow):")
fw = {p: check(p, i) for p, i in FRAMEWORKS}

has_framework = (fw["torch"] and fw["torchvision"]) or fw["tensorflow"]

# Quick functional test, not just an import test
print("\nFunctional checks:")
try:
    import cv2, numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"  OK       OpenCV Laplacian blur score runs ({blur_score:.1f})")
except Exception as e:
    print(f"  FAILED   OpenCV test: {e}")

try:
    import sqlite3
    sqlite3.connect(":memory:").close()
    print("  OK       sqlite3 (built into Python)")
except Exception as e:
    print(f"  FAILED   sqlite3: {e}")

print("\n" + "=" * 40)
if all(results) and has_framework:
    print("All set. Your environment is ready.")
else:
    print("Something is missing. Install the MISSING items with:")
    print("  pip install <pip name>")
