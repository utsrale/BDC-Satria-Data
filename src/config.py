"""
Konfigurasi global project BDC Satria Data 2026.

Auto-detect environment (lokal / Google Colab / cloud runtime) dan
menyediakan path yang konsisten untuk seluruh notebook & script.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Deteksi environment
# ---------------------------------------------------------------------------
def detect_environment() -> str:
    """Return 'colab', 'kaggle', atau 'local'."""
    import importlib.util
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


ENV = detect_environment()


# ---------------------------------------------------------------------------
# 2. Path root project
# ---------------------------------------------------------------------------
def _get_project_root() -> Path:
    """Cari root project (folder yang punya subfolder BDC2026/)."""
    if ENV == "colab":
        # Diasumsikan user mount Drive dan taruh project di:
        # /content/drive/MyDrive/BISMILLAH BDC 2026/BDC SatDat 2026/
        candidate = Path("/content/drive/MyDrive/BISMILLAH BDC 2026/BDC SatDat 2026")
        if candidate.exists():
            return candidate
        # Fallback: /content jika user upload zip
        return Path("/content")

    if ENV == "kaggle":
        # Kaggle biasanya dataset di /kaggle/input/<slug>/
        # Ubah sesuai kebutuhan
        return Path("/kaggle/working")

    # LOCAL: cari upward dari file ini
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "BDC2026").exists() or (parent.parent / "BDC2026").exists():
            return parent
    # Fallback: parent dari src/
    return here.parent.parent


PROJECT_ROOT: Path = _get_project_root()

# ---------------------------------------------------------------------------
# 3. Path dataset & output (dataset BDC2026)
# ---------------------------------------------------------------------------
if ENV == "colab":
    # Prioritaskan folder dataset luar (sejajar dengan folder project) yang lengkap
    colab_outer_data = Path("/content/drive/MyDrive/BISMILLAH BDC 2026/BDC2026")
    colab_inner_data = PROJECT_ROOT / "BDC2026"
    
    if colab_outer_data.exists():
        DATA_DIR: Path = colab_outer_data
    else:
        DATA_DIR = colab_inner_data
else:
    if (PROJECT_ROOT / "BDC2026").exists():
        DATA_DIR = PROJECT_ROOT / "BDC2026"
    else:
        DATA_DIR = PROJECT_ROOT.parent / "BDC2026"

TRAIN_DIR: Path = DATA_DIR / "train"
TEST_DIR: Path = DATA_DIR / "test"
SUBMISSION_TEMPLATE: Path = DATA_DIR / "submission.csv"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Auto-create output folders
for _d in (OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 4. Konstanta kelas
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]
CLASS_TO_IDX = {"Recyclable": 0, "Electronic": 1, "Organic": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Folder train punya nama '0_Recyclable', '1_Electronic', '2_Organic'
TRAIN_SUBFOLDERS = ["0_Recyclable", "1_Electronic", "2_Organic"]

# ImageNet normalization (dipakai saat transfer learning nanti)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# 5. Random seed
# ---------------------------------------------------------------------------
SEED = 42


def summary() -> None:
    """Print ringkasan konfigurasi (dipanggil dari notebook)."""
    print(f"Environment      : {ENV}")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Data dir         : {DATA_DIR}  (exists={DATA_DIR.exists()})")
    print(f"Train dir        : {TRAIN_DIR} (exists={TRAIN_DIR.exists()})")
    print(f"Test dir         : {TEST_DIR}  (exists={TEST_DIR.exists()})")
    print(f"Submission tmpl  : {SUBMISSION_TEMPLATE} (exists={SUBMISSION_TEMPLATE.exists()})")
    print(f"Outputs dir      : {OUTPUTS_DIR}")
    print(f"Plots dir        : {PLOTS_DIR}")


if __name__ == "__main__":
    summary()
