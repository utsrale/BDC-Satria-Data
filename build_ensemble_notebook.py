import json

notebook_path = 'd:/Pancar/KULIAH UNS/LOMBA/Satria Data 2026/BDC SatDat 2026/ensemble_inference.ipynb'

cells = []

# Sel 0: Markdown Judul
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Big Data Challenge (BDC) Satria Data 2026\n",
        "## Notebook Ensemble Inference (DINOv2 Large + RegNetY SWAG)\n",
        "\n",
        "**Strategi:**\n",
        "1. **Ensemble Soft Voting:** Menggabungkan probabilitas keyakinan dari 5 model DINOv2 Large (Transformer) + 5 model RegNetY SWAG (CNN).\n",
        "2. **Automatic Path Detection:** Kode akan memindai folder input Kaggle secara otomatis untuk mencari file bobot model `.pth`.\n",
        "3. **Zero Training Time:** Notebook ini hanya melakukan prediksi (inference) saja, selesai dalam waktu kurang dari 2 menit."
    ]
})

# Sel 1: Konfigurasi Inline & Deteksi Environment
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 1. KONFIGURASI & DETEKSI PATH DATASET (Kaggle & Colab)\n",
        "# ============================================================\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "def detect_environment():\n",
        "    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:\n",
        "        return 'kaggle'\n",
        "    try:\n",
        "        import google.colab\n",
        "        return 'colab'\n",
        "    except ImportError:\n",
        "        pass\n",
        "    return 'local'\n",
        "\n",
        "ENV = detect_environment()\n",
        "print(f'Environment terdeteksi: {ENV}')\n",
        "\n",
        "if ENV == 'colab':\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    project_path = '/content/drive/MyDrive/BISMILLAH BDC 2026/BDC SatDat 2026'\n",
        "    if os.path.exists(project_path):\n",
        "        sys.path.append(project_path)\n",
        "        os.chdir(project_path)\n",
        "        print(f'Berhasil berpindah ke direktori project: {os.getcwd()}')\n",
        "\n",
        "# Setup Path Dataset\n",
        "if ENV == 'colab':\n",
        "    if Path('/content/BDC2026/train').exists():\n",
        "        DATA_DIR = Path('/content/BDC2026')\n",
        "    elif Path('/content/drive/MyDrive/BISMILLAH BDC 2026/BDC2026/train').exists():\n",
        "        DATA_DIR = Path('/content/drive/MyDrive/BISMILLAH BDC 2026/BDC2026')\n",
        "    else:\n",
        "        DATA_DIR = Path('/content/drive/MyDrive/BISMILLAH BDC 2026/BDC SatDat 2026/BDC2026')\n",
        "elif ENV == 'kaggle':\n",
        "    kaggle_input = Path('/kaggle/input')\n",
        "    found = False\n",
        "    for root, dirs, files in os.walk(kaggle_input):\n",
        "        if 'train' in dirs:\n",
        "            train_candidate = Path(root) / 'train'\n",
        "            sub_items = os.listdir(train_candidate)\n",
        "            if any('ecyclable' in s or 'lectronic' in s or 'rganic' in s for s in sub_items):\n",
        "                DATA_DIR = Path(root)\n",
        "                found = True\n",
        "                break\n",
        "    if not found:\n",
        "        DATA_DIR = kaggle_input / 'bdc2026-dataset'\n",
        "        print('WARNING: Folder train tidak ditemukan.')\n",
        "else:\n",
        "    DATA_DIR = Path('BDC2026')\n",
        "    if not DATA_DIR.exists():\n",
        "        DATA_DIR = Path('../BDC2026')\n",
        "\n",
        "TRAIN_DIR = DATA_DIR / 'train'\n",
        "TEST_DIR = DATA_DIR / 'test'\n",
        "SUBMISSION_TEMPLATE = DATA_DIR / 'submission.csv'\n",
        "\n",
        "if ENV == 'kaggle':\n",
        "    PROJECT_ROOT = Path('/kaggle/working')\n",
        "elif ENV == 'colab':\n",
        "    PROJECT_ROOT = Path('/content/drive/MyDrive/BISMILLAH BDC 2026/BDC SatDat 2026')\n",
        "else:\n",
        "    PROJECT_ROOT = Path('.')\n",
        "\n",
        "CLASS_NAMES = ['Recyclable', 'Electronic', 'Organic']\n",
        "CLASS_TO_IDX = {'Recyclable': 0, 'Electronic': 1, 'Organic': 2}\n",
        "IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}\n",
        "IMAGENET_MEAN = (0.485, 0.456, 0.406)\n",
        "IMAGENET_STD = (0.229, 0.224, 0.225)\n",
        "SEED = 42\n",
        "\n",
        "print(f'Data dir         : {DATA_DIR} (exists={DATA_DIR.exists()})')\n",
        "print(f'Test dir         : {TEST_DIR} (exists={TEST_DIR.exists()})')\n",
        "print(f'Submission tmpl  : {SUBMISSION_TEMPLATE} (exists={SUBMISSION_TEMPLATE.exists()})')"
    ]
})

# Sel 2: Install Library
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 2. INSTALL LIBRARY\n",
        "!pip install -q timm pandas pillow"
    ]
})

# Sel 3: Import Library & Setup Seed
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3. IMPORT LIBRARIES\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torchvision import transforms\n",
        "from PIL import Image\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import glob\n",
        "import timm\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Device: {device}')"
    ]
})

# Sel 4: Transformasi Evaluasi (Resolusi 224x224 untuk kedua model)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 4. IMAGE TRANSFORMS\n",
        "IMG_SIZE = 224\n",
        "val_transforms = transforms.Compose([\n",
        "    transforms.Resize((IMG_SIZE, IMG_SIZE)),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),\n",
        "])"
    ]
})

# Sel 5: Definisi Arsitektur Model DINOv2 dan RegNetY SWAG
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 5. MODEL DEFINITIONS FOR INFERENCE\n",
        "class Dinov2WasteClassifier(nn.Module):\n",
        "    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m', num_classes=3):\n",
        "        super().__init__()\n",
        "        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, dynamic_img_size=True)\n",
        "        num_features = self.backbone.num_features\n",
        "        self.head = nn.Sequential(\n",
        "            nn.Linear(num_features, 512),\n",
        "            nn.BatchNorm1d(512),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.4),\n",
        "            nn.Linear(512, num_classes)\n",
        "        )\n",
        "    def forward(self, x):\n",
        "        features = self.backbone(x)\n",
        "        out = self.head(features)\n",
        "        return out\n",
        "\n",
        "class RegNetYSwagClassifier(nn.Module):\n",
        "    def __init__(self, model_name='regnety_160_swag_in1k', num_classes=3):\n",
        "        super().__init__()\n",
        "        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)\n",
        "        num_features = self.backbone.num_features\n",
        "        self.head = nn.Sequential(\n",
        "            nn.Linear(num_features, 512),\n",
        "            nn.BatchNorm1d(512),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.3),\n",
        "            nn.Linear(512, num_classes)\n",
        "        )\n",
        "    def forward(self, x):\n",
        "        features = self.backbone(x)\n",
        "        out = self.head(features)\n",
        "        return out"
    ]
})

# Sel 6: Skrip Pemindai Otomatis Berkas Bobot Model (.pth)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 6. AUTO DETECT MODEL WEIGHTS IN KAGGLE DIRECTORIES\n",
        "def find_model_weights(name_pattern):\n",
        "    candidates = []\n",
        "    # Scan local paths\n",
        "    candidates.extend(glob.glob(f\"models/{name_pattern}\"))\n",
        "    candidates.extend(glob.glob(f\"../models/{name_pattern}\"))\n",
        "    # Scan Kaggle input and working directories recursively\n",
        "    candidates.extend(glob.glob(f\"/kaggle/input/**/{name_pattern}\", recursive=True))\n",
        "    candidates.extend(glob.glob(f\"/kaggle/working/**/{name_pattern}\", recursive=True))\n",
        "    paths = sorted(list(set(os.path.abspath(p) for p in candidates)))\n",
        "    return paths\n",
        "\n",
        "dinov2_weights_paths = find_model_weights(\"best_dinov2_fold_*.pth\")\n",
        "regnet_weights_paths = find_model_weights(\"best_regnet_fold_*.pth\")\n",
        "\n",
        "print(\"Berkas bobot DINOv2 ditemukan:\")\n",
        "for p in dinov2_weights_paths:\n",
        "    print(f\"  - {p}\")\n",
        "\n",
        "print(\"\\nBerkas bobot RegNetY ditemukan:\")\n",
        "for p in regnet_weights_paths:\n",
        "    print(f\"  - {p}\")"
    ]
})

# Sel 7: Memuat Model untuk Ensemble
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 7. LOAD MODELS FOR ENSEMBLE\n",
        "models = []\n",
        "\n",
        "# Load DINOv2 Models\n",
        "for p in dinov2_weights_paths:\n",
        "    model_instance = Dinov2WasteClassifier(model_name='vit_large_patch14_dinov2.lvd142m', num_classes=len(CLASS_NAMES))\n",
        "    model_instance.load_state_dict(torch.load(p, map_location=device))\n",
        "    model_instance = model_instance.to(device)\n",
        "    model_instance.eval()\n",
        "    models.append((model_instance, 0.6)) # Bobot pengaruh DINOv2 = 0.6\n",
        "    print(f\"Loaded DINOv2 model: {os.path.basename(p)} (weight: 0.6)\")\n",
        "\n",
        "# Load RegNetY Models\n",
        "for p in regnet_weights_paths:\n",
        "    model_instance = RegNetYSwagClassifier(model_name='regnety_160_swag_in1k', num_classes=len(CLASS_NAMES))\n",
        "    model_instance.load_state_dict(torch.load(p, map_location=device))\n",
        "    model_instance = model_instance.to(device)\n",
        "    model_instance.eval()\n",
        "    models.append((model_instance, 0.4)) # Bobot pengaruh RegNetY = 0.4\n",
        "    print(f\"Loaded RegNetY model: {os.path.basename(p)} (weight: 0.4)\")\n",
        "\n",
        "print(f\"\\nEnsemble siap dengan total {len(models)} model!\")"
    ]
})

# Sel 8: Eksekusi Soft-Voting Inference & Export CSV
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 8. SOFT VOTING INFERENCE\n",
        "df_submission = pd.read_csv(SUBMISSION_TEMPLATE)\n",
        "df_submission['id'] = df_submission['id'].astype(float).astype(int).astype(str)\n",
        "\n",
        "predictions = []\n",
        "softmax = nn.Softmax(dim=1)\n",
        "total_test = len(df_submission)\n",
        "\n",
        "with torch.no_grad():\n",
        "    for idx, row in df_submission.iterrows():\n",
        "        if (idx + 1) % 200 == 0 or (idx + 1) == total_test:\n",
        "            print(f'Predicting: {idx + 1}/{total_test}', flush=True)\n",
        "            \n",
        "        test_id = row['id']\n",
        "        img_name_candidates = [f'{test_id}.jpg', f'{test_id}.jpeg', f'{test_id}.png']\n",
        "        img_path = None\n",
        "        for cand in img_name_candidates:\n",
        "            cand_path = TEST_DIR / cand\n",
        "            if cand_path.exists():\n",
        "                img_path = cand_path\n",
        "                break\n",
        "        \n",
        "        if img_path is None:\n",
        "            print(f'WARNING: Image dengan ID {test_id} tidak ditemukan.')\n",
        "            predictions.append(0)\n",
        "            continue\n",
        "            \n",
        "        image = Image.open(img_path).convert('RGB')\n",
        "        image = val_transforms(image).unsqueeze(0).to(device)\n",
        "        \n",
        "        # Soft voting weighted average\n",
        "        accumulated_prob = torch.zeros(1, len(CLASS_NAMES)).to(device)\n",
        "        total_weight = 0.0\n",
        "        \n",
        "        for model_instance, weight in models:\n",
        "            outputs = model_instance(image)\n",
        "            probs = softmax(outputs)\n",
        "            accumulated_prob += probs * weight\n",
        "            total_weight += weight\n",
        "            \n",
        "        # Ambil argmax dari rata-rata probabilitas berbobot\n",
        "        final_pred = torch.argmax(accumulated_prob / total_weight, dim=1).item()\n",
        "        predictions.append(final_pred)\n",
        "\n",
        "df_submission['predicted'] = predictions\n",
        "output_file = PROJECT_ROOT / 'submission_SD2026040000187.csv'\n",
        "df_submission.to_csv(output_file, index=False)\n",
        "\n",
        "print(f'\\nEnsemble Submission sukses disimpan di: {output_file}')\n",
        "print(df_submission['predicted'].value_counts().rename(index=IDX_TO_CLASS))"
    ]
})

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook Ensemble Inference sukses dibangun!")
