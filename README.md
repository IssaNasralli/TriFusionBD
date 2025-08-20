# TriFusionBD – Building Segmentation from GeoTIFF (Test Repository)

This repository provides a **lightweight test harness** for the pretrained **TriFusionBD** model to segment **buildings** on updated test images that combine **RGB + DEM + Slope** layers (5 bands). You will (1) run inference to obtain per-pixel **probabilities (0–1)** and (2) apply a threshold to get **binary masks** suitable for GIS visualization.

---

## 📦 Requirements

- Python 3.8+
- TensorFlow (2.12–2.15 recommended)
- numpy
- rasterio
Install dependencies:

    pip install -r requirements.txt

Create `requirements.txt`:

    tensorflow==2.15.0
    numpy
    rasterio

---

## 📂 Repository Structure

    TriFusionBD-Test/
    │── README.md
    │── TriFusion_Gate_Atrous_Gate.py   # Model definition
    │── predict.py                       # Run inference on all GeoTIFFs in test_dem/
    │── threshold.py                     # Apply threshold to probability maps
    │
    ├── test_updated/                        # UPDATED test GeoTIFFs (5 bands: R,G,B,DEM,Slope)
    │   ├── 1_updated.tif
    │   └── 2_updated.tif
    │
    └── test/                            # ORIGINAL test GeoTIFFs (reference RGB)
        ├── 1.tif
        └── 2.tif

**Input format**: GeoTIFFs with **5 bands** ordered as **[R, G, B, DEM, Slope]**.

---

## 🚀 Quick Start

### 1) Run Prediction (probability maps)
Runs on **all `.tif` files inside `test_dem/`** and writes per-pixel building probabilities (0–1) to `output/`.

    python predict.py

This creates:

    output/
      ├── 1_updated.tif    # float32, values in [0,1]
      └── 2_updated.tif

**Color meaning (for visualization):**
- 0.0 → **black** (non-building)
- 1.0 → **white** (building)
- (0.0–1.0) → **grayscale** probability

---

### 2) Apply Threshold (binary masks)
Converts probability maps to **binary** masks using a user-defined threshold (e.g., 0.50). The result contains **white (1)** for building and **black (0)** for background.

    python threshold.py --threshold 0.9

This creates:

    output_threshold_0.9/
      ├── sample_updated_1_mask.tif    # uint8 or bool, {0,1}
      └── sample_updated_2_mask.tif

> You can repeat with different thresholds. A new folder named `output_threshold_X.XX/` is created each time.

---

## 🛰️ Visualizing in QGIS

1. Open **QGIS**.
2. Drag the files from `output/` (probabilities) or `output_threshold_X.XX/` (binary) into the **Layers** panel in QGIS, **or simply open them with Paint/Image Viewer for a quick check**.
3. Add corresponding originals from `test/` or updated inputs from `test_updated/` for overlay comparison.

Download QGIS (Windows):

    https://download.osgeo.org/qgis/windows/QGIS-OSGeo4W-3.44.2-1.msi?US

---

## ℹ️ Notes & Tips

- **Band order matters**: inputs in `test_updated/` must be **[R, G, B, DEM, Slope]**.
- **Value ranges**: probabilities are written in **[0,1]**. Binary masks are **{0,1}**.
- **Performance**: large tiles benefit from running on a machine with sufficient RAM/VRAM; consider tiling if needed.

---

## 🧾 Citation

If you find this repository useful, please cite **TriFusionBD** (add your full citation here):

    @article{trifusionbd2025,
      title   = {TriFusionBD: Building Segmentation with RGB, DEM, and Slope Fusion},
      author  = {Author Names},
      journal = {GeoInformatica},
      year    = {2025}
    }

---
