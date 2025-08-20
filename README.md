# TriFusionBD – Building Segmentation from GeoTIFF (Test Repository)

This repository provides a **lightweight test harness** for the pretrained **TriFusionBD** model to segment **buildings** on updated test images that combine **RGB + DEM + Slope** layers (5 bands).  
The two provided test images (**1.tif** and **2.tif**) are extracted from the **test partition of the Massachusetts Buildings Dataset** and have been extended with DEM and slope channels for evaluation.  

---

## 📦 Requirements

- Python 3.8+
- TensorFlow
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
    │── predict.py                       # Run inference on all GeoTIFFs in test_updated/
    │── threshold.py                     # Apply threshold to probability maps
    │
    ├── test_updated/                    # UPDATED test GeoTIFFs (5 bands: R,G,B,DEM,Slope)
    │   ├── 1_updated.tif
    │   └── 2_updated.tif
    │
    └── test/                            # ORIGINAL test GeoTIFFs (RGB only, from Massachusetts dataset)
        ├── 1.tif
        └── 2.tif

**Input format**: GeoTIFFs with **5 bands** ordered as **[R, G, B, DEM, Slope]**.

---

## 🚀 Quick Start

### 1) Run Prediction (probability maps)
Run inference on **all `.tif` files inside `test_updated/`** and write per-pixel building probabilities (0–1) to `output/`.

    python predict.py

This creates:

    output/
      ├── 1_updated_pred.tif    # float32, values in [0,1]
      └── 2_updated_pred.tif

**Color meaning (for visualization):**
- 0.0 → **black** (non-building)
- 1.0 → **white** (building)
- (0.0–1.0) → **grayscale** probability

---

### 2) Apply Threshold (binary masks)
Convert probability maps to **binary masks** using a user-defined threshold (e.g., 0.90).  
The result contains **white (1)** for building and **black (0)** for background.

    python threshold.py --threshold 0.9

This creates:

    output_threshold_0.9/
      ├── 1_updated_mask.tif    # uint8 or bool, {0,1}
      └── 2_updated_mask.tif

> You can repeat with different thresholds. A new folder named `output_threshold_X.XX/` is created each time.

---

## 🛰️ Visualizing in QGIS

1. Open **QGIS**.  
2. Drag the files from `output/` (probabilities) or `output_threshold_X.XX/` (binary) into the **Layers** panel in QGIS, **or simply open them with Paint/Image Viewer for a quick check**.  
3. Optionally, overlay the originals from `test/` or the updated inputs from `test_updated/` for comparison.

Download QGIS (Windows):

    https://download.osgeo.org/qgis/windows/QGIS-OSGeo4W-3.44.2-1.msi?US

---

## ℹ️ Notes & Tips

- **Data origin**: `1.tif` and `2.tif` come from the **Massachusetts Buildings Dataset (test partition)**.  
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
