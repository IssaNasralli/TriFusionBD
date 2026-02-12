# Training and Data Capping Module  
## TriFusionBD – Principled Class Balancing, Training and Evaluation

This folder contains the Python implementation of the **Principled Data Capping and Class-Balancing Mechanism** and the full training/evaluation pipeline of the TriFusionBD model.

This module implements:

1. Pixel index generation
2. Balanced sampling (building / non-building)
3. Parameter-aware 10×M data capping
4. 70–15–15 split control
5. Model definition (TriFusion)
6. Training
7. Evaluation
8. Full-image inference

---

# Folder Contents

- `create_csv_files.py`
- `Test_TriFusion_Gate_Atrous_Gate.py`
- `TriFusion_Gate_Atrous_Gate.py`

---

# 1️⃣ Step 1 – Pixel Index Generation  
File: `create_csv_files.py`

## Purpose

Generate pixel-level indices for:
- Building class (label = 255)
- Non-building class (label = 0)

These indices define valid patch centers for training.

---

## Workflow

For each raster in:

