# TriFusionBD

# TriFusionBD: VAE-Driven Context Fusion with Atrous Convolutions for Satellite-Based Building Segmentation

This repository provides the implementation and supporting resources for the research paper:

**"TriFusionBD: VAE-Driven Context Fusion with Atrous Convolutions for Satellite-Based Building Segmentation"**

Our approach, **TriFusion**, leverages a Variational Autoencoder (VAE)-based context fusion mechanism with multi-scale atrous convolutions to enhance building segmentation accuracy from satellite imagery. We also integrate auxiliary features such as Digital Elevation Model (DEM) and slope data to improve segmentation performance in challenging environments.

---

## 📂 Repository Structure

Here's a guide to the contents of this repository:
```bash

TriFusionBD/
│
├── paper/
│ └── TriFusionBD.pdf
│ 📄 The PDF of our published paper, providing detailed methodology, results, and discussions.
│
├── dem_slope_extraction/
│ └── [JavaScript code for GEE]
│ 📁 JavaScript code designed for the Google Earth Engine (GEE) platform to extract DEM and slope data.
│
├── arcgis_processing/
│ └── [Python scripts for ArcGIS]
│ 📁 Python scripts to preprocess spatial data, such as generating terrain features and preparing datasets.
│
├── model_training/
│ └── [Python code for model]
│ 📁 Code to define and train the TriFusionBD deep learning model, including VAE context fusion and atrous convolution layers.
│
├── model_testing/
│ └── [Python code for evaluation]
│ 📁 Code to test the trained model and perform post-processing on the output (e.g., thresholding, morphological operations).
│
└── README.md
📄 This file, describing the repository structure and how to use it.
```



---

## 🚀 TriFusionBD Highlights

- **VAE-Driven Context Fusion**: Our architecture introduces a variational autoencoder to capture and merge multi-modal contextual information, including spatial features from satellite imagery and terrain data.
- **Atrous Convolutions**: Multi-scale atrous convolutions expand the receptive field, allowing the model to capture fine-grained details without increasing computational burden.
- **Auxiliary Data**: We incorporate DEM and slope data to enhance the model's understanding of the physical landscape, crucial for accurate building segmentation.

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.15 (for model training and testing)
- ArcGIS Python API (for spatial data processing)
- Google Earth Engine (for DEM and slope extraction)

### Suggested Workflow

1. **Extract DEM and Slope Data**:  
   Use the JavaScript code in `dem_slope_extraction/` via Google Earth Engine to obtain DEM and slope data for your region of interest.
   
2. **Preprocess Spatial Data**:  
   Run the Python scripts in `arcgis_processing/` to generate the necessary terrain features and prepare datasets.

3. **Train the Model**:  
   Use the code in `model_training/` to define and train the TriFusionBD model.

4. **Test and Post-process**:  
   Apply the trained model to new data using `model_testing/`, and refine the outputs with post-processing techniques.

---

## 📄 Citation

If you use this repository in your work, please cite:

@inproceedings{TriFusionBD2024,
title={TriFusionBD: VAE-Driven Context Fusion with Atrous Convolutions for Satellite-Based Building Segmentation},
author={Issa Nasralli},
booktitle={Journal ....},
year={2025}
}


---

## 📬 Contact

For questions or collaborations, feel free to reach out at [aissanasralli@gmail.com].

---

Thank you for your interest in **TriFusionBD**! We hope this repository helps advance research and applications in remote sensing and building segmentation.


