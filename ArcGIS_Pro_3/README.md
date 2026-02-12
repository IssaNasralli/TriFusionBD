# ArcGIS Pro 3 – Preprocessing Pipeline

This folder documents the complete ArcGIS Pro 3 preprocessing workflow used in the TriFusionBD study.  
All steps are implemented using ArcPy (Python API for ArcGIS Pro 3).

The objective of this pipeline is to:

1. Mosaic satellite tiles (train/val/test) into a single raster.
2. Generate a shapefile representing the full spatial extent.
3. Use that shapefile in Google Earth Engine (GEE) to extract DEM and slope data.
4. Reproject and resample DEM and slope layers.
5. Add DEM and slope as additional bands to each satellite image.
6. Produce final augmented datasets for training, validation, and testing.

---

# Folder Contents

- `mosaic.py`
- `shp_from_mosaic_all.py`
- `adding_dem_slope.py`

---

# Software Requirements

- ArcGIS Pro 3.x
- Spatial Analyst Extension (required)
- Python environment provided by ArcGIS Pro
- arcpy library

To enable Spatial Analyst:

```python
arcpy.CheckOutExtension("Spatial")
