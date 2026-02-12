# Google Earth Engine – DEM and Slope Preprocessing

This folder contains the Google Earth Engine (GEE) script used in the TriFusionBD study for generating Digital Elevation Model (DEM) and slope layers for Massachusetts (USA).

File included:
- `gee_dem_slop.js` – Script for DEM extraction, slope computation, and export.

---

## 1. Data Sources

### Administrative Boundary
- **Dataset:** FAO GAUL 2015 Level 1
- **Asset ID:** `FAO/GAUL/2015/level1`
- Used to extract the Massachusetts state boundary.

### Digital Elevation Model (DEM)
- **Dataset:** SRTM Global 1 Arc-Second
- **Asset ID:** `USGS/SRTMGL1_003`
- Native spatial resolution: ~30 meters
- Elevation unit: meters above sea level

Reference:
https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003

---

## 2. Processing Workflow

The script performs the following steps:

1. Load GAUL administrative boundaries.
2. Filter for Massachusetts (ADM1_NAME = "Massachusetts").
3. Load the SRTM DEM dataset.
4. Clip DEM to Massachusetts boundary.
5. Compute slope using `ee.Terrain.slope()`:
   - Slope is expressed in **degrees**
   - Range: 0° (flat) to 90° (vertical)
   - Computed using 4-connected neighbors
6. Export DEM and slope layers to Google Drive as GeoTIFF.

---

## 3. Export Parameters

### DEM Export
- Scale: 30 meters (native SRTM resolution)
- Projection: EPSG:26986 (Massachusetts State Plane)
- Format: GeoTIFF
- Max pixels: 1e13

### Slope Export
- Computed from DEM
- Export scale: 1 meter (upsampled to match satellite imagery grid)
- Format: GeoTIFF
- Max pixels: 1e13

Note:
SRTM native resolution is 30 m. When integrating with 1 m satellite imagery in TriFusionBD, bilinear interpolation was used during resampling to align the DEM/slope grid with the imagery resolution. The DEM is treated as low-frequency topographic context rather than fine-scale elevation detail.

---

## 4. How to Reproduce

1. Open Google Earth Engine Code Editor:
   https://code.earthengine.google.com/

2. Create a new script.
3. Copy the contents of `gee_dem_slop.js`.
4. Click "Run".
5. Start the export tasks from the Tasks tab.

Exports will appear in your Google Drive.

---

## 5. Notes for Reproducibility

- The exact dataset IDs are provided above.
- No additional preprocessing was applied in GEE beyond clipping and slope computation.
- Final resampling and alignment to the 1 m satellite imagery grid were performed locally during the TriFusionBD training pipeline.

---

## 6. Relation to the Paper

The generated DEM and slope layers are used as auxiliary topographic inputs in the TriFusionBD probabilistic–deterministic fusion network to reduce terrain-induced false positives and improve building detection in complex topography.

