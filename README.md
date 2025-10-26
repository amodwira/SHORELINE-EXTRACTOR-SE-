# Shoreline Extraction Toolbox (Segments)

Automates shoreline extraction from a multispectral composite using **MNDWI + Otsu**, with adaptive smoothing, morphology cleanup, polygon→line conversion (keeps boundary), and projection to a target CRS.

## Requirements
- **ArcGIS Pro 3.x (Windows)**
- **Spatial Analyst** extension
- Python libs:
  - **Bundled**: `arcpy`, `numpy`
  - **Install**: `scikit-image` (for `threshold_otsu`, `gaussian`, `disk/opening/closing/remove_small_objects`)

### Install (recommended)
```bat
# Open "Python Command Prompt" from ArcGIS Pro
conda create -n shoreline-env --clone arcgispro-py3
conda activate shoreline-env
conda install -c conda-forge scikit-image
```
Quick check:
```bat
python -c "import arcpy, numpy; from skimage.filters import threshold_otsu, gaussian; from skimage.morphology import disk, opening, closing, remove_small_objects; print('OK')"
```

## Inputs (Tool Parameters)
- **Composite Raster** (GPRasterLayer)
- **Green Band Name** (e.g., `B3`, `Band_3`, `Layer_3`)
- **SWIR Band Name** (e.g., `B11`, `Band_11`, `Layer_11`)
- **Study Area** *(optional, polygon)*
- **Output Folder**
- **Output File Prefix**
- **Output Coordinate System** (target CRS)

## What the Tool Does (Pipeline)
1. Compute **MNDWI = (Green−SWIR)/(Green+SWIR)**.
2. **Adaptive Gaussian smoothing** (sigma chosen from raster cell size).
3. **Otsu threshold** to binary (fallback: `MNDWI > 0`).
4. Morphology: **opening → small-object removal → closing**.
5. Raster→Polygon, select water class, small-area filter (map-units²).
6. Polygon→**FeatureToLine** (fallback: PolygonToLine + SplitLine).
7. **MultipartToSinglepart** + **SplitLine** → fine segments (keep boundary).
8. Define source projection from raster → **Project** to user CRS.
9. Add outputs to current map (if running inside ArcGIS Pro).

## Outputs (in Output Folder)
- `{prefix}_mndwi.tif`
- `{prefix}_binary.tif`
- `{prefix}_water_polygon.shp`
- `{prefix}_valid_polygons.shp` *(and possibly `{prefix}_filtered_polygons.shp`)*
- `{prefix}_raster_extent.shp` *(when no study area)*
- `{prefix}_mask_boundary.shp` *(reference only; kept)*
- `{prefix}_shoreline_segments.shp` *(source CRS)*
- `{prefix}_shoreline_segments_projected.shp` *(final)*
- `{prefix}_mndwi_debug.tif` *(only if no water pixels were detected)*

## Usage (ArcGIS Pro)
1. Add the `.pyt` to your project (Catalog → **Add**).
2. Open **Shoreline Extractor (Segments, keep boundary)**.
3. Set parameters (see above) and **Run**.

## Tips & Troubleshooting
- **No water found**: Verify band names and AOI; inspect `{prefix}_mndwi_debug.tif`.
- **Band names**: Must match the internal band labels in the composite (e.g., `B3`, `B11`).
- **Performance**: Clip with Study Area; write to a fast local drive.
- **Licensing**: Requires **Spatial Analyst** (tool checks out at runtime).

