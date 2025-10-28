# -*- coding: utf-8 -*-
"""
Shoreline Extraction Toolbox (SE+ v2.5) — ArcGIS Pro .pyt

Fixes & Features
- FIX: getParameterInfo uses correct arcpy.Parameter(...) keyword args (no DataType error)
- Uses Describe(...).catalogPath/extent for env locks (snapRaster/cellSize/extent)
- Avoids passing parameter objects to arcpy.sa.Raster(); uses comp_path consistently
- ExtractByMask uses arcpy.Raster(comp_path) for stability
- Robust metadata JSON (reads projected layer SR name or parameter text)
- Overwrite enabled; dissolved-water boundary → shoreline; optional border removal
- SEG_ID on raw & projected outputs; autoscale reflectance; robust Otsu (0.5–99.5% clip)
- Writes <prefix>_metadata.json

Requires: ArcGIS Pro, Spatial Analyst; scikit-image
Developed by Curtis Amo Dwira, MSc in Civil Engineering, Louisiana State University.
With support from Dr. Ahmmed Abdullah, PhD, Assistant Professor, Louisiana State University.
Prof. Bernard Kumi-Boateng, PhD, Professor, University of Mines and Technology, Ghana.
"""

import arcpy
import numpy as np
import os
import json
import datetime
import warnings

from arcpy.sa import *
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, opening, closing, remove_small_objects

warnings.filterwarnings("ignore")


# -----------------------
# Helpers
# -----------------------

def _resolve_band(raster_path, user_key):
    """Resolve a band by exact child name, fuzzy child match, or numeric index."""
    # 1) Exact child name/path
    try:
        cand = arcpy.Raster(f"{raster_path}/{user_key}")
        _ = cand.meanCellWidth
        return cand
    except Exception:
        pass
    # 2) Fuzzy child match
    try:
        desc = arcpy.Describe(raster_path)
        children = getattr(desc, "children", []) or []
        for ch in children:
            if user_key.lower() in ch.name.lower():
                return arcpy.Raster(ch.catalogPath)
    except Exception:
        pass
    # 3) Numeric index
    try:
        idx = int(user_key)
        cand = arcpy.Raster(raster_path + f"/Band_{idx}")
        _ = cand.meanCellWidth
        return cand
    except Exception:
        raise RuntimeError(
            f"Could not locate band '{user_key}' in composite: {raster_path}. "
            f"Provide an exact sublayer name (e.g., SR_B2) or a numeric index."
        )


def _autoscale(arr):
    """If array looks like 0..10000 scaled reflectance, normalize ~0..1."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    p99 = np.nanpercentile(finite, 99)
    return arr / 10000.0 if p99 > 10 else arr


def _define_projection_safely(path, spatial_ref, messages):
    try:
        arcpy.DefineProjection_management(path, spatial_ref)
    except Exception as e:
        messages.addWarningMessage(f"DefineProjection failed for {os.path.basename(path)}: {e}")


def _clip_percentiles(a, lo=0.5, hi=99.5):
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return a, None, None
    low, high = np.nanpercentile(finite, [lo, hi])
    b = np.clip(a, low, high)
    return b, float(low), float(high)


class Toolbox(object):
    def __init__(self):
        self.label = "Shoreline Extraction Toolbox (Segments)"
        self.alias = "shoreline_toolbox_segments"
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        self.label = "Shoreline Extractor (SE+ v2.5)"
        self.description = (
            "Semi-automated shoreline extraction using MNDWI + Otsu with adaptive smoothing, morphology, "
            "dissolved-water boundary → shoreline segments, optional border-segment removal, and projection."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        params = [
            arcpy.Parameter(
                displayName="Composite Raster",
                name="composite_raster",
                datatype="GPRasterLayer",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Green Band (name or index, e.g., SR_B2 or 2)",
                name="green_band_name",
                datatype="GPString",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="SWIR Band (name or index, e.g., SR_B5 or 5)",
                name="swir_band_name",
                datatype="GPString",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Study Area (optional)",
                name="study_area",
                datatype="GPFeatureLayer",
                parameterType="Optional",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Output Folder",
                name="output_folder",
                datatype="DEFolder",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Output File Prefix",
                name="output_prefix",
                datatype="GPString",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Output Coordinate System",
                name="output_crs",
                datatype="GPCoordinateSystem",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Minimum polygon area (map units²)",
                name="min_poly_area",
                datatype="GPDouble",
                parameterType="Optional",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Remove study-area/extent border segments",
                name="drop_border",
                datatype="GPBoolean",
                parameterType="Optional",
                direction="Input",
            ),
        ]
        # Default: drop border segments
        params[8].value = True
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters): return
    def updateMessages(self, parameters): return

    def execute(self, parameters, messages):
        arcpy.CheckOutExtension("Spatial")
        arcpy.env.overwriteOutput = True

        try:
            # Inputs
            composite          = parameters[0].value
            green_band_name    = parameters[1].valueAsText
            swir_band_name     = parameters[2].valueAsText
            study_area         = parameters[3].value
            output_folder      = parameters[4].valueAsText
            prefix             = parameters[5].valueAsText
            output_crs_param   = parameters[6]
            user_min_poly_area = parameters[7].value
            drop_border        = bool(parameters[8].value) if parameters[8].altered else True

            # Describe input & lock envs using dataset path/extent (NOT parameter object)
            try:
                comp_desc   = arcpy.Describe(composite)
                comp_path   = comp_desc.catalogPath
                comp_extent = comp_desc.extent
                comp_sr     = comp_desc.spatialReference
                messages.addMessage(f"Raster: {comp_path} | SR: {comp_sr.name if comp_sr else 'Unknown'}")
            except Exception as e:
                messages.addErrorMessage("Could not describe composite raster: " + str(e))
                raise

            arcpy.env.snapRaster = comp_path
            arcpy.env.cellSize   = comp_path
            arcpy.env.extent     = comp_extent
            arcpy.env.mask       = None

            raster_path   = comp_path
            raster_extent = comp_extent
            raster_sr     = comp_sr

            # Paths
            binary_raster_path   = os.path.join(output_folder, f"{prefix}_binary.tif")
            mndwi_raster_path    = os.path.join(output_folder, f"{prefix}_mndwi.tif")
            mndwi_debug_path     = os.path.join(output_folder, f"{prefix}_mndwi_debug.tif")
            polygon_path         = os.path.join(output_folder, f"{prefix}_water_polygon.shp")
            valid_polygons       = os.path.join(output_folder, f"{prefix}_valid_polygons.shp")
            filtered_polygons    = os.path.join(output_folder, f"{prefix}_filtered_polygons.shp")
            dissolved_water      = "in_memory/water_diss"
            shoreline_segments   = os.path.join(output_folder, f"{prefix}_shoreline_segments.shp")
            shoreline_proj       = os.path.join(output_folder, f"{prefix}_shoreline_segments_projected.shp")
            projected_study_area = os.path.join(output_folder, f"{prefix}_projected_study_area.shp")
            mask_boundary        = os.path.join(output_folder, f"{prefix}_mask_boundary.shp")
            extent_fc            = os.path.join(output_folder, f"{prefix}_raster_extent.shp")
            metadata_path        = os.path.join(output_folder, f"{prefix}_metadata.json")

            # Optional clipping
            if study_area:
                try:
                    arcpy.Project_management(study_area, projected_study_area, raster_sr)
                    arcpy.env.mask   = projected_study_area
                    arcpy.env.extent = arcpy.Describe(projected_study_area).extent
                    processed_raster = ExtractByMask(arcpy.Raster(comp_path), projected_study_area)
                    messages.addMessage("Study area reprojected; env.mask set; raster clipped.")
                except Exception as e:
                    messages.addWarningMessage("Study area project/clip failed, using full raster: " + str(e))
                    arcpy.env.mask   = None
                    arcpy.env.extent = raster_extent
                    processed_raster = arcpy.Raster(comp_path)
            else:
                processed_raster = arcpy.Raster(comp_path)
                messages.addMessage("No study area provided. Using full raster.")

            # Cell size & auto-params
            try:
                cs = float(processed_raster.meanCellWidth)
                if cs <= 0: cs = abs(cs) if cs != 0 else 1.0
                messages.addMessage(f"Detected raster cell size (map units): {cs}")
            except Exception as e:
                cs = 1.0
                messages.addWarningMessage("Could not determine cell size; using fallback cs=1.0. " + str(e))

            if cs <= 0.5: gaussian_sigma = 2.0
            elif cs <= 2: gaussian_sigma = 1.5
            elif cs <=10: gaussian_sigma = 1.0
            elif cs <=30: gaussian_sigma = 0.7
            else:         gaussian_sigma = 0.5
            morph_radius = max(1, int(round(gaussian_sigma)))
            default_min_pixels    = 25
            min_polygon_area_auto = float(default_min_pixels) * (cs * cs)
            min_polygon_area      = float(user_min_poly_area) if user_min_poly_area not in (None, "") else min_polygon_area_auto
            messages.addMessage(f"Auto params: sigma={gaussian_sigma}, morph_radius={morph_radius}, min_polygon_area={min_polygon_area:.2f}")

            # Read bands
            green_ras = _resolve_band(raster_path, green_band_name)
            swir_ras  = _resolve_band(raster_path, swir_band_name)
            messages.addMessage(f"Opening bands: {green_band_name}, {swir_band_name}")

            green = _autoscale(arcpy.RasterToNumPyArray(green_ras).astype(float))
            swir  = _autoscale(arcpy.RasterToNumPyArray(swir_ras).astype(float))

            # MNDWI + smoothing
            eps = 1e-10
            mndwi = (green - swir) / (green + swir + eps)
            try:
                if gaussian_sigma and gaussian_sigma > 0:
                    nan_mask = ~np.isfinite(mndwi)
                    work = mndwi.copy(); work[nan_mask] = 0.0
                    smoothed = gaussian(work, sigma=gaussian_sigma, preserve_range=True)
                    smoothed[nan_mask] = np.nan
                    mndwi = smoothed
                    messages.addMessage(f"Applied Gaussian smoothing (sigma={gaussian_sigma}).")
            except Exception as e:
                messages.addWarningMessage("Gaussian smoothing failed; proceeding unsmoothed: " + str(e))

            # Otsu threshold (robust)
            finite_mask = np.isfinite(mndwi)
            finite_vals = mndwi[finite_mask]
            if finite_vals.size == 0:
                try:
                    ll = processed_raster.extent.lowerLeft
                    cs_save = processed_raster.meanCellWidth
                    arcpy.NumPyArrayToRaster(mndwi, ll, cs_save, cs_save).save(mndwi_debug_path)
                except Exception:
                    pass
                messages.addErrorMessage("MNDWI array has no valid data to threshold.")
                raise RuntimeError("Empty MNDWI data for thresholding.")

            try:
                clipped_vals, lo_p, hi_p = _clip_percentiles(finite_vals, 0.5, 99.5)
                thresh = float(threshold_otsu(clipped_vals))
                messages.addMessage(f"Otsu threshold (robust): {thresh:.6f} (clip {lo_p:.4f}–{hi_p:.4f})")
            except Exception:
                thresh = None
                messages.addWarningMessage("Otsu thresholding failed; fallback to MNDWI > 0.0")

            binary_bool = (mndwi > thresh) if thresh is not None else (mndwi > 0.0)
            water_count = int(np.sum(np.logical_and(binary_bool, finite_mask)))
            messages.addMessage(f"Water pixel count: {water_count}")
            if water_count == 0:
                try:
                    ll = processed_raster.extent.lowerLeft
                    cs_save = processed_raster.meanCellWidth
                    arcpy.NumPyArrayToRaster(mndwi, ll, cs_save, cs_save).save(mndwi_debug_path)
                    messages.addMessage(f"Saved MNDWI debug raster: {mndwi_debug_path}")
                except Exception:
                    messages.addWarningMessage("Failed to save MNDWI debug raster.")
                raise RuntimeError("No water pixels detected; check band names, clipping, or scene quality.")

            # Morphology
            try:
                if morph_radius and morph_radius > 0:
                    selem = disk(morph_radius)
                    binary_bool = opening(binary_bool, selem)
                    small_pixels = max(8, morph_radius * morph_radius)
                    binary_bool = remove_small_objects(binary_bool.astype(bool), min_size=small_pixels)
                    binary_bool = closing(binary_bool, selem)
                    messages.addMessage(f"Morphology: open/remove_small/close (r={morph_radius}px).")
            except Exception as e:
                messages.addWarningMessage("Morphological cleanup failed/partial: " + str(e))

            # Save rasters (define SR)
            ll = processed_raster.extent.lowerLeft
            cs_save = processed_raster.meanCellWidth

            try:
                mndwi_r = arcpy.NumPyArrayToRaster(mndwi, ll, cs_save, cs_save)
                if study_area:
                    mndwi_r = ExtractByMask(mndwi_r, projected_study_area)
                mndwi_r.save(mndwi_raster_path)
                _define_projection_safely(mndwi_raster_path, raster_sr, messages)
                messages.addMessage(f"MNDWI saved: {mndwi_raster_path}")
            except Exception as e:
                messages.addWarningMessage("Failed to save MNDWI: " + str(e))

            try:
                binary_uint8 = np.full_like(mndwi, 255, dtype=np.uint8)
                binary_uint8[finite_mask] = binary_bool[finite_mask].astype(np.uint8)
                binary_r = arcpy.NumPyArrayToRaster(binary_uint8, ll, cs_save, cs_save, value_to_nodata=255)
                if study_area:
                    binary_r = ExtractByMask(binary_r, projected_study_area)
                binary_r.save(binary_raster_path)
                _define_projection_safely(binary_raster_path, raster_sr, messages)
                messages.addMessage(f"Binary saved: {binary_raster_path}")
            except Exception as e:
                messages.addErrorMessage("Failed to save binary raster: " + str(e))
                raise

            # Raster → Polygon (water)
            try:
                arcpy.RasterToPolygon_conversion(binary_raster_path, polygon_path, "SIMPLIFY", "Value")
                messages.addMessage("RasterToPolygon completed.")
            except Exception as e:
                messages.addErrorMessage("RasterToPolygon failed: " + str(e))
                raise
            try:
                arcpy.RepairGeometry_management(polygon_path)
            except Exception:
                pass

            # Detect value field
            value_field = None
            try:
                fields = list(arcpy.ListFields(polygon_path))
                for cand in ("GRIDCODE", "VALUE", "GRID_CODE", "DN"):
                    for f in fields:
                        if f.name.upper() == cand:
                            value_field = f.name; break
                    if value_field: break
                if value_field is None:
                    for f in fields:
                        if f.type in ("SmallInteger", "Integer", "Double", "Single"):
                            value_field = f.name; break
                messages.addMessage(f"Polygon value field detected: {value_field}")
            except Exception:
                value_field = None

            # Keep water polygons (value == 1)
            try:
                if not value_field:
                    arcpy.AddWarning("Could not detect value field; copying all polygons.")
                    arcpy.CopyFeatures_management(polygon_path, valid_polygons)
                else:
                    arcpy.MakeFeatureLayer_management(polygon_path, "poly_lyr")
                    arcpy.SelectLayerByAttribute_management("poly_lyr", "NEW_SELECTION", f'"{value_field}" = 1')
                    arcpy.CopyFeatures_management("poly_lyr", valid_polygons)
                    arcpy.Delete_management("poly_lyr")
                messages.addMessage("Water polygons isolated.")
            except Exception as e:
                messages.addErrorMessage("Filtering water polygons failed: " + str(e))
                raise

            # Small polygon filter by SHAPE@AREA
            try:
                if min_polygon_area and min_polygon_area > 0:
                    desc = arcpy.Describe(valid_polygons)
                    oid_field = desc.OIDFieldName
                    keep_oids = []
                    with arcpy.da.SearchCursor(valid_polygons, [oid_field, "SHAPE@AREA"]) as scur:
                        for oid, area in scur:
                            if area is not None and area >= min_polygon_area:
                                keep_oids.append(oid)
                    if keep_oids:
                        where = f'"{oid_field}" IN ({",".join(map(str, keep_oids))})'
                        arcpy.MakeFeatureLayer_management(valid_polygons, "valid_lyr")
                        arcpy.SelectLayerByAttribute_management("valid_lyr", "NEW_SELECTION", where)
                        arcpy.CopyFeatures_management("valid_lyr", filtered_polygons)
                        arcpy.Delete_management("valid_lyr")
                        valid_polygons = filtered_polygons
                        messages.addMessage(f"Removed small polygons < {min_polygon_area:.2f} (map units²).")
                    else:
                        messages.addWarningMessage("No polygons met min area; kept originals.")
                else:
                    messages.addMessage("Min area filtering skipped (<= 0).")
            except Exception as e:
                messages.addWarningMessage("Small-polygon filtering failed: " + str(e))
                messages.addWarningMessage("GP messages:\n" + arcpy.GetMessages(2))

            # Mask boundary (for optional border filtering)
            try:
                if study_area and arcpy.Exists(projected_study_area):
                    if arcpy.Exists(mask_boundary): arcpy.Delete_management(mask_boundary)
                    arcpy.PolygonToLine_management(projected_study_area, mask_boundary)
                    messages.addMessage("Mask boundary from study area created.")
                else:
                    ext = processed_raster.extent
                    arr = arcpy.Array([
                        arcpy.Point(ext.XMin, ext.YMin),
                        arcpy.Point(ext.XMin, ext.YMax),
                        arcpy.Point(ext.XMax, ext.YMax),
                        arcpy.Point(ext.XMax, ext.YMin),
                        arcpy.Point(ext.XMin, ext.YMin),
                    ])
                    extent_poly = arcpy.Polygon(arr, raster_sr)
                    if arcpy.Exists(extent_fc): arcpy.Delete_management(extent_fc)
                    arcpy.CopyFeatures_management([extent_poly], extent_fc)
                    if arcpy.Exists(mask_boundary): arcpy.Delete_management(mask_boundary)
                    arcpy.PolygonToLine_management(extent_fc, mask_boundary)
                    messages.addMessage("Mask boundary from raster extent created.")
            except Exception as e:
                messages.addWarningMessage("Failed to create mask boundary: " + str(e))

            # Dissolve water → boundary-only shoreline segments
            try:
                if arcpy.Exists(dissolved_water): arcpy.Delete_management(dissolved_water)
                arcpy.Dissolve_management(valid_polygons, dissolved_water)
                shoreline_lines_tmp = "in_memory/shoreline_lines_tmp"
                if arcpy.Exists(shoreline_lines_tmp): arcpy.Delete_management(shoreline_lines_tmp)
                arcpy.PolygonToLine_management(dissolved_water, shoreline_lines_tmp)

                single_parts = "in_memory/shoreline_singleparts"
                split_segments = "in_memory/shoreline_split_segments"
                for tmp in (single_parts, split_segments):
                    if arcpy.Exists(tmp): arcpy.Delete_management(tmp)

                arcpy.MultipartToSinglepart_management(shoreline_lines_tmp, single_parts)
                arcpy.SplitLine_management(single_parts, split_segments)
                try: arcpy.RepairGeometry_management(split_segments)
                except Exception: pass

                if drop_border and arcpy.Exists(mask_boundary):
                    arcpy.MakeFeatureLayer_management(split_segments, "seg_lyr")
                    arcpy.SelectLayerByLocation_management("seg_lyr", "SHARE_A_LINE_SEGMENT_WITH", mask_boundary, selection_type="NEW_SELECTION")
                    arcpy.SelectLayerByAttribute_management("seg_lyr", "SWITCH_SELECTION")
                    if arcpy.Exists(shoreline_segments): arcpy.Delete_management(shoreline_segments)
                    arcpy.CopyFeatures_management("seg_lyr", shoreline_segments)
                    arcpy.Delete_management("seg_lyr")
                    messages.addMessage("Removed study-area/extent border segments.")
                else:
                    if arcpy.Exists(shoreline_segments): arcpy.Delete_management(shoreline_segments)
                    arcpy.CopyFeatures_management(split_segments, shoreline_segments)

                for tmp in (shoreline_lines_tmp, single_parts, split_segments):
                    try:
                        if arcpy.Exists(tmp): arcpy.Delete_management(tmp)
                    except Exception:
                        pass

                seg_count = int(arcpy.GetCount_management(shoreline_segments).getOutput(0)) if arcpy.Exists(shoreline_segments) else 0
                messages.addMessage(f"Final shoreline segments (boundary-only): {seg_count}")
                if seg_count == 0:
                    messages.addWarningMessage("No shoreline segments saved. Inspect intermediate outputs.")
            except Exception as e_seg:
                messages.addErrorMessage("Boundary extraction failed: " + str(e_seg))
                messages.addErrorMessage("GP messages:\n" + arcpy.GetMessages(2))
                raise

            # Define + Project to Output CRS
            proj_sr_name = None
            try:
                if arcpy.Exists(shoreline_segments):
                    arcpy.DefineProjection_management(shoreline_segments, raster_sr)
                    arcpy.Project_management(shoreline_segments, shoreline_proj, output_crs_param.value)
                    messages.addMessage(f"Projected shoreline segments: {shoreline_proj}")
                    if shoreline_proj and arcpy.Exists(shoreline_proj):
                        proj_sr_name = arcpy.Describe(shoreline_proj).spatialReference.name
                else:
                    messages.addWarningMessage("Shoreline segments not found; skipping projection.")
            except Exception as e:
                messages.addWarningMessage("Projecting shoreline segments failed: " + str(e))
                messages.addWarningMessage("GP messages:\n" + arcpy.GetMessages(2))
                shoreline_proj = shoreline_segments if arcpy.Exists(shoreline_segments) else None
                try:
                    if shoreline_proj and arcpy.Exists(shoreline_proj):
                        proj_sr_name = arcpy.Describe(shoreline_proj).spatialReference.name
                except Exception:
                    proj_sr_name = None

            # SEG_ID on both outputs
            try:
                if arcpy.Exists(shoreline_segments):
                    names = [f.name.upper() for f in arcpy.ListFields(shoreline_segments)]
                    if "SEG_ID" not in names:
                        arcpy.AddField_management(shoreline_segments, "SEG_ID", "LONG")
                        with arcpy.da.UpdateCursor(shoreline_segments, ["SEG_ID"]) as u:
                            i = 1
                            for r in u:
                                r[0] = i; u.updateRow(r); i += 1
                if shoreline_proj and arcpy.Exists(shoreline_proj):
                    names = [f.name.upper() for f in arcpy.ListFields(shoreline_proj)]
                    if "SEG_ID" not in names:
                        arcpy.AddField_management(shoreline_proj, "SEG_ID", "LONG")
                        with arcpy.da.UpdateCursor(shoreline_proj, ["SEG_ID"]) as u:
                            i = 1
                            for r in u:
                                r[0] = i; u.updateRow(r); i += 1
                    messages.addMessage("SEG_ID added to shoreline segments (raw & projected).")
            except Exception as e:
                messages.addWarningMessage("Failed to add SEG_ID to outputs: " + str(e))

            # Metadata JSON (robust CRS handling)
            try:
                out_crs_label = proj_sr_name
                if not out_crs_label:
                    try:
                        out_crs_label = output_crs_param.valueAsText
                    except Exception:
                        out_crs_label = None

                meta = {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "composite": str(raster_path),
                    "green_band": green_band_name,
                    "swir_band": swir_band_name,
                    "cellsize_map_units": cs,
                    "gaussian_sigma": gaussian_sigma,
                    "morph_radius_px": morph_radius,
                    "min_polygon_area_map_units2": float(min_polygon_area),
                    "otsu_threshold": None if 'thresh' not in locals() or thresh is None else float(thresh),
                    "water_pixel_count": water_count,
                    "output_crs": out_crs_label,
                    "drop_border_segments": drop_border,
                    "paths": {
                        "mndwi": mndwi_raster_path,
                        "binary": binary_raster_path,
                        "water_polygons": valid_polygons,
                        "shoreline_segments": shoreline_segments,
                        "shoreline_segments_projected": shoreline_proj
                    }
                }
                with open(metadata_path, "w") as f:
                    json.dump(meta, f, indent=2)
                messages.addMessage(f"Metadata written: {metadata_path}")
            except Exception as e:
                messages.addWarningMessage("Failed to write metadata JSON: " + str(e))

            # Add outputs to current map
            try:
                aprx = arcpy.mp.ArcGISProject("CURRENT")
                m = aprx.activeMap
                if arcpy.Exists(mndwi_raster_path): m.addDataFromPath(mndwi_raster_path)
                if shoreline_proj and arcpy.Exists(shoreline_proj):
                    m.addDataFromPath(shoreline_proj)
                elif arcpy.Exists(shoreline_segments):
                    m.addDataFromPath(shoreline_segments)
                messages.addMessage("Added outputs to current map (if available).")
            except Exception as e:
                messages.addWarningMessage("Could not add layers to map: " + str(e))

        finally:
            arcpy.CheckInExtension("Spatial")

    def postExecute(self, parameters):
        "JUST BELIEVE"
        return
