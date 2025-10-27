import arcpy
import numpy as np
import os
from arcpy.sa import *
# scikit-image is used for Otsu, gaussian and morphology. If not available, install it in the ArcGIS Pro Python environment.
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, opening, closing, remove_small_objects
import warnings

warnings.filterwarnings("ignore")


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "Shoreline Extraction Toolbox (Segments)"
        self.alias = "shoreline_toolbox_segments"
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Tool definition."""
        self.label = "Shoreline Extractor (SE+)"
        self.description = '''Automated shoreline extraction from a multispectral composite using MNDWI + Otsu with optional study-area clipping, adaptive Gaussian smoothing, 	morphological cleanup, polygon → fine line-segments conversion (keeps boundary), and projection to a user-selected CRS.
        This tool is designed for use in ArcGIS Pro and requires the Spatial Analyst extension.
        The output will be a shapefile of the shoreline and MNDWI in tif format in the specified coordinate system.
        Developed by Curtis Amo Dwira, MSc in Civil Engineering, Louisiana State University.
        With support from Dr. Ahmmed Abdullah, PhD, Assistant Professor, Louisiana State University.
        Prof. Bernard Kumi-Boateng, PhD, Professor, University of Mines and Technology, Ghana.

        '''
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Tool parameters (fully automated smoothing parameters are hidden)."""
        params = [
            arcpy.Parameter(
                displayName="Composite Raster",
                name="composite_raster",
                datatype="GPRasterLayer",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Green Band Name (e.g., B3, Band_3, Layer_3)",
                name="green_band_name",
                datatype="GPString",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="SWIR Band Name (e.g., B11, Band_11, Layer_11)",
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
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        arcpy.CheckOutExtension("Spatial")

        try:
            # Inputs
            composite = parameters[0].value
            green_band_name = parameters[1].valueAsText
            swir_band_name = parameters[2].valueAsText
            study_area = parameters[3].value  # optional
            output_folder = parameters[4].valueAsText
            prefix = parameters[5].valueAsText
            output_crs = parameters[6].value

            # Output / intermediate paths
            binary_raster_path = os.path.join(output_folder, f"{prefix}_binary.tif")
            mndwi_raster_path = os.path.join(output_folder, f"{prefix}_mndwi.tif")
            polygon_path = os.path.join(output_folder, f"{prefix}_water_polygon.shp")
            valid_polygons = os.path.join(output_folder, f"{prefix}_valid_polygons.shp")
            filtered_polygons = os.path.join(output_folder, f"{prefix}_filtered_polygons.shp")
            shoreline_segments_path = os.path.join(output_folder, f"{prefix}_shoreline_segments.shp")  # final segments (unprojected)
            shoreline_segments_projected = os.path.join(output_folder, f"{prefix}_shoreline_segments_projected.shp")
            projected_study_area = os.path.join(output_folder, f"{prefix}_projected_study_area.shp")
            mask_boundary = os.path.join(output_folder, f"{prefix}_mask_boundary.shp")  # reference (kept)
            extent_fc = os.path.join(output_folder, f"{prefix}_raster_extent.shp")
            mndwi_debug_path = os.path.join(output_folder, f"{prefix}_mndwi_debug.tif")

            # Describe raster and set processed_raster (clip if study area provided)
            try:
                raster_desc = arcpy.Describe(composite)
                raster_sr = raster_desc.spatialReference
                raster_path = raster_desc.catalogPath
                messages.addMessage(f"Raster: {raster_path}  SR: {raster_sr.name if raster_sr else 'Unknown'}")
            except Exception as e:
                messages.addErrorMessage("Could not describe composite raster: " + str(e))
                arcpy.CheckInExtension("Spatial")
                raise

            if study_area:
                try:
                    arcpy.Project_management(study_area, projected_study_area, raster_sr)
                    processed_raster = ExtractByMask(composite, projected_study_area)
                    messages.addMessage("Study area reprojected and raster clipped.")
                except Exception as e:
                    messages.addWarningMessage("Study area project/clip failed, using full raster: " + str(e))
                    processed_raster = arcpy.Raster(composite)
            else:
                processed_raster = arcpy.Raster(composite)
                messages.addMessage("No study area provided. Using full raster.")

            # Determine raster cell size (map units) to auto-select parameters
            try:
                cs = float(processed_raster.meanCellWidth)
                if cs <= 0:
                    cs = abs(cs) if cs != 0 else 1.0
                messages.addMessage(f"Detected raster cell size (map units): {cs}")
            except Exception as e:
                cs = 1.0
                messages.addWarningMessage("Could not determine cell size; using fallback cs=1.0. " + str(e))

            # Automatic parameter heuristics (not exposed)
            if cs <= 0.5:
                gaussian_sigma = 2.0
            elif cs <= 2:
                gaussian_sigma = 1.5
            elif cs <= 10:
                gaussian_sigma = 1.0
            elif cs <= 30:
                gaussian_sigma = 0.7
            else:
                gaussian_sigma = 0.5
            morph_radius = max(1, int(round(gaussian_sigma)))
            min_pixels = 25
            min_polygon_area = float(min_pixels) * (cs * cs)
            smooth_tolerance = max(1.0, 2.5 * cs)
            messages.addMessage(f"Auto params: gaussian_sigma={gaussian_sigma}, morph_radius={morph_radius}, min_polygon_area={min_polygon_area:.2f}, smooth_tolerance={smooth_tolerance}")

            # Read specified bands from composite (stack)
            try:
                green_raster = arcpy.Raster(f"{raster_path}/{green_band_name}")
                swir_raster = arcpy.Raster(f"{raster_path}/{swir_band_name}")
                messages.addMessage(f"Opening bands: {green_band_name}, {swir_band_name}")
            except Exception as e:
                messages.addErrorMessage("Could not open specified bands from composite: " + str(e))
                arcpy.CheckInExtension("Spatial")
                raise

            # Convert bands to numpy arrays
            green_array = arcpy.RasterToNumPyArray(green_raster).astype(float)
            swir_array = arcpy.RasterToNumPyArray(swir_raster).astype(float)

            # Compute MNDWI
            eps = 1e-10
            mndwi_array = (green_array - swir_array) / (green_array + swir_array + eps)

            # Apply Gaussian smoothing (automated, non-fatal)
            try:
                if gaussian_sigma and gaussian_sigma > 0:
                    nan_mask = ~np.isfinite(mndwi_array)
                    working = mndwi_array.copy()
                    working[nan_mask] = 0.0
                    smoothed = gaussian(working, sigma=gaussian_sigma, preserve_range=True)
                    smoothed[nan_mask] = np.nan
                    mndwi_array = smoothed
                    messages.addMessage(f"Applied Gaussian smoothing to MNDWI with sigma={gaussian_sigma}.")
            except Exception as e:
                messages.addWarningMessage("Gaussian smoothing failed; proceeding with unsmoothed MNDWI: " + str(e))

            # Threshold with Otsu (and fallback)
            finite_mask = np.isfinite(mndwi_array)
            finite_vals = mndwi_array[finite_mask]
            if finite_vals.size == 0:
                messages.addErrorMessage("MNDWI array has no valid data to threshold.")
                arcpy.CheckInExtension("Spatial")
                raise RuntimeError("Empty MNDWI data for thresholding.")

            thresh = None
            try:
                thresh = float(threshold_otsu(finite_vals))
                messages.addMessage(f"Otsu threshold computed: {thresh:.6f}")
            except Exception:
                thresh = None
                messages.addWarningMessage("Otsu thresholding failed; will try MNDWI>0 fallback.")

            if thresh is not None:
                binary_bool = (mndwi_array > thresh)
            else:
                binary_bool = (mndwi_array > 0.0)

            water_count = int(np.sum(np.logical_and(binary_bool, finite_mask)))
            messages.addMessage(f"Water pixel count: {water_count}")

            if water_count == 0:
                try:
                    ll = processed_raster.extent.lowerLeft
                    cs_save = processed_raster.meanCellWidth
                    arcpy.NumPyArrayToRaster(mndwi_array, ll, cs_save, cs_save).save(mndwi_debug_path)
                    messages.addMessage(f"Saved MNDWI debug raster to: {mndwi_debug_path}")
                except Exception:
                    messages.addWarningMessage("Failed to save MNDWI debug raster for inspection.")
                messages.addErrorMessage("No water pixels detected. Check band names, clipping, or the debug MNDWI raster.")
                arcpy.CheckInExtension("Spatial")
                raise RuntimeError("No water pixels detected; aborting.")

            # Morphological cleaning
            try:
                if morph_radius and morph_radius > 0:
                    selem = disk(morph_radius)
                    binary_bool = opening(binary_bool, selem)
                    small_pixels = max(8, morph_radius * morph_radius)
                    binary_bool = remove_small_objects(binary_bool.astype(bool), min_size=small_pixels)
                    binary_bool = closing(binary_bool, selem)
                    messages.addMessage(f"Applied morphological opening/closing with radius={morph_radius} pixels.")
            except Exception as e:
                messages.addWarningMessage("Morphological cleanup failed/partially failed: " + str(e))

            # Prepare binary for saving (water=1, land=0, nodata=255)
            binary_uint8 = np.full_like(mndwi_array, 255, dtype=np.uint8)
            binary_uint8[finite_mask] = binary_bool[finite_mask].astype(np.uint8)

            # Save MNDWI raster
            try:
                ll = processed_raster.extent.lowerLeft
                cs_save = processed_raster.meanCellWidth
                mndwi_raster = arcpy.NumPyArrayToRaster(mndwi_array, ll, cs_save, cs_save)
                if study_area:
                    mndwi_raster = ExtractByMask(mndwi_raster, projected_study_area)
                mndwi_raster.save(mndwi_raster_path)
                messages.addMessage(f"MNDWI raster saved: {mndwi_raster_path}")
            except Exception as e:
                messages.addWarningMessage("Failed to save MNDWI raster: " + str(e))

            # Save binary raster
            try:
                binary_raster = arcpy.NumPyArrayToRaster(binary_uint8, ll, cs_save, cs_save, value_to_nodata=255)
                if study_area:
                    binary_raster = ExtractByMask(binary_raster, projected_study_area)
                binary_raster.save(binary_raster_path)
                messages.addMessage(f"Binary raster saved: {binary_raster_path}")
            except Exception as e:
                messages.addErrorMessage("Failed to save binary raster: " + str(e))
                arcpy.CheckInExtension("Spatial")
                raise

            # Raster -> Polygon
            try:
                arcpy.RasterToPolygon_conversion(binary_raster_path, polygon_path, "SIMPLIFY", "Value")
                messages.addMessage("RasterToPolygon completed (simplified polygons).")
            except Exception as e:
                messages.addErrorMessage("RasterToPolygon_conversion failed: " + str(e))
                arcpy.CheckInExtension("Spatial")
                raise

            # Repair polygon geometry
            try:
                arcpy.RepairGeometry_management(polygon_path)
                messages.addMessage("Polygon geometry repaired.")
            except Exception as e:
                messages.addWarningMessage("RepairGeometry failed: " + str(e))

            # Detect polygon value field (avoid assuming 'gridcode')
            try:
                fields = [f.name.upper() for f in arcpy.ListFields(polygon_path)]
                value_field = None
                for candidate in ("GRIDCODE", "VALUE", "GRID_CODE", "DN"):
                    if candidate in fields:
                        value_field = candidate
                        break
                if value_field is None:
                    # fallback: pick the first numeric field
                    for f in arcpy.ListFields(polygon_path):
                        if f.type in ("SmallInteger", "Integer", "Double", "Single"):
                            value_field = f.name
                            break
                messages.addMessage(f"Polygon value field detected: {value_field}")
            except Exception:
                value_field = None

            # Select only water polygons (value == 1) to valid_polygons
            try:
                if not value_field:
                    arcpy.AddWarning("Could not detect polygon value field; copying all polygons to valid_polygons for inspection.")
                    arcpy.CopyFeatures_management(polygon_path, valid_polygons)
                else:
                    arcpy.MakeFeatureLayer_management(polygon_path, "poly_lyr")
                    sel_clause = f'"{value_field}" = 1'
                    arcpy.SelectLayerByAttribute_management("poly_lyr", "NEW_SELECTION", sel_clause)
                    arcpy.CopyFeatures_management("poly_lyr", valid_polygons)
                    arcpy.Delete_management("poly_lyr")
                messages.addMessage("Water polygons copied to valid_polygons.")
            except Exception as e:
                messages.addErrorMessage("Filtering water polygons failed: " + str(e))
                arcpy.CheckInExtension("Spatial")
                raise

            # Small-polygon filtering using SHAPE@AREA (map units^2)
            try:
                if min_polygon_area and min_polygon_area > 0:
                    desc = arcpy.Describe(valid_polygons)
                    oid_field = desc.OIDFieldName
                    keep_oids = []
                    with arcpy.da.SearchCursor(valid_polygons, [oid_field, "SHAPE@AREA"]) as scur:
                        for oid, area in scur:
                            if area is not None and area >= min_polygon_area:
                                keep_oids.append(oid)

                    if len(keep_oids) == 0:
                        messages.addWarningMessage(f"No polygons meet min_polygon_area={min_polygon_area}. Keeping original polygons.")
                    else:
                        where = f'"{oid_field}" IN ({",".join(map(str, keep_oids))})'
                        arcpy.MakeFeatureLayer_management(valid_polygons, "valid_lyr")
                        arcpy.SelectLayerByAttribute_management("valid_lyr", "NEW_SELECTION", where)
                        arcpy.CopyFeatures_management("valid_lyr", filtered_polygons)
                        arcpy.Delete_management("valid_lyr")
                        valid_polygons = filtered_polygons
                        messages.addMessage(f"Filtered polygons smaller than {min_polygon_area} map units².")
                else:
                    messages.addMessage("No polygon area filtering applied (min_polygon_area <= 0).")
            except Exception as e:
                messages.addWarningMessage("Small-polygon filtering failed: " + str(e))
                messages.addWarningMessage("GP messages:\n" + arcpy.GetMessages(2))

            # ------------------------
            # Segment-based processing (KEEP BOUNDARY, SPLIT TO SINGLE PARTS):
            # - Create segments from water polygons (FeatureToLine or fallback)
            # - Convert multipart->singlepart and SplitLine so features are not joined
            # - Save all segments (including boundary) to shoreline_segments_path for downstream editing
            # ------------------------
            try:
                # Create mask boundary for reference (kept but not used)
                if study_area and arcpy.Exists(projected_study_area):
                    try:
                        if arcpy.Exists(mask_boundary):
                            arcpy.Delete_management(mask_boundary)
                        arcpy.PolygonToLine_management(projected_study_area, mask_boundary)
                        messages.addMessage("Mask boundary created from projected study area (kept for reference).")
                    except Exception as e:
                        messages.addWarningMessage("Failed to create mask boundary from study_area (kept for reference): " + str(e))
                else:
                    # create extent polygon for reference (kept)
                    try:
                        ext = processed_raster.extent
                        arr = arcpy.Array([
                            arcpy.Point(ext.XMin, ext.YMin),
                            arcpy.Point(ext.XMin, ext.YMax),
                            arcpy.Point(ext.XMax, ext.YMax),
                            arcpy.Point(ext.XMax, ext.YMin),
                            arcpy.Point(ext.XMin, ext.YMin),
                        ])
                        extent_poly = arcpy.Polygon(arr, raster_sr)
                        if arcpy.Exists(extent_fc):
                            arcpy.Delete_management(extent_fc)
                        arcpy.CopyFeatures_management([extent_poly], extent_fc)
                        if arcpy.Exists(mask_boundary):
                            arcpy.Delete_management(mask_boundary)
                        arcpy.PolygonToLine_management(extent_fc, mask_boundary)
                        messages.addMessage("Mask boundary created from raster extent (kept for reference).")
                    except Exception as e:
                        messages.addWarningMessage("Failed to create mask boundary from extent (kept for reference): " + str(e))

                # Create fine-grained line segments from polygons (in_memory)
                segments = "in_memory/shoreline_segments"
                if arcpy.Exists(segments):
                    arcpy.Delete_management(segments)

                try:
                    arcpy.FeatureToLine_management(valid_polygons, segments)
                    messages.addMessage("FeatureToLine produced segments in memory.")
                except Exception as e:
                    messages.addWarningMessage("FeatureToLine failed: " + str(e))
                    # fallback: polygon->line then split
                    try:
                        temp_line = "in_memory/temp_polyline"
                        if arcpy.Exists(temp_line):
                            arcpy.Delete_management(temp_line)
                        arcpy.PolygonToLine_management(valid_polygons, temp_line)
                        arcpy.SplitLine_management(temp_line, segments)
                        messages.addMessage("Fallback polygon->line + SplitLine created segments in memory.")
                    except Exception as e2:
                        messages.addErrorMessage("Failed to create segments from polygons: " + str(e2))
                        messages.addErrorMessage("GP messages:\n" + arcpy.GetMessages(2))
                        raise

                seg_count = int(arcpy.GetCount_management(segments).getOutput(0)) if arcpy.Exists(segments) else 0
                messages.addMessage(f"Segment count produced: {seg_count}")
                if seg_count == 0:
                    messages.addErrorMessage("No segments created from polygons; aborting.")
                    raise RuntimeError("No segments produced.")

                # Ensure segments are singlepart and split at vertices/intersections:
                single_parts = "in_memory/shoreline_singleparts"
                split_segments = "in_memory/shoreline_split_segments"
                # clean up if existing
                for tmp in (single_parts, split_segments):
                    if arcpy.Exists(tmp):
                        try:
                            arcpy.Delete_management(tmp)
                        except Exception:
                            pass

                # Multipart -> Singlepart
                arcpy.MultipartToSinglepart_management(segments, single_parts)
                messages.addMessage("Converted multipart segments to singlepart.")

                # Split lines at vertices/intersections so features are not joined
                arcpy.SplitLine_management(single_parts, split_segments)
                messages.addMessage("Split singlepart lines at vertices/intersections.")

                # Repair geometry (non-fatal)
                try:
                    arcpy.RepairGeometry_management(split_segments)
                except Exception:
                    pass

                # Copy split segments to final shapefile (keep boundary segments)
                if arcpy.Exists(shoreline_segments_path):
                    arcpy.Delete_management(shoreline_segments_path)
                arcpy.CopyFeatures_management(split_segments, shoreline_segments_path)
                messages.addMessage(f"Shoreline segments saved to: {shoreline_segments_path}")

                # Add segment ID field
                try:
                    fld_names = [f.name.upper() for f in arcpy.ListFields(shoreline_segments_path)]
                    if "SEG_ID" not in fld_names:
                        arcpy.AddField_management(shoreline_segments_path, "SEG_ID", "LONG")
                        with arcpy.da.UpdateCursor(shoreline_segments_path, ["SEG_ID"]) as ucur:
                            i = 1
                            for row in ucur:
                                row[0] = i
                                ucur.updateRow(row)
                                i += 1
                        messages.addMessage("Added SEG_ID field to shoreline segments.")
                except Exception:
                    # non-fatal
                    pass

                # Clean up in_memory intermediates
                for tmp in (segments, single_parts, split_segments, "in_memory/temp_polyline"):
                    try:
                        if arcpy.Exists(tmp):
                            arcpy.Delete_management(tmp)
                    except Exception:
                        pass

                final_count = int(arcpy.GetCount_management(shoreline_segments_path).getOutput(0)) if arcpy.Exists(shoreline_segments_path) else 0
                messages.addMessage(f"Final shoreline segments count (including boundary): {final_count}")
                if final_count == 0:
                    messages.addWarningMessage("No shoreline segments saved. Inspect intermediate outputs.")
            except Exception as e_seg:
                messages.addErrorMessage("Segment creation failed: " + str(e_seg))
                messages.addErrorMessage("GP messages:\n" + arcpy.GetMessages(2))
                arcpy.CheckInExtension("Spatial")
                raise

            # Define projection and project segments to requested CRS (if requested)
            try:
                if arcpy.Exists(shoreline_segments_path):
                    arcpy.DefineProjection_management(shoreline_segments_path, raster_sr)
                    arcpy.Project_management(shoreline_segments_path, shoreline_segments_projected, output_crs)
                    messages.addMessage(f"Projected shoreline segments saved to: {shoreline_segments_projected}")
                else:
                    messages.addWarningMessage("Shoreline segments not found; skipping projection.")
            except Exception as e:
                messages.addWarningMessage("Projecting shoreline segments failed: " + str(e))
                messages.addWarningMessage("GP messages:\n" + arcpy.GetMessages(2))
                shoreline_segments_projected = shoreline_segments_path if arcpy.Exists(shoreline_segments_path) else None

            # Add outputs to current map (if running inside ArcGIS Pro)
            try:
                aprx = arcpy.mp.ArcGISProject("CURRENT")
                m = aprx.activeMap
                if arcpy.Exists(mndwi_raster_path):
                    m.addDataFromPath(mndwi_raster_path)
                if shoreline_segments_projected and arcpy.Exists(shoreline_segments_projected):
                    m.addDataFromPath(shoreline_segments_projected)
                messages.addMessage("Added MNDWI raster and shoreline segments to current map (if running inside ArcGIS Pro).")
            except Exception as e:
                messages.addWarningMessage("Could not add layers to map: " + str(e))

        finally:
            arcpy.CheckInExtension("Spatial")
            return

    def postExecute(self, parameters):
        "JUST BELIEVE"
        return
