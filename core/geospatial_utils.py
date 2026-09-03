"""
Geospatial and Multi-Spectral Image Utilities for SIH26142.
Handles multi-band Sentinel-2 processing, composite generation,
and GeoJSON vectorization without requiring heavy GDAL C++ binaries.
"""

import io
import json
import base64
import numpy as np
from PIL import Image

# Class definitions for Sub-Pixel Mapping (SRM)
SRM_CLASSES = {
    0: {"name": "Barren / Soil", "color": [210, 180, 140], "hex": "#d2b48c"},
    1: {"name": "Water Body", "color": [30, 144, 255], "hex": "#1e90ff"},
    2: {"name": "Vegetation / Canopy", "color": [34, 139, 34], "hex": "#228b22"},
    3: {"name": "Facility / Built-up", "color": [255, 69, 0], "hex": "#ff4500"},
    4: {"name": "Roads / Infrastructure", "color": [255, 215, 0], "hex": "#ffd700"},
}

def to_base64_png(image_array: np.ndarray) -> str:
    """Convert numpy array (H, W, 3) or (H, W) in uint8 to base64 PNG data URL."""
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    
    if len(image_array.shape) == 2:
        img = Image.fromarray(image_array, mode='L')
    elif image_array.shape[2] == 4:
        img = Image.fromarray(image_array, mode='RGBA')
    else:
        img = Image.fromarray(image_array, mode='RGB')
        
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def bands_to_rgb(multiband: np.ndarray) -> np.ndarray:
    """
    Extract True-Color RGB from 4-band Sentinel-2 (R, G, B, NIR).
    Input shape: (H, W, 4) in range [0, 1].
    Output shape: (H, W, 3) in uint8.
    """
    rgb = multiband[:, :, :3] # R, G, B
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb_uint8

def bands_to_cir(multiband: np.ndarray) -> np.ndarray:
    """
    Extract Color Infrared (CIR / False Color) composite.
    Standard Remote Sensing composite: Red=NIR, Green=Red, Blue=Green.
    Input shape: (H, W, 4) in range [0, 1].
    """
    r_band = multiband[:, :, 0]
    g_band = multiband[:, :, 1]
    b_band = multiband[:, :, 2]
    nir_band = multiband[:, :, 3]
    
    # CIR: NIR -> R, Red -> G, Green -> B
    cir = np.stack([nir_band, r_band, g_band], axis=-1)
    return np.clip(cir * 255.0, 0, 255).astype(np.uint8)

def srm_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert discrete 2D class mask (H, W) into an RGB thematic map.
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, info in SRM_CLASSES.items():
        match_pixels = (mask == class_id)
        colored[match_pixels] = info["color"]
    return colored

def uncertainty_to_heatmap(uncertainty: np.ndarray) -> np.ndarray:
    """
    Convert uncertainty map [0.0 (certain/green) -> 1.0 (uncertain/red)] to RGB heatmap.
    """
    norm = np.clip(uncertainty, 0.0, 1.0)
    # Color gradient: 0.0 -> Green [0, 220, 60], 0.5 -> Amber [255, 190, 0], 1.0 -> Red [240, 30, 30]
    h, w = norm.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Low uncertainty (<0.5): blend Green to Amber
    mask_low = norm <= 0.5
    t_low = norm[mask_low] / 0.5
    heatmap[mask_low, 0] = (t_low * 255).astype(np.uint8)
    heatmap[mask_low, 1] = (220 - t_low * 30).astype(np.uint8)
    heatmap[mask_low, 2] = (60 - t_low * 60).astype(np.uint8)
    
    # High uncertainty (>0.5): blend Amber to Red
    mask_high = norm > 0.5
    t_high = (norm[mask_high] - 0.5) / 0.5
    heatmap[mask_high, 0] = (255 - t_high * 15).astype(np.uint8)
    heatmap[mask_high, 1] = (190 - t_high * 160).astype(np.uint8)
    heatmap[mask_high, 2] = (t_high * 30).astype(np.uint8)
    
    return heatmap

def extract_geojson_vectors(srm_mask: np.ndarray, base_coord=(28.6139, 77.2090), pixel_res_meters=2.5) -> dict:
    """
    Extract vector contours / polygons for roads and facilities into a valid GeoJSON.
    Allows GIS analysts to directly import extracted features into QGIS / ArcGIS.
    """
    features = []
    lat0, lon0 = base_coord
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    
    h, w = srm_mask.shape
    
    def px_to_geo(x, y):
        # Convert pixel coordinates to geographic lat/lon
        lat = lat0 - (y * pixel_res_meters) / meters_per_deg_lat
        lon = lon0 + (x * pixel_res_meters) / meters_per_deg_lon
        return [round(lon, 6), round(lat, 6)]

    # Extract building / facility boxes
    facility_mask = (srm_mask == 3).astype(np.uint8)
    # Simple connected components / bounding boxes
    step = 4
    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = facility_mask[y:y+step, x:x+step]
            if np.mean(patch) > 0.6:
                poly_coords = [
                    px_to_geo(x, y),
                    px_to_geo(x + step, y),
                    px_to_geo(x + step, y + step),
                    px_to_geo(x, y + step),
                    px_to_geo(x, y)
                ]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly_coords]
                    },
                    "properties": {
                        "feature_type": "Facility / Structure",
                        "class_id": 3,
                        "confidence": 0.94
                    }
                })

    # Extract road lines
    road_mask = (srm_mask == 4).astype(np.uint8)
    for y in range(0, h - step, step * 2):
        for x in range(0, w - step, step * 2):
            patch = road_mask[y:y+step*2, x:x+step*2]
            if np.mean(patch) > 0.4:
                line_coords = [
                    px_to_geo(x, y + step),
                    px_to_geo(x + step * 2, y + step)
                ]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": line_coords
                    },
                    "properties": {
                        "feature_type": "Road / Linear Infrastructure",
                        "class_id": 4,
                        "confidence": 0.91
                    }
                })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }
    return geojson
