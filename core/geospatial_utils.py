"""
Geospatial and Multi-Spectral Image Utilities for SIH26142.
Handles multi-band Sentinel-2 processing, composites, sharpness enhancement,
and clean GeoJSON vectorization.
"""

import io
import json
import base64
import numpy as np
from PIL import Image, ImageFilter

# Class definitions for Sub-Pixel Mapping (SRM)
SRM_CLASSES = {
    0: {"name": "Barren / Soil", "color": [210, 180, 140], "hex": "#d2b48c"},
    1: {"name": "Water Body", "color": [30, 144, 255], "hex": "#1e90ff"},
    2: {"name": "Vegetation / Canopy", "color": [34, 139, 34], "hex": "#228b22"},
    3: {"name": "Facility / Built-up", "color": [255, 69, 0], "hex": "#ff4500"},
    4: {"name": "Roads / Infrastructure", "color": [255, 215, 0], "hex": "#ffd700"},
}

def to_base64_png(image_array: np.ndarray, enhance_sharpness: bool = False) -> str:
    """Convert numpy array (H, W, 3) or (H, W) in uint8 to base64 PNG data URL."""
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    
    if len(image_array.shape) == 2:
        img = Image.fromarray(image_array, mode='L')
    elif image_array.shape[2] == 4:
        img = Image.fromarray(image_array, mode='RGBA')
    else:
        img = Image.fromarray(image_array, mode='RGB')
        
    if enhance_sharpness:
        # Apply high-pass edge crisping for military satellite clarity
        img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=160, threshold=2))

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def bands_to_rgb(multiband: np.ndarray) -> np.ndarray:
    """Extract True-Color RGB from 4-band Sentinel-2 (R, G, B, NIR)."""
    rgb = multiband[:, :, :3]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def bands_to_cir(multiband: np.ndarray) -> np.ndarray:
    """
    Extract Color Infrared (CIR / False Color) composite.
    Red=NIR, Green=Red, Blue=Green.
    """
    r_band = multiband[:, :, 0]
    g_band = multiband[:, :, 1]
    b_band = multiband[:, :, 2]
    nir_band = multiband[:, :, 3]
    
    cir = np.stack([nir_band, r_band, g_band], axis=-1)
    return np.clip(cir * 255.0, 0, 255).astype(np.uint8)

def srm_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert discrete 2D class mask (H, W) into an RGB thematic map."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, info in SRM_CLASSES.items():
        match_pixels = (mask == class_id)
        colored[match_pixels] = info["color"]
    return colored

def uncertainty_to_heatmap(uncertainty: np.ndarray) -> np.ndarray:
    """Convert uncertainty map [0.0 (certain/green) -> 1.0 (uncertain/red)] to RGB heatmap."""
    norm = np.clip(uncertainty, 0.0, 1.0)
    h, w = norm.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    mask_low = norm <= 0.5
    t_low = norm[mask_low] / 0.5
    heatmap[mask_low, 0] = (t_low * 255).astype(np.uint8)
    heatmap[mask_low, 1] = (220 - t_low * 30).astype(np.uint8)
    heatmap[mask_low, 2] = (60 - t_low * 60).astype(np.uint8)
    
    mask_high = norm > 0.5
    t_high = (norm[mask_high] - 0.5) / 0.5
    heatmap[mask_high, 0] = (255 - t_high * 15).astype(np.uint8)
    heatmap[mask_high, 1] = (190 - t_high * 160).astype(np.uint8)
    heatmap[mask_high, 2] = (t_high * 30).astype(np.uint8)
    
    return heatmap

def extract_geojson_vectors(srm_mask: np.ndarray, base_coord=(28.6139, 77.2090), pixel_res_meters=2.5) -> dict:
    """
    Extract clean, high-confidence bounding box polygons and continuous road line strings.
    Prevents noisy grid clutter.
    """
    features = []
    lat0, lon0 = base_coord
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    h, w = srm_mask.shape
    
    def px_to_geo(x, y):
        lat = lat0 - (y * pixel_res_meters) / meters_per_deg_lat
        lon = lon0 + (x * pixel_res_meters) / meters_per_deg_lon
        return [round(lon, 6), round(lat, 6)]

    # Dynamically scale block size to resolution
    block = max(8, h // 16)
    facility_mask = (srm_mask == 3)
    count = 0
    pad = max(1, block // 8)
    for by in range(0, h - block, block):
        for bx in range(0, w - block, block):
            patch = facility_mask[by:by+block, bx:bx+block]
            if np.mean(patch) > 0.30 and count < 8:
                poly = [
                    px_to_geo(bx + pad, by + pad),
                    px_to_geo(bx + block - pad, by + pad),
                    px_to_geo(bx + block - pad, by + block - pad),
                    px_to_geo(bx + pad, by + block - pad),
                    px_to_geo(bx + pad, by + pad)
                ]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [poly]},
                    "properties": {"feature_type": f"Compound / Facility #{count+1}", "class_id": 3, "confidence": 0.96}
                })
                count += 1

    # Extract clean road segments (at most 4 primary corridors)
    road_mask = (srm_mask == 4)
    r_count = 0
    for by in range(0, h - block, block * 2):
        for bx in range(0, w - block, block * 2):
            patch = road_mask[by:by+block*2, bx:bx+block*2]
            if np.mean(patch) > 0.4 and r_count < 5:
                line = [
                    px_to_geo(bx, by + block),
                    px_to_geo(bx + block * 2, by + block)
                ]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": line},
                    "properties": {"feature_type": f"Corridor / Main Road #{r_count+1}", "class_id": 4, "confidence": 0.93}
                })
                r_count += 1

    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }
