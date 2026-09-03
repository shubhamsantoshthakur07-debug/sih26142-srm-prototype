"""
Synthetic & Real Satellite Scene Generator for SIH26142.
Generates realistic multi-spectral 4-band scenes (Red, Green, Blue, NIR)
with realistic physical spectral responses, sub-pixel mixed boundaries,
and corresponding high-resolution (2.5m) ground truth.
"""

import numpy as np

# Standard spectral signatures for Sentinel-2 bands [Red, Green, Blue, NIR]
SPECTRAL_SIGNATURES = {
    0: np.array([0.55, 0.45, 0.35, 0.40], dtype=np.float32), # Barren soil / sand
    1: np.array([0.05, 0.15, 0.35, 0.02], dtype=np.float32), # Water body (absorbs NIR)
    2: np.array([0.08, 0.28, 0.05, 0.78], dtype=np.float32), # Dense vegetation (strong NIR peak)
    3: np.array([0.72, 0.70, 0.68, 0.62], dtype=np.float32), # Facilities / Structures (high concrete/metal albedo)
    4: np.array([0.22, 0.22, 0.22, 0.20], dtype=np.float32), # Roads / Asphalt runway
}

def generate_tactical_scene(scene_type: str = "border_facility", hr_size: int = 256) -> dict:
    """
    Generates a paired (HR 2.5m, MR 10m, SRM Ground Truth) multi-spectral scene.
    
    Args:
        scene_type: "border_facility", "naval_coastal", "airfield_base", or "river_valley"
        hr_size: High-resolution raster dimension (e.g. 256x256)
    """
    hr_mask = np.zeros((hr_size, hr_size), dtype=np.int64) # 0=Soil, 1=Water, 2=Veg, 3=Facility, 4=Road
    
    # 2D coordinates normalized to [0.0, 1.0]
    y_coords, x_coords = np.meshgrid(
        np.linspace(0.0, 1.0, hr_size),
        np.linspace(0.0, 1.0, hr_size),
        indexing='ij'
    )
    
    if scene_type == "border_facility":
        hr_mask[:] = 0
        # Vegetation patches
        veg_mask = (np.sin(x_coords * 25.0) * np.cos(y_coords * 22.0)) > 0.45
        hr_mask[veg_mask] = 2
        # Strategic highway running diagonally
        road_mask = np.abs((y_coords - 0.75 * x_coords) - 0.12) < 0.02
        hr_mask[road_mask] = 4
        # Secondary access road
        road2_mask = (x_coords > 0.42) & (x_coords < 0.46) & (y_coords > 0.25) & (y_coords < 0.75)
        hr_mask[road2_mask] = 4
        # Fortified outpost / compound buildings
        bldg1 = (y_coords >= 0.35) & (y_coords <= 0.52) & (x_coords >= 0.48) & (x_coords <= 0.65)
        bldg2 = (y_coords >= 0.56) & (y_coords <= 0.68) & (x_coords >= 0.50) & (x_coords <= 0.70)
        bldg3 = (y_coords >= 0.27) & (y_coords <= 0.33) & (x_coords >= 0.54) & (x_coords <= 0.68)
        hr_mask[bldg1 | bldg2 | bldg3] = 3
        # Water retention reservoir
        pond = ((x_coords - 0.82)**2 + (y_coords - 0.24)**2) < (0.12)**2
        hr_mask[pond] = 1

    elif scene_type == "naval_coastal":
        # Ocean water on left side with curved coastline
        shoreline = 0.42 + 0.08 * np.sin(y_coords * 12.0)
        water_mask = x_coords < shoreline
        hr_mask[water_mask] = 1
        hr_mask[~water_mask] = 0
        # Mangrove / coastal canopy
        veg_strip = (~water_mask) & (x_coords < shoreline + 0.14)
        hr_mask[veg_strip] = 2
        # Jetty / Docking pier
        pier = (y_coords >= 0.45) & (y_coords <= 0.52) & (x_coords >= 0.20) & (x_coords <= 0.55)
        hr_mask[pier] = 3
        # Coastal access road
        road_mask = (x_coords >= shoreline + 0.15) & (x_coords <= shoreline + 0.18)
        hr_mask[road_mask] = 4
        # Port warehouses
        for by in [0.22, 0.58, 0.75]:
            wh = (y_coords >= by) & (y_coords <= by + 0.10) & (x_coords >= shoreline + 0.21) & (x_coords <= shoreline + 0.34)
            hr_mask[wh] = 3

    elif scene_type == "airfield_base":
        hr_mask[:] = 0
        # Agricultural patchwork
        grid_veg = (((x_coords * 8).astype(int) % 2) == 0) & (((y_coords * 8).astype(int) % 2) == 1)
        hr_mask[grid_veg] = 2
        # Main military runway
        runway = (np.abs(y_coords - x_coords) < 0.035) & (x_coords > 0.08) & (x_coords < 0.92)
        hr_mask[runway] = 4
        # Parallel taxiway
        taxiway = (np.abs((y_coords - x_coords) - 0.09) < 0.02) & (x_coords > 0.15) & (x_coords < 0.85)
        hr_mask[taxiway] = 4
        # Hangars
        for pos in [0.28, 0.45, 0.62]:
            hangar = (y_coords >= pos) & (y_coords <= pos + 0.08) & (x_coords >= pos - 0.20) & (x_coords <= pos - 0.08)
            hr_mask[hangar] = 3
        # Drainage canal
        canal = (y_coords >= 0.88) & (y_coords <= 0.94)
        hr_mask[canal] = 1

    else: # "river_valley"
        hr_mask[:] = 0
        # Meandering mountain river
        river_center = 0.50 + 0.18 * np.sin(x_coords * 8.0) + 0.06 * np.cos(x_coords * 20.0)
        river_mask = np.abs(y_coords - river_center) < 0.06
        hr_mask[river_mask] = 1
        # Heavy forest on hill slopes
        forest = (y_coords < river_center - 0.12) | (y_coords > river_center + 0.16)
        hr_mask[forest] = 2
        # Valley road tracking river
        road = np.abs(y_coords - (river_center + 0.09)) < 0.016
        hr_mask[road] = 4
        # Valley settlements
        for bx in [0.20, 0.70]:
            center_y = 0.50 + 0.18 * np.sin(bx * 8.0) + 0.06 * np.cos(bx * 20.0) + 0.14
            settlement = (np.abs(x_coords - bx) < 0.07) & (np.abs(y_coords - center_y) < 0.06)
            hr_mask[settlement] = 3

    # Generate High-Resolution 4-Band Multi-Spectral Raster (HR 2.5m)
    hr_multiband = np.zeros((hr_size, hr_size, 4), dtype=np.float32)
    for class_id, signature in SPECTRAL_SIGNATURES.items():
        match_pixels = (hr_mask == class_id)
        hr_multiband[match_pixels] = signature
    
    # Add subtle sensor noise and texture
    noise = np.random.normal(0, 0.015, size=hr_multiband.shape).astype(np.float32)
    hr_multiband = np.clip(hr_multiband + noise, 0.0, 1.0)
    
    # Degrade through Satellite Point Spread Function (PSF) to generate authentic Medium-Res 10m image
    scale = 4
    lr_size = hr_size // scale
    lr_multiband = hr_multiband.reshape(lr_size, scale, lr_size, scale, 4).mean(axis=(1, 3))
    
    # Authentic mixed-pixel sensor noise at 10m
    lr_noise = np.random.normal(0, 0.008, size=lr_multiband.shape).astype(np.float32)
    lr_multiband = np.clip(lr_multiband + lr_noise, 0.0, 1.0)
    
    return {
        "scene_name": scene_type,
        "hr_size": hr_size,
        "lr_size": lr_size,
        "scale": scale,
        "hr_multiband": hr_multiband,
        "lr_multiband": lr_multiband,
        "hr_srm_mask": hr_mask
    }
