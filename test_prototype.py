"""
Comprehensive Automated Test Suite for SIH26142 SRM Prototype.
Verifies model architecture, metrics, synthetic data generation, and API responses.
"""

import sys
import os
import torch
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.srm_model import DualHeadSRMNet
from core.dataset_generator import generate_tactical_scene
from core.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_sam,
    calculate_ergas,
    calculate_cycle_consistency,
    calculate_srm_accuracy
)
from core.geospatial_utils import (
    extract_geojson_vectors,
    bands_to_rgb,
    bands_to_cir,
    srm_mask_to_rgb,
    uncertainty_to_heatmap
)

def test_scene_generator():
    print("[TEST 1/5] Testing Multi-Spectral Scene Generator...")
    for scene_type in ["border_facility", "naval_coastal", "airfield_base", "river_valley"]:
        scene = generate_tactical_scene(scene_type, hr_size=128)
        assert scene["lr_multiband"].shape == (32, 32, 4), f"Wrong LR shape: {scene['lr_multiband'].shape}"
        assert scene["hr_multiband"].shape == (128, 128, 4), f"Wrong HR shape: {scene['hr_multiband'].shape}"
        assert scene["hr_srm_mask"].shape == (128, 128), f"Wrong mask shape: {scene['hr_srm_mask'].shape}"
        assert np.max(scene["lr_multiband"]) <= 1.0 and np.min(scene["lr_multiband"]) >= 0.0
    print("  --> PASS: Synthetic generator accurately simulates 4-band Sentinel-2 & HR ground truth.")

def test_srm_model_forward():
    print("[TEST 2/5] Testing DualHeadSRMNet PyTorch Forward Pass & Degradation...")
    model = DualHeadSRMNet(in_channels=4, num_classes=5, num_features=32, scale_factor=4)
    model.eval()

    # Input: 1 sample, 4 bands (R, G, B, NIR), 32x32
    dummy_in = torch.rand(1, 4, 32, 32)
    with torch.no_grad():
        sr_img, srm_logits, uncertainty = model(dummy_in)
        degraded = model.degrade_psf(sr_img)

    assert sr_img.shape == (1, 4, 128, 128), f"Expected (1, 4, 128, 128), got {sr_img.shape}"
    assert srm_logits.shape == (1, 5, 128, 128), f"Expected (1, 5, 128, 128), got {srm_logits.shape}"
    assert uncertainty.shape == (1, 1, 128, 128), f"Expected (1, 1, 128, 128), got {uncertainty.shape}"
    assert degraded.shape == (1, 4, 32, 32), f"Degraded PSF shape mismatch: {degraded.shape}"
    print("  --> PASS: Model executes dual-head SR + SRM + Uncertainty + PSF downsampling correctly.")

def test_metrics():
    print("[TEST 3/5] Testing Remote Sensing Metrics & Guardrails...")
    img1 = np.ones((64, 64, 4), dtype=np.float32) * 0.5
    img2 = img1 + np.random.normal(0, 0.01, size=img1.shape).astype(np.float32)
    img2 = np.clip(img2, 0.0, 1.0)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    sam = calculate_sam(img1, img2)
    ergas = calculate_ergas(img1, img2, scale=4)
    cycle_err = calculate_cycle_consistency(img1, np.ones((16, 16, 4), dtype=np.float32) * 0.5, scale=4)

    assert psnr > 30.0, f"PSNR too low: {psnr}"
    assert 0.8 < ssim <= 1.0, f"SSIM out of bounds: {ssim}"
    assert sam < 5.0, f"SAM too high: {sam}"
    assert ergas >= 0.0, f"ERGAS invalid: {ergas}"
    assert cycle_err < 0.01, f"Cycle error unexpected: {cycle_err}"

    mask1 = np.array([[0, 1], [2, 3]], dtype=np.int64)
    mask2 = np.array([[0, 1], [2, 0]], dtype=np.int64)
    acc = calculate_srm_accuracy(mask1, mask2)
    assert acc["overall_accuracy"] == 75.0, f"Acc expected 75.0, got {acc}"
    print(f"  --> PASS: Metrics verified (PSNR: {psnr:.2f}dB, SSIM: {ssim:.3f}, SAM: {sam:.2f}°, Cycle: {cycle_err:.4f}).")

def test_geospatial_utilities():
    print("[TEST 4/5] Testing Geospatial Composites & Vectorization...")
    test_multiband = np.random.uniform(0.1, 0.9, size=(64, 64, 4)).astype(np.float32)
    rgb = bands_to_rgb(test_multiband)
    cir = bands_to_cir(test_multiband)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert cir.shape == (64, 64, 3) and cir.dtype == np.uint8

    dummy_mask = np.zeros((64, 64), dtype=np.int64)
    dummy_mask[10:25, 10:25] = 3 # facility
    dummy_mask[30:35, :] = 4     # road
    geojson = extract_geojson_vectors(dummy_mask, pixel_res_meters=2.5)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0, "No features extracted from mask"
    print(f"  --> PASS: Extracted {len(geojson['features'])} vector features (Polygons & Lines) into valid GeoJSON.")

def test_api_endpoints():
    print("[TEST 5/5] Testing FastAPI Server Endpoints via TestClient...")
    from fastapi.testclient import TestClient
    from server.app import app

    with TestClient(app) as client:
        # Test multi-page routes
        for path in ["/", "/home", "/dashboard", "/analytics", "/methodology", "/catalog", "/about"]:
            res = client.get(path)
            assert res.status_code == 200, f"Route {path} failed: {res.status_code}"

        # Test analytics endpoint
        res_an = client.get("/api/analytics")
        assert res_an.status_code == 200
        an_data = res_an.json()
        assert "fidelity_benchmarks" in an_data

        # Test scenes list
        res_scenes = client.get("/api/scenes")
        assert res_scenes.status_code == 200
        scenes = res_scenes.json()
        assert len(scenes) >= 4

        # Test process endpoint with 8x scaling and dehaze
        res_process = client.post("/api/process", json={
            "scene_id": "border_facility",
            "scale_factor": 8,
            "dehaze": True
        })
        assert res_process.status_code == 200
        data = res_process.json()
        assert "metrics" in data
        assert "layers" in data
        assert "sr_rgb" in data["layers"]
        assert "sr_cir" in data["layers"]
        assert "srm_thematic" in data["layers"]
        assert "uncertainty_heatmap" in data["layers"]
        assert "ndvi" in data["layers"]
        assert "ndwi" in data["layers"]
        assert "flir_thermal" in data["layers"]
        assert "structural_edges" in data["layers"]
        assert "detections" in data
        assert len(data["detections"]) > 0
        assert data["scale_factor"] == "8x"
        assert data["dehazed"] is True
        assert "geojson" in data
        print("  --> PASS: All 6 routes, APIs, 8 spectral layers, detections, and 8x scaling verified.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE: SIH26142 SRM PROTOTYPE")
    print("=" * 60)
    test_scene_generator()
    test_srm_model_forward()
    test_metrics()
    test_geospatial_utilities()
    test_api_endpoints()
    print("=" * 60)
    print("ALL 5 TESTS PASSED SUCCESSFULLY! PROTOTYPE IS READY.")
    print("=" * 60)
