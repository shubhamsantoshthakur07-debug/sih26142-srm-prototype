# SIH26142: Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries

**Sponsoring Organization:** National Technical Research Organisation (NTRO)  
**Category:** Software | Space Technology / Geospatial Intelligence  
**Scale Factor:** 4× Spatial Super-Resolution (10m $\rightarrow$ 2.5m Ground Sampling Distance)

---

## 1. System Overview

This prototype demonstrates an end-to-end, defense-grade solution for problem statement **SIH26142**, addressing the core trade-off between medium-resolution satellite swaths (e.g. Sentinel-2) and high-resolution spatial targeting.

### Key Capabilities
1. **Multi-Spectral Physical Super-Resolution (Head 1)**:
   - Processes 4-band multi-spectral inputs: Red (B4), Green (B3), Blue (B2), and Near-Infrared (B8).
   - Generates 4× enhanced 2.5m multi-spectral rasters while strictly preserving radiometric calibration.
2. **Sub-Pixel Mapping (SRM - Head 2)**:
   - Resolves the "mixed-pixel" dilemma where 10m coarse pixels contain mixtures of terrain.
   - Accurately classifies and allocates sub-pixels into discrete classes: Water bodies, Dense canopy/vegetation, Military/civilian facilities, Roads/runways, and Barren soil.
3. **Anti-Hallucination & Uncertainty Guardrails (Head 3)**:
   - Implements satellite sensor Point Spread Function (PSF) downsampling cycle-consistency validation.
   - Outputs a pixel-wise confidence/uncertainty heatmap to alert intelligence analysts if any sub-pixel lacks empirical spectral basis.
4. **Interactive Tactical WebGIS Console**:
   - Split-screen comparison slider with real-time before/after wipe.
   - True-Color RGB vs Color-Infrared (CIR False-Color) composite toggling.
   - GeoJSON vector contour extraction for direct import into QGIS/ArcGIS.

---

## 2. Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.14 on Windows)
- PyTorch, FastAPI, Uvicorn, NumPy, Pillow

### Launching the Prototype
1. Open a terminal in this directory:
   ```powershell
   cd "C:\Users\shubh\.gemini\antigravity\scratch\sih26142-srm-prototype"
   ```
2. Run the test suite:
   ```powershell
   python test_prototype.py
   ```
3. Start the interactive console:
   ```powershell
   python run.py
   ```
   *(Or double-click `run.bat` on Windows)*
4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

---

## 3. Project Structure

```
sih26142-srm-prototype/
├── core/
│   ├── srm_model.py            # DualHeadSRMNet: Multi-spectral SR + Sub-Pixel Mapping + Uncertainty
│   ├── metrics.py              # PSNR, SSIM, SAM, ERGAS, Cycle Consistency, mIoU
│   ├── dataset_generator.py    # Multi-spectral Sentinel-2 & HR ground-truth simulator
│   ├── geospatial_utils.py     # CIR composite generator, GeoJSON polygonizer, color palettes
│   └── weights_init.py         # Neural calibration & checkpoint manager
├── server/
│   └── app.py                  # FastAPI REST server & inference pipeline
├── static/
│   ├── index.html              # Tactical WebGIS interface
│   ├── style.css               # Defense-grade dark styling
│   └── app.js                  # Slider controller, layer switcher, vector renderer
├── test_prototype.py           # Verification test suite
├── run.py                      # Server runner
├── run.bat                     # Windows batch launcher
└── requirements.txt            # Python dependencies
```

---

## 4. Evaluation Metrics for NTRO

| Metric | Target | Description |
| :--- | :--- | :--- |
| **PSNR** | $> 32\text{ dB}$ | Peak Signal-to-Noise Ratio for image fidelity |
| **SSIM** | $> 0.90$ | Multi-band Structural Similarity Index |
| **SAM (Spectral Angle Mapper)** | $< 3.5^\circ$ | Measures preservation of physical material signatures |
| **ERGAS** | $< 3.0$ | Synthesis relative global dimensional error |
| **Cycle Consistency Error** | $< 0.02$ | $\| \text{PSF\_Degrade}(I_{SR}) - I_{LR} \|_1 \approx 0$ (Zero hallucination) |
| **SRM Accuracy / mIoU** | $> 85\%$ | Sub-pixel land cover classification precision |
