/**
 * Frontend JavaScript for SIH26142 - Super Resolution Mapping (SRM) Console
 * Manages split-screen slider, layer switching, API requests, and vector rendering.
 */

let scenesData = [];
let currentResult = null;
let activeLayerKey = "sr_rgb";
let customImageB64 = null;
let isDragging = false;

// DOM Elements
const sceneSelect = document.getElementById("scene-select");
const sceneDesc = document.getElementById("scene-desc");
const customFileInput = document.getElementById("custom-file-input");
const customFilename = document.getElementById("custom-filename");
const runBtn = document.getElementById("run-btn");
const loadingOverlay = document.getElementById("loading-overlay");

const comparisonContainer = document.getElementById("comparison-container");
const resizeWrapper = document.getElementById("resize-wrapper");
const sliderHandle = document.getElementById("slider-handle");
const lrImage = document.getElementById("lr-image");
const srImage = document.getElementById("sr-image");
const vectorCanvas = document.getElementById("vector-canvas");
const vectorToggle = document.getElementById("vector-toggle");
const activeLayerName = document.getElementById("active-layer-name");
const coordDisplay = document.getElementById("coord-display");

// Metrics DOM
const valPsnr = document.getElementById("val-psnr");
const valSsim = document.getElementById("val-ssim");
const valSam = document.getElementById("val-sam");
const valErgas = document.getElementById("val-ergas");
const valCycle = document.getElementById("val-cycle");
const valSrmAcc = document.getElementById("val-srm-acc");
const valSrmMiou = document.getElementById("val-srm-miou");
const classBreakdownList = document.getElementById("class-breakdown-list");
const exportGeojsonBtn = document.getElementById("export-geojson-btn");
const exportRasterBtn = document.getElementById("export-raster-btn");

// Initialize application
window.addEventListener("DOMContentLoaded", async () => {
    setupSlider();
    setupLayerToggles();
    setupCoordTracking();
    await loadScenes();
    // Automatically trigger initial inference
    await executeInference();
});

// Load Scenario Presets from Server
async function loadScenes() {
    try {
        const res = await fetch("/api/scenes");
        scenesData = await res.json();
        sceneSelect.innerHTML = "";
        scenesData.forEach(sc => {
            const opt = document.createElement("option");
            opt.value = sc.id;
            opt.textContent = `${sc.name} (${sc.region})`;
            sceneSelect.appendChild(opt);
        });
        updateSceneDescription();
    } catch (err) {
        console.error("Failed to load scenes:", err);
        sceneDesc.textContent = "Error connecting to SRM inference engine.";
    }
}

function updateSceneDescription() {
    const selected = scenesData.find(s => s.id === sceneSelect.value);
    if (selected) {
        sceneDesc.innerHTML = `<strong>${selected.region}</strong><br>${selected.description}`;
    }
}

sceneSelect.addEventListener("change", () => {
    customImageB64 = null;
    customFilename.textContent = "No custom file loaded";
    updateSceneDescription();
    executeInference();
});

// Custom Image File Upload
customFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    customFilename.textContent = file.name;
    const reader = new FileReader();
    reader.onload = (event) => {
        customImageB64 = event.target.result;
        executeInference();
    };
    reader.readAsDataURL(file);
});

// Execute Super Resolution Mapping (SRM) Inference
runBtn.addEventListener("click", () => executeInference());

async function executeInference() {
    loadingOverlay.classList.add("active");
    runBtn.disabled = true;

    try {
        const payload = {
            scene_id: sceneSelect.value,
            custom_image_base64: customImageB64
        };

        const res = await fetch("/api/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            throw new Error(`Inference error: ${res.statusText}`);
        }

        currentResult = await res.json();
        updateDisplay();
    } catch (err) {
        alert("Inference failed: " + err.message);
        console.error(err);
    } finally {
        loadingOverlay.classList.remove("active");
        runBtn.disabled = false;
    }
}

// Update Viewport & Telemetry HUD with Inference Results
function updateDisplay() {
    if (!currentResult) return;

    // Set Medium Resolution image (Left of split)
    lrImage.src = currentResult.layers.lr_rgb;

    // Set Super Resolved layer (Right of split)
    updateActiveLayer();

    // Render Metrics
    const m = currentResult.metrics;
    valPsnr.textContent = `${m.psnr_db} dB`;
    valSsim.textContent = m.ssim.toFixed(3);
    valSam.textContent = `${m.sam_deg}°`;
    valErgas.textContent = m.ergas.toFixed(2);
    valCycle.textContent = m.cycle_consistency_error.toFixed(4);
    valSrmAcc.textContent = `${m.subpixel_accuracy_pct}%`;
    valSrmMiou.textContent = `${m.subpixel_miou_pct}%`;

    // Render Sub-Pixel Class Breakdown
    renderClassBreakdown(currentResult.class_breakdown);

    // Draw Vector Overlay
    renderVectors();
}

function updateActiveLayer() {
    if (!currentResult) return;
    const layerSrc = currentResult.layers[activeLayerKey] || currentResult.layers.sr_rgb;
    srImage.src = layerSrc;

    const layerTitles = {
        "sr_rgb": "4x Super-Resolved True Color (2.5m)",
        "sr_cir": "Color-Infrared Composite (CIR False Color)",
        "srm_thematic": "Sub-Pixel Thematic Classification (SRM)",
        "uncertainty_heatmap": "Anti-Hallucination Confidence Heatmap"
    };
    activeLayerName.textContent = layerTitles[activeLayerKey] || "Super-Resolved 2.5m";
}

function renderClassBreakdown(breakdown) {
    classBreakdownList.innerHTML = "";
    breakdown.forEach(item => {
        const row = document.createElement("div");
        row.className = "class-row";
        row.innerHTML = `
            <div class="class-header">
                <span class="class-name">
                    <span class="class-dot" style="background:${item.color};"></span>
                    ${item.name}
                </span>
                <span class="class-pct">${item.percentage}% (${item.area_sq_km} km²)</span>
            </div>
            <div class="class-bar-bg">
                <div class="class-bar-fill" style="width: ${item.percentage}%; background: ${item.color};"></div>
            </div>
        `;
        classBreakdownList.appendChild(row);
    });
}

// Layer Switcher
function setupLayerToggles() {
    const radioInputs = document.querySelectorAll('input[name="display-layer"]');
    radioInputs.forEach(radio => {
        radio.addEventListener("change", (e) => {
            activeLayerKey = e.target.value;
            document.querySelectorAll(".layer-radio").forEach(el => el.classList.remove("active"));
            e.target.closest(".layer-radio").classList.add("active");
            updateActiveLayer();
        });
    });

    vectorToggle.addEventListener("change", () => renderVectors());
}

// Draw Extracted GIS Vectors onto Canvas
function renderVectors() {
    const canvas = vectorCanvas;
    const ctx = canvas.getContext("2d");
    canvas.width = 512;
    canvas.height = 512;
    ctx.clearRect(0, 0, 512, 512);

    if (!vectorToggle.checked || !currentResult || !currentResult.geojson) {
        return;
    }

    const features = currentResult.geojson.features || [];
    const scale = 512 / 256; // 2x display scaling

    features.forEach(feat => {
        const geom = feat.geometry;
        const props = feat.properties;

        if (geom.type === "Polygon") {
            // Draw building / facility boundary
            ctx.fillStyle = "rgba(255, 69, 0, 0.35)";
            ctx.strokeStyle = "#ff4500";
            ctx.lineWidth = 1.5;
            
            ctx.beginPath();
            geom.coordinates[0].forEach((pt, idx) => {
                // Approximate mapping to canvas space
                const x = ((pt[0] - 77.2090) * 40000 + 128) * scale;
                const y = ((28.6139 - pt[1]) * 40000 + 128) * scale;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        } else if (geom.type === "LineString") {
            // Draw road / linear infrastructure
            ctx.strokeStyle = "rgba(255, 235, 59, 0.9)";
            ctx.lineWidth = 2.5;
            ctx.setLineDash([4, 2]);

            ctx.beginPath();
            geom.coordinates.forEach((pt, idx) => {
                const x = ((pt[0] - 77.2090) * 40000 + 128) * scale;
                const y = ((28.6139 - pt[1]) * 40000 + 128) * scale;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });
}

// Interactive Draggable Comparison Slider
function setupSlider() {
    const startDrag = () => { isDragging = true; };
    const stopDrag = () => { isDragging = false; };

    sliderHandle.addEventListener("mousedown", startDrag);
    window.addEventListener("mouseup", stopDrag);

    sliderHandle.addEventListener("touchstart", startDrag);
    window.addEventListener("touchend", stopDrag);

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        moveSlider(e.clientX);
    });

    window.addEventListener("touchmove", (e) => {
        if (!isDragging) return;
        moveSlider(e.touches[0].clientX);
    });

    // Clicking anywhere on comparison container snaps slider
    comparisonContainer.addEventListener("click", (e) => {
        if (e.target !== sliderHandle && !sliderHandle.contains(e.target)) {
            moveSlider(e.clientX);
        }
    });
}

function moveSlider(clientX) {
    const rect = comparisonContainer.getBoundingClientRect();
    const imageWidth = 512;
    const imageLeft = rect.left + (rect.width - imageWidth) / 2;

    let posX = clientX - imageLeft;
    if (posX < 10) posX = 10;
    if (posX > imageWidth - 10) posX = imageWidth - 10;

    // Update resize wrapper and handle position
    resizeWrapper.style.width = `${posX}px`;
    sliderHandle.style.left = `${imageLeft - rect.left + posX}px`;
}

// Coordinate Tracking Readout
function setupCoordTracking() {
    comparisonContainer.addEventListener("mousemove", (e) => {
        const rect = comparisonContainer.getBoundingClientRect();
        const imageWidth = 512;
        const imageHeight = 512;
        const imgLeft = (rect.width - imageWidth) / 2;
        const imgTop = (rect.height - imageHeight) / 2;

        const x = e.clientX - rect.left - imgLeft;
        const y = e.clientY - rect.top - imgTop;

        if (x >= 0 && x <= imageWidth && y >= 0 && y <= imageHeight) {
            // Normalized to geographic coords
            const lat = (28.6139 - (y / imageHeight) * 0.006).toFixed(4);
            const lon = (77.2090 + (x / imageWidth) * 0.006).toFixed(4);
            coordDisplay.textContent = `${lat}° N, ${lon}° E`;
        }
    });
}

// Export GeoJSON Vector contours
exportGeojsonBtn.addEventListener("click", () => {
    if (!currentResult || !currentResult.geojson) {
        alert("Run SRM inference first to generate vectors.");
        return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentResult.geojson, null, 2));
    const dlAnchor = document.createElement("a");
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", `ntro_srm_vectors_${sceneSelect.value}.geojson`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
});

// Export Current Super-Resolved Layer
exportRasterBtn.addEventListener("click", () => {
    if (!currentResult) {
        alert("Run SRM inference first.");
        return;
    }
    const src = currentResult.layers[activeLayerKey];
    const dlAnchor = document.createElement("a");
    dlAnchor.setAttribute("href", src);
    dlAnchor.setAttribute("download", `srm_2.5m_${activeLayerKey}_${sceneSelect.value}.png`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
});
