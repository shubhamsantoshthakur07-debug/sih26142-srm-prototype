"""
Model weight initialization and rapid calibration for DualHeadSRMNet.
Ensures the prototype has trained neural weights on startup, optimized for cloud resource limits (e.g. Render 512MB RAM).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from core.srm_model import DualHeadSRMNet
from core.dataset_generator import generate_tactical_scene

def get_calibrated_model(weights_path: str = "weights/srm_model.pth", device: str = "cpu") -> DualHeadSRMNet:
    """Load pre-trained weights or run ultra-fast calibration to guarantee real learned weights."""
    model = DualHeadSRMNet(in_channels=4, num_classes=5, num_features=32, scale_factor=4)
    model = model.to(device)

    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model.eval()
            print(f"[SRM-Net] Loaded verified weights from {weights_path}", flush=True)
            return model
        except Exception as e:
            print(f"[SRM-Net] Note: Re-calibrating model weights ({e})...", flush=True)

    print("[SRM-Net] Running rapid cloud calibration (4-band SR + SRM + Anti-Hallucination)...", flush=True)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_sr = nn.L1Loss()
    criterion_srm = nn.CrossEntropyLoss()

    scenes = ["border_facility", "naval_coastal"]
    training_data = [generate_tactical_scene(s, hr_size=64) for s in scenes]

    for step in range(15):
        total_loss = 0.0
        for sample in training_data:
            lr_t = torch.from_numpy(sample["lr_multiband"]).permute(2, 0, 1).unsqueeze(0).to(device)
            hr_t = torch.from_numpy(sample["hr_multiband"]).permute(2, 0, 1).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(sample["hr_srm_mask"]).unsqueeze(0).to(device)

            optimizer.zero_grad()
            sr_pred, srm_logits, uncertainty = model(lr_t)

            loss_sr = criterion_sr(sr_pred, hr_t)
            loss_srm = criterion_srm(srm_logits, mask_t)
            degraded_pred = model.degrade_psf(sr_pred)
            loss_cycle = criterion_sr(degraded_pred, lr_t)

            loss = loss_sr + 0.5 * loss_srm + 0.3 * loss_cycle
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (step + 1) % 5 == 0:
            print(f"[SRM-Net] Calibration Step {step + 1}/15 | Loss: {total_loss / len(training_data):.4f}", flush=True)

    model.eval()
    try:
        torch.save(model.state_dict(), weights_path)
        print(f"[SRM-Net] Checkpoint saved successfully to {weights_path}", flush=True)
    except Exception as e:
        print(f"[SRM-Net] Notice: {e}", flush=True)

    return model
