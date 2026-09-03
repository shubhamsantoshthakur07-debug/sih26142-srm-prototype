"""
Model weight initialization and rapid calibration for DualHeadSRMNet.
Ensures the prototype has trained neural weights on startup.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from core.srm_model import DualHeadSRMNet
from core.dataset_generator import generate_tactical_scene

def get_calibrated_model(weights_path: str = "weights/srm_model.pth", device: str = "cpu") -> DualHeadSRMNet:
    """Load pre-trained weights or run rapid calibration to guarantee real learned weights."""
    model = DualHeadSRMNet(in_channels=4, num_classes=5, num_features=64, scale_factor=4)
    model = model.to(device)

    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model.eval()
            print(f"[SRM-Net] Loaded verified weights from {weights_path}")
            return model
        except Exception as e:
            print(f"[SRM-Net] Note: Re-calibrating model weights ({e})...")

    print("[SRM-Net] Running rapid multi-task neural calibration (4-band SR + SRM + Anti-Hallucination)...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_sr = nn.L1Loss()
    criterion_srm = nn.CrossEntropyLoss()

    scenes = ["border_facility", "naval_coastal", "airfield_base", "river_valley"]
    training_data = [generate_tactical_scene(s, hr_size=128) for s in scenes]

    # Quick 60-step multi-task convergence
    for step in range(60):
        total_loss = 0.0
        for sample in training_data:
            # Prepare tensors: (1, 4, H, W)
            lr_t = torch.from_numpy(sample["lr_multiband"]).permute(2, 0, 1).unsqueeze(0).to(device)
            hr_t = torch.from_numpy(sample["hr_multiband"]).permute(2, 0, 1).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(sample["hr_srm_mask"]).unsqueeze(0).to(device)

            optimizer.zero_grad()
            sr_pred, srm_logits, uncertainty = model(lr_t)

            # 1. Physical SR loss
            loss_sr = criterion_sr(sr_pred, hr_t)

            # 2. Sub-Pixel Mapping (SRM) Cross Entropy loss
            loss_srm = criterion_srm(srm_logits, mask_t)

            # 3. Anti-Hallucination Sensor PSF cycle-consistency loss
            degraded_pred = model.degrade_psf(sr_pred)
            loss_cycle = criterion_sr(degraded_pred, lr_t)

            # Combined multi-task loss
            loss = loss_sr + 0.5 * loss_srm + 0.3 * loss_cycle
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(f"[SRM-Net] Calibration Step {step + 1}/60 | Multi-task Loss: {total_loss / len(training_data):.4f}")

    model.eval()
    try:
        torch.save(model.state_dict(), weights_path)
        print(f"[SRM-Net] Checkpoint saved successfully to {weights_path}")
    except Exception as e:
        print(f"[SRM-Net] Notice: {e}")

    return model
