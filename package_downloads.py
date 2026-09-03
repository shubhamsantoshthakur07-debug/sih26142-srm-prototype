import os
import shutil
import zipfile

# Source paths
base_brain = r"C:\Users\shubh\.gemini\antigravity\brain\d0ebe830-ee5c-4299-b4bc-987e8221b026"
project_dir = r"C:\Users\shubh\.gemini\antigravity\scratch\sih26142-srm-prototype"

images = [
    # Presentation Slides
    (os.path.join(base_brain, "srm_satellite_comparison_1788430253502.jpg"), "01_Slide_Before_After_SuperResolution_Comparison.jpg"),
    (os.path.join(base_brain, "srm_multispectral_layers_1788430275666.jpg"), "02_Slide_MultiSpectral_4Panel_Analysis.jpg"),
    (os.path.join(base_brain, "srm_system_architecture_1788430305962.jpg"), "03_Slide_DeepLearning_System_Architecture.jpg"),
    # Real-World Satellite Scenes
    (os.path.join(base_brain, "real_border_outpost_1788430502715.jpg"), "04_RealSatellite_Himalayan_Border_Outpost.jpg"),
    (os.path.join(base_brain, "real_airbase_runway_1788430524082.jpg"), "05_RealSatellite_Forward_Airbase_Runway.jpg"),
    (os.path.join(base_brain, "real_naval_harbor_1788430548174.jpg"), "06_RealSatellite_Littoral_Naval_Harbor.jpg"),
]

# Destination Folder inside project
pack_dir = os.path.join(project_dir, "all_images_pack")
os.makedirs(pack_dir, exist_ok=True)

# Also check if Desktop exists
desktop_dir = os.path.expanduser(r"~\Desktop\SIH26142_Images_Pack")
try:
    os.makedirs(desktop_dir, exist_ok=True)
    has_desktop = True
except Exception:
    has_desktop = False

zip_path = os.path.join(project_dir, "SIH26142_Complete_Image_Pack.zip")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for src, target_name in images:
        if os.path.exists(src):
            dst_project = os.path.join(pack_dir, target_name)
            shutil.copyfile(src, dst_project)
            zipf.write(src, arcname=target_name)
            if has_desktop:
                shutil.copyfile(src, os.path.join(desktop_dir, target_name))
            print(f"Packaged: {target_name}")
        else:
            print(f"Missing: {src}")

print(f"\nAll images packaged into:")
print(f"1. Project Folder: {pack_dir}")
if has_desktop:
    print(f"2. Desktop Folder: {desktop_dir}")
print(f"3. Single ZIP file: {zip_path}")
