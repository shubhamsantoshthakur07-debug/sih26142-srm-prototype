import os
import shutil

src_files = [
    (r"C:\Users\shubh\.gemini\antigravity\brain\d0ebe830-ee5c-4299-b4bc-987e8221b026\real_border_outpost_1788430502715.jpg", "01_himalayan_border_outpost.jpg"),
    (r"C:\Users\shubh\.gemini\antigravity\brain\d0ebe830-ee5c-4299-b4bc-987e8221b026\real_airbase_runway_1788430524082.jpg", "02_forward_airbase_runway.jpg"),
    (r"C:\Users\shubh\.gemini\antigravity\brain\d0ebe830-ee5c-4299-b4bc-987e8221b026\real_naval_harbor_1788430548174.jpg", "03_naval_harbor_pier.jpg"),
]

target_dir = os.path.join(os.path.dirname(__file__), "sample_test_images")
os.makedirs(target_dir, exist_ok=True)

for src, name in src_files:
    if os.path.exists(src):
        dst = os.path.join(target_dir, name)
        shutil.copyfile(src, dst)
        print(f"Copied: {name} -> {dst}")
    else:
        print(f"Source not found: {src}")

print("\nSample images are ready in 'sample_test_images' folder!")
