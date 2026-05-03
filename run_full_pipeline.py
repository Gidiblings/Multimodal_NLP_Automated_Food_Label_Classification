#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path
import shutil

# Prepare raw assets: extract zip, move extracted assets to raw_images root
print("Step 1: Preparing raw assets...")
print("  - Extracting zip contents...")

from step1_image_processing import ImagePreprocessor

pre = ImagePreprocessor(
    input_dir='./raw_images',
    output_dir='./processed_images',
    ground_truth_path='./ground_truth.xlsx'
)

pre._extract_zip_contents()
print("  [OK] Zip extracted")

# Move all image files from zip_extracted to raw_images root for unified processing
src_dir = pre.input_dir / 'zip_extracted' / 'Complete Label List'
if src_dir.exists():
    for file in src_dir.rglob('*'):
        if file.is_file():
            dst = pre.input_dir / file.name
            if not dst.exists():
                shutil.copy2(file, dst)
    print("  [OK] Images consolidated to raw_images root")

# Convert HEIC files found anywhere
print("  Converting HEIC files...")
pre._convert_heic_files(pre.input_dir)
print("  [OK] HEICs converted")

# Now run full preprocessing
print("\nStep 1: Processing all product images...")
results_df = pre.process_all()
results_df.to_csv("processed_images_log.csv", index=False)
print("[OK] Step 1 complete\n")

# Step 2: Extract information
print("Step 2: Extracting information...")
subprocess.run([sys.executable, 'step2_extraction.py'], check=True)
print("[OK] Step 2 complete\n")

# Step 3: NOVA classification
print("Step 3: NOVA classification...")
subprocess.run([sys.executable, 'step3_nova_nutrient_profiles.py'], check=True)
print("[OK] Step 3 complete\n")

# Step 4: Nutrient profiling
print("Step 4: Nutrient profiling...")
subprocess.run([sys.executable, 'step4_model_classification.py'], check=True)
print("[OK] Step 4 complete\n")

# Step 5: Final evaluation
print("Step 5: Final evaluation and visualizations...")
subprocess.run([sys.executable, 'Step5_Healthiness_scoring_ranking.py'], check=True)
print("[OK] Step 5 complete\n")

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
