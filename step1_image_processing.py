#step1_image_processing.py
"""
Step 1: Data Preparation and Image Processing
Handles image selection, enhancement, and deduplication
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from pathlib import Path
import hashlib
from typing import List, Dict, Tuple
from zipfile import ZipFile
import fitz
import pillow_heif
import re

class ImagePreprocessor:
    def __init__(self, input_dir: str, output_dir: str, ground_truth_path: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Load ground truth and ensure Product_ID is a string for matching
        self.ground_truth = pd.read_excel(ground_truth_path)
        self.ground_truth['Product_ID'] = self.ground_truth['Product_ID'].astype(str)
        self.supported_image_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

        try:
            pillow_heif.register_heif_opener()
            self.supported_image_exts.add('.heic')
        except Exception:
            pass
        
    def enhance_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img
    
    def get_image_hash(self, img: np.ndarray) -> str:
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        diff = gray > avg
        return hashlib.md5(diff.tobytes()).hexdigest()
    
    def deduplicate_images(self, product_id: str, image_paths: List[str]) -> Dict[int, str]:
        unique_images = {}
        seen_hashes = set()
        
        for img_path in sorted(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_hash = self.get_image_hash(img)
            # Safely extract image number, default to sequential if parsing fails
            matches = re.findall(r'_(\d+)(?:$|[^\d])', Path(img_path).stem)
            if matches:
                img_num = int(matches[-1])
            else:
                img_num = len(unique_images) + 1
            
            if img_hash not in seen_hashes:
                unique_images[img_num] = img_path
                seen_hashes.add(img_hash)
                
        return unique_images
    
    def _extract_zip_contents(self) -> None:
        zip_files = list(self.input_dir.glob('*.zip'))
        if not zip_files:
            return

        extracted_dir = self.input_dir / 'zip_extracted'
        extracted_dir.mkdir(parents=True, exist_ok=True)

        for zip_file in zip_files:
            try:
                with ZipFile(zip_file, 'r') as zf:
                    zf.extractall(extracted_dir)
            except Exception as e:
                print(f"Failed to extract {zip_file}: {e}")

    def _convert_pdf_pages(self, root_dir: Path) -> None:
        pdf_paths = list(root_dir.rglob('*.pdf'))
        for pdf_path in pdf_paths:
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(alpha=False)
                    output_path = pdf_path.with_name(f"{pdf_path.stem}_{page_num+1}.jpg")
                    pix.save(str(output_path), output='jpeg')
            except Exception as e:
                print(f"Failed to convert PDF {pdf_path}: {e}")

    def _convert_heic_files(self, root_dir: Path) -> None:
        heic_paths = list(root_dir.rglob('*.heic'))
        for heic_path in heic_paths:
            try:
                img = Image.open(heic_path)
                output_path = heic_path.with_suffix('.jpg')
                img.save(output_path, format='JPEG', quality=95)
            except Exception as e:
                print(f"Failed to convert HEIC {heic_path}: {e}")

    def _prepare_raw_assets(self) -> None:
        self._extract_zip_contents()
        self._convert_pdf_pages(self.input_dir)
        self._convert_pdf_pages(self.input_dir / 'zip_extracted')
        self._convert_heic_files(self.input_dir)
        self._convert_heic_files(self.input_dir / 'zip_extracted')

    def process_product_images(self, product_id: str) -> Dict[int, str]:
        image_paths = []
        for ext in self.supported_image_exts:
            image_paths.extend(self.input_dir.rglob(f"{product_id}_*{ext}"))
        
        if not image_paths:
            return {}
            
        unique_images = self.deduplicate_images(product_id, [str(p) for p in image_paths])
        processed_paths = {}
        
        for img_num, img_path in unique_images.items():
            try:
                enhanced_img = self.enhance_image(img_path)
                output_path = self.output_dir / f"{product_id}_{img_num}.jpg"
                cv2.imwrite(str(output_path), enhanced_img)
                processed_paths[img_num] = str(output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return processed_paths
    
    def process_all(self) -> pd.DataFrame:
        self._prepare_raw_assets()
        # Use ALL unique Product IDs from the ground truth
        selected_products = self.ground_truth['Product_ID'].unique()
        total_products = len(selected_products)
        
        results = []
        successful_products = 0
        
        for i, product_id in enumerate(selected_products):
            print(f"Processing product {i+1}/{total_products}: {product_id}")
            processed = self.process_product_images(product_id)
            
            if processed:
                successful_products += 1

            # Append regardless of success to maintain baseline
            results.append({
                'Product_ID': product_id,
                'front_image': processed.get(1, None),
                'back_image_1': processed.get(2, None),
                'back_image_2': processed.get(3, None),
                'back_image_3': processed.get(4, None),  # Support up to 4 images per product
                'num_unique_images': len(processed),
                'image_processing_success': len(processed) > 0
            })
            
        # Print Summary Statistics
        success_rate = (successful_products / total_products) * 100
        print("\n" + "="*40)
        print("IMAGE PROCESSING SUMMARY")
        print("="*40)
        print(f"Total Products in Baseline: {total_products}")
        print(f"Products with Processed Images: {successful_products}")
        print(f"Image Processing Success Rate: {success_rate:.2f}%")
        print("="*40 + "\n")
        
        return pd.DataFrame(results)

# Usage example
if __name__ == "__main__":
    preprocessor = ImagePreprocessor(
        input_dir="./raw_images",
        output_dir="./processed_images",
        ground_truth_path="./ground_truth.xlsx"
    )
    
    # Process ALL products based on ground truth
    results_df = preprocessor.process_all()
    
    # Save processing results
    results_df.to_csv("processed_images_log.csv", index=False)