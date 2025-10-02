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
import shutil

class ImagePreprocessor:
    def __init__(self, input_dir: str, output_dir: str, ground_truth_path: str):
        """
        Initialize image preprocessor
        
        Args:
            input_dir: Directory containing raw JPG images
            output_dir: Directory for processed images
            ground_truth_path: Path to Excel file with ground truth
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth = pd.read_excel(ground_truth_path)
        
    def select_subset(self, n_products: int = 200) -> List[str]:
        """Select subset of products to process"""
        product_ids = self.ground_truth['Product_ID'].unique()[:n_products]
        return [str(pid) for pid in product_ids]
    
    def enhance_image(self, img_path: str) -> np.ndarray:
        """
        Enhance image quality for better OCR
        
        Args:
            img_path: Path to image file
            
        Returns:
            Enhanced image as numpy array
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Convert to PIL for enhancement
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(2.0)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Adaptive thresholding for text clarity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to BGR for consistency
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def get_image_hash(self, img: np.ndarray) -> str:
        """Generate perceptual hash for image deduplication"""
        # Resize to small size for comparison
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Create hash
        avg = gray.mean()
        diff = gray > avg
        return hashlib.md5(diff.tobytes()).hexdigest()
    
    def deduplicate_images(self, product_id: str, 
                          image_paths: List[str]) -> Dict[int, str]:
        """
        Deduplicate images for a product based on visual similarity
        
        Args:
            product_id: Product identifier
            image_paths: List of image paths for this product
            
        Returns:
            Dictionary mapping image number to path of unique images
        """
        unique_images = {}
        seen_hashes = set()
        
        for img_path in sorted(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_hash = self.get_image_hash(img)
            
            # Extract image number from filename (productID_1, _2, _3)
            img_num = int(Path(img_path).stem.split('_')[-1])
            
            if img_hash not in seen_hashes:
                unique_images[img_num] = img_path
                seen_hashes.add(img_hash)
        
        return unique_images
    
    def process_product_images(self, product_id: str) -> Dict[int, str]:
        """
        Process all images for a single product
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary mapping image number to saved path
        """
        # Find all images for this product
        pattern = f"{product_id}_*.jpg"
        image_paths = list(self.input_dir.glob(pattern))
        
        if not image_paths:
            print(f"No images found for product {product_id}")
            return {}
        
        # Deduplicate
        unique_images = self.deduplicate_images(product_id, 
                                                [str(p) for p in image_paths])
        
        processed_paths = {}
        
        # Enhance and save unique images
        for img_num, img_path in unique_images.items():
            try:
                enhanced_img = self.enhance_image(img_path)
                
                # Save enhanced image
                output_path = self.output_dir / f"{product_id}_{img_num}.jpg"
                cv2.imwrite(str(output_path), enhanced_img)
                processed_paths[img_num] = str(output_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return processed_paths
    
    def process_all(self, n_products: int = 200) -> pd.DataFrame:
        """
        Process all selected products
        
        Args:
            n_products: Number of products to process
            
        Returns:
            DataFrame with product IDs and processed image paths
        """
        selected_products = self.select_subset(n_products)
        
        results = []
        for i, product_id in enumerate(selected_products):
            print(f"Processing product {i+1}/{len(selected_products)}: {product_id}")
            processed = self.process_product_images(product_id)
            
            results.append({
                'Product_ID': product_id,
                'front_image': processed.get(1, None),
                'back_image_1': processed.get(2, None),
                'back_image_2': processed.get(3, None),
                'num_unique_images': len(processed)
            })
        
        return pd.DataFrame(results)


# Usage example
if __name__ == "__main__":
    preprocessor = ImagePreprocessor(
        input_dir="./raw_images",
        output_dir="./processed_images",
        ground_truth_path="./ground_truth.xlsx"
    )
    
    # Process 200 products
    results_df = preprocessor.process_all(n_products=200)
    
    # Save processing results
    results_df.to_csv("processed_images_log.csv", index=False)
    print(f"Processed {len(results_df)} products")
    print(f"Results saved to processed_images_log.csv")
