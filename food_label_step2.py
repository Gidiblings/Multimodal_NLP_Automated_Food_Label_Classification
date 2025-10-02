"""
Step 2: Information Extraction using OCR and NLP
Extracts product names, claims, ingredients, and nutrition info
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import json
from paddleocr import PaddleOCR
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

class MultimodalExtractor:
    def __init__(self, ground_truth_path: str, use_layoutlm: bool = False):
        """
        Initialize multimodal information extractor
        
        Args:
            ground_truth_path: Path to ground truth Excel
            use_layoutlm: Whether to use LayoutLM for structured extraction
        """
        self.ground_truth = pd.read_excel(ground_truth_path)
        self.use_layoutlm = use_layoutlm
        
        # Initialize PaddleOCR (better for food labels than Tesseract)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Train/test split
        self.train_data, self.test_data = self._split_data()
        
        # Build extraction patterns from training data
        self.claim_patterns = self._learn_claim_patterns()
        self.ingredient_patterns = self._learn_ingredient_patterns()
        
    def _split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        shuffled = self.ground_truth.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled) * (1 - test_size))
        return shuffled[:split_idx], shuffled[split_idx:]
    
    def _learn_claim_patterns(self) -> List[str]:
        """Learn patterns from claim_statement column"""
        claims = self.train_data['Claim_statement'].dropna()
        
        # Common claim keywords
        patterns = [
            r'\b(high|low|reduced|free|rich|source|fortified|enriched)\s+\w+',
            r'\b(natural|organic|whole|pure|real|fresh)\b',
            r'\b(vitamin|protein|fiber|calcium|iron)\b',
            r'\b(\d+%)\s+(less|more|reduced)\b',
            r'\b(no|zero)\s+(sugar|fat|calories|artificial)\b',
        ]
        return patterns
    
    def _learn_ingredient_patterns(self) -> List[str]:
        """Learn patterns for ingredient list identification"""
        patterns = [
            r'ingredients?\s*:',
            r'contains?\s*:',
            r'made with\s*:',
        ]
        return patterns
    
    def extract_text_paddleocr(self, image_path: str) -> Dict:
        """
        Extract text using PaddleOCR
        
        Returns:
            Dictionary with text and bounding boxes
        """
        result = self.ocr.ocr(image_path, cls=True)
        
        extracted = {
            'full_text': '',
            'lines': [],
            'boxes': []
        }
        
        if result and result[0]:
            for line in result[0]:
                box, (text, conf) = line
                extracted['lines'].append({
                    'text': text,
                    'confidence': conf,
                    'box': box
                })
                extracted['full_text'] += text + ' '
        
        return extracted
    
    def extract_product_name_claims(self, front_image_path: str) -> Dict:
        """
        Extract product name and claims from front of pack
        
        Args:
            front_image_path: Path to front image (productID_1)
            
        Returns:
            Dictionary with product_name and claims
        """
        ocr_result = self.extract_text_paddleocr(front_image_path)
        full_text = ocr_result['full_text']
        
        # Extract product name (usually in larger font at top)
        # Heuristic: Take first 1-2 lines with high confidence
        lines = ocr_result['lines']
        product_name = ''
        if lines:
            # Sort by y-coordinate (top to bottom)
            sorted_lines = sorted(lines, key=lambda x: x['box'][0][1])
            product_name = ' '.join([l['text'] for l in sorted_lines[:2]])
        
        # Extract claims using learned patterns
        claims = []
        for pattern in self.claim_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            claims.extend(matches)
        
        return {
            'product_name': product_name.strip(),
            'claims': list(set(claims)),  # Remove duplicates
            'full_front_text': full_text
        }
    
    def extract_ingredients(self, back_image_path: str) -> Dict:
        """
        Extract ingredient list from back of pack
        
        Args:
            back_image_path: Path to back image (productID_2 or _3)
            
        Returns:
            Dictionary with ingredient_list
        """
        ocr_result = self.extract_text_paddleocr(back_image_path)
        full_text = ocr_result['full_text']
        
        # Find ingredient section
        ingredient_list = ''
        for pattern in self.ingredient_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Extract text after "Ingredients:"
                start_idx = match.end()
                # Find next section (usually allergen info or nutrition)
                next_section = re.search(
                    r'(allergen|nutrition|may contain|storage)',
                    full_text[start_idx:],
                    re.IGNORECASE
                )
                end_idx = next_section.start() if next_section else len(full_text)
                ingredient_list = full_text[start_idx:start_idx+end_idx].strip()
                break
        
        return {
            'ingredient_list': ingredient_list,
            'full_back_text': full_text
        }
    
    def extract_nutrition_table(self, back_image_path: str) -> Dict:
        """
        Extract nutrition information from table
        
        Args:
            back_image_path: Path to back image
            
        Returns:
            Dictionary with nutrition values per 100g/100ml
        """
        ocr_result = self.extract_text_paddleocr(back_image_path)
        full_text = ocr_result['full_text']
        
        # Initialize nutrition dict
        nutrition = {
            'energy_kcal': None,
            'carbohydrates_g': None,
            'protein_g': None,
            'fat_g': None,
            'saturated_fat_g': None,
            'sugar_g': None,
            'sodium_mg': None,
            'serving_size': None
        }
        
        # Extract nutrition values using regex
        patterns = {
            'energy_kcal': r'energy[:\s]*(\d+\.?\d*)\s*kcal',
            'carbohydrates_g': r'carbohydrate[s]?[:\s]*(\d+\.?\d*)\s*g',
            'protein_g': r'protein[:\s]*(\d+\.?\d*)\s*g',
            'fat_g': r'(?:total\s+)?fat[:\s]*(\d+\.?\d*)\s*g',
            'saturated_fat_g': r'saturated[:\s]*(\d+\.?\d*)\s*g',
            'sugar_g': r'sugar[s]?[:\s]*(\d+\.?\d*)\s*g',
            'sodium_mg': r'sodium[:\s]*(\d+\.?\d*)\s*mg',
        }
        
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                try:
                    nutrition[nutrient] = float(match.group(1))
                except ValueError:
                    pass
        
        # Check if values are per 100g/100ml
        per_100 = re.search(r'per\s+100\s*[gml]', full_text, re.IGNORECASE)
        nutrition['is_per_100'] = bool(per_100)
        
        return nutrition
    
    def convert_to_per_100(self, nutrition: Dict, serving_size: float = None) -> Dict:
        """
        Convert nutrition values to per 100g/100ml
        
        Args:
            nutrition: Dictionary with nutrition values
            serving_size: Serving size in grams if known
            
        Returns:
            Converted nutrition dictionary
        """
        if nutrition.get('is_per_100'):
            return nutrition
        
        if serving_size and serving_size > 0:
            conversion_factor = 100.0 / serving_size
            
            for key in ['energy_kcal', 'carbohydrates_g', 'protein_g', 
                       'fat_g', 'saturated_fat_g', 'sugar_g', 'sodium_mg']:
                if nutrition.get(key) is not None:
                    nutrition[key] = nutrition[key] * conversion_factor
            
            nutrition['is_per_100'] = True
        
        return nutrition
    
    def extract_product_info(self, product_id: str, 
                            front_img: str,
                            back_imgs: List[str]) -> Dict:
        """
        Extract all information for a product
        
        Args:
            product_id: Product identifier
            front_img: Path to front image
            back_imgs: List of paths to back images
            
        Returns:
            Complete product information dictionary
        """
        product_info = {'product_id': product_id}
        
        # Extract from front
        if front_img:
            front_data = self.extract_product_name_claims(front_img)
            product_info.update(front_data)
        
        # Extract from back images
        all_ingredients = []
        all_nutrition = []
        
        for back_img in back_imgs:
            if not back_img:
                continue
            
            # Ingredients
            ing_data = self.extract_ingredients(back_img)
            if ing_data['ingredient_list']:
                all_ingredients.append(ing_data['ingredient_list'])
            
            # Nutrition
            nutr_data = self.extract_nutrition_table(back_img)
            all_nutrition.append(nutr_data)
        
        # Combine ingredients (take longest/most complete)
        product_info['ingredient_list'] = max(all_ingredients, 
                                             key=len, 
                                             default='')
        
        # Combine nutrition (take most complete)
        combined_nutrition = {}
        for nutr in all_nutrition:
            for key, value in nutr.items():
                if value is not None and combined_nutrition.get(key) is None:
                    combined_nutrition[key] = value
        
        product_info['nutrition'] = combined_nutrition
        
        return product_info
    
    def batch_extract(self, processed_images_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from all products
        
        Args:
            processed_images_df: DataFrame with processed image paths
            
        Returns:
            DataFrame with extracted information
        """
        results = []
        
        for idx, row in processed_images_df.iterrows():
            print(f"Extracting {idx+1}/{len(processed_images_df)}: {row['Product_ID']}")
            
            back_imgs = [
                row.get('back_image_1'),
                row.get('back_image_2')
            ]
            back_imgs = [img for img in back_imgs if img and pd.notna(img)]
            
            try:
                info = self.extract_product_info(
                    product_id=row['Product_ID'],
                    front_img=row.get('front_image'),
                    back_imgs=back_imgs
                )
                results.append(info)
            except Exception as e:
                print(f"Error extracting {row['Product_ID']}: {e}")
                results.append({'product_id': row['Product_ID'], 'error': str(e)})
        
        return pd.DataFrame(results)


# Usage example
if __name__ == "__main__":
    # Load processed images
    processed_df = pd.read_csv("processed_images_log.csv")
    
    # Initialize extractor
    extractor = MultimodalExtractor(
        ground_truth_path="./ground_truth.xlsx",
        use_layoutlm=False
    )
    
    # Extract information
    extracted_df = extractor.batch_extract(processed_df)
    
    # Save results
    extracted_df.to_json("extracted_data.json", orient='records', indent=2)
    print(f"Extracted data for {len(extracted_df)} products")
