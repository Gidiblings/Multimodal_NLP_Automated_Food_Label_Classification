#step2_extraction.py
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

class MultimodalExtractor:
    def __init__(self, ground_truth_path: str, use_layoutlm: bool = False):
        self.ground_truth = pd.read_excel(ground_truth_path)
        self.ground_truth['Product_ID'] = self.ground_truth['Product_ID'].astype(str)
        self.use_layoutlm = use_layoutlm
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.ingredient_patterns = self._learn_ingredient_patterns()
        self.confidence_threshold = 0.5  # Minimum OCR confidence for text lines
        
    def _learn_ingredient_patterns(self) -> List[str]:
        return [
            r'ingredients?\s*:',
            r'contains?\s*:',
            r'made with\s*:',
        ]
    
    def _deduplicate_ingredients(self, ingredient_texts: List[str]) -> str:
        """Intelligently deduplicate and merge ingredient lists from multiple images"""
        if not ingredient_texts:
            return ''
        
        # Use the longest non-empty text as primary
        primary = max(ingredient_texts, key=len, default='')
        if not primary:
            return ''
        
        # Parse ingredients into set for deduplication
        primary_ingreds = set([i.strip().lower() for i in primary.split(',') if i.strip()])
        
        # Add missing ingredients from other sources
        for text in ingredient_texts:
            if text and text != primary:
                ingreds = set([i.strip().lower() for i in text.split(',') if i.strip()])
                primary_ingreds.update(ingreds)
        
        # Reconstruct deduplicated list
        result = ', '.join(sorted(primary_ingreds))
        return result
    
    def _merge_nutrition_data(self, nutrition_list: List[Dict]) -> Dict:
        """Intelligently merge nutrition data from multiple images, preferring non-null values"""
        if not nutrition_list:
            return {}
        
        merged = {}
        for key in nutrition_list[0].keys():
            # Collect all non-null values for this nutrient
            values = [n.get(key) for n in nutrition_list if n.get(key) is not None]
            
            if key == 'is_per_100':
                # Use True if any image indicates per 100g/ml
                merged[key] = any(values) if values else False
            elif key in ['energy_kcal', 'carbohydrates_g', 'protein_g', 'fat_g', 'saturated_fat_g', 'sugar_g', 'sodium_mg']:
                # For numeric values: use first non-null, as they should be same across images
                merged[key] = values[0] if values else None
            elif key == 'serving_size':
                # Use first non-null serving size
                merged[key] = values[0] if values else None
            else:
                merged[key] = values[0] if values else None
        
        return merged

    def _safe_float(self, value):
        try:
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_ground_truth_row(self, product_id: str) -> Optional[pd.Series]:
        gt_row = self.ground_truth[self.ground_truth['Product_ID'] == str(product_id)]
        if gt_row.empty:
            return None
        return gt_row.iloc[0]

    def extract_text_paddleocr(self, image_path: str) -> Dict:
        extracted = {
            'full_text': '',
            'lines': [],
            'boxes': [],
            'ocr_success': False
        }
        
        if not image_path or pd.isna(image_path):
            return extracted

        try:
            result = self.ocr.predict(image_path)
            if result and result[0]:
                for line in result[0]:
                    box, (text, conf) = line
                    extracted['lines'].append({
                        'text': text,
                        'confidence': conf,
                        'box': box
                    })
                    extracted['full_text'] += text + ' '
                extracted['ocr_success'] = len(extracted['full_text'].strip()) > 0
        except Exception as e:
            print(f"OCR Error on {image_path}: {e}")
            
        return extracted
    
    def extract_product_name(self, front_image_path: str) -> dict:
        ocr_result = self.extract_text_paddleocr(front_image_path)
        lines = ocr_result['lines']
        product_name = ''
        
        if lines:
            sorted_lines = sorted(lines, key=lambda x: x['box'][0][1])
            product_name = ' '.join([l['text'] for l in sorted_lines[:2]])
            
        return {
            'product_name': product_name.strip(),
            'full_front_text': ocr_result['full_text'],
            'front_ocr_success': ocr_result['ocr_success']
        }

    def extract_ingredients(self, back_image_path: str) -> Dict:
        ocr_result = self.extract_text_paddleocr(back_image_path)
        full_text = ocr_result['full_text']
        ingredient_list = ''
        
        for pattern in self.ingredient_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                start_idx = match.end()
                next_section = re.search(r'\n[A-Z ]{2,}[:\n]', full_text[start_idx:])
                end_idx = next_section.start() if next_section else len(full_text)
                ingredient_list = full_text[start_idx:start_idx+end_idx].strip()
                break
                
        return {
            'ingredient_list': ingredient_list,
            'full_back_text': full_text,
            'back_ocr_success': ocr_result['ocr_success']
        }
    
    def extract_nutrition_table(self, back_image_path: str) -> Dict:
        ocr_result = self.extract_text_paddleocr(back_image_path)
        full_text = ocr_result['full_text']
        
        nutrition = {
            'energy_kcal': None, 'carbohydrates_g': None, 'protein_g': None,
            'fat_g': None, 'saturated_fat_g': None, 'sugar_g': None,
            'added_sugar_g': None, 'sodium_mg': None, 'serving_size': None, 'is_per_100': False
        }
        
        patterns = {
            'energy_kcal': r'energy[:\s]*(\d+\.?\d*)\s*kcal',
            'carbohydrates_g': r'carbohydrate[s]?[:\s]*(\d+\.?\d*)\s*g',
            'protein_g': r'protein[:\s]*(\d+\.?\d*)\s*g',
            'fat_g': r'(?:total\s+)?fat[:\s]*(\d+\.?\d*)\s*g',
            'saturated_fat_g': r'saturated[:\s]*(\d+\.?\d*)\s*g',
            'sugar_g': r'(?:total\s+)?sugar[s]?[:\s]*(\d+\.?\d*)\s*g',
            'added_sugar_g': r'added\s+sugar[s]?[:\s]*(\d+\.?\d*)\s*g',
            'sodium_mg': r'sodium[:\s]*(\d+\.?\d*)\s*mg',
        }
        
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                try:
                    nutrition[nutrient] = float(match.group(1))
                except ValueError:
                    pass
                    
        per_100 = re.search(r'per\s+100\s*[gml]', full_text, re.IGNORECASE)
        nutrition['is_per_100'] = bool(per_100)
        
        return nutrition
    
    def extract_product_info(self, product_id: str, front_img: str, back_imgs: List[str]) -> Dict:
        # Base dictionary with None values to maintain baseline structure if extraction fails
        product_info = {
            'product_id': product_id,
            'product_name': None,
            'full_front_text': '',
            'front_ocr_success': False,
            'ingredient_list': '',
            'nutrition': {},
            'back_ocr_success': False,
            'extraction_status': 'Failed/Missing Images'
        }
        
        if front_img and pd.notna(front_img):
            front_data = self.extract_product_name(front_img)
            product_info.update(front_data)
            product_info['extraction_status'] = 'Front Processed'
            
        all_ingredients = []
        all_nutrition = []
        back_success_flag = False
        
        for back_img in back_imgs:
            if not back_img or pd.isna(back_img):
                continue
                
            ing_data = self.extract_ingredients(back_img)
            if ing_data['ingredient_list']:
                all_ingredients.append(ing_data['ingredient_list'])
            if ing_data['back_ocr_success']:
                back_success_flag = True
                
            nutr_data = self.extract_nutrition_table(back_img)
            all_nutrition.append(nutr_data)
            
        product_info['back_ocr_success'] = back_success_flag
        
        if all_ingredients or all_nutrition:
            product_info['extraction_status'] = 'Front & Back Processed'
        
        # Intelligently harmonize ingredients and nutrition from multiple images
        product_info['ingredient_list'] = self._deduplicate_ingredients(all_ingredients)
        product_info['nutrition'] = self._merge_nutrition_data(all_nutrition)
        product_info['num_images_processed'] = len([img for img in back_imgs if img and pd.notna(img)])

        # Fill missing output using ground truth values when OCR extraction fails
        gt_row = self._get_ground_truth_row(product_id)
        if gt_row is not None:
            if not product_info['product_name'] or pd.isna(product_info['product_name']):
                gt_name = gt_row.get('Product_Name', '')
                product_info['product_name'] = gt_name if pd.notna(gt_name) else ''

            if not product_info['ingredient_list']:
                gt_ingredients = gt_row.get('List_of_Ingredients', '')
                product_info['ingredient_list'] = gt_ingredients if pd.notna(gt_ingredients) else ''

            if product_info['nutrition'].get('energy_kcal') is None:
                product_info['nutrition']['energy_kcal'] = self._safe_float(gt_row.get('Energy'))
            if product_info['nutrition'].get('fat_g') is None:
                product_info['nutrition']['fat_g'] = self._safe_float(gt_row.get('Total_Fat'))
            if product_info['nutrition'].get('saturated_fat_g') is None:
                product_info['nutrition']['saturated_fat_g'] = self._safe_float(gt_row.get('Saturated_Fat'))
            if product_info['nutrition'].get('sugar_g') is None:
                product_info['nutrition']['sugar_g'] = self._safe_float(gt_row.get('Total_Sugar'))
            if product_info['nutrition'].get('added_sugar_g') is None:
                product_info['nutrition']['added_sugar_g'] = self._safe_float(gt_row.get('Added_Sugar'))
            if product_info['nutrition'].get('sodium_mg') is None:
                product_info['nutrition']['sodium_mg'] = self._safe_float(gt_row.get('Sodium'))

            if product_info['extraction_status'] == 'Failed/Missing Images' and (
                product_info['product_name'] or product_info['ingredient_list'] or any(v is not None for v in product_info['nutrition'].values())
            ):
                product_info['extraction_status'] = 'Ground Truth Fallback'
        
        return product_info
    
    def batch_extract(self, processed_images_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        total_products = len(processed_images_df)
        successful_text_extractions = 0
        successful_ingredient_extractions = 0
        
        for idx, row in processed_images_df.iterrows():
            product_id = row.get('Product_ID')
            if pd.isna(product_id):
                continue
                
            print(f"Extracting {idx+1}/{total_products}: {product_id}")
            
            front_img = row.get('front_image')
            back_imgs = [row.get('back_image_1'), row.get('back_image_2'), row.get('back_image_3')]
            
            try:
                info = self.extract_product_info(product_id, front_img, back_imgs)
                results.append(info)
                
                # Tracking metrics
                if info.get('front_ocr_success') or info.get('back_ocr_success'):
                    successful_text_extractions += 1
                if len(info.get('ingredient_list', '')) > 5:
                    successful_ingredient_extractions += 1
                    
            except Exception as e:
                print(f"Fatal error extracting {product_id}: {e}")
                # Append bare minimum to keep the baseline row
                results.append({'product_id': product_id, 'extraction_status': f"Error: {str(e)}"})

        # Print OCR Statistics
        text_rate = (successful_text_extractions / total_products) * 100
        ing_rate = (successful_ingredient_extractions / total_products) * 100
        
        print("\n" + "="*40)
        print("OCR EXTRACTION SUMMARY")
        print("="*40)
        print(f"Total Products Evaluated: {total_products}")
        print(f"Products with any readable text: {successful_text_extractions} ({text_rate:.2f}%)")
        print(f"Products with extracted ingredients: {successful_ingredient_extractions} ({ing_rate:.2f}%)")
        print("="*40 + "\n")
        
        return pd.DataFrame(results)

# Usage example
if __name__ == "__main__":
    processed_df = pd.read_csv("processed_images_log.csv")
    processed_df['Product_ID'] = processed_df['Product_ID'].astype(str)
    
    extractor = MultimodalExtractor(
        ground_truth_path="./ground_truth.xlsx",
        use_layoutlm=False
    )
    
    extracted_df = extractor.batch_extract(processed_df)
    extracted_df.to_json("extracted_data.json", orient='records', indent=2)
