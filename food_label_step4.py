"""
Step 4: Nutrient Profiling based on WHO African Region thresholds
Determines Codex Category and checks nutrient thresholds
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, classification_report
import json

class NutrientProfiler:
    def __init__(self, ground_truth_path: str):
        """
        Initialize Nutrient Profiler
        
        Args:
            ground_truth_path: Path to ground truth Excel
        """
        self.ground_truth = pd.read_excel(ground_truth_path)
        
        # Build Codex category mapping
        self._build_codex_categories()
        
        # Build WHO African Region thresholds
        self._build_who_thresholds()
        
        # Train/test split
        self.train_data, self.test_data = self._split_data()
    
    def _split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        shuffled = self.ground_truth.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled) * (1 - test_size))
        return shuffled[:split_idx], shuffled[split_idx:]
    
    def _build_codex_categories(self):
        """
        Build Codex category mapping based on FAO GSFA
        Reference: https://www.fao.org/gsfaonline/foods/index.html
        """
        self.codex_patterns = {
            '5.1-5.4_chocolate_confectionery': {
                'keywords': [
                    r'\bchocolate\b', r'\bcocoa\b', r'\bcandy\b', r'\bcandie[s]?\b',
                    r'\btoffee\b', r'\bfudge\b', r'\bcaramel\b', r'\bgum\b',
                    r'\bconfection\b', r'\bsweet[s]?\b'
                ],
                'category': '5.1-5.4'
            },
            '7.2_cakes_biscuits': {
                'keywords': [
                    r'\bcake[s]?\b', r'\bpastry\b', r'\bpastr[y|ies]\b',
                    r'\bcookie[s]?\b', r'\bbiscuit[s]?\b', r'\bdoughnut[s]?\b',
                    r'\bmuffin[s]?\b', r'\bbrownie[s]?\b', r'\bwafer[s]?\b'
                ],
                'category': '7.2'
            },
            '7.1_bread_products': {
                'keywords': [
                    r'\bbread[s]?\b', r'\broll[s]?\b', r'\bbaguette\b',
                    r'\bcracker[s]?\b', r'\bcrispbread\b', r'\bpita\b',
                    r'\bbagel[s]?\b', r'\btortilla\b'
                ],
                'category': '7.1'
            },
            '6.1_6.3_6.7_breakfast_cereals': {
                'keywords': [
                    r'\boat[s]?\b', r'\bmuesli\b', r'\bgranola\b', r'\bcereal[s]?\b',
                    r'\bporridge\b', r'\bcorn\s+flake[s]?\b', r'\brice\s+cake[s]?\b',
                    r'\bwheat\s+flake[s]?\b', r'\bbran\b'
                ],
                'category': '6.1/6.3/6.7'
            },
            '15.1_savory_snacks': {
                'keywords': [
                    r'\bchip[s]?\b', r'\bcrisp[s]?\b', r'\bpopcorn\b',
                    r'\bpretzels?\b', r'\bsnack[s]?\b', r'\bnachos?\b',
                    r'\btortilla\s+chip[s]?\b', r'\bcorn\s+chip[s]?\b'
                ],
                'category': '15.1'
            },
            '14.1.4_fruit_juices': {
                'keywords': [
                    r'\bjuice[s]?\b', r'\bsmoothie[s]?\b', r'\bnectar\b',
                    r'\bfruit\s+drink\b'
                ],
                'category': '14.1.4'
            },
            '1.1_1.2_dairy_milk': {
                'keywords': [
                    r'\bmilk\b', r'\bcream\b', r'\byogu?h?rt\b',
                    r'\bdairy\b', r'\bcheese\b', r'\bbutter\b'
                ],
                'category': '1.1/1.2'
            },
            '5.3_desserts': {
                'keywords': [
                    r'\bdessert[s]?\b', r'\bpudding[s]?\b', r'\bice\s+cream\b',
                    r'\bgelato\b', r'\bsorbet\b', r'\bmousse\b'
                ],
                'category': '5.3'
            },
            '9_fish_meat': {
                'keywords': [
                    r'\bfish\b', r'\bmeat\b', r'\bchicken\b', r'\bbeef\b',
                    r'\bpork\b', r'\bsausage[s]?\b', r'\bham\b'
                ],
                'category': '9'
            },
            '4_fruits_vegetables': {
                'keywords': [
                    r'\bfruit[s]?\b', r'\bvegetable[s]?\b', r'\bsalad\b',
                    r'\btomato\b', r'\bapple\b', r'\bbanana\b'
                ],
                'category': '4'
            },
        }
    
    def _build_who_thresholds(self):
        """
        Build WHO African Region nutrient thresholds
        Reference: WHO African Region nutrient profiling model
        """
        # Thresholds per 100g/100ml
        self.who_thresholds = {
            '5.1-5.4': {  # Chocolate/confectionery
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0,
                'sodium_mg': 120.0
            },
            '7.2': {  # Cakes/biscuits
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0,
                'sodium_mg': 120.0
            },
            '7.1': {  # Bread products
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0,
                'sodium_mg': 120.0
            },
            '6.1/6.3/6.7': {  # Breakfast cereals
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0,
                'sodium_mg': 120.0
            },
            '15.1': {  # Savory snacks
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0,
                'sodium_mg': 400.0  # Higher for savory
            },
            '14.1.4': {  # Fruit juices
                'total_sugar_g': 6.0,
                'energy_kcal': 40.0
            },
            '1.1/1.2': {  # Dairy/milk
                'total_fat_g': 3.0,
                'saturated_fat_g': 2.0,
                'total_sugar_g': 6.0
            },
            '5.3': {  # Desserts
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0
            },
            'default': {  # Default thresholds
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'total_sugar_g': 6.0,
                'sodium_mg': 120.0
            }
        }
    
    def determine_codex_category(self, product_name: str, 
                                 ingredient_list: str) -> Dict:
        """
        Determine Codex category from product name and ingredients
        
        Args:
            product_name: Product name/description
            ingredient_list: Ingredient list
            
        Returns:
            Dictionary with category and confidence
        """
        combined_text = f"{product_name} {ingredient_list}".lower()
        
        scores = {}
        for cat_name, cat_info in self.codex_patterns.items():
            score = 0
            matched_keywords = []
            
            for pattern in cat_info['keywords']:
                matches = re.findall(pattern, combined_text)
                if matches:
                    score += len(matches)
                    matched_keywords.extend(matches)
            
            if score > 0:
                scores[cat_name] = {
                    'score': score,
                    'category': cat_info['category'],
                    'keywords': matched_keywords
                }
        
        if not scores:
            return {
                'codex_category': 'Unknown',
                'confidence': 0.0,
                'matched_keywords': []
            }
        
        # Get category with highest score
        best_match = max(scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'codex_category': best_match[1]['category'],
            'confidence': min(best_match[1]['score'] / 3.0, 1.0),
            'matched_keywords': best_match[1]['keywords'][:3]
        }
    
    def check_nutrient_thresholds(self, nutrition: Dict, 
                                  codex_category: str) -> Dict:
        """
        Check if nutrients exceed WHO thresholds
        
        Args:
            nutrition: Dictionary with nutrition values per 100g/100ml
            codex_category: Codex category
            
        Returns:
            Dictionary with threshold check results
        """
        # Get thresholds for category
        thresholds = self.who_thresholds.get(
            codex_category,
            self.who_thresholds['default']
        )
        
        exceeded = []
        missing = []
        
        # Map nutrition keys to threshold keys
        nutrient_mapping = {
            'fat_g': 'total_fat_g',
            'saturated_fat_g': 'saturated_fat_g',
            'sugar_g': 'total_sugar_g',
            'sodium_mg': 'sodium_mg',
            'energy_kcal': 'energy_kcal'
        }
        
        for nutr_key, thresh_key in nutrient_mapping.items():
            if thresh_key in thresholds:
                threshold_value = thresholds[thresh_key]
                nutrition_value = nutrition.get(nutr_key)
                
                if nutrition_value is None or pd.isna(nutrition_value):
                    missing.append(nutr_key)
                elif nutrition_value > threshold_value:
                    exceeded.append({
                        'nutrient': nutr_key,
                        'value': nutrition_value,
                        'threshold': threshold_value,
                        'excess': nutrition_value - threshold_value
                    })
        
        # Format result
        if exceeded:
            exceeded_names = [e['nutrient'] for e in exceeded]
            result = ', '.join(exceeded_names) + ' exceeded'
        elif missing:
            result = f"Missing: {', '.join(missing)}"
        else:
            result = 'None'
        
        return {
            'nutrient_profile': result,
            'exceeded_details': exceeded,
            'missing_nutrients': missing,
            'compliant': len(exceeded) == 0 and len(missing) == 0
        }
    
    def profile_product(self, product_data: Dict) -> Dict:
        """
        Complete nutrient profiling for a product
        
        Args:
            product_data: Dictionary with product information
            
        Returns:
            Dictionary with profiling results
        """
        product_name = product_data.get('product_name', '')
        ingredient_list = product_data.get('ingredient_list', '')
        nutrition = product_data.get('nutrition', {})
        
        # Step 4.1: Determine Codex category
        codex_result = self.determine_codex_category(product_name, ingredient_list)
        
        # Step 4.2: Check nutrient thresholds
        threshold_result = self.check_nutrient_thresholds(
            nutrition,
            codex_result['codex_category']
        )
        
        return {
            'product_id': product_data.get('product_id'),
            'codex_category': codex_result['codex_category'],
            'codex_confidence': codex_result['confidence'],
            'nutrient_profile': threshold_result['nutrient_profile'],
            'compliant': threshold_result['compliant'],
            'exceeded_details': threshold_result['exceeded_details'],
            'missing_nutrients': threshold_result['missing_nutrients']
        }
    
    def batch_profile(self, extracted_data: pd.DataFrame) -> pd.DataFrame:
        """
        Profile all products in batch
        
        Args:
            extracted_data: DataFrame with extracted product information
            
        Returns:
            DataFrame with profiling results
        """
        results = []
        
        for idx, row in extracted_data.iterrows():
            product_data = row.to_dict()
            
            try:
                profile = self.profile_product(product_data)
                results.append(profile)
            except Exception as e:
                print(f"Error profiling {row.get('product_id')}: {e}")
                results.append({
                    'product_id': row.get('product_id'),
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def evaluate_on_test_set(self) -> Dict:
        """
        Evaluate profiler on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        codex_predictions = []
        codex_actuals = []
        profile_predictions = []
        profile_actuals = []
        
        for idx, row in self.test_data.iterrows():
            product_name = row.get('Claim_statement', '')
            ingredient_list = row.get('List_of_Ingredients', '')
            
            # Build nutrition dict from row
            nutrition = {
                'energy_kcal': row.get('Energy_kcal'),
                'carbohydrates_g': row.get('Carbohydrates_g'),
                'protein_g': row.get('Protein_g'),
                'fat_g': row.get('Fat_g'),
                'saturated_fat_g': row.get('Saturated_Fat_g'),
                'sugar_g': row.get('Sugar_g'),
                'sodium_mg': row.get('Sodium_mg')
            }
            
            product_data = {
                'product_id': row.get('Product_ID'),
                'product_name': product_name,
                'ingredient_list': ingredient_list,
                'nutrition': nutrition
            }
            
            # Predict
            profile = self.profile_product(product_data)
            
            # Codex category evaluation
            actual_codex = row.get('Codex_Category')
            if actual_codex and pd.notna(actual_codex):
                codex_predictions.append(profile['codex_category'])
                codex_actuals.append(actual_codex)
            
            # Nutrient profile evaluation
            actual_profile = row.get('Nutrient_Profile')
            if actual_profile and pd.notna(actual_profile):
                profile_predictions.append(profile['nutrient_profile'])
                profile_actuals.append(actual_profile)
        
        results = {}
        
        # Codex accuracy
        if codex_actuals:
            results['codex_accuracy'] = accuracy_score(codex_actuals, codex_predictions)
            results['codex_samples'] = len(codex_actuals)
        
        # Nutrient profile accuracy
        if profile_actuals:
            results['profile_accuracy'] = accuracy_score(profile_actuals, profile_predictions)
            results['profile_samples'] = len(profile_actuals)
        
        return results


# Usage example
if __name__ == "__main__":
    # Load extracted data
    with open("extracted_data.json", 'r') as f:
        extracted_data = pd.DataFrame(json.load(f))
    
    # Initialize profiler
    profiler = NutrientProfiler(ground_truth_path="./ground_truth.xlsx")
    
    # Profile all products
    profile_results = profiler.batch_profile(extracted_data)
    
    # Save results
    profile_results.to_csv("nutrient_profiles.csv", index=False)
    
    # Evaluate on test set
    eval_results = profiler.evaluate_on_test_set()
    print("\nNutrient Profiling Evaluation:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")
    
    # Save evaluation
    with open("nutrient_evaluation.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
