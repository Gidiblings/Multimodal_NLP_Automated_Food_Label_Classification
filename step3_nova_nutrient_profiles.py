# step3_nova_nutrient_profiles.py
"""
Step 3: NOVA Classification based on ingredient processing level
Classifies foods as Minimally Processed, Processed, or Ultra Processed
"""

import pandas as pd
from thefuzz import fuzz
import re
from typing import Dict, List, Tuple

class NOVAClassifier:
    def __init__(self, fuzzy_threshold: int = 85, ground_truth_path: str = "./ground_truth.xlsx"):
        self.fuzzy_threshold = fuzzy_threshold
        self.ground_truth = pd.read_excel(ground_truth_path, dtype=str).fillna('')
        self.ground_truth['Product_ID'] = self.ground_truth['Product_ID'].astype(str)
        self._build_nova_dictionaries()
        self._build_who_thresholds()

    def _build_nova_dictionaries(self):
        # Converted from Regex to plain text for Fuzzy Matching
        self.minimally_processed = [
            # whole foods
            "whole", "fresh", "frozen", "dried",
            "raw", "plain", "unsweetened", "unflavoured", "unflavored",

            # grains & staples
            "whole grain", "wholegrain", "whole wheat", "wholemeal",
            "rolled oats", "steel cut oats", "brown rice", "quinoa", "millet", "barley",

            # animal products
            "fresh milk", "pasteurized milk", "egg", "fresh meat", "fish",

            # fermentation (traditional)
            "live cultures", "active cultures", "lactobacillus"
        ]
        self.culinary_ingredients = [
            # sugars
            "sugar", "brown sugar", "cane sugar", "raw sugar",
            "honey", "maple syrup", "molasses",

            # fats & oils
            "vegetable oil", "olive oil", "palm oil", "sunflower oil",
            "butter", "lard",

            # salts
            "salt", "sea salt", "himalayan salt"
        ]
        self.processed_foods = [
            # preservation methods
            "canned", "jarred", "bottled", "smoked", "cured", "pickled",

            # traditional processing
            "cheese", "bread", "yogurt",

            # ingredients used in simple processing
            "barley malt extract", "malt extract", "vinegar",

            # powdered but still simple
            "milk powder", "dried milk",

            # fortification/enrichment signals
            "vitamin", "vitamins", "fortified", "enriched"
        ]
        # Group 4 overrides all others
        self.ultra_processed = [
            # industrial additives
            "bha", "bht", "tbhq",
            "sodium nitrite", "sodium nitrate",
            "potassium sorbate", "sodium benzoate", "sulfite",

            # artificial sweeteners
            "aspartame", "sucralose", "acesulfame k", "saccharin",

            # flavor enhancers
            "monosodium glutamate", "msg", "disodium inosinate", "disodium guanylate",

            # emulsifiers
            "monoglycerides", "diglycerides", "polysorbate", "soy lecithin",

            # thickeners/gums
            "xanthan gum", "guar gum", "carrageenan", "cellulose gum",

            # industrial carbs
            "maltodextrin", "modified starch", "modified corn starch",

            # reconstructed ingredients
            "protein isolate", "soy protein isolate", "whey protein isolate",

            # sensory manipulation
            "artificial flavor", "artificial colour", "artificial color",
            "caramel color", "food coloring", "tartrazine",

            # ambiguous but suspicious
            "flavoring", "colour", "color", "emulsifier", "stabilizer", "thickener"
        ]

        # UPF pattern detection for chemical/compound names that indicate industrial processing
        self.upf_patterns = [
            r"\b\w+ate\b",
            r"\b\w+ite\b",
            r"\b\w+acid\b",
            r"\bsodium\b",
            r"\bpotassium\b",
            r"\bcalcium\b",
            r"\bammonium\b",
            r"\bmono[- ]?glyceride\b",
            r"\bdi[- ]?glyceride\b",
            r"\bmodified\b",
            r"\bhydrogenated\b",
            r"\bemulsifier\b",
            r"\bstabilizer\b",
            r"\bthickener\b",
            r"\bacidity regulator\b"
        ]

    def _build_who_thresholds(self):
        self.who_thresholds = {
            '5.1-5.4': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'sugar_g': 6.0,
                'sodium_mg': 120.0
            },

            '7.2': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'sugar_g': 6.0,
                'sodium_mg': 120.0
            },

            '7.1': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'sugar_g': 6.0,
                'sodium_mg': 120.0
            },

            '6.1/6.3/6.7': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'sugar_g': 6.0,
                'sodium_mg': 120.0
            },

            '15.1': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'sugar_g': 6.0,
                'sodium_mg': 400.0
            },

            '15.2': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 3.0,
                'sodium_mg': 400.0
            },

            '14.1.4': {
                'sugar_g': 6.0
            },

            '14.1.2/14.1.3': {
                'sugar_g': 6.0
            },

            '1.1/1.2': {
                'total_fat_g': 3.0,
                'saturated_fat_g': 2.0,
                'sugar_g': 6.0
            },

            '9': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 3.0,
                'sodium_mg': 400.0
            },

            '12.6/12.9.2': {
                'total_fat_g': 8.0,
                'added_sugar_g': 0.0,
                'sodium_mg': 300.0
            },

            'default': {
                'total_fat_g': 8.0,
                'saturated_fat_g': 4.0,
                'sugar_g': 6.0,
                'sodium_mg': 120.0
            }
        }

    def fuzzy_match(self, text: str, target_list: List[str]) -> Tuple[bool, List[str]]:
        if not text or pd.isna(text): return False, []
        text_lower = str(text).lower()
        matches = []
        for target in target_list:
            if fuzz.partial_ratio(target, text_lower) >= self.fuzzy_threshold:
                matches.append(target)
        return len(matches) > 0, matches

    def has_upf_pattern(self, ingredients: str) -> bool:
        if not ingredients or pd.isna(ingredients):
            return False
        ingredients_lower = str(ingredients).lower()
        for pattern in self.upf_patterns:
            if re.search(pattern, ingredients_lower):
                return True
        return False

    def classify_nova(self, ingredients: str) -> str:
        if not ingredients or pd.isna(ingredients):
            return "Minimally Processed"

        ingredients = str(ingredients).lower()
        ing_list = [i.strip() for i in ingredients.split(",") if i.strip()]
        n_ing = len(ing_list)

        # Priority 4: Ultra Processed
        is_upf, _ = self.fuzzy_match(ingredients, self.ultra_processed)
        if is_upf or self.has_upf_pattern(ingredients):
            return "Ultra Processed"
        
        # Priority 3: Processed
        is_processed, _ = self.fuzzy_match(ingredients, self.processed_foods)
        if is_processed:
            return "Processed"
        
        # Priority 2: Culinary (≤2 ingredients only)
        is_culinary, _ = self.fuzzy_match(ingredients, self.culinary_ingredients)
        if is_culinary:
            if n_ing <= 2:
                return "Processed Culinary Ingredients"
            return "Processed"
        
        # Priority 1: Minimally Processed
        is_minimal, _ = self.fuzzy_match(ingredients, self.minimally_processed)
        if is_minimal:
            return "Minimally Processed"
        
        # Default to minimally processed when no rule matches
        return "Minimally Processed"

    def get_codex_from_ground_truth(self, product_id: str) -> str:
        gt_row = self.ground_truth[self.ground_truth['Product_ID'] == str(product_id)]
        if gt_row.empty:
            return 'default'
        codex = gt_row['Codex_Category'].iloc[0]
        return codex if pd.notna(codex) and codex.strip() else 'default'

    def profile_nutrients(self, nutrition: Dict, codex: str) -> Dict:
        thresholds = self.who_thresholds.get(codex, self.who_thresholds['default'])
        profile = {
            'high_sodium': False,
            'high_sugar': False,
            'high_fat': False,
            'high_sat_fat': False,
            'high_added_sugar': False,
            'high_energy': False
        }
        
        if not nutrition: 
            return profile
        
        sodium_value = nutrition.get('sodium_mg')
        sugar_value = nutrition.get('sugar_g')
        fat_value = nutrition.get('fat_g', nutrition.get('total_fat_g'))
        sat_fat_value = nutrition.get('saturated_fat_g')
        added_sugar_value = nutrition.get('added_sugar_g')
        energy_value = nutrition.get('energy_kcal')

        if sodium_value is not None and 'sodium_mg' in thresholds:
            profile['high_sodium'] = float(sodium_value) > thresholds['sodium_mg']
        if sugar_value is not None and 'sugar_g' in thresholds:
            profile['high_sugar'] = float(sugar_value) > thresholds['sugar_g']
        if fat_value is not None and 'total_fat_g' in thresholds:
            profile['high_fat'] = float(fat_value) > thresholds['total_fat_g']
        if sat_fat_value is not None and 'saturated_fat_g' in thresholds:
            profile['high_sat_fat'] = float(sat_fat_value) > thresholds['saturated_fat_g']
        if added_sugar_value is not None and 'added_sugar_g' in thresholds:
            profile['high_added_sugar'] = float(added_sugar_value) > thresholds['added_sugar_g']
        if energy_value is not None and 'energy_kcal' in thresholds:
            profile['high_energy'] = float(energy_value) > thresholds['energy_kcal']
            
        return profile

    def batch_classify(self, extracted_data: List[Dict]) -> pd.DataFrame:
        results = []
        for item in extracted_data:
            product_id = item.get('product_id')
            ing_text = item.get('ingredient_list', '')
            nutrition = item.get('nutrition', {})
            
            nova = self.classify_nova(ing_text)
            codex = self.get_codex_from_ground_truth(product_id)
            nutr_profile = self.profile_nutrients(nutrition, codex)
            
            exceeded = [k for k, v in nutr_profile.items() if v]
            nutrient_profile_pred = str(len(exceeded))
            
            results.append({
                'product_id': product_id,
                'NOVA_Predicted': nova,
                'nutrient_profile': nutrient_profile_pred,
                'High_Sodium_Predicted': nutr_profile['high_sodium'],
                'High_Sugar_Predicted': nutr_profile['high_sugar'],
                'High_Fat_Predicted': nutr_profile['high_fat'],
                'High_Sat_Fat_Predicted': nutr_profile['high_sat_fat'],
                'High_Added_Sugar_Predicted': nutr_profile['high_added_sugar'],
                'High_Energy_Predicted': nutr_profile['high_energy']
            })
        return pd.DataFrame(results)

# Execution block
if __name__ == "__main__":
    import json
    with open("extracted_data.json", "r") as f:
        data = json.load(f)
    
    classifier = NOVAClassifier()
    df = classifier.batch_classify(data)
    df.to_csv("step3_comprehensive_classification.csv", index=False)
    df.to_csv("nova_classifications.csv", index=False)
    print("Step 3 Complete. Robust classifications generated.")