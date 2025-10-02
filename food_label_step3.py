"""
Step 3: NOVA Classification based on ingredient processing level
Classifies foods as Minimally Processed, Processed, or Ultra Processed
"""

import pandas as pd
import re
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class NOVAClassifier:
    def __init__(self, ground_truth_path: str):
        """
        Initialize NOVA classifier
        
        Args:
            ground_truth_path: Path to ground truth Excel
        """
        self.ground_truth = pd.read_excel(ground_truth_path)
        
        # Define NOVA classification dictionaries
        self._build_nova_dictionaries()
        
        # Train/test split
        self.train_data, self.test_data = self._split_data()
    
    def _split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        shuffled = self.ground_truth.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled) * (1 - test_size))
        return shuffled[:split_idx], shuffled[split_idx:]
    
    def _build_nova_dictionaries(self):
        """
        Build dictionaries for NOVA classification based on NOVA document
        Reference: https://doi.org/10.1017/S1368980018003762
        """
        
        # Group 1: Minimally Processed Foods
        self.minimally_processed = {
            'natural_ingredients': [
                r'\b100%\s+oat[s]?\b',
                r'\bwhole\s+grain[s]?\b',
                r'\bwhole\s+wheat\b',
                r'\bwhole\s+oat[s]?\b',
                r'\brolled\s+oat[s]?\b',
                r'\bbrown\s+rice\b',
                r'\bquinoa\b',
                r'\bmillet\b',
            ],
            'starter_cultures': [
                r'\blactobacillus\s+bulgaricus\b',
                r'\bstreptococcus\s+thermophilus\b',
                r'\blive\s+cultures\b',
                r'\bactive\s+cultures\b',
                r'\byogurt\s+cultures\b',
            ],
            'basic_ingredients': [
                r'\bwater\b',
                r'\bsea\s+salt\b',
                r'\bhimalayan\s+salt\b',
                r'\bnatural\s+flavou?r[s]?\b',
            ]
        }
        
        # Group 2: Processed Culinary Ingredients (when appearing ALONE)
        self.culinary_ingredients = [
            r'^\s*sugar\s*$',
            r'^\s*vegetable\s+oil\s*$',
            r'^\s*salt\s*$',
            r'^\s*butter\s*$',
            r'^\s*lard\s*$',
        ]
        
        # Group 3: Processed Foods
        self.processed_foods = [
            r'\bbarley\s+malt\s+extract\b',
            r'\bmalt\s+extract\b',
            r'\bcanned\s+\w+\b',
            r'\bsmoked\s+\w+\b',
            r'\bcured\s+\w+\b',
        ]
        
        # Group 4: Ultra Processed Foods (ANY presence overrides all others)
        self.ultra_processed = {
            'antioxidants': [
                r'\bascorbyl\s+palmitate\b',
                r'\bbha\b',
                r'\bbht\b',
                r'\btbhq\b',
                r'\btocopherol[s]?\b',
                r'\be\s*300\b',
                r'\be\s*304\b',
                r'\be\s*306\b',
                r'\be\s*307\b',
            ],
            'colors': [
                r'\be\s*1\d{2}\b',  # E100-E199
                r'\bcaramel\s+colou?r\b',
                r'\bartificial\s+colou?r[s]?\b',
                r'\bfood\s+colou?ring\b',
                r'\btartrazine\b',
                r'\bsunset\s+yellow\b',
            ],
            'emulsifiers': [
                r'\bsoy\s+lecithin\b',
                r'\blecithin\b',
                r'\bmono[-\s]?and\s+di[-\s]?glycerides\b',
                r'\bpolysorbate\b',
                r'\be\s*4\d{2}\b',  # E400-E499
                r'\bsodium\s+stearoyl\b',
            ],
            'flavor_enhancers': [
                r'\be\s*6\d{2}\b',  # E600-E699
                r'\bmsg\b',
                r'\bmonosodium\s+glutamate\b',
                r'\byeast\s+extract\b',
                r'\bhydrolyzed\s+\w+\s+protein\b',
                r'\bdisodium\s+inosinate\b',
                r'\bdisodium\s+guanylate\b',
            ],
            'preservatives': [
                r'\bsodium\s+benzoate\b',
                r'\bpotassium\s+sorbate\b',
                r'\bsodium\s+nitrite\b',
                r'\bsulfite[s]?\b',
                r'\be\s*2\d{2}\b',  # E200-E299
                r'\bcalcium\s+propionate\b',
            ],
            'sweeteners': [
                r'\baspartame\b',
                r'\bsucralose\b',
                r'\bacesulfame[-\s]?k\b',
                r'\bsaccharin\b',
                r'\bhigh\s+fructose\s+corn\s+syrup\b',
                r'\bcorn\s+syrup\b',
                r'\bglucose[-\s]?fructose\b',
            ],
            'thickeners': [
                r'\bxanthan\s+gum\b',
                r'\bguar\s+gum\b',
                r'\bcarrageenan\b',
                r'\bmodified\s+\w+\s+starch\b',
                r'\bmodified\s+starch\b',
                r'\bcmc\b',
                r'\bcellulose\s+gum\b',
            ],
            'artificial_ingredients': [
                r'\bartificial\s+flavou?r[s]?\b',
                r'\bartificial\s+\w+\b',
                r'\bprotein\s+isolate\b',
                r'\bsoy\s+protein\s+isolate\b',
                r'\bwhey\s+protein\s+isolate\b',
                r'\bmaltodextrin\b',
            ],
        }
    
    def _check_ultra_processed(self, ingredient_text: str) -> Tuple[bool, List[str]]:
        """
        Check if ingredients contain ultra-processed indicators
        
        Args:
            ingredient_text: Ingredient list text
            
        Returns:
            Tuple of (is_ultra_processed, list_of_found_indicators)
        """
        found_indicators = []
        ingredient_lower = ingredient_text.lower()
        
        for category, patterns in self.ultra_processed.items():
            for pattern in patterns:
                matches = re.findall(pattern, ingredient_lower)
                if matches:
                    found_indicators.extend(matches)
        
        return len(found_indicators) > 0, found_indicators
    
    def _check_processed(self, ingredient_text: str) -> bool:
        """Check if ingredients indicate processed food"""
        ingredient_lower = ingredient_text.lower()
        
        for pattern in self.processed_foods:
            if re.search(pattern, ingredient_lower):
                return True
        return False
    
    def _check_culinary_ingredient(self, ingredient_text: str) -> bool:
        """Check if ingredient list contains only culinary ingredients"""
        ingredient_lower = ingredient_text.lower().strip()
        
        for pattern in self.culinary_ingredients:
            if re.match(pattern, ingredient_lower):
                return True
        return False
    
    def _check_minimally_processed(self, ingredient_text: str) -> bool:
        """Check if ingredients indicate minimally processed food"""
        ingredient_lower = ingredient_text.lower()
        
        for category, patterns in self.minimally_processed.items():
            for pattern in patterns:
                if re.search(pattern, ingredient_lower):
                    return True
        return False
    
    def classify_nova(self, ingredient_text: str) -> Dict:
        """
        Classify food based on NOVA system
        
        Args:
            ingredient_text: Ingredient list as string
            
        Returns:
            Dictionary with classification and reasoning
        """
        if not ingredient_text or pd.isna(ingredient_text) or not ingredient_text.strip():
            return {
                'nova_class': 'Unknown',
                'confidence': 0.0,
                'reasoning': 'No ingredient information available'
            }
        
        # PRIORITY 1: Check for ultra-processed indicators (overrides all)
        is_ultra, ultra_indicators = self._check_ultra_processed(ingredient_text)
        if is_ultra:
            return {
                'nova_class': 'Ultra Processed',
                'confidence': 1.0,
                'reasoning': f'Contains ultra-processed indicators: {", ".join(ultra_indicators[:5])}'
            }
        
        # Check for processed foods
        if self._check_processed(ingredient_text):
            return {
                'nova_class': 'Processed',
                'confidence': 0.8,
                'reasoning': 'Contains processed food indicators'
            }
        
        # Check for culinary ingredients only
        if self._check_culinary_ingredient(ingredient_text):
            return {
                'nova_class': 'Processed Culinary Ingredient',
                'confidence': 0.7,
                'reasoning': 'Contains only culinary ingredients'
            }
        
        # Check for minimally processed
        if self._check_minimally_processed(ingredient_text):
            return {
                'nova_class': 'Minimally Processed',
                'confidence': 0.9,
                'reasoning': 'Contains natural/minimally processed ingredients'
            }
        
        # Default to minimally processed if no indicators found
        return {
            'nova_class': 'Minimally Processed',
            'confidence': 0.5,
            'reasoning': 'No processing indicators found, defaulting to minimally processed'
        }
    
    def batch_classify(self, extracted_data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all products in batch
        
        Args:
            extracted_data: DataFrame with extracted ingredient lists
            
        Returns:
            DataFrame with NOVA classifications
        """
        results = []
        
        for idx, row in extracted_data.iterrows():
            ingredient_text = row.get('ingredient_list', '')
            
            classification = self.classify_nova(ingredient_text)
            
            results.append({
                'product_id': row.get('product_id'),
                'nova_class': classification['nova_class'],
                'nova_confidence': classification['confidence'],
                'nova_reasoning': classification['reasoning']
            })
        
        return pd.DataFrame(results)
    
    def evaluate_on_test_set(self) -> Dict:
        """
        Evaluate classifier on test set using ground truth
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        actuals = []
        
        for idx, row in self.test_data.iterrows():
            ingredient_text = row.get('List_of_Ingredients', '')
            
            # Predict
            pred = self.classify_nova(ingredient_text)
            predicted_class = pred['nova_class']
            
            # Get actual
            actual_class = row.get('Nova_Classification', '')
            
            if actual_class and pd.notna(actual_class):
                predictions.append(predicted_class)
                actuals.append(actual_class)
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        report = classification_report(actuals, predictions, output_dict=True)
        conf_matrix = confusion_matrix(actuals, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'num_samples': len(actuals)
        }
    
    def refine_with_ground_truth(self):
        """
        Analyze misclassifications to refine patterns
        """
        misclassified = []
        
        for idx, row in self.test_data.iterrows():
            ingredient_text = row.get('List_of_Ingredients', '')
            actual_class = row.get('Nova_Classification', '')
            
            if not actual_class or pd.isna(actual_class):
                continue
            
            pred = self.classify_nova(ingredient_text)
            
            if pred['nova_class'] != actual_class:
                misclassified.append({
                    'product_id': row.get('Product_ID'),
                    'predicted': pred['nova_class'],
                    'actual': actual_class,
                    'ingredients': ingredient_text,
                    'reasoning': pred['reasoning']
                })
        
        return pd.DataFrame(misclassified)


# Usage example
if __name__ == "__main__":
    import json
    
    # Load extracted data
    with open("extracted_data.json", 'r') as f:
        extracted_data = pd.DataFrame(json.load(f))
    
    # Initialize classifier
    classifier = NOVAClassifier(ground_truth_path="./ground_truth.xlsx")
    
    # Classify all products
    nova_results = classifier.batch_classify(extracted_data)
    
    # Save results
    nova_results.to_csv("nova_classifications.csv", index=False)
    
    # Evaluate on test set
    eval_results = classifier.evaluate_on_test_set()
    print(f"\nNOVA Classification Evaluation:")
    print(f"Accuracy: {eval_results['accuracy']:.3f}")
    print(f"Test samples: {eval_results['num_samples']}")
    
    # Save evaluation
    with open("nova_evaluation.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Analyze misclassifications
    misclassified = classifier.refine_with_ground_truth()
    if not misclassified.empty:
        misclassified.to_csv("nova_misclassified.csv", index=False)
        print(f"\nMisclassified: {len(misclassified)} products")
        print("Saved to nova_misclassified.csv for analysis")