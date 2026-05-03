"""
Step 4: Model classifiation
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ComprehensiveEvaluator:
    def __init__(self, ground_truth_path: str, predicted_path: str):
        self.gt_df = pd.read_excel(ground_truth_path, dtype=str).fillna('')
        self.gt_df['Product_ID'] = self.gt_df['Product_ID'].astype(str)
        
        self.pred_df = pd.read_csv(predicted_path)
        self.pred_df['product_id'] = self.pred_df['product_id'].astype(str)
        
        self.merged_df = pd.merge(self.gt_df, self.pred_df, left_on='Product_ID', right_on='product_id', how='inner')

    def normalize_profile_string(self, profile_str: str) -> str:
        """Normalize nutrient profile strings for comparison"""
        if pd.isna(profile_str) or not profile_str:
            return 'None'
        profile_str = str(profile_str).strip().lower()
        if profile_str in ['', 'none', 'n/a', 'na']:
            return 'None'
        return profile_str

    def evaluate_nova(self):
        print("\n--- NOVA Classification Evaluation (4-Tier) ---")
        # Filter out rows with NaN values in either true or predicted
        valid_idx = self.merged_df['Nova_Classification'].notna() & self.merged_df['NOVA_Predicted'].notna()
        valid_df = self.merged_df[valid_idx]
        
        if len(valid_df) == 0:
            print("No valid data for NOVA evaluation")
            return
            
        y_true = valid_df['Nova_Classification'].astype(str)
        y_pred = valid_df['NOVA_Predicted'].astype(str)
        
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    def evaluate_nutrient_profiling(self):
        """Evaluate individual nutrient predictions against ground truth *_Profile columns"""
        print("\n--- Individual Nutrient Profile Evaluation ---")
        
        # Define the 6 nutrients: (nutrient_name, actual_profile_col, predicted_flag_col)
        nutrients = [
            ('Energy', 'Energy_Profile', 'High_Energy_Predicted'),
            ('Total_Fat', 'Total_Fat_Profile', 'High_Fat_Predicted'),
            ('Saturated_Fat', 'Saturated_Fat_Profile', 'High_Sat_Fat_Predicted'),
            ('Total_Sugar', 'Total_Sugar_Profile', 'High_Sugar_Predicted'),
            ('Added_Sugar', 'Added_Sugar_Profile', 'High_Added_Sugar_Predicted'),
            ('Sodium', 'Sodium_Profile', 'High_Sodium_Predicted')
        ]
        
        print("\nNutrient-by-Nutrient Accuracy (comparing predictions to *_Profile categories):")
        print("-" * 70)
        
        for nutrient_name, profile_col, pred_col in nutrients:
            if profile_col not in self.merged_df.columns:
                print(f"  {nutrient_name}: Column '{profile_col}' not found")
                continue
            
            if pred_col not in self.merged_df.columns:
                print(f"  {nutrient_name}: Predicted column '{pred_col}' not found")
                continue
            
            # Get actual profiles from ground truth (should be "Exceeded" or "Not exceeded/no limit")
            actual = self.merged_df[profile_col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else '')
            
            # Convert predicted boolean flags to category strings
            pred = self.merged_df[pred_col].apply(lambda x: 'exceeded' if x else 'not exceeded/no limit')
            
            # Filter for valid comparisons (remove empty/nan)
            valid_idx = (actual != '') & (actual != 'nan')
            valid_actual = actual[valid_idx]
            valid_pred = pred[valid_idx]
            
            if len(valid_actual) == 0:
                print(f"  {nutrient_name}: No valid data")
                continue
            
            # Calculate accuracy
            matches = (valid_actual == valid_pred).sum()
            accuracy = matches / len(valid_actual)
            
            # Count distribution for "Exceeded" class
            actual_exceeded = (valid_actual == 'exceeded').sum()
            pred_exceeded = (valid_pred == 'exceeded').sum()
            
            # Count correctly identified exceeded cases
            correct_exceeded = ((valid_actual == 'exceeded') & (valid_pred == 'exceeded')).sum()
            
            print(f"  {nutrient_name}:")
            print(f"    Overall Accuracy: {accuracy:.3f} ({matches}/{len(valid_actual)})")
            print(f"    Actual Exceeded: {actual_exceeded}, Predicted Exceeded: {pred_exceeded}, Correctly Identified: {correct_exceeded}")
            print()

# Execution
if __name__ == "__main__":
    print("Step 4: Model Evaluation")
    print("=" * 50)
    
    # Check if required files exist
    import os
    predicted_path = "./step3_comprehensive_classification.csv"
    if not os.path.exists(predicted_path):
        predicted_path = "./nova_classifications.csv"

    if not os.path.exists(predicted_path):
        print(f"ERROR: {predicted_path} not found!")
        print("Please run step3_nova_nutrient_profiles.py first.")
        exit(1)
    
    evaluator = ComprehensiveEvaluator(
        ground_truth_path="./ground_truth.xlsx",
        predicted_path=predicted_path
    )
    
    print(f"Loaded {len(evaluator.merged_df)} products for evaluation")
    
    evaluator.evaluate_nova()
    
    evaluator.evaluate_nutrient_profiling()
