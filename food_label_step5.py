"""
Step 5: Multimodal Framework Integration and Comprehensive Evaluation
Combines all components and provides final healthiness ranking
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

class MultimodalFramework:
    def __init__(self, ground_truth_path: str):
        """
        Initialize complete multimodal framework
        
        Args:
            ground_truth_path: Path to ground truth Excel
        """
        self.ground_truth = pd.read_excel(ground_truth_path)
        self.train_data, self.test_data = self._split_data()
        
        # Healthiness scoring weights
        self.healthiness_weights = {
            'nova': {
                'Minimally Processed': 3,
                'Processed Culinary Ingredient': 2,
                'Processed': 1,
                'Ultra Processed': 0
            },
            'nutrient_profile': {
                'None': 3,  # No nutrients exceeded
                'exceeded_1': 1,  # 1 nutrient exceeded
                'exceeded_2plus': 0  # 2+ nutrients exceeded
            }
        }
    
    def _split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        shuffled = self.ground_truth.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled) * (1 - test_size))
        return shuffled[:split_idx], shuffled[split_idx:]
    
    def run_complete_pipeline(self, n_products: int = 200) -> pd.DataFrame:
        """
        Run complete pipeline from images to classifications
        
        Args:
            n_products: Number of products to process
            
        Returns:
            Complete results DataFrame
        """
        from step1_image_processing import ImagePreprocessor
        from step2_extraction import MultimodalExtractor
        from step3_nova_classification import NOVAClassifier
        from step4_nutrient_profiling import NutrientProfiler
        
        print("="*60)
        print("MULTIMODAL FOOD LABEL ANALYSIS FRAMEWORK")
        print("="*60)
        
        # Step 1: Image Processing
        print("\n[1/4] Processing Images...")
        start_time = time.time()
        preprocessor = ImagePreprocessor(
            input_dir="./raw_images",
            output_dir="./processed_images",
            ground_truth_path=ground_truth_path
        )
        processed_df = preprocessor.process_all(n_products=n_products)
        print(f"Completed in {time.time() - start_time:.2f}s")
        
        # Step 2: Information Extraction
        print("\n[2/4] Extracting Information...")
        start_time = time.time()
        extractor = MultimodalExtractor(
            ground_truth_path=ground_truth_path,
            use_layoutlm=False
        )
        extracted_df = extractor.batch_extract(processed_df)
        print(f"Completed in {time.time() - start_time:.2f}s")
        
        # Step 3: NOVA Classification
        print("\n[3/4] NOVA Classification...")
        start_time = time.time()
        nova_classifier = NOVAClassifier(ground_truth_path=ground_truth_path)
        nova_df = nova_classifier.batch_classify(extracted_df)
        print(f"Completed in {time.time() - start_time:.2f}s")
        
        # Step 4: Nutrient Profiling
        print("\n[4/4] Nutrient Profiling...")
        start_time = time.time()
        profiler = NutrientProfiler(ground_truth_path=ground_truth_path)
        profile_df = profiler.batch_profile(extracted_df)
        print(f"Completed in {time.time() - start_time:.2f}s")
        
        # Merge all results
        results = extracted_df.copy()
        results = results.merge(nova_df, on='product_id', how='left')
        results = results.merge(profile_df, on='product_id', how='left')
        
        return results
    
    def calculate_healthiness_score(self, row: pd.Series) -> Dict:
        """
        Calculate overall healthiness score
        
        Args:
            row: DataFrame row with NOVA and nutrient profile data
            
        Returns:
            Dictionary with healthiness score and rank
        """
        # NOVA score (0-3)
        nova_class = row.get('nova_class', 'Unknown')
        nova_score = self.healthiness_weights['nova'].get(nova_class, 0)
        
        # Nutrient profile score (0-3)
        nutrient_profile = row.get('nutrient_profile', '')
        
        if 'None' in nutrient_profile or nutrient_profile == 'None':
            profile_score = 3
        elif 'exceeded' in nutrient_profile.lower():
            # Count how many nutrients exceeded
            exceeded_count = len(row.get('exceeded_details', []))
            if exceeded_count >= 2:
                profile_score = 0
            else:
                profile_score = 1
        else:
            profile_score = 1  # Missing data or unknown
        
        # Weighted total score (max 6)
        total_score = (nova_score * 0.5 + profile_score * 0.5) * 2
        
        # Determine healthiness rank
        if total_score >= 5:
            rank = 'Very Healthy'
        elif total_score >= 3.5:
            rank = 'Moderately Healthy'
        elif total_score >= 2:
            rank = 'Less Healthy'
        else:
            rank = 'Unhealthy'
        
        return {
            'nova_score': nova_score,
            'profile_score': profile_score,
            'total_healthiness_score': total_score,
            'healthiness_rank': rank
        }
    
    def add_healthiness_rankings(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Add healthiness scores and rankings to results"""
        healthiness_data = []
        
        for idx, row in results_df.iterrows():
            health_scores = self.calculate_healthiness_score(row)
            healthiness_data.append(health_scores)
        
        healthiness_df = pd.DataFrame(healthiness_data)
        results_with_health = pd.concat([results_df, healthiness_df], axis=1)
        
        # Sort by healthiness score
        results_with_health = results_with_health.sort_values(
            'total_healthiness_score', 
            ascending=False
        )
        
        return results_with_health
    
    def evaluate_extraction_accuracy(self, results_df: pd.DataFrame) -> Dict:
        """
        Evaluate extraction accuracy against ground truth
        
        Args:
            results_df: Results with extracted data
            
        Returns:
            Dictionary with extraction metrics
        """
        # Merge with test data
        test_results = results_df[
            results_df['product_id'].isin(self.test_data['Product_ID'])
        ].copy()
        
        test_with_gt = test_results.merge(
            self.test_data,
            left_on='product_id',
            right_on='Product_ID',
            how='inner'
        )
        
        metrics = {
            'n_samples': len(test_with_gt),
            'extraction_rates': {}
        }
        
        # Check extraction success rates
        fields = {
            'product_name': 'Claim_statement',
            'ingredient_list': 'List_of_Ingredients',
            'energy_kcal': 'Energy_kcal',
            'protein_g': 'Protein_g',
            'fat_g': 'Fat_g',
            'sugar_g': 'Sugar_g'
        }
        
        for extracted_field, gt_field in fields.items():
            extracted_count = 0
            
            for idx, row in test_with_gt.iterrows():
                if extracted_field in ['energy_kcal', 'protein_g', 'fat_g', 'sugar_g']:
                    extracted_val = row.get('nutrition', {}).get(extracted_field)
                else:
                    extracted_val = row.get(extracted_field)
                
                gt_val = row.get(gt_field)
                
                # Check if extracted (not empty/null)
                if extracted_val and pd.notna(extracted_val):
                    if str(extracted_val).strip():
                        extracted_count += 1
            
            extraction_rate = extracted_count / len(test_with_gt) if len(test_with_gt) > 0 else 0
            metrics['extraction_rates'][extracted_field] = {
                'rate': extraction_rate,
                'extracted': extracted_count,
                'total': len(test_with_gt)
            }
        
        return metrics
    
    def evaluate_nova_classification(self, results_df: pd.DataFrame) -> Dict:
        """Evaluate NOVA classification accuracy"""
        test_results = results_df[
            results_df['product_id'].isin(self.test_data['Product_ID'])
        ].copy()
        
        test_with_gt = test_results.merge(
            self.test_data,
            left_on='product_id',
            right_on='Product_ID',
            how='inner'
        )
        
        predictions = []
        actuals = []
        
        for idx, row in test_with_gt.iterrows():
            pred = row.get('nova_class')
            actual = row.get('Nova_Classification')
            
            if pred and actual and pd.notna(pred) and pd.notna(actual):
                predictions.append(pred)
                actuals.append(actual)
        
        if not predictions:
            return {'error': 'No valid predictions for evaluation'}
        
        return {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, average='weighted', zero_division=0),
            'recall': recall_score(actuals, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(actuals, predictions, average='weighted', zero_division=0),
            'n_samples': len(predictions),
            'classification_report': classification_report(actuals, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(actuals, predictions).tolist()
        }
    
    def evaluate_nutrient_profiling(self, results_df: pd.DataFrame) -> Dict:
        """Evaluate nutrient profiling accuracy"""
        test_results = results_df[
            results_df['product_id'].isin(self.test_data['Product_ID'])
        ].copy()
        
        test_with_gt = test_results.merge(
            self.test_data,
            left_on='product_id',
            right_on='Product_ID',
            how='inner'
        )
        
        # Codex category evaluation
        codex_pred = []
        codex_actual = []
        
        # Nutrient profile evaluation
        profile_pred = []
        profile_actual = []
        
        for idx, row in test_with_gt.iterrows():
            # Codex
            pred_codex = row.get('codex_category')
            actual_codex = row.get('Codex_Category')
            if pred_codex and actual_codex and pd.notna(pred_codex) and pd.notna(actual_codex):
                codex_pred.append(pred_codex)
                codex_actual.append(actual_codex)
            
            # Profile
            pred_profile = row.get('nutrient_profile')
            actual_profile = row.get('Nutrient_Profile')
            if pred_profile and actual_profile and pd.notna(pred_profile) and pd.notna(actual_profile):
                profile_pred.append(pred_profile)
                profile_actual.append(actual_profile)
        
        results = {}
        
        if codex_pred:
            results['codex'] = {
                'accuracy': accuracy_score(codex_actual, codex_pred),
                'n_samples': len(codex_pred)
            }
        
        if profile_pred:
            results['nutrient_profile'] = {
                'accuracy': accuracy_score(profile_actual, profile_pred),
                'n_samples': len(profile_pred)
            }
        
        return results
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame, 
                                     output_dir: str = "./results") -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            results_df: Complete results DataFrame
            output_dir: Directory to save reports
            
        Returns:
            Dictionary with all evaluation metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # 1. Extraction Evaluation
        print("\n[1] Information Extraction Performance:")
        extraction_metrics = self.evaluate_extraction_accuracy(results_df)
        print(f"  Test Samples: {extraction_metrics['n_samples']}")
        for field, metrics in extraction_metrics['extraction_rates'].items():
            print(f"  {field}: {metrics['rate']*100:.1f}% ({metrics['extracted']}/{metrics['total']})")
        
        # 2. NOVA Classification Evaluation
        print("\n[2] NOVA Classification Performance:")
        nova_metrics = self.evaluate_nova_classification(results_df)
        if 'error' not in nova_metrics:
            print(f"  Accuracy: {nova_metrics['accuracy']:.3f}")
            print(f"  Precision: {nova_metrics['precision']:.3f}")
            print(f"  Recall: {nova_metrics['recall']:.3f}")
            print(f"  F1-Score: {nova_metrics['f1_score']:.3f}")
            print(f"  Test Samples: {nova_metrics['n_samples']}")
        else:
            print(f"  {nova_metrics['error']}")
        
        # 3. Nutrient Profiling Evaluation
        print("\n[3] Nutrient Profiling Performance:")
        profile_metrics = self.evaluate_nutrient_profiling(results_df)
        if 'codex' in profile_metrics:
            print(f"  Codex Category Accuracy: {profile_metrics['codex']['accuracy']:.3f}")
            print(f"  Codex Samples: {profile_metrics['codex']['n_samples']}")
        if 'nutrient_profile' in profile_metrics:
            print(f"  Nutrient Profile Accuracy: {profile_metrics['nutrient_profile']['accuracy']:.3f}")
            print(f"  Profile Samples: {profile_metrics['nutrient_profile']['n_samples']}")
        
        # 4. Healthiness Distribution
        print("\n[4] Healthiness Distribution:")
        health_dist = results_df['healthiness_rank'].value_counts()
        for rank, count in health_dist.items():
            print(f"  {rank}: {count} ({count/len(results_df)*100:.1f}%)")
        
        # Compile all metrics
        report = {
            'extraction_metrics': extraction_metrics,
            'nova_metrics': nova_metrics,
            'profile_metrics': profile_metrics,
            'healthiness_distribution': health_dist.to_dict(),
            'total_products_analyzed': len(results_df)
        }
        
        # Save report
        with open(output_path / "evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed results
        results_df.to_csv(output_path / "complete_results.csv", index=False)
        results_df.to_json(output_path / "complete_results.json", orient='records', indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_dir}/")
        print(f"  - evaluation_report.json")
        print(f"  - complete_results.csv")
        print(f"  - complete_results.json")
        print(f"{'='*60}\n")
        
        return report
    
    def visualize_results(self, results_df: pd.DataFrame, 
                         output_dir: str = "./results"):
        """Generate visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Healthiness Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Healthiness Rank Distribution
        health_counts = results_df['healthiness_rank'].value_counts()
        axes[0, 0].bar(health_counts.index, health_counts.values, color='skyblue')
        axes[0, 0].set_title('Healthiness Rank Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Healthiness Rank')
        axes[0, 0].set_ylabel('Number of Products')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # NOVA Classification Distribution
        nova_counts = results_df['nova_class'].value_counts()
        axes[0, 1].bar(nova_counts.index, nova_counts.values, color='lightcoral')
        axes[0, 1].set_title('NOVA Classification Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('NOVA Class')
        axes[0, 1].set_ylabel('Number of Products')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Healthiness Score Distribution
        axes[1, 0].hist(results_df['total_healthiness_score'].dropna(), 
                       bins=20, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Healthiness Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Healthiness Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Compliant vs Non-Compliant
        compliant_counts = results_df['compliant'].value_counts()
        axes[1, 1].pie(compliant_counts.values, labels=['Non-Compliant', 'Compliant'],
                      autopct='%1.1f%%', colors=['#ff9999', '#99ff99'])
        axes[1, 1].set_title('WHO Nutrient Profile Compliance', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'healthiness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {output_dir}/healthiness_analysis.png")


# Main execution
if __name__ == "__main__":
    import sys
    
    # Configuration
    ground_truth_path = "./ground_truth.xlsx"
    n_products = 200  # Adjust based on compute resources
    
    # Initialize framework
    framework = MultimodalFramework(ground_truth_path=ground_truth_path)
    
    # Option 1: Run complete pipeline (from images)
    # Uncomment if running from scratch
    # results = framework.run_complete_pipeline(n_products=n_products)
    
    # Option 2: Load existing results from individual steps
    # Use this if you've already run steps 1-4 separately
    print("Loading results from previous steps...")
    
    try:
        # Load extracted data
        with open("extracted_data.json", 'r') as f:
            extracted_df = pd.DataFrame(json.load(f))
        
        # Load NOVA classifications
        nova_df = pd.read_csv("nova_classifications.csv")
        
        # Load nutrient profiles
        profile_df = pd.read_csv("nutrient_profiles.csv")
        
        # Merge all results
        results = extracted_df.merge(nova_df, on='product_id', how='left')
        results = results.merge(profile_df, on='product_id', how='left')
        
        print(f"Loaded {len(results)} products")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files. Please run steps 1-4 first.")
        print(f"Missing file: {e}")
        sys.exit(1)
    
    # Add healthiness rankings
    print("\nCalculating healthiness scores...")
    results_with_health = framework.add_healthiness_rankings(results)
    
    # Generate comprehensive report
    report = framework.generate_comprehensive_report(
        results_with_health,
        output_dir="./results"
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    framework.visualize_results(results_with_health, output_dir="./results")
    
    # Display top 10 healthiest products
    print("\n" + "="*60)
    print("TOP 10 HEALTHIEST PRODUCTS")
    print("="*60)
    top_10 = results_with_health.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{idx+1}. {row.get('product_name', 'N/A')}")
        print(f"   Product ID: {row.get('product_id')}")
        print(f"   Healthiness Score: {row.get('total_healthiness_score', 0):.2f}/6.0")
        print(f"   Rank: {row.get('healthiness_rank')}")
        print(f"   NOVA: {row.get('nova_class')}")
        print(f"   Nutrient Profile: {row.get('nutrient_profile')}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)