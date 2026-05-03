# file: Step5_Healthiness_scoring_ranking.py
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
from matplotlib.lines import Line2D
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
        self.ground_truth_path = ground_truth_path
        # read as string to avoid dtype mismatches
        self.ground_truth = pd.read_excel(ground_truth_path, dtype=str).fillna('')
        # normalize GT column names by trimming whitespace only
        self.ground_truth.columns = [c.strip() for c in self.ground_truth.columns]

        self.train_data, self.test_data = self._split_data()
        ...
        # Healthiness scoring weights
        self.healthiness_weights = {
            'nova': {
                'Minimally Processed': 3,
                'Processed Culinary Ingredients': 2,
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
    
    def run_complete_pipeline(self, n_products: int = 138, allow_overwrite: bool = False, overwrite_files: bool = False) -> pd.DataFrame:
        """
        Run pipeline. By default this does NOT overwrite existing non-empty fields from 'extracted_df'.
        Set allow_overwrite=True to permit later-step outputs to replace existing values.
        """
        from step1_image_processing import ImagePreprocessor
        from step2_extraction import MultimodalExtractor
        from step3_nova_nutrient_profiles import NOVAClassifier

        print("="*60)
        print("MULTIMODAL FOOD LABEL ANALYSIS FRAMEWORK")
        print("="*60)

        # Step 1: Image Processing
        print("\n[1/3] Processing Images...")
        start_time = time.time()
        preprocessor = ImagePreprocessor(
            input_dir="./raw_images",
            output_dir="./processed_images",
            ground_truth_path=self.ground_truth_path
        )
        processed_df = preprocessor.process_all()
        print(f"Completed in {time.time() - start_time:.2f}s")

        # Step 2: Information Extraction
        print("\n[2/3] Extracting Information...")
        start_time = time.time()
        extractor = MultimodalExtractor(
            ground_truth_path=self.ground_truth_path
        )
        extracted_df = extractor.batch_extract(processed_df)
        print(f"Completed in {time.time() - start_time:.2f}s")

        # Step 3: NOVA Classification
        print("\n[3/3] NOVA Classification...")
        start_time = time.time()
        nova_classifier = NOVAClassifier()
        nova_df = nova_classifier.batch_classify(extracted_df)
        nova_df.to_csv("nova_classifications.csv", index=False)
        nova_df.to_csv("step3_comprehensive_classification.csv", index=False)
        print(f"Completed in {time.time() - start_time:.2f}s")

        # Combine extracted information with NOVA predictions
        results = extracted_df.merge(
            nova_df.rename(columns={
                'NOVA_Predicted': 'nova_class'
            }),
            on='product_id',
            how='left'
        )

        results = self._coerce_types_for_results(results)

        out_dir = Path("./results")
        out_dir.mkdir(parents=True, exist_ok=True)
        results_csv = out_dir / "complete_results.csv"
        if not results_csv.exists() or overwrite_files:
            results.to_csv(results_csv, index=False)
        else:
            print(f"{results_csv} exists — skipped (set overwrite_files=True to overwrite).")

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
        nova_class = row.get('nova_class', row.get('NOVA_Predicted', 'Unknown'))
        nova_score = self.healthiness_weights['nova'].get(nova_class, 0)
        
        # Nutrient profile score (0-3)
        nutrient_profile = row.get('nutrient_profile', '')
        nutrient_profile = str(nutrient_profile).strip() if pd.notna(nutrient_profile) else ''

        if nutrient_profile in ['', 'None', 'nan']:
            profile_score = 3
        else:
            try:
                exceeded_count = int(nutrient_profile)
            except (ValueError, TypeError):
                exceeded_count = 0

            if exceeded_count == 0:
                profile_score = 3
            elif exceeded_count == 1:
                profile_score = 1
            else:
                profile_score = 0
        
        # Weighted total score (max 6)
        total_score = (nova_score * 0.5 + profile_score * 0.5) * 2
        
        # Determine healthiness rank (3 categories)
        if total_score >= 5.0:
            rank = 'Healthier'
        elif total_score >= 3.0:
            rank = 'Moderately Healthy'
        else:
            rank = 'Less Healthy'
        
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
        
        # Ensure compliance field exists for visualization and reporting
        nutrition_profile_series = results_with_health['nutrient_profile'].fillna('0').astype(str).str.strip()
        results_with_health['compliant'] = nutrition_profile_series == '0'
        
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
            'product_name': 'Product_Name',
            'ingredient_list': 'List_of_Ingredients',
            'energy_kcal': 'Energy',
            'protein_g': 'Protein',
            'fat_g': 'Total_Fat',
            'sugar_g': 'Total_Sugar',
            'added_sugar_g': 'Added_Sugar'
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
            pred = row.get('nova_class', row.get('NOVA_Predicted'))
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
        
        # Nutrient profile evaluation
        profile_pred = []
        profile_actual = []
        
        for idx, row in test_with_gt.iterrows():
            # Profile
            pred_profile = row.get('nutrient_profile')
            actual_profile = row.get('Overall_Nutrient_Profile')
            if pred_profile and actual_profile and pd.notna(pred_profile) and pd.notna(actual_profile):
                profile_pred.append(str(pred_profile))
                profile_actual.append(str(actual_profile))
        
        results = {}
        
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
        plt.style.use('seaborn-v0_8-darkgrid')

        merged_df = results_df.merge(
            self.ground_truth[[
                'Product_ID', 'Nova_Classification', 'Energy_Profile', 'Total_Sugar_Profile',
                'Total_Fat_Profile', 'Saturated_Fat_Profile', 'Added_Sugar_Profile',
                'Sodium_Profile', 'Overall_Nutrient_Profile'
            ]],
            left_on='product_id',
            right_on='Product_ID',
            how='left'
        )

        labels = [
            'Minimally Processed',
            'Processed Culinary Ingredients',
            'Processed',
            'Ultra Processed'
        ]
        predicted_col = 'nova_class' if 'nova_class' in merged_df.columns else 'NOVA_Predicted'
        nova_plot_df = merged_df.dropna(subset=['Nova_Classification', predicted_col])
        if not nova_plot_df.empty:
            cm = confusion_matrix(
                nova_plot_df['Nova_Classification'],
                nova_plot_df[predicted_col],
                labels=labels
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=labels, yticklabels=labels, cbar=True, ax=ax
            )
            ax.set_xlabel('Predicted NOVA Category')
            ax.set_ylabel('Actual NOVA Category')
            ax.legend([Line2D([0], [0], color='black')], ['Count'], loc='upper right')
            fig.tight_layout()
            fig.savefig(output_path / 'nova_confusion_matrix.tiff', dpi=300, bbox_inches='tight')
            plt.close(fig)

        nutrient_map = [
            ('Energy', 'Energy_Profile', 'High_Energy_Predicted'),
            ('Fat', 'Total_Fat_Profile', 'High_Fat_Predicted'),
            ('Saturated Fat', 'Saturated_Fat_Profile', 'High_Sat_Fat_Predicted'),
            ('Sugar', 'Total_Sugar_Profile', 'High_Sugar_Predicted'),
            ('Added Sugar', 'Added_Sugar_Profile', 'High_Added_Sugar_Predicted'),
            ('Sodium', 'Sodium_Profile', 'High_Sodium_Predicted')
        ]
        accuracy_results = []
        for display_name, actual_col, pred_col in nutrient_map:
            if actual_col in merged_df.columns and pred_col in merged_df.columns:
                actual = merged_df[actual_col].fillna('').astype(str).str.strip().str.lower()
                pred = merged_df[pred_col].apply(lambda x: 'exceeded' if x else 'not exceeded/no limit')
                valid = actual.isin(['exceeded', 'not exceeded/no limit'])
                if valid.sum() > 0:
                    accuracy = (actual[valid] == pred[valid]).mean()
                    accuracy_results.append({'Nutrient': display_name, 'Accuracy': accuracy})

        # Add Overall Nutrient Profile accuracy
        if 'nutrient_profile' in merged_df.columns and 'Overall_Nutrient_Profile' in merged_df.columns:
            gt_overall = merged_df['Overall_Nutrient_Profile'].fillna('').astype(str).str.strip()
            pred_overall = merged_df['nutrient_profile'].fillna('0').astype(str).str.strip()
            valid_overall = (gt_overall != '') & (gt_overall != 'nan')
            if valid_overall.sum() > 0:
                overall_accuracy = (gt_overall[valid_overall] == pred_overall[valid_overall]).mean()
                accuracy_results.append({'Nutrient': 'Overall Nutrient Profile', 'Accuracy': overall_accuracy})

        if accuracy_results:
            accuracy_df = pd.DataFrame(accuracy_results)
            accuracy_df.to_csv(output_path / 'nutrient_accuracy_data.csv', index=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=accuracy_df,
                x='Nutrient',
                y='Accuracy',
                palette='viridis',
                ax=ax
            )
            ax.set_xlabel('Nutrient Profiles')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)
            ax.legend(['Accuracy'], loc='upper right')
            ax.tick_params(axis='x', rotation=45)
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9
                )
            fig.tight_layout()
            fig.savefig(output_path / 'nutrient_accuracy_bar_chart.tiff', dpi=300, bbox_inches='tight')
            plt.close(fig)

        if 'healthiness_rank' in results_df.columns:
            order = ['Healthier', 'Moderately Healthy', 'Less Healthy']
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Count values and calculate percentages
            rank_counts = results_df['healthiness_rank'].value_counts().reindex(order, fill_value=0)
            total = rank_counts.sum()
            rank_percentages = (rank_counts / total * 100).round(1)
            
            # Define colors (traffic light)
            colors = {'Healthier': '#2ecc71', 'Moderately Healthy': '#f39c12', 'Less Healthy': '#e74c3c'}
            color_list = [colors[rank] for rank in order if rank in results_df['healthiness_rank'].unique()]
            
            bars = ax.bar(
                [rank for rank in order if rank in results_df['healthiness_rank'].unique()],
                [rank_counts[rank] for rank in order if rank in results_df['healthiness_rank'].unique()],
                color=color_list
            )
            
            # Add labels with count and percentage
            for bar, rank in zip(bars, [r for r in order if r in results_df['healthiness_rank'].unique()]):
                height = bar.get_height()
                percentage = rank_percentages[rank]
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'(n={int(height)}; {percentage}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )
            
            ax.set_xlabel('Healthiness Category', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_ylim(0, rank_counts.max() * 1.15)
            ax.tick_params(axis='x', rotation=15)
            fig.tight_layout()
            fig.savefig(output_path / 'healthiness_distribution.tiff', dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"Visualizations saved to: {output_dir}")


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

        # Prefer the full step3 comprehensive output if available
        if Path("step3_comprehensive_classification.csv").exists():
            nova_df = pd.read_csv("step3_comprehensive_classification.csv")
        elif Path("nova_classifications.csv").exists():
            nova_df = pd.read_csv("nova_classifications.csv")
        else:
            raise FileNotFoundError("No NOVA prediction file found.")

        # Normalize older and newer column names for compatibility
        if 'NOVA_Predicted' in nova_df.columns:
            nova_df = nova_df.rename(columns={
                'NOVA_Predicted': 'nova_class',
                'Codex_Predicted': 'codex_category'
            })

        # Merge extracted data with NOVA predictions
        results = extracted_df.merge(nova_df, on='product_id', how='left')
        
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