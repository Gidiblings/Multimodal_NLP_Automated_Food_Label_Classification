import pandas as pd

# Load the results
results_df = pd.read_csv('nova_classifications.csv')
gt = pd.read_excel('ground_truth.xlsx', dtype=str)

# Simulate the merge
results_df['product_id'] = results_df['product_id'].astype(str)
gt['Product_ID'] = gt['Product_ID'].astype(str)

merged_df = results_df.merge(
    gt[[
        'Product_ID', 'Nova_Classification', 'Energy_Profile', 'Total_Sugar_Profile',
        'Total_Fat_Profile', 'Saturated_Fat_Profile', 'Added_Sugar_Profile',
        'Sodium_Profile', 'Overall_Nutrient_Profile'
    ]],
    left_on='product_id',
    right_on='Product_ID',
    how='left'
)

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
    print(f"\n--- Processing {display_name} ---")
    if actual_col in merged_df.columns and pred_col in merged_df.columns:
        print(f"  Columns found: {actual_col}, {pred_col}")
        actual = merged_df[actual_col].fillna('').astype(str).str.strip().str.lower()
        pred = merged_df[pred_col].apply(lambda x: 'exceeded' if x else 'not exceeded/no limit')
        print(f"  Actual values (first 5): {actual.head().tolist()}")
        print(f"  Predicted values (first 5): {pred.head().tolist()}")
        valid = actual.isin(['exceeded', 'not exceeded/no limit'])
        print(f"  Valid entries: {valid.sum()}/{len(valid)}")
        if valid.sum() > 0:
            accuracy = (actual[valid] == pred[valid]).mean()
            print(f"  Accuracy: {accuracy:.4f}")
            accuracy_results.append({'Nutrient': display_name, 'Accuracy': accuracy})
        else:
            print(f"  No valid entries!")
    else:
        print(f"  Columns NOT found!")

print("\n\n--- Overall Nutrient Profile ---")
if 'nutrient_profile' in merged_df.columns and 'Overall_Nutrient_Profile' in merged_df.columns:
    print("  Columns found: nutrient_profile, Overall_Nutrient_Profile")
    gt_overall = merged_df['Overall_Nutrient_Profile'].fillna('').astype(str).str.strip()
    pred_overall = merged_df['nutrient_profile'].fillna('0').astype(str).str.strip()
    print(f"  GT Overall values (first 5): {gt_overall.head().tolist()}")
    print(f"  Pred Overall values (first 5): {pred_overall.head().tolist()}")
    valid_overall = (gt_overall != '') & (gt_overall != 'nan')
    print(f"  Valid entries: {valid_overall.sum()}/{len(valid_overall)}")
    if valid_overall.sum() > 0:
        overall_accuracy = (gt_overall[valid_overall] == pred_overall[valid_overall]).mean()
        print(f"  Accuracy: {overall_accuracy:.4f}")
        accuracy_results.append({'Nutrient': 'Overall Nutrient Profile', 'Accuracy': overall_accuracy})
    else:
        print(f"  No valid entries!")
else:
    print("  Columns NOT found!")

print(f"\n\nFinal accuracy_results list:")
for result in accuracy_results:
    print(f"  {result}")

accuracy_df = pd.DataFrame(accuracy_results)
print("\n\nAccuracy DataFrame:")
print(accuracy_df)
