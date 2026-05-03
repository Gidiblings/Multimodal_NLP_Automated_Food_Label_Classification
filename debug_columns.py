import pandas as pd
import json

# Load the results
results_df = pd.read_csv('nova_classifications.csv')
nutrient_profiles = pd.read_csv('nutrient_profiles.csv')
complete_results = pd.read_csv('processed_images_log.csv')

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

print('Columns in merged_df:')
for col in merged_df.columns:
    print(f'  - {col}')

print(f'\nEnergy_Profile in merged_df: {"Energy_Profile" in merged_df.columns}')
print(f'Overall_Nutrient_Profile in merged_df: {"Overall_Nutrient_Profile" in merged_df.columns}')
print(f'nutrient_profile in merged_df: {"nutrient_profile" in merged_df.columns}')

print("\n--- Checking High_Energy_Predicted column ---")
print(f'High_Energy_Predicted in merged_df: {"High_Energy_Predicted" in merged_df.columns}')
if "High_Energy_Predicted" in merged_df.columns:
    print(f'High_Energy_Predicted values (first 10):')
    print(merged_df['High_Energy_Predicted'].head(10))

print("\n--- Checking Energy_Profile column ---")
print(f'Energy_Profile values (first 10):')
print(merged_df['Energy_Profile'].head(10))
