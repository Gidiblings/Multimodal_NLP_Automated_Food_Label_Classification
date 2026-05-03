import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the accuracy data
accuracy_df = pd.read_csv('results/nutrient_accuracy_data.csv')

print("Accuracy DataFrame:")
print(accuracy_df)
print(f"\nShape: {accuracy_df.shape}")
print(f"Columns: {accuracy_df.columns.tolist()}")

# Create the plot
fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(
    data=accuracy_df,
    x='Nutrient',
    y='Accuracy',
    palette='viridis',
    ax=ax
)
ax.set_xlabel('Nutrient Profiles', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=45, labelsize=10)

# Add value labels on bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.2f}",
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom', fontsize=10
    )

fig.tight_layout()
fig.savefig('test_nutrient_chart.png', dpi=100, bbox_inches='tight')
fig.savefig('test_nutrient_chart.tiff', dpi=300, bbox_inches='tight')

print("\nChart saved as:")
print("  - test_nutrient_chart.png")
print("  - test_nutrient_chart.tiff")

plt.close(fig)
