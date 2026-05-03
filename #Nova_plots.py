#Nova_plots.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Replace this JSON string with your evaluation output ===
results_json = {
  "accuracy": 0.5681818181818182,
  "classification_report": {
    "Minimally Processed": {
      "precision": 0.05555555555555555,
      "recall": 0.09090909090909091,
      "f1-score": 0.06896551724137931,
      "support": 11.0
    },
    "Processed": {
      "precision": 0.38235294117647056,
      "recall": 0.8125,
      "f1-score": 0.52,
      "support": 16.0
    },
    "Processed Culinary Ingredients": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 1.0
    },
    "Ultra Processed": {
      "precision": 1.0,
      "recall": 0.6,
      "f1-score": 0.75,
      "support": 60.0
    },
    "accuracy": 0.5681818181818182,
    "macro avg": {
      "precision": 0.35947712418300654,
      "recall": 0.37585227272727273,
      "f1-score": 0.33474137931034487,
      "support": 88.0
    },
    "weighted avg": {
      "precision": 0.7582813428401664,
      "recall": 0.5681818181818182,
      "f1-score": 0.6145297805642633,
      "support": 88.0
    }
  },
  "confusion_matrix": [
    [1, 10, 0, 0],
    [3, 13, 0, 0],
    [0, 1, 0, 0],
    [14, 10, 0, 36]
  ],
  "num_samples": 88
}

# === Extract report & confusion matrix ===
report = results_json["classification_report"]
cm = results_json["confusion_matrix"]
labels = ["Minimally Processed", "Processed", "Processed Culinary Ingredients", "Ultra Processed"]

# --- Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (NOVA Classifier)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --- Bar Chart: Precision, Recall, F1 ---
df = pd.DataFrame({cls: metrics for cls, metrics in report.items() if cls in labels}).T
df = df[['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(10, 6))
plt.title('Classification Metrics per NOVA Class')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()