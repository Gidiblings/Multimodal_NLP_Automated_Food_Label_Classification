# AI Coding Assistant Instructions for Food Label Analysis Framework

## Project Overview
This is a multimodal food label analysis framework for MSc thesis work. It processes food product images through a 5-step pipeline: image preprocessing, OCR-based information extraction, NOVA food classification, nutrient profiling against WHO thresholds, and comprehensive evaluation with healthiness ranking.

## Architecture & Data Flow
- **Pipeline Steps**: `step1_image_processing.py` → `step2_extraction.py` → `step3_nova_classification.py` → `step4_nutrient_profiling.py` → `food_label_step5.py`
- **Data Flow**: Images → Processed images → Extracted text/nutrition → Classifications → Profiles → Final ranked results
- **Key Data Structures**: Pandas DataFrames with `product_id` as primary key
- **Ground Truth**: Excel file (`ground_truth.xlsx`) with manual annotations for evaluation

## Critical Patterns & Conventions

### Fuzzy Matching for Classification
Use `thefuzz` library with partial ratio matching (threshold: 85) for ingredient-based classifications:
```python
from thefuzz import fuzz
if fuzz.partial_ratio(target, text_lower) >= self.fuzzy_threshold:
    matches.append(target)
```

### NOVA Classification Priority
Ultra-processed foods override all others. Check in order: Ultra Processed (4) → Processed (3) → Culinary Ingredients (2) → Minimally Processed (1).

### Nutrient Profiling Structure
- Codex categories determined by product name/ingredients using regex patterns
- WHO African Region thresholds applied per category (e.g., `{'total_fat_g': 8.0, 'saturated_fat_g': 4.0, 'sugar_g': 6.0, 'sodium_mg': 120.0}`)
- Profile flags: `high_sodium`, `high_sugar`, `high_fat`, `high_sat_fat`

### OCR Integration
- Primary: PaddleOCR for text extraction from food labels
- Fallback: Tesseract (pytesseract) if needed
- Extract from front images (product names) and back images (ingredients/nutrition)

### File Organization
- Raw images: `raw_images/` (JPG format, named `{product_id}_{image_num}.jpg` where product_id matches ground truth, e.g., BC1_1.jpg, BC1_2.jpg)
- Processed images: `processed_images/` (enhanced/deduplicated, one per angle)
- Results: `results/` directory with CSVs and evaluation JSONs
- Logs: `logs/` directory with timestamped execution logs

### Multi-Image Harmonization
Products may have 2-4 images from different angles (e.g., BC1_1.jpg, BC1_2.jpg, BC1_3.jpg). The pipeline intelligently harmonizes extractions:
- **Ingredients**: Deduplicated by comma-parsing and merging unique items across images (removes duplicates from different angles)
- **Nutrition data**: Merged using first non-null values (should be identical across angles; uses any available extraction)
- **Confidence handling**: Filters low-confidence OCR lines (threshold: 0.5)

## Developer Workflows

### Running Individual Steps
```python
# Step 1: Process images
python step1_image_processing.py  # Uses ground_truth.xlsx, outputs processed_images_log.csv

# Step 2: Extract information
python step2_extraction.py  # Reads processed_images_log.csv, outputs extracted_data.json

# Step 3: NOVA classification
python step3_nova_classification.py  # Reads extracted_data.json, outputs nova_classifications.csv

# Step 4: Nutrient profiling
python step4_nutrient_profiling.py  # Reads extracted_data.json, outputs nutrient_profiles.csv

# Step 5: Final evaluation
python food_label_step5.py  # Combines all outputs, generates complete_results.csv
```

### Full Pipeline Execution
```python
python master_pipeline.py  # Orchestrates all steps with logging and timing
```

### Dependencies & Environment
- Install: `pip install -r requirements.txt`
- Key libraries: pandas, opencv-python, paddleocr, thefuzz, scikit-learn
- Python 3.8+ required for PaddleOCR compatibility

## Common Patterns

### Data Loading & Type Handling
```python
# Load ground truth with string dtypes to avoid ID parsing issues
ground_truth = pd.read_excel("ground_truth.xlsx", dtype=str).fillna('')

# Ensure Product_ID is string for consistent matching
df['Product_ID'] = df['Product_ID'].astype(str)
```

### Error Handling for OCR/Image Processing
```python
try:
    result = ocr.ocr(image_path)
    if result and result[0]:
        # Process results
except Exception as e:
    print(f"OCR failed for {image_path}: {e}")
    # Continue with empty results rather than failing
```

### Evaluation Metrics
Use sklearn metrics for classification accuracy:
```python
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(true_labels, predicted_labels)
```

### Healthiness Scoring
Combined NOVA + nutrient profile scoring:
- NOVA: Minimally Processed (3) → Ultra Processed (0)
- Nutrients: None exceeded (3) → 2+ exceeded (0)
- Final score = weighted average, ranked as Very Healthy/Moderately Healthy/Less Healthy/Unhealthy

## Key Files to Reference
- `step3_nova_classification.py`: Fuzzy matching implementation and classification logic
- `step4_nutrient_profiling.py`: Codex category patterns and WHO thresholds
- `food_label_step5.py`: Final integration and evaluation methods
- `master_pipeline.py`: Execution orchestration and logging setup