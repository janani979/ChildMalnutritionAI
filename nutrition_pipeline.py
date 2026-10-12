"""
Child Nutrition AI Assessment System
Standalone Parts 1-14 pipeline extracted from ChildNutritionAI (2).ipynb.

Part 15 (interactive input()) and the final PDF-only cell are intentionally
excluded. Training-only cells are retained as callable functions but are not
executed during import; saved models are used by the existing Part 5/8 logic.
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# Streamlit/Python compatibility for notebook display() calls.
try:
    display
except NameError:
    def display(*args, **kwargs):
        return None


# ================= ORIGINAL NOTEBOOK CELL 1 =================
# ============================================================
# PART 1 — PROJECT SETUP
# Agentic AI Framework for Explainable Child Nutrition
# Recommendation and Malnutrition Prevention
# ============================================================

from pathlib import Path
import os
import json
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("AGENTIC AI CHILD NUTRITION SYSTEM")
print("PART 1 — PROJECT SETUP")
print("=" * 70)


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"


# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\nProject directories:")
print("BASE_DIR   :", BASE_DIR)
print("DATA_DIR   :", DATA_DIR)
print("MODEL_DIR  :", MODEL_DIR)
print("REPORT_DIR :", REPORT_DIR)
print("PLOT_DIR   :", PLOT_DIR)
print("OUTPUT_DIR :", OUTPUT_DIR)


# ============================================================
# 2. INDIAN FOOD IMAGE DATASET PATH
# ============================================================

IMAGE_ROOT = Path(
    r"C:\indian_food_images\Indian Food Images\Indian Food Images"
)

print("\nIndian food image dataset:")
print(IMAGE_ROOT)


# ============================================================
# 3. REQUIRED DATASET FILES
# ============================================================

CHILD_DATASET = DATA_DIR / "child_dataset.csv"

FOOD_DATASETS = [
    DATA_DIR / "FOOD-DATA-GROUP1.csv",
    DATA_DIR / "FOOD-DATA-GROUP2.csv",
    DATA_DIR / "FOOD-DATA-GROUP3.csv",
    DATA_DIR / "FOOD-DATA-GROUP4.csv",
    DATA_DIR / "FOOD-DATA-GROUP5.csv"
]


# ============================================================
# 4. CHECK DATASET FILES
# ============================================================

print("\nChecking child dataset...")

if CHILD_DATASET.exists():
    print("✓ child_dataset.csv found")
else:
    print("❌ child_dataset.csv NOT FOUND")
    print("Expected location:")
    print(CHILD_DATASET)


print("\nChecking food nutrition datasets...")

for food_file in FOOD_DATASETS:

    if food_file.exists():
        print("✓", food_file.name)

    else:
        print("❌ NOT FOUND:", food_file)


# ============================================================
# 5. CHECK INDIAN FOOD IMAGE DATASET
# ============================================================

print("\nChecking Indian food image dataset...")

if IMAGE_ROOT.exists():

    print("✓ Indian food image dataset found")

    food_folders = [
        folder
        for folder in IMAGE_ROOT.iterdir()
        if folder.is_dir()
    ]

    print(
        "Number of food categories found:",
        len(food_folders)
    )

else:

    print("❌ Indian food image dataset NOT FOUND")

    print(
        "Expected location:",
        IMAGE_ROOT
    )


# ============================================================
# 6. IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# 7. CHILD AGE RANGE
# ============================================================

MIN_AGE_MONTHS = 0
MAX_AGE_MONTHS = 60


# ============================================================
# 8. MODEL FEATURE NAMES
# ============================================================

SELECTED_FEATURES = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi"
]


# ============================================================
# 9. NUTRIENT COLUMN MAPPING
# ============================================================

NUTRIENT_COLUMNS = {

    "energy": "Caloric Value",

    "protein": "Protein",

    "fat": "Fat",

    "carbohydrate": "Carbohydrates",

    "fiber": "Dietary Fiber",

    "iron": "Iron",

    "calcium": "Calcium",

    "zinc": "Zinc",

    "vitamin_a": "Vitamin A",

    "vitamin_b12": "Vitamin B12",

    "vitamin_c": "Vitamin C",

    "vitamin_d": "Vitamin D",

    "vitamin_e": "Vitamin E",

    "vitamin_k": "Vitamin K",

    "potassium": "Potassium",

    "magnesium": "Magnesium",

    "phosphorus": "Phosphorus"
}


# ============================================================
# 10. VERIFY IMPORTANT PATHS
# ============================================================

print("\n" + "=" * 70)
print("PART 1 SETUP COMPLETED")
print("=" * 70)

print("\nChild dataset:")
print(CHILD_DATASET)

print("\nNutrition datasets:")
for file in FOOD_DATASETS:
    print(" -", file.name)

print("\nImage dataset:")
print(IMAGE_ROOT)

print("\nSupported child age:")
print(
    f"{MIN_AGE_MONTHS} to {MAX_AGE_MONTHS} months"
)

print("\nSelected ML features:")
for feature in SELECTED_FEATURES:
    print(" -", feature)

print("\nNutrient mappings:")
for key, value in NUTRIENT_COLUMNS.items():
    print(f" - {key} -> {value}")

print("\n✓ Ready for Part 2")
# ================= ORIGINAL NOTEBOOK CELL 2 =================
# ============================================================
# PART 2
# LOAD AND INSPECT CHILD + FOOD NUTRITION DATASETS
# ============================================================

import pandas as pd
import numpy as np

print("=" * 75)
print("PART 2 — DATASET LOADING AND INSPECTION")
print("=" * 75)


# ============================================================
# 1. LOAD CHILD DATASET
# ============================================================

print("\n" + "=" * 75)
print("LOADING CHILD DATASET")
print("=" * 75)

if not CHILD_DATASET.exists():
    raise FileNotFoundError(
        f"Child dataset not found:\n{CHILD_DATASET}"
    )

child_df = pd.read_csv(CHILD_DATASET)

print("\n✓ Child dataset loaded successfully")

print("Rows    :", child_df.shape[0])
print("Columns :", child_df.shape[1])


# ============================================================
# 2. DISPLAY CHILD DATASET
# ============================================================

print("\nFirst 5 rows of child dataset:")

display(
    child_df.head()
)


# ============================================================
# 3. DISPLAY CHILD DATASET COLUMNS
# ============================================================

print("\nChild dataset columns:")

for column in child_df.columns:
    print(" -", column)


# ============================================================
# 4. REQUIRED CHILD COLUMNS
# ============================================================

required_child_columns = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "nutrition_status"
]


print("\nChecking required child columns...")

missing_child_columns = [
    column
    for column in required_child_columns
    if column not in child_df.columns
]


if missing_child_columns:

    print(
        "\n❌ Missing child dataset columns:"
    )

    for column in missing_child_columns:
        print(" -", column)

    raise ValueError(
        "Required columns are missing from child_dataset.csv"
    )

else:

    print(
        "✓ All required child columns are present"
    )


# ============================================================
# 5. CHILD DATA TYPES
# ============================================================

print("\nChild dataset data types:")

display(
    child_df[
        required_child_columns
    ].dtypes.to_frame(
        name="Data Type"
    )
)


# ============================================================
# 6. CHILD MISSING VALUES
# ============================================================

print("\nMissing values in child dataset:")

child_missing = (
    child_df[
        required_child_columns
    ]
    .isnull()
    .sum()
    .to_frame(
        name="Missing Values"
    )
)

display(child_missing)


# ============================================================
# 7. CHILD BASIC STATISTICS
# ============================================================

print("\nChild dataset numerical summary:")

display(
    child_df[
        [
            "age_months",
            "weight_kg",
            "height_cm",
            "muac_cm"
        ]
    ].describe()
)


# ============================================================
# 8. NUTRITION STATUS DISTRIBUTION
# ============================================================

print("\nNutrition status distribution:")

status_counts = (
    child_df[
        "nutrition_status"
    ]
    .value_counts(
        dropna=False
    )
)

display(status_counts.to_frame(
    name="Number of Children"
))


# ============================================================
# 9. LOAD ALL FIVE FOOD DATASETS
# ============================================================

print("\n" + "=" * 75)
print("LOADING FOOD NUTRITION DATASETS")
print("=" * 75)


food_dataframes = []

for food_file in FOOD_DATASETS:

    if not food_file.exists():

        raise FileNotFoundError(
            f"Food nutrition dataset not found:\n{food_file}"
        )

    current_food_df = pd.read_csv(
        food_file
    )

    print(
        f"\n✓ {food_file.name}"
    )

    print(
        "  Rows    :",
        current_food_df.shape[0]
    )

    print(
        "  Columns :",
        current_food_df.shape[1]
    )

    food_dataframes.append(
        current_food_df
    )


# ============================================================
# 10. COMBINE ALL FIVE FOOD DATASETS
# ============================================================

food_df = pd.concat(
    food_dataframes,
    ignore_index=True
)


print("\n" + "=" * 75)
print("COMBINED FOOD NUTRITION DATASET")
print("=" * 75)

print(
    "Total rows    :",
    food_df.shape[0]
)

print(
    "Total columns :",
    food_df.shape[1]
)


# ============================================================
# 11. DISPLAY FOOD DATASET
# ============================================================

print("\nFirst 5 rows of combined food dataset:")

display(
    food_df.head()
)


# ============================================================
# 12. FOOD DATASET COLUMNS
# ============================================================

print("\nFood dataset columns:")

for column in food_df.columns:
    print(" -", column)


# ============================================================
# 13. CHECK FOOD COLUMN
# ============================================================

if "food" not in food_df.columns:

    raise ValueError(
        "The combined food dataset must contain a 'food' column."
    )

else:

    print(
        "\n✓ 'food' column found"
    )


# ============================================================
# 14. CHECK REQUIRED NUTRIENT COLUMNS
# ============================================================

print("\nChecking nutrient columns...")

missing_nutrient_columns = []

for nutrient, column in NUTRIENT_COLUMNS.items():

    if column not in food_df.columns:

        missing_nutrient_columns.append(
            column
        )


if missing_nutrient_columns:

    print(
        "\n❌ Missing nutrient columns:"
    )

    for column in missing_nutrient_columns:
        print(" -", column)

    raise ValueError(
        "One or more required nutrient columns are missing."
    )

else:

    print(
        "✓ All required nutrient columns are present"
    )


# ============================================================
# 15. FOOD DATA TYPES
# ============================================================

print("\nFood nutrient data types:")

nutrient_columns_available = [
    column
    for column in NUTRIENT_COLUMNS.values()
    if column in food_df.columns
]

display(
    food_df[
        nutrient_columns_available
    ].dtypes.to_frame(
        name="Data Type"
    )
)


# ============================================================
# 16. FOOD MISSING VALUES
# ============================================================

print("\nMissing values in food dataset:")

food_missing = (
    food_df[
        nutrient_columns_available
    ]
    .isnull()
    .sum()
    .to_frame(
        name="Missing Values"
    )
)

display(food_missing)


# ============================================================
# 17. CONVERT NUTRIENT COLUMNS TO NUMERIC
# ============================================================

print("\nConverting nutrient columns to numeric...")

for column in nutrient_columns_available:

    food_df[column] = pd.to_numeric(
        food_df[column],
        errors="coerce"
    )


print(
    "✓ Nutrient columns converted to numeric"
)


# ============================================================
# 18. REMOVE DUPLICATE FOOD RECORDS
# ============================================================

before_duplicates = len(food_df)

food_df = food_df.drop_duplicates(
    subset=["food"]
).reset_index(
    drop=True
)

after_duplicates = len(food_df)

print("\nDuplicate food records removed:")

print(
    "Before :",
    before_duplicates
)

print(
    "After  :",
    after_duplicates
)

print(
    "Removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 19. FOOD NAME NORMALIZATION
# ============================================================

def normalize_food_name(name):

    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    name = name.replace(
        "_",
        " "
    )

    name = name.replace(
        "-",
        " "
    )

    name = " ".join(
        name.split()
    )

    return name


food_df[
    "_normalized_food_name"
] = (
    food_df[
        "food"
    ]
    .apply(
        normalize_food_name
    )
)


# ============================================================
# 20. CHECK DUPLICATES AFTER NORMALIZATION
# ============================================================

normalized_duplicates = (
    food_df[
        "_normalized_food_name"
    ]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate normalized food names:",
    normalized_duplicates
)


# ============================================================
# 21. DISPLAY SAMPLE FOOD NAMES
# ============================================================

print("\nSample food names:")

display(
    food_df[
        [
            "food",
            "_normalized_food_name"
        ]
    ].head(20)
)


# ============================================================
# 22. FOOD NUTRIENT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FOOD NUTRIENT SUMMARY")
print("=" * 75)

display(
    food_df[
        nutrient_columns_available
    ].describe().T
)


# ============================================================
# 23. SAVE CLEAN COMBINED FOOD DATASET
# ============================================================

clean_food_dataset_path = (
    DATA_DIR /
    "combined_food_nutrition.csv"
)

food_df.to_csv(
    clean_food_dataset_path,
    index=False
)

print(
    "\n✓ Combined food dataset saved:"
)

print(
    clean_food_dataset_path
)


# ============================================================
# 24. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PART 2 DATASET SUMMARY")
print("=" * 75)

print(
    "\nChild dataset"
)

print(
    "Rows:",
    len(child_df)
)

print(
    "Columns:",
    len(child_df.columns)
)


print(
    "\nCombined food nutrition dataset"
)

print(
    "Rows:",
    len(food_df)
)

print(
    "Columns:",
    len(food_df.columns)
)


print(
    "\nRequired child columns:"
)

for column in required_child_columns:
    print(
        " ✓",
        column
    )


print(
    "\nRequired nutrition columns:"
)

for nutrient, column in NUTRIENT_COLUMNS.items():

    print(
        " ✓",
        nutrient,
        "->",
        column
    )


print("\n" + "=" * 75)
print("✓ PART 2 COMPLETED SUCCESSFULLY")
print("=" * 75)
# ================= ORIGINAL NOTEBOOK CELL 3 =================
# ============================================================
# PART 3
# CHILD DATA PREPROCESSING
# BMI CALCULATION
# DATA QUALITY CHECKS
# EXPLORATORY VISUALIZATION
# TARGET ENCODING
# TRAIN / TEST SPLIT
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


print("=" * 75)
print("PART 3 — CHILD DATA PREPROCESSING")
print("=" * 75)


# ============================================================
# 1. COPY ORIGINAL CHILD DATASET
# ============================================================

child_processed_df = child_df.copy()

print("\nOriginal dataset shape:")
print(child_processed_df.shape)


# ============================================================
# 2. REQUIRED NUMERIC COLUMNS
# ============================================================

numeric_child_columns = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm"
]


# ============================================================
# 3. CONVERT NUMERIC COLUMNS
# ============================================================

print("\nConverting child measurements to numeric...")

for column in numeric_child_columns:

    child_processed_df[column] = pd.to_numeric(
        child_processed_df[column],
        errors="coerce"
    )

print("✓ Numeric conversion completed")


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")

display(
    child_processed_df[
        numeric_child_columns + ["nutrition_status"]
    ].isnull().sum().to_frame(
        name="Missing Values"
    )
)


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

before_missing = len(child_processed_df)

child_processed_df = child_processed_df.dropna(
    subset=numeric_child_columns + ["nutrition_status"]
).copy()

after_missing = len(child_processed_df)

print("\nMissing-value cleaning:")
print("Rows before :", before_missing)
print("Rows after  :", after_missing)
print("Removed     :", before_missing - after_missing)


# ============================================================
# 6. AGE VALIDATION
# ============================================================

print("\nValidating age range 0–60 months...")

invalid_age_mask = (
    (child_processed_df["age_months"] < 0) |
    (child_processed_df["age_months"] > 60)
)

invalid_age_count = invalid_age_mask.sum()

print(
    "Invalid age records:",
    invalid_age_count
)

if invalid_age_count > 0:

    child_processed_df = child_processed_df[
        ~invalid_age_mask
    ].copy()

    print(
        "✓ Invalid age records removed"
    )

else:

    print(
        "✓ All ages are within 0–60 months"
    )


# ============================================================
# 7. WEIGHT VALIDATION
# ============================================================

invalid_weight_mask = (
    child_processed_df["weight_kg"] <= 0
)

invalid_weight_count = (
    invalid_weight_mask.sum()
)

print(
    "\nInvalid weight records:",
    invalid_weight_count
)

if invalid_weight_count > 0:

    child_processed_df = child_processed_df[
        ~invalid_weight_mask
    ].copy()


# ============================================================
# 8. HEIGHT VALIDATION
# ============================================================

invalid_height_mask = (
    child_processed_df["height_cm"] <= 0
)

invalid_height_count = (
    invalid_height_mask.sum()
)

print(
    "Invalid height records:",
    invalid_height_count
)

if invalid_height_count > 0:

    child_processed_df = child_processed_df[
        ~invalid_height_mask
    ].copy()


# ============================================================
# 9. MUAC VALIDATION
# ============================================================

invalid_muac_mask = (
    child_processed_df["muac_cm"] <= 0
)

invalid_muac_count = (
    invalid_muac_mask.sum()
)

print(
    "Invalid MUAC records:",
    invalid_muac_count
)

if invalid_muac_count > 0:

    child_processed_df = child_processed_df[
        ~invalid_muac_mask
    ].copy()


# ============================================================
# 10. REMOVE DUPLICATE CHILD RECORDS
# ============================================================

print("\nChecking duplicate child records...")

before_duplicates = len(
    child_processed_df
)

child_processed_df = child_processed_df.drop_duplicates(
    subset=[
        "age_months",
        "weight_kg",
        "height_cm",
        "muac_cm",
        "nutrition_status"
    ]
).reset_index(drop=True)

after_duplicates = len(
    child_processed_df
)

duplicates_removed = (
    before_duplicates -
    after_duplicates
)

print(
    "Duplicate records removed:",
    duplicates_removed
)


# ============================================================
# 11. CALCULATE BMI
# ============================================================

# BMI = weight in kg / height in metres squared

child_processed_df["height_m"] = (
    child_processed_df["height_cm"] / 100
)

child_processed_df["bmi"] = (

    child_processed_df["weight_kg"]

    /

    (
        child_processed_df["height_m"] ** 2
    )
)


# ============================================================
# 12. REMOVE INVALID BMI VALUES
# ============================================================

invalid_bmi_mask = (
    ~np.isfinite(
        child_processed_df["bmi"]
    )
    |
    (child_processed_df["bmi"] <= 0)
)

invalid_bmi_count = (
    invalid_bmi_mask.sum()
)

print(
    "\nInvalid BMI records:",
    invalid_bmi_count
)

if invalid_bmi_count > 0:

    child_processed_df = child_processed_df[
        ~invalid_bmi_mask
    ].copy()


print(
    "✓ BMI calculation completed"
)


# ============================================================
# 13. CLEAN NUTRITION STATUS
# ============================================================

child_processed_df[
    "nutrition_status"
] = (

    child_processed_df[
        "nutrition_status"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


invalid_target_values = [
    "",
    "nan",
    "none"
]

child_processed_df = child_processed_df[
    ~child_processed_df[
        "nutrition_status"
    ].isin(
        invalid_target_values
    )
].copy()


# ============================================================
# 14. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("FINAL DATA QUALITY CHECK")
print("=" * 75)

final_columns = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi",
    "nutrition_status"
]

print("\nMissing values:")

display(
    child_processed_df[
        final_columns
    ].isnull().sum().to_frame(
        name="Missing Values"
    )
)


# ============================================================
# 15. PROCESSED DATASET PREVIEW
# ============================================================

print("\nProcessed child dataset:")

display(
    child_processed_df[
        final_columns
    ].head(10)
)


# ============================================================
# 16. BASIC STATISTICS
# ============================================================

print("\nNumerical feature statistics:")

display(
    child_processed_df[
        [
            "age_months",
            "weight_kg",
            "height_cm",
            "muac_cm",
            "bmi"
        ]
    ].describe().T
)


# ============================================================
# 17. NUTRITION STATUS DISTRIBUTION
# ============================================================

print("\nNutrition status distribution:")

target_distribution = (
    child_processed_df[
        "nutrition_status"
    ]
    .value_counts()
    .sort_index()
)

display(
    target_distribution.to_frame(
        name="Number of Children"
    )
)


# ============================================================
# 18. NUTRITION STATUS VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

target_distribution.plot(
    kind="bar"
)

plt.title(
    "Child Nutrition Status Distribution"
)

plt.xlabel(
    "Nutrition Status"
)

plt.ylabel(
    "Number of Children"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

target_plot_path = (
    PLOT_DIR /
    "nutrition_status_distribution.png"
)

plt.savefig(
    target_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


print(
    "\n✓ Target distribution plot saved:"
)

print(
    target_plot_path
)


# ============================================================
# 19. FEATURE DISTRIBUTIONS
# ============================================================

features_to_plot = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi"
]

print("\nGenerating feature distributions...")

for feature in features_to_plot:

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        child_processed_df[
            feature
        ].dropna(),
        bins=30
    )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.xlabel(
        feature
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plot_path = (
        PLOT_DIR /
        f"{feature}_distribution.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


print(
    "✓ Feature distribution plots generated"
)


# ============================================================
# 20. CORRELATION ANALYSIS
# ============================================================

correlation_features = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi"
]

correlation_matrix = (
    child_processed_df[
        correlation_features
    ]
    .corr()
)


print("\nFeature correlation matrix:")

display(
    correlation_matrix
)


# ============================================================
# 21. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    correlation_matrix,
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation_features)),
    correlation_features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_features)),
    correlation_features
)

plt.title(
    "Child Nutrition Feature Correlation"
)

plt.tight_layout()

correlation_plot_path = (
    PLOT_DIR /
    "feature_correlation_matrix.png"
)

plt.savefig(
    correlation_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


print(
    "✓ Correlation matrix saved:"
)

print(
    correlation_plot_path
)


# ============================================================
# 22. DEFINE ML FEATURES
# ============================================================

selected_features = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi"
]


print("\nML input features:")

for feature in selected_features:

    print(
        " ✓",
        feature
    )


# ============================================================
# 23. CREATE X AND TARGET
# ============================================================

X = child_processed_df[
    selected_features
].copy()

y_text = child_processed_df[
    "nutrition_status"
].copy()


print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget vector shape:")
print(y_text.shape)


# ============================================================
# 24. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)


print("\nLabel encoding:")

for encoded_value, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f" {encoded_value} -> {class_name}"
    )


# ============================================================
# 25. CLASS COUNTS
# ============================================================

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)


print("\nEncoded class counts:")

for class_index, count in class_counts.items():

    print(
        label_encoder.classes_[class_index],
        "->",
        count
    )


# ============================================================
# 26. CHECK STRATIFIED SPLIT POSSIBILITY
# ============================================================

minimum_class_count = (
    class_counts.min()
)

if minimum_class_count < 2:

    raise ValueError(
        "Every nutrition-status class must contain "
        "at least 2 records for a stratified split."
    )


# ============================================================
# 27. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )
)


print("\n" + "=" * 75)
print("TRAIN / TEST SPLIT")
print("=" * 75)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 28. STANDARD SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


print(
    "\n✓ StandardScaler fitted only on training data"
)


# ============================================================
# 29. SCALED DATAFRAMES
# ============================================================

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=selected_features,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=selected_features,
    index=X_test.index
)


# ============================================================
# 30. DISPLAY SCALED DATA
# ============================================================

print("\nTraining data before scaling:")

display(
    X_train.head()
)


print("\nTraining data after scaling:")

display(
    X_train_scaled_df.head()
)


# ============================================================
# 31. SAVE LABEL ENCODER
# ============================================================

label_encoder_path = (
    MODEL_DIR /
    "label_encoder.pkl"
)

joblib.dump(
    label_encoder,
    label_encoder_path
)


# ============================================================
# 32. SAVE SCALER
# ============================================================

scaler_path = (
    MODEL_DIR /
    "standard_scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)


# ============================================================
# 33. SAVE PROCESSED CHILD DATASET
# ============================================================

processed_child_path = (
    DATA_DIR /
    "processed_child_dataset.csv"
)

child_processed_df.to_csv(
    processed_child_path,
    index=False
)


# ============================================================
# 34. SAVE FEATURE INFORMATION
# ============================================================

feature_info = {

    "selected_features":
        selected_features,

    "target_column":
        "nutrition_status",

    "age_min_months":
        0,

    "age_max_months":
        60,

    "bmi_formula":
        "weight_kg / (height_m ** 2)"
}


feature_info_path = (
    MODEL_DIR /
    "feature_info.json"
)

with open(
    feature_info_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_info,
        file,
        indent=4
    )


# ============================================================
# 35. SAVE CORRELATION MATRIX
# ============================================================

correlation_path = (
    REPORT_DIR /
    "feature_correlation.csv"
)

correlation_matrix.to_csv(
    correlation_path
)


# ============================================================
# 36. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PART 3 COMPLETED")
print("=" * 75)

print(
    "\nFinal processed records:",
    len(child_processed_df)
)

print(
    "ML features:",
    selected_features
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "\nNutrition classes:"
)

for class_name in label_encoder.classes_:

    print(
        " ✓",
        class_name
    )


print("\nSaved files:")

print(
    "✓",
    processed_child_path
)

print(
    "✓",
    label_encoder_path
)

print(
    "✓",
    scaler_path
)

print(
    "✓",
    feature_info_path
)

print(
    "✓",
    correlation_path
)

print("\n" + "=" * 75)
print("✓ PART 3 READY FOR PART 4")
print("=" * 75)
# ================= PART 4 (callable training logic) =================
def train_part4_models():
    # ============================================================
    # PART 4
    # TRAIN, EVALUATE AND COMPARE 5 ML MODELS
    #
    # Models:
    # 1. Logistic Regression
    # 2. Decision Tree
    # 3. Random Forest
    # 4. XGBoost
    # 5. LightGBM
    #
    # Evaluation:
    # Accuracy
    # Precision
    # Recall
    # F1-score
    # Classification Report
    # Confusion Matrix
    # ============================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    import json

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay
    )

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier


    print("=" * 80)
    print("PART 4 — ML MODEL TRAINING AND EVALUATION")
    print("=" * 80)


    # ============================================================
    # 1. CHECK REQUIRED VARIABLES FROM PART 3
    # ============================================================

    required_variables = [
        "X_train",
        "X_test",
        "X_train_scaled",
        "X_test_scaled",
        "y_train",
        "y_test",
        "label_encoder",
        "selected_features"
    ]

    missing_variables = []

    for variable_name in required_variables:

        if variable_name not in globals():

            missing_variables.append(
                variable_name
            )


    if missing_variables:

        raise NameError(
            "The following variables are missing from Part 3:\n"
            + "\n".join(
                f" - {name}"
                for name in missing_variables
            )
            + "\n\nPlease run Part 3 first."
        )


    print("\n✓ All required Part 3 variables are available")


    # ============================================================
    # 2. DISPLAY DATASET INFORMATION
    # ============================================================

    print("\n" + "=" * 80)
    print("DATA INFORMATION")
    print("=" * 80)

    print(
        "\nSelected features:"
    )

    for feature in selected_features:

        print(
            " ✓",
            feature
        )


    print(
        "\nNumber of training samples:",
        len(X_train)
    )

    print(
        "Number of testing samples :",
        len(X_test)
    )

    print(
        "Number of classes         :",
        len(label_encoder.classes_)
    )

    print(
        "\nNutrition classes:"
    )

    for index, class_name in enumerate(
        label_encoder.classes_
    ):

        print(
            f" {index} → {class_name}"
        )


    # ============================================================
    # 3. DETERMINE NUMBER OF CLASSES
    # ============================================================

    number_of_classes = len(
        label_encoder.classes_
    )

    if number_of_classes < 2:

        raise ValueError(
            "At least two nutrition-status classes are required."
        )


    # ============================================================
    # 4. CREATE MODEL DIRECTORY
    # ============================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ============================================================
    # 5. CREATE FIVE ML MODELS
    # ============================================================

    print("\n" + "=" * 80)
    print("CREATING ML MODELS")
    print("=" * 80)


    # ------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )


    # ------------------------------------------------------------
    # Decision Tree
    # ------------------------------------------------------------

    decision_tree_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2
    )


    # ------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )


    # ------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------

    if number_of_classes == 2:

        xgboost_model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )

    else:

        xgboost_model = XGBClassifier(
            objective="multi:softprob",
            num_class=number_of_classes,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1
        )


    # ------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------

    if number_of_classes == 2:

        lightgbm_model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

    else:

        lightgbm_model = LGBMClassifier(
            objective="multiclass",
            num_class=number_of_classes,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )


    # ============================================================
    # 6. STORE MODELS
    # ============================================================

    models = {

        "Logistic Regression":
            logistic_model,

        "Decision Tree":
            decision_tree_model,

        "Random Forest":
            random_forest_model,

        "XGBoost":
            xgboost_model,

        "LightGBM":
            lightgbm_model
    }


    print("\nModels created:")

    for model_name in models:

        print(
            " ✓",
            model_name
        )


    # ============================================================
    # 7. PREPARE TRAINING DATA
    # ============================================================

    # Logistic Regression works with scaled features.
    #
    # Tree-based models do not require scaling.
    #
    # Therefore:
    #
    # Logistic Regression → scaled data
    # Decision Tree       → original data
    # Random Forest       → original data
    # XGBoost             → original data
    # LightGBM            → original data


    model_training_data = {

        "Logistic Regression": (
            X_train_scaled,
            X_test_scaled
        ),

        "Decision Tree": (
            X_train,
            X_test
        ),

        "Random Forest": (
            X_train,
            X_test
        ),

        "XGBoost": (
            X_train,
            X_test
        ),

        "LightGBM": (
            X_train,
            X_test
        )
    }


    # ============================================================
    # 8. TRAIN ALL MODELS
    # ============================================================

    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)


    trained_models = {}

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        train_data, test_data = (
            model_training_data[model_name]
        )

        model.fit(
            train_data,
            y_train
        )

        trained_models[
            model_name
        ] = model

        print(
            f"✓ {model_name} training completed"
        )


    # ============================================================
    # 9. EVALUATE ALL MODELS
    # ============================================================

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)


    results = []

    classification_reports = {}

    predictions = {}


    for model_name, model in trained_models.items():

        print("\n")
        print("#" * 80)
        print(
            f"# {model_name.upper()}"
        )
        print("#" * 80)


        train_data, test_data = (
            model_training_data[model_name]
        )


        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        y_pred = model.predict(
            test_data
        )


        predictions[
            model_name
        ] = y_pred


        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )


        # --------------------------------------------------------
        # Store metrics
        # --------------------------------------------------------

        results.append({

            "Model":
                model_name,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1 Score":
                f1
        })


        # --------------------------------------------------------
        # Classification Report
        # --------------------------------------------------------

        report = classification_report(

            y_test,

            y_pred,

            labels=np.arange(
                number_of_classes
            ),

            target_names=label_encoder.classes_,

            digits=4,

            zero_division=0
        )


        classification_reports[
            model_name
        ] = report


        # --------------------------------------------------------
        # Display Classification Report
        # --------------------------------------------------------

        print(
            "\nClassification Report:"
        )

        print(
            report
        )


        # --------------------------------------------------------
        # Display Overall Metrics
        # --------------------------------------------------------

        print(
            "Overall Metrics:"
        )

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )


    # ============================================================
    # 10. CREATE MODEL COMPARISON DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(
        results
    )


    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    display(
        results_df.style.format({

            "Accuracy": "{:.4f}",

            "Precision": "{:.4f}",

            "Recall": "{:.4f}",

            "F1 Score": "{:.4f}"
        })
    )


    # ============================================================
    # 11. AUTOMATICALLY SELECT BEST MODEL
    # ============================================================

    # Primary criterion:
    # Weighted F1 Score
    #
    # Tie-break:
    # Accuracy


    results_sorted_df = (
        results_df
        .sort_values(
            by=[
                "F1 Score",
                "Accuracy"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )


    best_model_name = (
        results_sorted_df
        .iloc[0]["Model"]
    )


    best_model = trained_models[
        best_model_name
    ]


    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)

    print(
        "\nSelected model:",
        best_model_name
    )


    best_model_row = results_sorted_df.iloc[0]


    print(
        f"Accuracy  : {best_model_row['Accuracy']:.4f}"
    )

    print(
        f"Precision : {best_model_row['Precision']:.4f}"
    )

    print(
        f"Recall    : {best_model_row['Recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_model_row['F1 Score']:.4f}"
    )


    # ============================================================
    # 12. MODEL COMPARISON BAR CHART
    # ============================================================

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]


    x = np.arange(
        len(results_df["Model"])
    )

    width = 0.18


    plt.figure(
        figsize=(14, 7)
    )


    for index, metric in enumerate(
        metrics_to_plot
    ):

        plt.bar(

            x +
            (
                index -
                1.5
            ) * width,

            results_df[metric],

            width,

            label=metric
        )


    plt.xticks(
        x,
        results_df["Model"],
        rotation=25,
        ha="right"
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.title(
        "Comparison of ML Model Performance"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.tight_layout()


    comparison_plot_path = (
        PLOT_DIR /
        "ml_model_comparison.png"
    )


    plt.savefig(
        comparison_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    print(
        "\n✓ Model comparison plot saved:"
    )

    print(
        comparison_plot_path
    )


    # ============================================================
    # 13. CONFUSION MATRICES FOR ALL 5 MODELS
    # ============================================================

    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)


    for model_name, y_pred in predictions.items():

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=np.arange(
                number_of_classes
            )
        )


        fig, ax = plt.subplots(
            figsize=(8, 6)
        )


        display_object = (
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=label_encoder.classes_
            )
        )


        display_object.plot(
            ax=ax,
            xticks_rotation=45,
            values_format="d"
        )


        ax.set_title(
            f"Confusion Matrix — {model_name}"
        )


        plt.tight_layout()


        safe_model_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )


        confusion_plot_path = (
            PLOT_DIR /
            f"confusion_matrix_{safe_model_name}.png"
        )


        plt.savefig(
            confusion_plot_path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.show()

        plt.close()


        print(
            f"✓ Saved: {confusion_plot_path}"
        )


    # ============================================================
    # 14. SAVE MODEL COMPARISON
    # ============================================================

    comparison_csv_path = (
        REPORT_DIR /
        "model_comparison.csv"
    )


    results_df.to_csv(
        comparison_csv_path,
        index=False
    )


    print(
        "\n✓ Model comparison saved:"
    )

    print(
        comparison_csv_path
    )


    # ============================================================
    # 15. SAVE ALL CLASSIFICATION REPORTS
    # ============================================================

    classification_report_path = (
        REPORT_DIR /
        "classification_reports.txt"
    )


    with open(
        classification_report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CHILD NUTRITION ML MODEL CLASSIFICATION REPORTS\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )


        for model_name, report in (
            classification_reports.items()
        ):

            file.write(
                "\n"
            )

            file.write(
                "#" * 80
            )

            file.write(
                "\n"
            )

            file.write(
                model_name
            )

            file.write(
                "\n"
            )

            file.write(
                "#" * 80
            )

            file.write(
                "\n\n"
            )

            file.write(
                report
            )

            file.write(
                "\n\n"
            )


    print(
        "✓ Classification reports saved:"
    )

    print(
        classification_report_path
    )


    # ============================================================
    # 16. SAVE EACH TRAINED MODEL
    # ============================================================

    print("\n" + "=" * 80)
    print("SAVING TRAINED MODELS")
    print("=" * 80)


    model_file_paths = {}


    for model_name, model in trained_models.items():

        safe_model_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )


        model_path = (
            MODEL_DIR /
            f"{safe_model_name}.pkl"
        )


        joblib.dump(
            model,
            model_path
        )


        model_file_paths[
            model_name
        ] = str(model_path)


        print(
            f"✓ {model_name} → {model_path}"
        )


    # ============================================================
    # 17. SAVE BEST MODEL
    # ============================================================

    best_model_path = (
        MODEL_DIR /
        "best_nutrition_model.pkl"
    )


    joblib.dump(
        best_model,
        best_model_path
    )


    print(
        "\n✓ Best model saved:"
    )

    print(
        best_model_path
    )


    # ============================================================
    # 18. SAVE BEST MODEL INFORMATION
    # ============================================================

    best_model_information = {

        "best_model":
            best_model_name,

        "selection_criterion":
            "Weighted F1 Score, with Accuracy as tie-breaker",

        "accuracy":
            float(
                best_model_row["Accuracy"]
            ),

        "precision":
            float(
                best_model_row["Precision"]
            ),

        "recall":
            float(
                best_model_row["Recall"]
            ),

        "f1_score":
            float(
                best_model_row["F1 Score"]
            ),

        "features":
            selected_features,

        "classes":
            label_encoder.classes_.tolist()
    }


    best_model_info_path = (
        MODEL_DIR /
        "best_model_information.json"
    )


    with open(
        best_model_info_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_model_information,
            file,
            indent=4
        )


    print(
        "✓ Best-model information saved:"
    )

    print(
        best_model_info_path
    )


    # ============================================================
    # 19. FINAL MODEL COMPARISON
    # ============================================================

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON — SORTED")
    print("=" * 80)


    display(
        results_sorted_df.style.format({

            "Accuracy": "{:.4f}",

            "Precision": "{:.4f}",

            "Recall": "{:.4f}",

            "F1 Score": "{:.4f}"
        })
    )


    # ============================================================
    # 20. FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 80)
    print("✓ PART 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        "\nFive models trained:"
    )

    for model_name in models:

        print(
            " ✓",
            model_name
        )


    print(
        "\nAutomatically selected best model:"
    )

    print(
        " ★",
        best_model_name
    )


    print(
        "\nBest-model F1 Score:",
        f"{best_model_row['F1 Score']:.4f}"
    )


    print(
        "\nSaved model:"
    )

    print(
        best_model_path
    )


    print(
        "\nSaved comparison:"
    )

    print(
        comparison_csv_path
    )


    print(
        "\nSaved classification reports:"
    )

    print(
        classification_report_path
    )


    print(
        "\n" + "=" * 80
    )
    print(
        "READY FOR PART 5"
    )
    print(
        "=" * 80
    )

# ================= ORIGINAL NOTEBOOK CELL 5 =================
# ============================================================
# PART 5
# CHILD NUTRITION PREDICTION AGENT
#
# Loads the best trained model from Part 4
# Accepts:
#   age_months
#   weight_kg
#   height_cm
#   muac_cm
#
# BMI is calculated automatically.
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


print("=" * 80)
print("PART 5 — CHILD NUTRITION PREDICTION AGENT")
print("=" * 80)


# ============================================================
# 1. CHECK REQUIRED VARIABLES
# ============================================================

required_variables = [
    "MODEL_DIR",
    "selected_features",
    "label_encoder"
]

missing_variables = []

for variable_name in required_variables:

    if variable_name not in globals():

        missing_variables.append(
            variable_name
        )


if missing_variables:

    raise NameError(
        "The following variables are missing:\n"
        + "\n".join(
            f" - {name}"
            for name in missing_variables
        )
        + "\n\nPlease run Part 3 first."
    )


print("\n✓ Required variables are available")


# ============================================================
# 2. MODEL PATH
# ============================================================

best_model_path = (
    MODEL_DIR /
    "best_nutrition_model.pkl"
)


if not best_model_path.exists():

    raise FileNotFoundError(
        "\nBest model file was not found:\n"
        f"{best_model_path}\n\n"
        "Please run Part 4 completely first."
    )


print(
    "\n✓ Best model file found:"
)

print(
    best_model_path
)


# ============================================================
# 3. LOAD BEST MODEL USING JOBLIB
# ============================================================

best_model = joblib.load(
    best_model_path
)


# IMPORTANT:
# best_model must be the actual trained model object.
# It must NOT be the model name string.

if not hasattr(
    best_model,
    "predict"
):

    raise TypeError(
        "\nLoaded best model is not a trained "
        "machine-learning model.\n"
        "Please run Part 4 again and then run Part 5."
    )


print(
    "\n✓ Best trained model loaded successfully"
)


# ============================================================
# 4. LOAD BEST MODEL INFORMATION
# ============================================================

best_model_info_path = (
    MODEL_DIR /
    "best_model_information.json"
)


if best_model_info_path.exists():

    with open(
        best_model_info_path,
        "r",
        encoding="utf-8"
    ) as file:

        best_model_information = json.load(
            file
        )

else:

    best_model_information = {}


# ------------------------------------------------------------
# Model name
# ------------------------------------------------------------

best_model_name = (
    best_model_information.get(
        "best_model",
        type(best_model).__name__
    )
)


print(
    "✓ Best model:",
    best_model_name
)


# ============================================================
# 5. LOAD STANDARD SCALER
# ============================================================

scaler_path = (
    MODEL_DIR /
    "standard_scaler.pkl"
)


if not scaler_path.exists():

    raise FileNotFoundError(
        "\nStandard scaler was not found:\n"
        f"{scaler_path}\n\n"
        "Please run Part 3 first."
    )


standard_scaler = joblib.load(
    scaler_path
)


print(
    "✓ Standard scaler loaded"
)


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_child(
    age_months,
    weight_kg,
    height_cm,
    muac_cm
):
    """
    Predict child nutritional status.

    Inputs:
        age_months : Child age in months
        weight_kg   : Weight in kilograms
        height_cm   : Height in centimeters
        muac_cm     : MUAC in centimeters

    BMI is calculated automatically.

    Returns:
        Dictionary containing:
        - child measurements
        - BMI
        - predicted class
        - predicted status
        - probabilities
        - best model
    """


    # ========================================================
    # INPUT CONVERSION
    # ========================================================

    age_months = float(
        age_months
    )

    weight_kg = float(
        weight_kg
    )

    height_cm = float(
        height_cm
    )

    muac_cm = float(
        muac_cm
    )


    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not 0 <= age_months <= 60:

        raise ValueError(
            "Age must be between 0 and 60 months."
        )


    if weight_kg <= 0:

        raise ValueError(
            "Weight must be greater than zero."
        )


    if height_cm <= 0:

        raise ValueError(
            "Height must be greater than zero."
        )


    if muac_cm <= 0:

        raise ValueError(
            "MUAC must be greater than zero."
        )


    # ========================================================
    # BMI CALCULATION
    # ========================================================

    height_m = (
        height_cm / 100.0
    )


    bmi = (
        weight_kg /
        (height_m ** 2)
    )


    if not np.isfinite(bmi):

        raise ValueError(
            "Calculated BMI is invalid."
        )


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(

        [[
            age_months,
            weight_kg,
            height_cm,
            muac_cm,
            bmi
        ]],

        columns=selected_features
    )


    # ========================================================
    # PREPARE DATA FOR MODEL
    # ========================================================

    # Part 4 trained:
    #
    # Logistic Regression → scaled features
    # Tree models         → original features
    #
    # Therefore only Logistic Regression requires scaling.

    if best_model_name == "Logistic Regression":

        model_input = (
            standard_scaler.transform(
                input_data[
                    selected_features
                ]
            )
        )

    else:

        model_input = (
            input_data[
                selected_features
            ]
        )


    # ========================================================
    # PREDICT CLASS
    # ========================================================

    predicted_class_encoded = (
        best_model.predict(
            model_input
        )[0]
    )


    # Convert numpy integer to Python integer

    predicted_class_encoded = int(
        predicted_class_encoded
    )


    # ========================================================
    # CONVERT CLASS TO NUTRITION STATUS
    # ========================================================

    predicted_status = (
        label_encoder.inverse_transform(
            [predicted_class_encoded]
        )[0]
    )


    predicted_status = str(
        predicted_status
    )


    # ========================================================
    # PREDICTION PROBABILITIES
    # ========================================================

    probabilities = {}


    if hasattr(
        best_model,
        "predict_proba"
    ):

        probability_values = (
            best_model.predict_proba(
                model_input
            )[0]
        )


        # Use model.classes_ so probability
        # values are mapped to the correct
        # encoded class.

        if hasattr(
            best_model,
            "classes_"
        ):

            model_classes = (
                best_model.classes_
            )

        else:

            model_classes = np.arange(
                len(probability_values)
            )


        for class_id, probability in zip(
            model_classes,
            probability_values
        ):

            class_id = int(
                class_id
            )

            class_name = (
                label_encoder.inverse_transform(
                    [class_id]
                )[0]
            )

            probabilities[
                str(class_name)
            ] = float(
                probability
            )


    else:

        probabilities[
            predicted_status
        ] = 1.0


    # ========================================================
    # RETURN PREDICTION RESULT
    # ========================================================

    prediction_result = {

        "age_months":
            float(age_months),

        "weight_kg":
            float(weight_kg),

        "height_cm":
            float(height_cm),

        "muac_cm":
            float(muac_cm),

        "bmi":
            float(bmi),

        "predicted_class":
            int(
                predicted_class_encoded
            ),

        "predicted_status":
            str(
                predicted_status
            ),

        "probabilities":
            probabilities,

        "best_model":
            str(
                best_model_name
            )
    }


    return prediction_result


# ============================================================
# 7. SAVE INFERENCE INFORMATION
# ============================================================

inference_information = {

    "best_model":
        str(best_model_name),

    "model_file":
        str(best_model_path),

    "features":
        list(selected_features),

    "classes":
        label_encoder.classes_.tolist(),

    "input_features": [

        "age_months",

        "weight_kg",

        "height_cm",

        "muac_cm"
    ],

    "calculated_features": [

        "bmi"
    ],

    "age_range_months": {

        "minimum": 0,

        "maximum": 60
    }
}


inference_information_path = (
    MODEL_DIR /
    "inference_information.json"
)


with open(
    inference_information_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        inference_information,
        file,
        indent=4
    )


print(
    "\n✓ Inference information saved:"
)

print(
    inference_information_path
)


# ============================================================
# 8. PART 5 COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("✓ PART 5 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    "\nPrediction function:"
)

print(
    "predict_child(age_months, weight_kg, height_cm, muac_cm)"
)

print(
    "\nBest model:"
)

print(
    best_model_name
)

print(
    "\nModel object type:"
)

print(
    type(best_model).__name__
)

print(
    "\nReady for Part 6"
)

print("=" * 80)
# ================= ORIGINAL NOTEBOOK CELL 6 =================
# ============================================================
# PART 6
# CHILD-SUITABLE NUTRITION RECOMMENDATION AGENT
# ============================================================

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path.cwd() / "data"
REPORT_DIR = Path.cwd() / "reports"
MODEL_DIR = Path.cwd() / "saved_models"

FOOD_FILE = DATA_DIR / "combined_food_nutrition.csv"


# ============================================================
# NUTRIENT MAPPING
# ============================================================

NUTRIENT_COLUMNS = {
    "energy": "Caloric Value",
    "protein": "Protein",
    "fat": "Fat",
    "carbohydrate": "Carbohydrates",
    "fiber": "Dietary Fiber",
    "iron": "Iron",
    "calcium": "Calcium",
    "zinc": "Zinc",
    "vitamin_a": "Vitamin A",
    "vitamin_b12": "Vitamin B12",
    "vitamin_c": "Vitamin C",
    "vitamin_d": "Vitamin D",
    "vitamin_e": "Vitamin E",
    "vitamin_k": "Vitamin K",
    "potassium": "Potassium",
    "magnesium": "Magnesium",
    "phosphorus": "Phosphorus"
}


# ============================================================
# NUTRIENT PRIORITY WEIGHTS
# ============================================================

RECOMMENDATION_PRIORITY_WEIGHTS = {

    "normal": {
        "protein": 1.00,
        "calcium": 0.90,
        "iron": 0.90,
        "fiber": 0.75,
        "vitamin_a": 0.75,
        "vitamin_c": 0.70
    },

    "moderate": {
        "energy": 1.00,
        "protein": 1.00,
        "iron": 0.95,
        "calcium": 0.90,
        "zinc": 0.85,
        "vitamin_a": 0.80,
        "vitamin_c": 0.75
    },

    "severe": {
        "energy": 1.00,
        "protein": 1.00,
        "iron": 0.95,
        "calcium": 0.90,
        "zinc": 0.90,
        "vitamin_a": 0.85,
        "vitamin_c": 0.80,
        "vitamin_b12": 0.75
    }
}


# ============================================================
# LOAD FOOD DATA
# ============================================================

food_df = pd.read_csv(
    FOOD_FILE
)

food_df = food_df.copy()

food_df["food"] = (
    food_df["food"]
    .astype(str)
    .str.strip()
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in NUTRIENT_COLUMNS.values():

    if column in food_df.columns:

        food_df[column] = pd.to_numeric(
            food_df[column],
            errors="coerce"
        )

food_df = food_df.drop_duplicates(
    subset=["food"]
).reset_index(drop=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

food_df["_food_lower"] = (
    food_df["food"]
    .str.lower()
    .str.replace("_", " ", regex=False)
    .str.replace("-", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ============================================================
# REMOVE CLEARLY INAPPROPRIATE FOODS
# ============================================================

EXCLUDE_TERMS = [

    "raw",
    "uncooked",
    "frozen",
    "powder",
    "supplement",
    "protein powder",

    # alcohol
    "alcohol",
    "wine",
    "beer",

    # desserts / highly processed foods
    "pie",
    "cake",
    "donut",
    "doughnut",
    "candy",
    "chocolate",
    "ice cream",
    "cookie",
    "biscuit",

    # junk / fast food
    "pizza",
    "burger",
    "hot dog",
    "fries",
    "chips",

    # strongly unsuitable meat entries for this project
    "pork",
    "turkey",
    "bacon",
    "ham",
    "sausage",
    "salami",

    # very specific processed foods
    "cola",
    "soft drink",
    "soda"
]


def is_excluded_food(food_name):

    name = str(food_name).lower()

    for term in EXCLUDE_TERMS:

        if term in name:
            return True

    return False


food_df = food_df[
    ~food_df["_food_lower"].apply(
        is_excluded_food
    )
].copy()


# ============================================================
# INDIAN / CHILD-FRIENDLY FOOD KEYWORDS
# ============================================================

CHILD_FOOD_KEYWORDS = [

    # cereals / staples
    "chapati",
    "roti",
    "bread",
    "rice",
    "dosa",
    "idli",
    "upma",
    "porridge",
    "oat",
    "oats",

    # pulses
    "lentil",
    "dal",
    "dhal",
    "chickpea",
    "chana",
    "bean",
    "peas",

    # vegetables
    "spinach",
    "greens",
    "mustard",
    "okra",
    "bhindi",
    "carrot",
    "pumpkin",
    "potato",
    "sweet potato",
    "tomato",
    "peas",
    "vegetable",

    # fruits
    "banana",
    "mango",
    "papaya",
    "orange",
    "apple",

    # dairy
    "milk",
    "curd",
    "yogurt",
    "yoghurt",
    "paneer",
    "cheese",

    # nuts / seeds
    "peanut",
    "sesame",
    "almond",
    "cashew",

    # egg
    "egg"
]


def has_child_food_keyword(food_name):

    name = str(food_name).lower()

    return any(
        keyword in name
        for keyword in CHILD_FOOD_KEYWORDS
    )


food_df["child_food_candidate"] = (
    food_df["food"]
    .apply(has_child_food_keyword)
)


# ============================================================
# REQUIRE READY-TO-EAT / PREPARED FOOD
# ============================================================

PREPARED_TERMS = [

    "cooked",
    "boiled",
    "prepared",
    "masala",
    "curry",
    "tadka",
    "dal",
    "dosa",
    "idli",
    "chapati",
    "roti",
    "porridge",
    "rice",
    "upma",
    "yogurt",
    "curd",
    "milk",
    "paneer",
    "bread"
]


def has_prepared_form(food_name):

    name = str(food_name).lower()

    return any(
        term in name
        for term in PREPARED_TERMS
    )


food_df["prepared_food"] = (
    food_df["food"]
    .apply(has_prepared_form)
)


# ============================================================
# AGE GROUP
# ============================================================

def get_age_group(age_months):

    age_months = float(age_months)

    if age_months <= 6:
        return "infant"

    if age_months <= 12:
        return "young_infant"

    if age_months <= 24:
        return "young_child"

    return "child"


# ============================================================
# PRIORITY PROFILE
# ============================================================

def determine_priority_profile(predicted_status):

    status = str(
        predicted_status
    ).strip().lower()

    if "severe" in status:

        return "severe"

    if "moderate" in status:

        return "moderate"

    return "normal"


# ============================================================
# NUTRIENT PRIORITY ANALYSIS
# ============================================================

def analyze_nutrient_priorities(
    predicted_status
):

    profile = determine_priority_profile(
        predicted_status
    )

    weights = (
        RECOMMENDATION_PRIORITY_WEIGHTS[
            profile
        ]
    )

    rows = []

    for nutrient, weight in weights.items():

        rows.append({

            "nutrient": nutrient,

            "nutrient_name":
                NUTRIENT_COLUMNS[nutrient],

            "priority_weight":
                float(weight)
        })


    priority_df = pd.DataFrame(
        rows
    )

    priority_df = priority_df.sort_values(
        "priority_weight",
        ascending=False
    ).reset_index(drop=True)


    return {

        "profile": profile,

        "priority_nutrients":
            rows,

        "priority_dataframe":
            priority_df,

        "priority_names":
            priority_df[
                "nutrient_name"
            ].tolist()
    }


# ============================================================
# NUTRIENT NORMALIZATION
# ============================================================

def normalize_series(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    minimum = series.min()
    maximum = series.max()

    if maximum <= minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# CHILD-SUITABLE FOOD POOL
# ============================================================

def create_child_food_pool(
    age_months
):

    age_months = float(age_months)

    candidates = food_df[
        food_df["child_food_candidate"]
    ].copy()


    # For children older than 6 months,
    # favour prepared foods.

    if age_months > 6:

        prepared = candidates[
            candidates["prepared_food"]
        ].copy()

        if len(prepared) >= 10:

            candidates = prepared


    # Remove obvious raw entries again.

    candidates = candidates[
        ~candidates["food"]
        .str.lower()
        .str.contains(
            r"\braw\b|\buncooked\b",
            regex=True,
            na=False
        )
    ]


    return candidates.reset_index(
        drop=True
    )


# ============================================================
# RECOMMENDATION SCORE
# ============================================================

def calculate_recommendation_scores(
    candidates,
    priority_weights
):

    candidates = candidates.copy()

    nutritional_score = np.zeros(
        len(candidates),
        dtype=float
    )


    # --------------------------------------------------------
    # NUTRIENT CONTRIBUTION
    # --------------------------------------------------------

    for nutrient, weight in (
        priority_weights.items()
    ):

        column = NUTRIENT_COLUMNS[
            nutrient
        ]

        if column not in candidates.columns:

            continue

        normalized = normalize_series(
            candidates[column]
        )

        nutritional_score += (
            normalized.values
            *
            float(weight)
        )


    candidates[
        "nutritional_score"
    ] = nutritional_score


    # --------------------------------------------------------
    # PREPARED FOOD BONUS
    # --------------------------------------------------------

    candidates[
        "prepared_bonus"
    ] = np.where(
        candidates["prepared_food"],
        0.25,
        0.0
    )


    # --------------------------------------------------------
    # INDIAN FOOD / IMAGE BONUS
    # --------------------------------------------------------

    image_keywords = [

        "chapati",
        "roti",
        "dal",
        "lentil",
        "chickpea",
        "chana",
        "okra",
        "bhindi",
        "mustard",
        "spinach",
        "greens",
        "rice",
        "dosa",
        "idli"
    ]


    def image_food_bonus(name):

        name = str(name).lower()

        if any(
            keyword in name
            for keyword in image_keywords
        ):

            return 0.35

        return 0.0


    candidates[
        "image_food_bonus"
    ] = candidates["food"].apply(
        image_food_bonus
    )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    candidates[
        "recommendation_score"
    ] = (

        candidates[
            "nutritional_score"
        ]

        +

        candidates[
            "prepared_bonus"
        ]

        +

        candidates[
            "image_food_bonus"
        ]
    )


    return candidates


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_food_recommendations(
    predicted_status,
    age_months,
    top_n=10
):

    nutrient_result = (
        analyze_nutrient_priorities(
            predicted_status
        )
    )

    priority_weights = {

        row["nutrient"]:
            row["priority_weight"]

        for row
        in nutrient_result[
            "priority_nutrients"
        ]
    }


    candidates = create_child_food_pool(
        age_months
    )


    if candidates.empty:

        raise ValueError(
            "No child-suitable food candidates "
            "were found in the nutrition dataset."
        )


    candidates = calculate_recommendation_scores(
        candidates,
        priority_weights
    )


    candidates = candidates.sort_values(
        "recommendation_score",
        ascending=False
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # DIVERSITY
    # --------------------------------------------------------

    selected = []

    categories_used = set()


    def category_for_food(name):

        name = str(name).lower()

        if any(
            x in name
            for x in [
                "dal",
                "lentil",
                "chickpea",
                "chana",
                "bean",
                "peas"
            ]
        ):

            return "pulse"

        if any(
            x in name
            for x in [
                "chapati",
                "roti",
                "rice",
                "dosa",
                "idli",
                "upma",
                "oat"
            ]
        ):

            return "grain"

        if any(
            x in name
            for x in [
                "spinach",
                "greens",
                "mustard",
                "okra",
                "bhindi",
                "carrot",
                "pumpkin",
                "potato",
                "vegetable"
            ]
        ):

            return "vegetable"

        if any(
            x in name
            for x in [
                "banana",
                "mango",
                "papaya",
                "orange",
                "apple"
            ]
        ):

            return "fruit"

        if any(
            x in name
            for x in [
                "milk",
                "curd",
                "yogurt",
                "paneer",
                "cheese"
            ]
        ):

            return "dairy"

        return "other"


    # First pass: diversity

    for _, row in candidates.iterrows():

        category = category_for_food(
            row["food"]
        )

        if (
            category not in categories_used
            or len(selected) < 5
        ):

            selected.append(
                row
            )

            categories_used.add(
                category
            )

        if len(selected) >= top_n:

            break


    recommendations = pd.DataFrame(
        selected
    ).reset_index(
        drop=True
    )


    # If fewer than requested,
    # fill from remaining high scores.

    if len(recommendations) < top_n:

        already = set(
            recommendations["food"]
        )

        for _, row in candidates.iterrows():

            if row["food"] in already:

                continue

            recommendations = pd.concat(
                [
                    recommendations,
                    pd.DataFrame(
                        [row]
                    )
                ],
                ignore_index=True
            )

            already.add(
                row["food"]
            )

            if len(recommendations) >= top_n:

                break


    # --------------------------------------------------------
    # NUTRIENT CONTRIBUTION TEXT
    # --------------------------------------------------------

    priority_names = (
        nutrient_result[
            "priority_nutrients"
        ]
    )


    contribution_text = []


    for _, row in recommendations.iterrows():

        nutrients = []

        for item in priority_names:

            nutrient = item[
                "nutrient"
            ]

            column = NUTRIENT_COLUMNS[
                nutrient
            ]

            if column not in row:

                continue

            value = row[column]

            if pd.notna(value):

                try:

                    if float(value) > 0:

                        nutrients.append(
                            item[
                                "nutrient_name"
                            ]
                        )

                except Exception:

                    pass


        contribution_text.append(
            ", ".join(
                nutrients[:5]
            )
        )


    recommendations[
        "nutrient_contribution"
    ] = contribution_text


    # Clean helper columns

    recommendations = (
        recommendations.drop(
            columns=[
                "_food_lower",
                "child_food_candidate",
                "prepared_food",
                "prepared_bonus",
                "image_food_bonus"
            ],
            errors="ignore"
        )
    )


    return (
        nutrient_result,
        recommendations
    )


# ============================================================
# AGENT
# ============================================================

class NutritionRecommendationAgent:

    def run(
        self,
        predicted_status,
        age_months,
        top_n=10
    ):

        nutrient_result, recommendations = (
            generate_food_recommendations(
                predicted_status,
                age_months,
                top_n
            )
        )

        return {

            "nutrient_analysis":
                nutrient_result,

            "priority_nutrients":
                nutrient_result[
                    "priority_nutrients"
                ],

            "recommendations":
                recommendations
        }


nutrition_recommendation_agent = (
    NutritionRecommendationAgent()
)


# ============================================================
# SAVE INFORMATION
# ============================================================

with open(
    MODEL_DIR /
    "nutrition_recommendation_agent_information.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        {
            "agent":
                "Child Nutrition Recommendation Agent",

            "logic":
                "Nutrition-first child-suitable food recommendation",

            "excluded":
                EXCLUDE_TERMS,

            "priority_profiles":
                RECOMMENDATION_PRIORITY_WEIGHTS
        },

        f,
        indent=4
    )


# ============================================================
# VISUALIZATION
# ============================================================

print()
print("=" * 75)
print("PART 6 COMPLETED")
print("=" * 75)
print()
print(
    "Child-suitable nutrition recommendation agent created."
)
print(
    "Recommendations are generated dynamically from nutrient priorities."
)
print(
    "Raw foods, desserts and unsuitable processed foods are excluded."
)
# ================= ORIGINAL NOTEBOOK CELL 7 =================
# ============================================================
# PART 6 → PART 7 COMPATIBILITY
# ============================================================

# Part 7 expects this exact variable name.
# Create it using a neutral empty template.
# The actual child-specific recommendations will be generated
# by Part 15.

recommendation_food_df = pd.DataFrame()

print("✓ Part 6 → Part 7 compatibility variable created.")
print("✓ recommendation_food_df is ready for Part 7.")
# ================= ORIGINAL NOTEBOOK CELL 8 =================
# ============================================================
# PART 7 — INDIAN FOOD IMAGE MATCHING AGENT
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

print("=" * 75)
print("PART 7 — INDIAN FOOD IMAGE MATCHING AGENT")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "saved_models"

IMAGE_ROOT = Path(
    r"C:\indian_food_images\Indian Food Images\Indian Food Images"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDATE IMAGE DATASET
# ============================================================

if not IMAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Indian food image dataset not found:\n{IMAGE_ROOT}"
    )


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_food_name(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = value.replace("_", " ")
    value = value.replace("-", " ")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# CONTROLLED EXACT IMAGE MAPPING
# ============================================================

EXACT_IMAGE_MAPPING = {

    "chickpeas cooked":
        "chana_masala",

    "lentils cooked":
        "dal_tadka",

    "mustard greens cooked":
        "makki_di_roti_sarson_da_saag",

    "okra cooked":
        "bhindi_masala",

    "chaptti roti indian bread":
        "chapati",

    "chapati":
        "chapati",

    "roti":
        "chapati"
}


# ============================================================
# CONTROLLED RELATED IMAGE MAPPING
# ============================================================

RELATED_IMAGE_MAPPING = {

    "chickpeas":
        ["chana_masala", "dal_tadka"],

    "chickpea":
        ["chana_masala", "dal_tadka"],

    "lentils":
        ["dal_tadka", "chana_masala"],

    "lentil":
        ["dal_tadka", "chana_masala"],

    "beans":
        ["chana_masala", "dal_tadka"],

    "bean":
        ["chana_masala", "dal_tadka"],

    "peas":
        ["chana_masala"],

    "pea":
        ["chana_masala"],

    "spinach":
        ["makki_di_roti_sarson_da_saag"],

    "mustard greens":
        ["makki_di_roti_sarson_da_saag"],

    "mustard green":
        ["makki_di_roti_sarson_da_saag"],

    "sarson":
        ["makki_di_roti_sarson_da_saag"],

    "greens":
        ["makki_di_roti_sarson_da_saag"],

    "okra":
        ["bhindi_masala"],

    "bhindi":
        ["bhindi_masala"],

    "wheat":
        ["chapati"],

    "roti":
        ["chapati"],

    "chapati":
        ["chapati"]
}


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

image_index = []


for class_dir in sorted(IMAGE_ROOT.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    image_files = sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    for image_path in image_files:

        image_index.append(
            {
                "image_food_class": class_name,
                "image_food_name":
                    normalize_food_name(class_name),
                "image_path":
                    str(image_path)
            }
        )


image_index_df = pd.DataFrame(image_index)


if image_index_df.empty:

    raise RuntimeError(
        "No food images were found in the Indian food image dataset."
    )


print(
    f"Image classes found : "
    f"{image_index_df['image_food_class'].nunique()}"
)

print(
    f"Images found        : "
    f"{len(image_index_df)}"
)


# ============================================================
# SAVE IMAGE INDEX
# ============================================================

image_index_df.to_csv(
    REPORT_DIR / "indian_food_image_index.csv",
    index=False
)


# ============================================================
# FIND IMAGE FOR FOOD
# ============================================================

def find_food_image(food_name):

    normalized_food = normalize_food_name(food_name)

    # --------------------------------------------------------
    # 1. EXACT MAPPING
    # --------------------------------------------------------

    exact_class = EXACT_IMAGE_MAPPING.get(
        normalized_food
    )

    if exact_class is not None:

        matches = image_index_df[
            image_index_df["image_food_class"].str.lower()
            == exact_class.lower()
        ]

        if not matches.empty:

            row = matches.iloc[0]

            return {
                "image_path":
                    row["image_path"],

                "image_food_class":
                    row["image_food_class"],

                "image_relation":
                    "exact",

                "image_match_score":
                    1.0,

                "image_match_reason":
                    "Exact controlled food-image mapping."
            }


    # --------------------------------------------------------
    # 2. CONTROLLED RELATED MAPPING
    # --------------------------------------------------------

    related_classes = RELATED_IMAGE_MAPPING.get(
        normalized_food,
        []
    )

    for related_class in related_classes:

        matches = image_index_df[
            image_index_df["image_food_class"].str.lower()
            == related_class.lower()
        ]

        if not matches.empty:

            row = matches.iloc[0]

            return {
                "image_path":
                    row["image_path"],

                "image_food_class":
                    row["image_food_class"],

                "image_relation":
                    "related",

                "image_match_score":
                    0.75,

                "image_match_reason":
                    "Related controlled Indian food image."
            }


    # --------------------------------------------------------
    # 3. NO IMAGE
    # --------------------------------------------------------

    return {

        "image_path": "",

        "image_food_class": "",

        "image_relation": "none",

        "image_match_score": 0.0,

        "image_match_reason":
            "No exact or controlled related image found."
    }


# ============================================================
# MAIN IMAGE MATCHING AGENT
# ============================================================

def run_image_matching_agent(
    recommendations_df,
    nutrient_priority_result=None
):

    if recommendations_df is None:

        raise ValueError(
            "recommendations_df cannot be None."
        )


    if not isinstance(
        recommendations_df,
        pd.DataFrame
    ):

        recommendations_df = pd.DataFrame(
            recommendations_df
        )


    if recommendations_df.empty:

        result = recommendations_df.copy()

        result["image_path"] = ""
        result["image_food_class"] = ""
        result["image_relation"] = "none"
        result["image_match_score"] = 0.0
        result["image_match_reason"] = ""

        return result


    result = recommendations_df.copy()


    # --------------------------------------------------------
    # FIND FOOD COLUMN
    # --------------------------------------------------------

    food_column = None

    possible_columns = [
        "food",
        "recommended_food",
        "Food",
        "food_name"
    ]

    for column in possible_columns:

        if column in result.columns:

            food_column = column
            break


    if food_column is None:

        raise ValueError(
            "Recommendations DataFrame must contain "
            "a food column."
        )


    # --------------------------------------------------------
    # MATCH IMAGES
    # --------------------------------------------------------

    matched_rows = []


    for _, row in result.iterrows():

        food_name = row[food_column]

        image_result = find_food_image(
            food_name
        )

        updated_row = row.to_dict()

        updated_row.update(
            image_result
        )

        # Preserve the actual recommended food
        updated_row["recommended_food"] = food_name

        # Human-readable image food
        updated_row["image_food"] = (
            image_result["image_food_class"]
        )

        matched_rows.append(
            updated_row
        )


    matched_df = pd.DataFrame(
        matched_rows
    )


    # ========================================================
    # SAVE MATCHING RESULT
    # ========================================================

    matched_df.to_csv(
        REPORT_DIR /
        "verified_food_image_mapping.csv",
        index=False
    )


    # ========================================================
    # SAVE INFORMATION
    # ========================================================

    information = {

        "image_dataset":
            str(IMAGE_ROOT),

        "total_image_classes":
            int(
                image_index_df[
                    "image_food_class"
                ].nunique()
            ),

        "total_images":
            int(len(image_index_df)),

        "matching_order":
            [
                "exact",
                "related",
                "none"
            ],

        "random_selection":
            False
    }


    with open(
        REPORT_DIR /
        "image_dataset_information.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            information,
            file,
            indent=4
        )


    return matched_df


print()
print("Part 7 image matching agent loaded successfully.")
print("Function available:")
print("    run_image_matching_agent(recommendations_df, nutrient_priority_result=None)")
print("=" * 75)
# ================= PART 8 (callable training logic) =================
def train_part8_mobilenetv2():
    # ============================================================
    # PART 8
    # DEEP LEARNING INDIAN FOOD IMAGE CLASSIFICATION & VALIDATION
    # ============================================================

    from pathlib import Path
    import json
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )

    import tensorflow as tf

    from tensorflow.keras import layers
    from tensorflow.keras import models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint
    )


    # ============================================================
    # 8.1 CHECK REQUIRED VARIABLES
    # ============================================================

    required_part8_variables = [
        "IMAGE_ROOT",
        "IMAGE_EXTENSIONS",
        "image_index_df"
    ]

    missing_variables = [
        variable
        for variable in required_part8_variables
        if variable not in globals()
    ]

    if missing_variables:
        raise NameError(
            f"Missing variables: {missing_variables}. "
            "Run Parts 1, 2, 6, and 7 before Part 8."
        )

    print("Part 8: required variables are available.")


    # ============================================================
    # 8.2 CREATE LOCAL PATHS
    # ============================================================

    DL_MODEL_PATH = (
        MODEL_DIR /
        "indian_food_mobilenetv2.keras"
    )

    DL_CHECKPOINT_PATH = (
        MODEL_DIR /
        "indian_food_mobilenetv2_best.keras"
    )

    DL_CLASS_MAPPING_PATH = (
        MODEL_DIR /
        "food_class_mapping.json"
    )

    DL_RESULTS_PATH = (
        REPORT_DIR /
        "deep_learning_results.csv"
    )

    DL_CLASSIFICATION_REPORT_PATH = (
        REPORT_DIR /
        "deep_learning_classification_report.txt"
    )

    DL_CONFUSION_MATRIX_PATH = (
        PLOT_DIR /
        "deep_learning_confusion_matrix.png"
    )

    DL_TRAINING_ACCURACY_PATH = (
        PLOT_DIR /
        "deep_learning_training_accuracy.png"
    )

    DL_TRAINING_LOSS_PATH = (
        PLOT_DIR /
        "deep_learning_training_loss.png"
    )


    # ============================================================
    # 8.3 DEEP LEARNING CONFIGURATION
    # ============================================================

    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    IMG_SIZE = (
        IMG_HEIGHT,
        IMG_WIDTH
    )

    BATCH_SIZE = 32

    RANDOM_STATE = 42

    EPOCHS = 20

    LEARNING_RATE = 0.001

    print("\nDeep Learning configuration:")
    print("Image size:", IMG_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)


    # ============================================================
    # 8.4 CLEAN IMAGE INDEX
    # ============================================================

    dl_image_df = image_index_df.copy()

    dl_image_df["image_path"] = (
        dl_image_df["image_path"]
        .astype(str)
    )

    dl_image_df["image_food_class"] = (
        dl_image_df["image_food_class"]
        .astype(str)
    )


    # Remove files that no longer exist
    dl_image_df = dl_image_df[
        dl_image_df["image_path"].apply(
            lambda path: Path(path).exists()
        )
    ].copy()


    dl_image_df.reset_index(
        drop=True,
        inplace=True
    )


    if dl_image_df.empty:
        raise ValueError(
            "No valid image files are available."
        )


    # ============================================================
    # 8.5 REMOVE CLASSES WITH TOO FEW IMAGES
    # ============================================================

    class_counts = (
        dl_image_df[
            "image_food_class"
        ]
        .value_counts()
    )


    print("\nImage count per class:")
    display(
        class_counts
        .reset_index()
        .rename(
            columns={
                "count": "image_count"
            }
        )
    )


    # Need at least 3 images for a 70/15/15 split
    valid_classes = class_counts[
        class_counts >= 3
    ].index.tolist()


    dl_image_df = dl_image_df[
        dl_image_df[
            "image_food_class"
        ].isin(valid_classes)
    ].copy()


    dl_image_df.reset_index(
        drop=True,
        inplace=True
    )


    # ============================================================
    # 8.6 CREATE CLASS LIST
    # ============================================================

    food_classes = sorted(
        dl_image_df[
            "image_food_class"
        ]
        .unique()
        .tolist()
    )

    NUM_CLASSES = len(food_classes)


    if NUM_CLASSES < 2:
        raise ValueError(
            "Deep Learning requires at least two food classes."
        )


    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(food_classes)
    }


    index_to_class = {
        str(index): class_name
        for class_name, index
        in class_to_index.items()
    }


    dl_image_df["label"] = (
        dl_image_df[
            "image_food_class"
        ]
        .map(class_to_index)
    )


    print("\nNumber of food classes:", NUM_CLASSES)

    print("\nFood classes:")
    for index, class_name in enumerate(food_classes):
        print(index, "→", class_name)


    # ============================================================
    # 8.7 SAVE CLASS MAPPING
    # ============================================================

    with open(
        DL_CLASS_MAPPING_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            index_to_class,
            file,
            indent=4
        )


    print(
        "\nClass mapping saved:",
        DL_CLASS_MAPPING_PATH
    )


    # ============================================================
    # 8.8 TRAIN / VALIDATION / TEST SPLIT
    # ============================================================

    """
    70% Training
    15% Validation
    15% Testing
    """

    train_df, temporary_df = train_test_split(
        dl_image_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=dl_image_df["label"]
    )


    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_df["label"]
    )


    train_df = train_df.reset_index(
        drop=True
    )

    validation_df = validation_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )


    print("\nDataset split:")
    print(
        "Training images   :",
        len(train_df)
    )

    print(
        "Validation images :",
        len(validation_df)
    )

    print(
        "Testing images    :",
        len(test_df)
    )


    # ============================================================
    # 8.9 CREATE TF.DATA DATASET
    # ============================================================

    def load_and_preprocess_image(
        image_path,
        label
    ):
        """
        Load one image, resize it to 224x224 and apply
        MobileNetV2 preprocessing.
        """

        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape(
            [None, None, 3]
        )

        image = tf.image.resize(
            image,
            IMG_SIZE
        )

        image = tf.cast(
            image,
            tf.float32
        )

        image = preprocess_input(
            image
        )

        return image, label


    def create_dataset(
        dataframe,
        shuffle=False
    ):

        paths = dataframe[
            "image_path"
        ].values

        labels = dataframe[
            "label"
        ].values.astype(
            np.int32
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                paths,
                labels
            )
        )

        dataset = dataset.map(
            load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(dataframe),
                seed=RANDOM_STATE
            )

        dataset = dataset.batch(
            BATCH_SIZE
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset


    train_ds = create_dataset(
        train_df,
        shuffle=True
    )

    validation_ds = create_dataset(
        validation_df,
        shuffle=False
    )

    test_ds = create_dataset(
        test_df,
        shuffle=False
    )


    print("\nTensorFlow datasets created.")


    # ============================================================
    # 8.10 DATA AUGMENTATION
    # ============================================================

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.08
            ),

            layers.RandomZoom(
                0.10
            ),

            layers.RandomContrast(
                0.10
            )
        ],
        name="food_data_augmentation"
    )


    # ============================================================
    # 8.11 LOAD MOBILENETV2
    # ============================================================

    print(
        "\nLoading MobileNetV2..."
    )


    base_model = MobileNetV2(
        input_shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        ),

        include_top=False,

        weights="imagenet"
    )


    # Freeze pretrained layers
    base_model.trainable = False


    print(
        "MobileNetV2 loaded successfully."
    )

    print(
        "Pretrained layers frozen:",
        not base_model.trainable
    )


    # ============================================================
    # 8.12 BUILD DEEP LEARNING MODEL
    # ============================================================

    inputs = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    )


    x = data_augmentation(
        inputs
    )


    x = base_model(
        x,
        training=False
    )


    x = layers.GlobalAveragePooling2D()(
        x
    )


    x = layers.Dropout(
        0.30
    )(
        x
    )


    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(
        x
    )


    food_model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="IndianFood_MobileNetV2"
    )


    # ============================================================
    # 8.13 COMPILE MODEL
    # ============================================================

    food_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]
    )


    print("\nModel architecture:")
    food_model.summary()


    # ============================================================
    # 8.14 CALLBACKS
    # ============================================================

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6
        ),

        ModelCheckpoint(
            filepath=str(
                DL_CHECKPOINT_PATH
            ),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]


    # ============================================================
    # 8.15 TRAIN MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("STARTING DEEP LEARNING TRAINING")
    print("=" * 70)


    history = food_model.fit(
        train_ds,

        validation_data=validation_ds,

        epochs=EPOCHS,

        callbacks=callbacks
    )


    print("\nDeep Learning training completed.")


    # ============================================================
    # 8.16 SAVE FINAL MODEL
    # ============================================================

    food_model.save(
        DL_MODEL_PATH
    )


    print(
        "\nFinal MobileNetV2 model saved:"
    )

    print(
        DL_MODEL_PATH
    )


    # ============================================================
    # 8.17 TRAINING HISTORY DATAFRAME
    # ============================================================

    history_df = pd.DataFrame(
        history.history
    )

    history_df["epoch"] = (
        np.arange(
            1,
            len(history_df) + 1
        )
    )


    display(
        history_df
    )


    # ============================================================
    # 8.18 TRAINING ACCURACY VISUALIZATION
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(
        history.history[
            "accuracy"
        ],
        label="Training Accuracy"
    )


    plt.plot(
        history.history[
            "val_accuracy"
        ],
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "MobileNetV2 Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        DL_TRAINING_ACCURACY_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    # ============================================================
    # 8.19 TRAINING LOSS VISUALIZATION
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(
        history.history[
            "loss"
        ],
        label="Training Loss"
    )


    plt.plot(
        history.history[
            "val_loss"
        ],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "MobileNetV2 Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        DL_TRAINING_LOSS_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    # ============================================================
    # 8.20 EVALUATE ON TEST DATA
    # ============================================================

    print("\n" + "=" * 70)
    print("DEEP LEARNING TEST EVALUATION")
    print("=" * 70)


    test_loss, test_accuracy = (
        food_model.evaluate(
            test_ds,
            verbose=1
        )
    )


    print(
        "\nTest Loss:",
        round(float(test_loss), 4)
    )

    print(
        "Test Accuracy:",
        round(float(test_accuracy), 4)
    )


    # ============================================================
    # 8.21 GENERATE TEST PREDICTIONS
    # ============================================================

    test_probabilities = (
        food_model.predict(
            test_ds,
            verbose=1
        )
    )


    test_predictions = np.argmax(
        test_probabilities,
        axis=1
    )


    test_true_labels = test_df[
        "label"
    ].values


    # ============================================================
    # 8.22 CALCULATE DL METRICS
    # ============================================================

    dl_accuracy = accuracy_score(
        test_true_labels,
        test_predictions
    )


    dl_precision = precision_score(
        test_true_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )


    dl_recall = recall_score(
        test_true_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )


    dl_f1 = f1_score(
        test_true_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )


    dl_results = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            "Score": [
                dl_accuracy,
                dl_precision,
                dl_recall,
                dl_f1
            ]
        }
    )


    print("\nDeep Learning performance:")
    display(dl_results)


    dl_results.to_csv(
        DL_RESULTS_PATH,
        index=False
    )


    # ============================================================
    # 8.23 FULL CLASSIFICATION REPORT
    # ============================================================

    classification_report_text = (
        classification_report(
            test_true_labels,
            test_predictions,
            labels=np.arange(
                NUM_CLASSES
            ),
            target_names=food_classes,
            zero_division=0
        )
    )


    print(
        "\nDeep Learning Classification Report:"
    )

    print(
        classification_report_text
    )


    with open(
        DL_CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            classification_report_text
        )


    # ============================================================
    # 8.24 CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        test_true_labels,
        test_predictions,
        labels=np.arange(
            NUM_CLASSES
        )
    )


    plt.figure(
        figsize=(16, 14)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        "MobileNetV2 Indian Food Classification Confusion Matrix"
    )


    plt.colorbar()


    tick_marks = np.arange(
        NUM_CLASSES
    )


    plt.xticks(
        tick_marks,
        food_classes,
        rotation=90,
        fontsize=7
    )


    plt.yticks(
        tick_marks,
        food_classes,
        fontsize=7
    )


    plt.xlabel(
        "Predicted Food Class"
    )


    plt.ylabel(
        "True Food Class"
    )


    plt.tight_layout()


    plt.savefig(
        DL_CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    # ============================================================
    # 8.25 SINGLE IMAGE PREDICTION FUNCTION
    # ============================================================

    def predict_food_image(
        image_path,
        model=None
    ):
        """
        Predict the food class of one image.

        Returns:
            predicted_class
            confidence
            probabilities
        """

        if model is None:
            model = food_model

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )


        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )


        image_array = (
            tf.keras.utils.img_to_array(
                image
            )
        )


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        image_array = preprocess_input(
            image_array
        )


        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        predicted_class = (
            index_to_class[
                str(predicted_index)
            ]
        )


        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        probability_dict = {
            index_to_class[str(index)]:
            float(probability)

            for index, probability
            in enumerate(probabilities)
        }


        return {
            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "probabilities":
                probability_dict
        }


    # ============================================================
    # 8.26 NORMALIZE IMAGE CLASS NAMES
    # ============================================================

    def normalize_image_class_name(
        class_name
    ):
        """
        Normalize a class name for comparison.
        """

        if class_name is None:
            return ""

        name = str(
            class_name
        ).lower().strip()

        name = name.replace(
            "_",
            " "
        )

        name = name.replace(
            "-",
            " "
        )

        name = " ".join(
            name.split()
        )

        return name


    # ============================================================
    # 8.27 CHECK WHETHER DL PREDICTION MATCHES EXPECTED CLASS
    # ============================================================

    def image_prediction_matches_expected_class(
        predicted_class,
        expected_class
    ):
        """
        Compare MobileNetV2 prediction with the expected
        image class from Part 7.
        """

        predicted = (
            normalize_image_class_name(
                predicted_class
            )
        )

        expected = (
            normalize_image_class_name(
                expected_class
            )
        )

        return predicted == expected


    # ============================================================
    # 8.28 VALIDATE ONE IMAGE SELECTED BY PART 7
    # ============================================================

    def validate_selected_food_image(
        image_path,
        expected_class,
        confidence_threshold=0.50
    ):
        """
        Validate an image selected by Part 7.

        Conditions for MATCH:

            1. Image exists
            2. MobileNetV2 predicts expected class
            3. Prediction confidence >= threshold

        Otherwise the image is rejected for report use.
        """

        if image_path is None:
            return {
                "validation_status":
                    "NO_IMAGE",

                "predicted_class":
                    None,

                "confidence":
                    0.0,

                "expected_class":
                    expected_class
            }


        image_path = Path(
            image_path
        )


        if not image_path.exists():
            return {
                "validation_status":
                    "IMAGE_NOT_FOUND",

                "predicted_class":
                    None,

                "confidence":
                    0.0,

                "expected_class":
                    expected_class
            }


        prediction = predict_food_image(
            image_path
        )


        predicted_class = prediction[
            "predicted_class"
        ]


        confidence = prediction[
            "confidence"
        ]


        class_match = (
            image_prediction_matches_expected_class(
                predicted_class,
                expected_class
            )
        )


        if (
            class_match
            and
            confidence >= confidence_threshold
        ):

            validation_status = (
                "MATCH"
            )

        else:

            validation_status = (
                "MISMATCH"
            )


        return {
            "validation_status":
                validation_status,

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "expected_class":
                expected_class
        }


    # ============================================================
    # 8.29 VALIDATE ALL IMAGES FROM PART 7
    # ============================================================

    def validate_recommendation_images(
        recommendations_with_images,
        confidence_threshold=0.50
    ):
        """
        Validate images selected by Part 7.

        IMPORTANT:

        This function does not change the recommended foods.

        It only determines whether the selected image can be
        trusted as a visually validated representation.
        """

        result_df = (
            recommendations_with_images.copy()
        )


        validation_statuses = []
        predicted_classes = []
        confidences = []
        expected_classes = []
        validated_image_paths = []


        for _, row in result_df.iterrows():

            image_path = row.get(
                "image_path",
                None
            )

            expected_class = row.get(
                "image_food_class",
                None
            )


            validation_result = (
                validate_selected_food_image(
                    image_path=image_path,
                    expected_class=expected_class,
                    confidence_threshold=
                        confidence_threshold
                )
            )


            validation_statuses.append(
                validation_result[
                    "validation_status"
                ]
            )


            predicted_classes.append(
                validation_result[
                    "predicted_class"
                ]
            )


            confidences.append(
                validation_result[
                    "confidence"
                ]
            )


            expected_classes.append(
                validation_result[
                    "expected_class"
                ]
            )


            if (
                validation_result[
                    "validation_status"
                ]
                ==
                "MATCH"
            ):

                validated_image_paths.append(
                    image_path
                )

            else:

                # Do not use an unvalidated image
                validated_image_paths.append(
                    None
                )


        result_df[
            "dl_validation_status"
        ] = validation_statuses


        result_df[
            "dl_predicted_image_class"
        ] = predicted_classes


        result_df[
            "dl_confidence"
        ] = confidences


        result_df[
            "dl_expected_image_class"
        ] = expected_classes


        result_df[
            "validated_image_path"
        ] = validated_image_paths


        return result_df


    # ============================================================
    # 8.30 MAIN DEEP LEARNING VALIDATION AGENT
    # ============================================================

    def run_deep_learning_image_agent(
        recommendations_with_images,
        confidence_threshold=0.50
    ):
        """
        Main Part 8 function.

        Input:
            Part 7 recommendation/image table

        Output:
            Same recommendations with DL validation.
        """

        return validate_recommendation_images(
            recommendations_with_images=
                recommendations_with_images,
            confidence_threshold=
                confidence_threshold
        )


    # ============================================================
    # 8.31 SAVE MODEL INFORMATION
    # ============================================================

    deep_learning_information = {

        "model": "MobileNetV2",

        "transfer_learning": True,

        "pretrained_weights": "ImageNet",

        "input_size": [
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        ],

        "number_of_classes": int(
            NUM_CLASSES
        ),

        "number_of_training_images": int(
            len(train_df)
        ),

        "number_of_validation_images": int(
            len(validation_df)
        ),

        "number_of_test_images": int(
            len(test_df)
        ),

        "epochs_requested": int(
            EPOCHS
        ),

        "batch_size": int(
            BATCH_SIZE
        ),

        "learning_rate": float(
            LEARNING_RATE
        ),

        "test_accuracy": float(
            dl_accuracy
        ),

        "test_precision": float(
            dl_precision
        ),

        "test_recall": float(
            dl_recall
        ),

        "test_f1": float(
            dl_f1
        ),

        "random_image_selection": False,

        "image_validation_enabled": True
    }


    with open(
        MODEL_DIR /
        "deep_learning_information.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            deep_learning_information,
            file,
            indent=4
        )


    # ============================================================
    # 8.32 FINAL STATUS
    # ============================================================

    print("\n" + "=" * 70)
    print("PART 8 COMPLETED")
    print("=" * 70)

    print(
        "Deep Learning model             : MobileNetV2"
    )

    print(
        "Transfer learning               : ENABLED"
    )

    print(
        "Indian food image classification : ENABLED"
    )

    print(
        "Image validation                 : ENABLED"
    )

    print(
        "Random image selection           : NOT USED"
    )

    print(
        "Unrelated image fallback        : NOT USED"
    )

    print(
        "Training images                 :",
        len(train_df)
    )

    print(
        "Validation images               :",
        len(validation_df)
    )

    print(
        "Test images                     :",
        len(test_df)
    )

    print(
        "Food classes                    :",
        NUM_CLASSES
    )

    print(
        "Test accuracy                   :",
        round(float(dl_accuracy), 4)
    )

    print(
        "Test precision                  :",
        round(float(dl_precision), 4)
    )

    print(
        "Test recall                     :",
        round(float(dl_recall), 4)
    )

    print(
        "Test F1                         :",
        round(float(dl_f1), 4)
    )

    print("\nSaved model:")
    print(
        DL_MODEL_PATH
    )

    print("\nSaved class mapping:")
    print(
        DL_CLASS_MAPPING_PATH
    )

    print("\nSaved reports:")
    print(
        DL_RESULTS_PATH
    )

    print(
        DL_CLASSIFICATION_REPORT_PATH
    )

# ================= ORIGINAL NOTEBOOK CELL 11 =================
# ============================================================
# PART 8 — SAFE IMAGE VALIDATION FIX
# Uses the already-trained MobileNetV2 model
# NO TRAINING
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. PREDICT FOOD IMAGE
# ============================================================

def predict_food_image(image_path, model=None):

    # --------------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------------

    if model is None:

        model_path = (
            MODEL_DIR /
            "indian_food_mobilenetv2.keras"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Saved DL model not found:\n{model_path}"
            )

        model = tf.keras.models.load_model(
            model_path
        )

    # --------------------------------------------------------
    # CHECK IMAGE PATH
    # --------------------------------------------------------

    if image_path is None:

        raise FileNotFoundError(
            "Image path is None."
        )

    image_path = Path(
        str(image_path).strip()
    )

    # Empty path
    if str(image_path) == "":

        raise FileNotFoundError(
            "Image path is empty."
        )

    # File does not exist
    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # Directory instead of image
    if image_path.is_dir():

        raise FileNotFoundError(
            "Image path points to a directory, "
            "not an image:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # CHECK IMAGE EXTENSION
    # --------------------------------------------------------

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    if image_path.suffix.lower() not in valid_extensions:

        raise ValueError(
            f"Unsupported image format:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # LOAD CLASS MAPPING
    # --------------------------------------------------------

    class_mapping_path = (
        MODEL_DIR /
        "food_class_mapping.json"
    )

    if not class_mapping_path.exists():

        raise FileNotFoundError(
            "Food class mapping file not found:\n"
            f"{class_mapping_path}"
        )

    with open(
        class_mapping_path,
        "r",
        encoding="utf-8"
    ) as file:

        class_mapping = json.load(file)

    # --------------------------------------------------------
    # CONVERT INDEX TO CLASS NAME
    # --------------------------------------------------------

    if all(
        str(key).isdigit()
        for key in class_mapping.keys()
    ):

        predicted_class = class_mapping.get(
            str(predicted_index),
            class_mapping.get(
                predicted_index,
                str(predicted_index)
            )
        )

    else:

        reverse_mapping = {
            int(value): key
            for key, value in class_mapping.items()
        }

        predicted_class = reverse_mapping.get(
            predicted_index,
            str(predicted_index)
        )

    return {

        "predicted_class":
            str(predicted_class),

        "confidence":
            confidence,

        "predicted_index":
            predicted_index,

        "probabilities":
            probabilities.tolist(),

        "image_path":
            str(image_path)
    }


# ============================================================
# 2. VALIDATE ONE FOOD IMAGE
# ============================================================

def validate_selected_food_image(
    image_path,
    expected_class,
    confidence_threshold=0.50
):

    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if (
        image_path is None
        or str(image_path).strip() == ""
    ):

        return {

            "validation_status":
                "IMAGE_NOT_FOUND",

            "predicted_class":
                "",

            "confidence":
                0.0,

            "expected_class":
                (
                    str(expected_class)
                    if expected_class is not None
                    else ""
                ),

            "image_path":
                "",

            "validation_message":
                "No image was matched for this food."
        }

    image_path = Path(
        str(image_path).strip()
    )

    # --------------------------------------------------------
    # DIRECTORY CHECK
    # --------------------------------------------------------

    if image_path.is_dir():

        return {

            "validation_status":
                "IMAGE_NOT_FOUND",

            "predicted_class":
                "",

            "confidence":
                0.0,

            "expected_class":
                (
                    str(expected_class)
                    if expected_class is not None
                    else ""
                ),

            "image_path":
                "",

            "validation_message":
                "The selected path is a directory, not an image."
        }

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    if not image_path.exists():

        return {

            "validation_status":
                "IMAGE_NOT_FOUND",

            "predicted_class":
                "",

            "confidence":
                0.0,

            "expected_class":
                (
                    str(expected_class)
                    if expected_class is not None
                    else ""
                ),

            "image_path":
                str(image_path),

            "validation_message":
                "The selected image file does not exist."
        }

    # --------------------------------------------------------
    # DL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = predict_food_image(
            image_path=image_path
        )

    except Exception as error:

        return {

            "validation_status":
                "IMAGE_VALIDATION_ERROR",

            "predicted_class":
                "",

            "confidence":
                0.0,

            "expected_class":
                (
                    str(expected_class)
                    if expected_class is not None
                    else ""
                ),

            "image_path":
                str(image_path),

            "validation_message":
                str(error)
        }

    # --------------------------------------------------------
    # GET PREDICTION
    # --------------------------------------------------------

    predicted_class = str(
        prediction["predicted_class"]
    )

    confidence = float(
        prediction["confidence"]
    )

    expected_class = (
        str(expected_class)
        if expected_class is not None
        else ""
    )

    # --------------------------------------------------------
    # COMPARE CLASS
    # --------------------------------------------------------

    class_matches = (

        predicted_class.lower().strip()

        ==

        expected_class.lower().strip()
    )

    confidence_ok = (
        confidence >= confidence_threshold
    )

    # --------------------------------------------------------
    # VALIDATION STATUS
    # --------------------------------------------------------

    if class_matches and confidence_ok:

        validation_status = "VALIDATED"

    elif not class_matches:

        validation_status = "CLASS_MISMATCH"

    else:

        validation_status = "LOW_CONFIDENCE"

    return {

        "validation_status":
            validation_status,

        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "expected_class":
            expected_class,

        "image_path":
            str(image_path),

        "validation_message":

            (
                "Image successfully validated."

                if validation_status == "VALIDATED"

                else

                "Image did not satisfy the DL validation criteria."
            )
    }


# ============================================================
# 3. VALIDATE ALL RECOMMENDED FOOD IMAGES
# ============================================================

def validate_recommendation_images(
    recommendations_with_images,
    confidence_threshold=0.50
):

    if recommendations_with_images is None:

        return pd.DataFrame()

    if not isinstance(
        recommendations_with_images,
        pd.DataFrame
    ):

        recommendations_with_images = pd.DataFrame(
            recommendations_with_images
        )

    result = (
        recommendations_with_images.copy()
    )

    validation_statuses = []
    predicted_classes = []
    confidences = []
    expected_classes = []
    validation_messages = []

    # --------------------------------------------------------
    # PROCESS EACH RECOMMENDATION
    # --------------------------------------------------------

    for _, row in result.iterrows():

        image_path = row.get(
            "image_path",
            ""
        )

        expected_class = row.get(
            "image_food_class",
            ""
        )

        validation_result = (
            validate_selected_food_image(

                image_path=
                    image_path,

                expected_class=
                    expected_class,

                confidence_threshold=
                    confidence_threshold
            )
        )

        validation_statuses.append(
            validation_result[
                "validation_status"
            ]
        )

        predicted_classes.append(
            validation_result[
                "predicted_class"
            ]
        )

        confidences.append(
            validation_result[
                "confidence"
            ]
        )

        expected_classes.append(
            validation_result[
                "expected_class"
            ]
        )

        validation_messages.append(
            validation_result[
                "validation_message"
            ]
        )

    # --------------------------------------------------------
    # ADD VALIDATION RESULTS
    # --------------------------------------------------------

    result[
        "dl_validation_status"
    ] = validation_statuses

    result[
        "dl_predicted_class"
    ] = predicted_classes

    result[
        "dl_confidence"
    ] = confidences

    result[
        "dl_expected_class"
    ] = expected_classes

    result[
        "dl_validation_message"
    ] = validation_messages

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result.to_csv(

        REPORT_DIR /
        "deep_learning_image_validation.csv",

        index=False
    )

    return result


# ============================================================
# 4. MAIN DEEP LEARNING IMAGE AGENT
# ============================================================

def run_deep_learning_image_agent(
    recommendations_with_images,
    confidence_threshold=0.50
):

    return validate_recommendation_images(

        recommendations_with_images=
            recommendations_with_images,

        confidence_threshold=
            confidence_threshold
    )


# ============================================================
# CONFIRMATION
# ============================================================

print("=" * 70)
print("PART 8 SAFE IMAGE VALIDATION FIX LOADED")
print("=" * 70)
print("Saved MobileNetV2 model will be reused.")
print("No Deep Learning training was performed.")
print("Directory paths will not be treated as images.")
print("=" * 70)
# ================= ORIGINAL NOTEBOOK CELL 13 =================
import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
# ================= ORIGINAL NOTEBOOK CELL 14 =================
BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_nutrition_model.pkl"
SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
BEST_MODEL_INFO_PATH = MODEL_DIR / "best_model_information.json"

required_files = [
    BEST_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    BEST_MODEL_INFO_PATH
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "The following required files are missing:\n" +
        "\n".join(missing_files) +
        "\n\nPlease run Parts 3–5 first."
    )

print("✓ Required Part 3–5 files found.")
# ================= ORIGINAL NOTEBOOK CELL 15 =================
best_model = joblib.load(BEST_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

with open(BEST_MODEL_INFO_PATH, "r", encoding="utf-8") as file:
    best_model_information = json.load(file)

best_model_name = best_model_information["best_model"]

selected_features = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi"
]

print("Best Model:", best_model_name)
print("Features:", selected_features)
print("Classes:", list(label_encoder.classes_))
# ================= ORIGINAL NOTEBOOK CELL 16 =================
PROCESSED_CHILD_DATASET = DATA_DIR / "processed_child_dataset.csv"

if not PROCESSED_CHILD_DATASET.exists():
    raise FileNotFoundError(
        f"Processed child dataset not found:\n{PROCESSED_CHILD_DATASET}\n"
        "Please run Part 3 first."
    )

processed_child_df = pd.read_csv(PROCESSED_CHILD_DATASET)

print("Processed child dataset shape:", processed_child_df.shape)

display(processed_child_df.head())
# ================= ORIGINAL NOTEBOOK CELL 17 =================
missing_features = [
    feature
    for feature in selected_features
    if feature not in processed_child_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing SHAP features: {missing_features}"
    )

X_background = processed_child_df[selected_features].copy()

X_background = X_background.apply(
    pd.to_numeric,
    errors="coerce"
)

X_background = X_background.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

print("SHAP background shape:", X_background.shape)
# ================= ORIGINAL NOTEBOOK CELL 18 =================
def prepare_shap_model_input(X):
    """
    Prepare feature values in the same representation
    used when the best model was trained.
    """

    X = X[selected_features].copy()

    if best_model_name == "Logistic Regression":
        return pd.DataFrame(
            scaler.transform(X),
            columns=selected_features,
            index=X.index
        )

    return X
# ================= ORIGINAL NOTEBOOK CELL 19 =================
X_background_model = prepare_shap_model_input(X_background)

print("Creating SHAP explainer...")

if best_model_name == "Logistic Regression":

    shap_explainer = shap.LinearExplainer(
        best_model,
        X_background_model
    )

elif best_model_name in [
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM"
]:

    shap_explainer = shap.TreeExplainer(
        best_model
    )

else:

    raise ValueError(
        f"Unsupported model for SHAP: {best_model_name}"
    )

print("✓ SHAP explainer created successfully.")
# ================= ORIGINAL NOTEBOOK CELL 20 =================
print("Calculating SHAP values...")

shap_background_values = shap_explainer(
    X_background_model
)

print("✓ SHAP values calculated.")
# ================= ORIGINAL NOTEBOOK CELL 21 =================
def get_class_shap_values(shap_result, class_index):
    """
    Extract numeric SHAP values for one class.

    Supports modern SHAP Explanation objects,
    NumPy arrays, and older list-based SHAP outputs.
    """

    # Modern SHAP Explanation object
    if isinstance(shap_result, shap.Explanation):

        values = np.asarray(shap_result.values)

        # samples × features
        if values.ndim == 2:
            return values

        # samples × features × classes
        if values.ndim == 3:

            if values.shape[2] == n_classes:
                return values[:, :, class_index]

            # samples × classes × features
            if values.shape[1] == n_classes:
                return values[:, class_index, :]

        raise ValueError(
            f"Unsupported SHAP Explanation shape: {values.shape}"
        )

    # Older SHAP list output
    if isinstance(shap_result, list):

        return np.asarray(
            shap_result[class_index]
        )

    # NumPy array
    values = np.asarray(shap_result)

    if values.ndim == 2:
        return values

    if values.ndim == 3:

        if values.shape[2] == n_classes:
            return values[:, :, class_index]

        if values.shape[1] == n_classes:
            return values[:, class_index, :]

    raise ValueError(
        f"Unsupported SHAP value shape: {values.shape}"
    )
# ================= ORIGINAL NOTEBOOK CELL 22 =================
print("Calculating global SHAP feature importance...")

n_classes = len(label_encoder.classes_)

global_importance = np.zeros(
    len(selected_features),
    dtype=float
)

for class_index in range(n_classes):

    class_values = get_class_shap_values(
        shap_background_values,
        class_index
    )

    # Make absolutely sure we are working with
    # numeric SHAP values, not an Explanation object.
    class_values = np.asarray(
        class_values,
        dtype=float
    )

    class_importance = np.mean(
        np.abs(class_values),
        axis=0
    )

    global_importance += class_importance


global_importance = (
    global_importance / n_classes
)


shap_importance_df = pd.DataFrame({
    "feature": selected_features,
    "mean_absolute_shap": global_importance
})

shap_importance_df = shap_importance_df.sort_values(
    "mean_absolute_shap",
    ascending=False
).reset_index(drop=True)


print("✓ Global SHAP importance calculated.")

display(shap_importance_df)
# ================= ORIGINAL NOTEBOOK CELL 23 =================
plt.figure(figsize=(9, 6))

plt.barh(
    shap_importance_df["feature"][::-1],
    shap_importance_df["mean_absolute_shap"][::-1]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    f"Global SHAP Feature Importance — {best_model_name}"
)

plt.tight_layout()

global_shap_plot = (
    PLOT_DIR / "shap_global_feature_importance.png"
)

plt.savefig(
    global_shap_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Saved:", global_shap_plot)
# ================= ORIGINAL NOTEBOOK CELL 24 =================
global_shap_report = REPORT_DIR / "shap_global_feature_importance.csv"

shap_importance_df.to_csv(
    global_shap_report,
    index=False
)

print("Saved:", global_shap_report)
# ================= ORIGINAL NOTEBOOK CELL 25 =================
def explain_child_prediction(
    age_months,
    weight_kg,
    height_cm,
    muac_cm,
    predicted_class_encoded=None
):
    """
    Generate a SHAP explanation for one child prediction.

    No user input is collected here.
    Part 15 will call this function.
    """

    # -----------------------------
    # Validate input
    # -----------------------------

    age_months = float(age_months)
    weight_kg = float(weight_kg)
    height_cm = float(height_cm)
    muac_cm = float(muac_cm)

    if not 0 <= age_months <= 60:
        raise ValueError(
            "Age must be between 0 and 60 months."
        )

    if weight_kg <= 0:
        raise ValueError(
            "Weight must be greater than zero."
        )

    if height_cm <= 0:
        raise ValueError(
            "Height must be greater than zero."
        )

    if muac_cm <= 0:
        raise ValueError(
            "MUAC must be greater than zero."
        )

    # -----------------------------
    # Calculate BMI
    # -----------------------------

    height_m = height_cm / 100

    bmi = weight_kg / (height_m ** 2)

    # -----------------------------
    # Create child DataFrame
    # -----------------------------

    child_input = pd.DataFrame([{
        "age_months": age_months,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "muac_cm": muac_cm,
        "bmi": bmi
    }])

    # -----------------------------
    # Prepare model input
    # -----------------------------

    child_model_input = prepare_shap_model_input(
        child_input
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    encoded_prediction = int(
        best_model.predict(child_model_input)[0]
    )

    if predicted_class_encoded is not None:
        encoded_prediction = int(predicted_class_encoded)

    predicted_status = label_encoder.inverse_transform(
        [encoded_prediction]
    )[0]

    # -----------------------------
    # SHAP explanation
    # -----------------------------

    shap_result = shap_explainer(
        child_model_input
    )

    child_class_values = get_class_shap_values(
        shap_result,
        encoded_prediction
    )

    feature_contributions = child_class_values[0]

    # -----------------------------
    # Base value
    # -----------------------------

    base_values = np.asarray(
        shap_result.base_values
    )

    if base_values.ndim == 0:
        base_value = float(base_values)

    elif base_values.ndim == 1:

        if len(base_values) == n_classes:
            base_value = float(
                base_values[encoded_prediction]
            )
        else:
            base_value = float(base_values[0])

    elif base_values.ndim >= 2:

        base_value = float(
            base_values[0][encoded_prediction]
        )

    else:
        base_value = 0.0

    # -----------------------------
    # Feature contribution table
    # -----------------------------

    explanation_df = pd.DataFrame({
        "feature": selected_features,
        "feature_value": [
            age_months,
            weight_kg,
            height_cm,
            muac_cm,
            bmi
        ],
        "shap_value": feature_contributions
    })

    explanation_df["absolute_shap"] = (
        explanation_df["shap_value"].abs()
    )

    explanation_df["direction"] = np.where(
        explanation_df["shap_value"] >= 0,
        "positive",
        "negative"
    )

    explanation_df = explanation_df.sort_values(
        "absolute_shap",
        ascending=False
    ).reset_index(drop=True)

    # -----------------------------
    # Return everything
    # -----------------------------

    return {
        "predicted_class_encoded": encoded_prediction,
        "predicted_status": str(predicted_status),
        "bmi": float(bmi),
        "base_value": base_value,
        "explanation": explanation_df,
        "model_name": best_model_name
    }
# ================= ORIGINAL NOTEBOOK CELL 26 =================
def create_shap_waterfall_plot(
    explanation_result,
    output_path=None
):
    """
    Create a waterfall-style SHAP contribution plot
    for one predicted child.
    """

    explanation_df = explanation_result["explanation"]
    predicted_status = explanation_result["predicted_status"]
    base_value = explanation_result["base_value"]

    values = explanation_df["shap_value"].to_numpy()

    feature_values = explanation_df["feature_value"].to_numpy()

    features = explanation_df["feature"].tolist()

    shap_explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=feature_values,
        feature_names=features
    )

    plt.figure(figsize=(9, 6))

    shap.plots.waterfall(
        shap_explanation,
        max_display=len(features),
        show=False
    )

    plt.title(
        f"SHAP Explanation — Predicted Class: {predicted_status}"
    )

    plt.tight_layout()

    if output_path is None:
        output_path = (
            PLOT_DIR /
            "shap_individual_explanation.png"
        )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return output_path
# ================= ORIGINAL NOTEBOOK CELL 27 =================
def generate_shap_explanation_text(
    explanation_result,
    top_n=5
):
    """
    Convert SHAP feature contributions into
    human-readable explanation sentences.
    """

    explanation_df = explanation_result["explanation"]
    predicted_status = explanation_result["predicted_status"]

    top_features = explanation_df.head(top_n)

    explanation_lines = []

    explanation_lines.append(
        f"Predicted nutrition status: {predicted_status}"
    )

    explanation_lines.append(
        "Important prediction-related features:"
    )

    for _, row in top_features.iterrows():

        feature = row["feature"]
        value = row["feature_value"]
        shap_value = row["shap_value"]

        if shap_value > 0:
            direction = "increased the model's output toward the predicted class"
        elif shap_value < 0:
            direction = "decreased the model's output toward the predicted class"
        else:
            direction = "had little contribution to the predicted class"

        explanation_lines.append(
            f"{feature}: value={value:.2f}, "
            f"SHAP={shap_value:.4f}; {direction}."
        )

    return "\n".join(explanation_lines)
# ================= ORIGINAL NOTEBOOK CELL 28 =================
shap_information = {
    "best_model": best_model_name,
    "features": selected_features,
    "classes": [
        str(value)
        for value in label_encoder.classes_
    ],
    "explanation_type": "SHAP",
    "global_importance_file": str(
        global_shap_report
    )
}

SHAP_INFORMATION_PATH = (
    MODEL_DIR / "shap_information.json"
)

with open(
    SHAP_INFORMATION_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        shap_information,
        file,
        indent=4
    )

print(
    "Saved:",
    SHAP_INFORMATION_PATH
)
# ================= ORIGINAL NOTEBOOK CELL 29 =================
class SHAPExplanationAgent:

    def __init__(
        self,
        explainer,
        model_name,
        label_encoder
    ):

        self.explainer = explainer
        self.model_name = model_name
        self.label_encoder = label_encoder

    def explain(
        self,
        age_months,
        weight_kg,
        height_cm,
        muac_cm,
        predicted_class_encoded=None
    ):

        result = explain_child_prediction(
            age_months=age_months,
            weight_kg=weight_kg,
            height_cm=height_cm,
            muac_cm=muac_cm,
            predicted_class_encoded=predicted_class_encoded
        )

        result["agent"] = "SHAP Explanation Agent"

        result["explanation_text"] = (
            generate_shap_explanation_text(result)
        )

        return result


shap_agent = SHAPExplanationAgent(
    explainer=shap_explainer,
    model_name=best_model_name,
    label_encoder=label_encoder
)

print("✓ SHAP Explanation Agent created.")
# ================= ORIGINAL NOTEBOOK CELL 30 =================
print("==========================================")
print("SHAP EXPLANATION AGENT")
print("==========================================")
print("Agent:", "SHAP Explanation Agent")
print("Model:", best_model_name)
print("Features:", selected_features)
print(
    "Classes:",
    list(label_encoder.classes_)
)
print("Status:", "Ready for Part 15")
print("==========================================")
# ================= ORIGINAL NOTEBOOK CELL 31 =================
# ============================================================
# PART 10
# CHILD-SUITABLE MEAL PLAN GENERATION AGENT
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path.cwd() / "data"
MODEL_DIR = Path.cwd() / "saved_models"
REPORT_DIR = Path.cwd() / "reports"

FOOD_FILE = (
    DATA_DIR /
    "combined_food_nutrition.csv"
)


# ============================================================
# NUTRIENTS
# ============================================================

NUTRIENT_COLUMNS = {

    "energy": "Caloric Value",
    "protein": "Protein",
    "fat": "Fat",
    "carbohydrate": "Carbohydrates",
    "fiber": "Dietary Fiber",
    "iron": "Iron",
    "calcium": "Calcium",
    "zinc": "Zinc",
    "vitamin_a": "Vitamin A",
    "vitamin_b12": "Vitamin B12",
    "vitamin_c": "Vitamin C",
    "vitamin_d": "Vitamin D"
}


# ============================================================
# LOAD
# ============================================================

meal_food_df = pd.read_csv(
    FOOD_FILE
).copy()


meal_food_df["food"] = (
    meal_food_df["food"]
    .astype(str)
    .str.strip()
)


for column in NUTRIENT_COLUMNS.values():

    if column in meal_food_df.columns:

        meal_food_df[column] = pd.to_numeric(
            meal_food_df[column],
            errors="coerce"
        )


meal_food_df["_food_lower"] = (
    meal_food_df["food"]
    .str.lower()
)


# ============================================================
# EXCLUDE INAPPROPRIATE FOODS
# ============================================================

MEAL_EXCLUDE_TERMS = [

    "raw",
    "uncooked",
    "pie",
    "cake",
    "donut",
    "doughnut",
    "candy",
    "chocolate",
    "ice cream",
    "cookie",
    "biscuit",
    "pizza",
    "burger",
    "hot dog",
    "fries",
    "chips",
    "pork",
    "turkey",
    "bacon",
    "ham",
    "sausage",
    "salami",
    "alcohol",
    "wine",
    "beer",
    "cola",
    "soda"
]


def meal_food_allowed(name):

    name = str(name).lower()

    return not any(
        term in name
        for term in MEAL_EXCLUDE_TERMS
    )


meal_food_df = meal_food_df[
    meal_food_df["food"].apply(
        meal_food_allowed
    )
].copy()


# ============================================================
# CHILD FOOD KEYWORDS
# ============================================================

CHILD_MEAL_KEYWORDS = [

    "chapati",
    "roti",
    "rice",
    "dosa",
    "idli",
    "upma",
    "porridge",
    "oat",
    "lentil",
    "dal",
    "dhal",
    "chickpea",
    "chana",
    "bean",
    "peas",
    "spinach",
    "greens",
    "mustard",
    "okra",
    "bhindi",
    "carrot",
    "pumpkin",
    "potato",
    "sweet potato",
    "vegetable",
    "banana",
    "mango",
    "papaya",
    "orange",
    "apple",
    "milk",
    "curd",
    "yogurt",
    "paneer",
    "cheese",
    "peanut",
    "egg"
]


def child_meal_food(name):

    name = str(name).lower()

    return any(
        keyword in name
        for keyword in CHILD_MEAL_KEYWORDS
    )


meal_food_df = meal_food_df[
    meal_food_df["food"].apply(
        child_meal_food
    )
].copy()


# ============================================================
# PRIORITY PROFILES
# ============================================================

MEAL_PRIORITY_PROFILES = {

    "normal": [
        "protein",
        "calcium",
        "iron",
        "fiber",
        "vitamin_a",
        "vitamin_c"
    ],

    "moderate": [
        "energy",
        "protein",
        "iron",
        "calcium",
        "zinc",
        "vitamin_a",
        "vitamin_c"
    ],

    "severe": [
        "energy",
        "protein",
        "iron",
        "calcium",
        "zinc",
        "vitamin_a",
        "vitamin_c",
        "vitamin_b12"
    ]
}


def determine_meal_priority_profile(
    predicted_status
):

    status = str(
        predicted_status
    ).lower()

    if "severe" in status:

        return "severe"

    if "moderate" in status:

        return "moderate"

    return "normal"


# ============================================================
# MEAL CATEGORIES
# ============================================================

MEAL_KEYWORDS = {

    "Breakfast": [
        "idli",
        "dosa",
        "upma",
        "porridge",
        "oat",
        "chapati",
        "roti",
        "rice",
        "milk",
        "banana"
    ],

    "Mid-Morning": [
        "banana",
        "mango",
        "papaya",
        "apple",
        "orange",
        "curd",
        "yogurt",
        "milk"
    ],

    "Lunch": [
        "rice",
        "dal",
        "lentil",
        "chickpea",
        "chana",
        "bean",
        "peas",
        "spinach",
        "greens",
        "okra",
        "bhindi",
        "vegetable",
        "chapati",
        "roti"
    ],

    "Evening Snack": [
        "banana",
        "fruit",
        "curd",
        "yogurt",
        "milk",
        "chapati",
        "roti",
        "chickpea"
    ],

    "Dinner": [
        "rice",
        "dal",
        "lentil",
        "chickpea",
        "chana",
        "spinach",
        "greens",
        "okra",
        "bhindi",
        "vegetable",
        "chapati",
        "roti"
    ]
}


# ============================================================
# FOOD SCORE
# ============================================================

def normalize_column(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    if series.max() == series.min():

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - series.min())
        /
        (series.max() - series.min())
    )


def calculate_food_priority_score(
    row,
    priority_nutrients
):

    score = 0.0

    for nutrient in priority_nutrients:

        column = NUTRIENT_COLUMNS.get(
            nutrient
        )

        if (
            column is None
            or column not in row
        ):

            continue

        value = row[column]

        if pd.notna(value):

            score += float(value)


    return score


# ============================================================
# CATEGORY FILTER
# ============================================================

def get_meal_candidates(
    meal_name
):

    keywords = MEAL_KEYWORDS[
        meal_name
    ]

    mask = meal_food_df[
        "food"
    ].str.lower().apply(

        lambda name:
        any(
            keyword in name
            for keyword in keywords
        )
    )

    candidates = meal_food_df[
        mask
    ].copy()

    return candidates


# ============================================================
# FALLBACK CANDIDATES
# ============================================================

def get_fallback_candidates():

    return meal_food_df.copy()


# ============================================================
# GENERATE ONE MEAL
# ============================================================

def generate_single_meal(
    meal_name,
    priority_nutrients
):

    candidates = get_meal_candidates(
        meal_name
    )


    if candidates.empty:

        candidates = get_fallback_candidates()


    if candidates.empty:

        return None


    # Score using normalized nutrient values.

    score = np.zeros(
        len(candidates),
        dtype=float
    )


    for nutrient in priority_nutrients:

        column = NUTRIENT_COLUMNS.get(
            nutrient
        )

        if (
            column is None
            or column not in candidates.columns
        ):

            continue


        normalized = normalize_column(
            candidates[column]
        )


        score += (
            normalized.values
        )


    candidates = candidates.copy()

    candidates[
        "meal_priority_score"
    ] = score


    # Prefer Indian-food image classes.

    image_terms = [

        "chapati",
        "roti",
        "dal",
        "lentil",
        "chickpea",
        "chana",
        "okra",
        "bhindi",
        "mustard",
        "greens",
        "spinach"
    ]


    candidates[
        "image_food_bonus"
    ] = candidates[
        "food"
    ].str.lower().apply(

        lambda name:
        0.5
        if any(
            term in name
            for term in image_terms
        )
        else 0.0
    )


    candidates[
        "final_meal_score"
    ] = (
        candidates[
            "meal_priority_score"
        ]
        +
        candidates[
            "image_food_bonus"
        ]
    )


    candidates = candidates.sort_values(
        "final_meal_score",
        ascending=False
    )


    return candidates.iloc[0]


# ============================================================
# GENERATE MEAL PLAN
# ============================================================

def generate_meal_plan(
    predicted_status,
    age_months,
    recommendations_df=None
):

    age_months = float(
        age_months
    )


    profile = (
        determine_meal_priority_profile(
            predicted_status
        )
    )


    priority_nutrients = (
        MEAL_PRIORITY_PROFILES[
            profile
        ]
    )


    meal_names = [

        "Breakfast",
        "Mid-Morning",
        "Lunch",
        "Evening Snack",
        "Dinner"
    ]


    rows = []


    for meal_name in meal_names:

        selected = generate_single_meal(
            meal_name,
            priority_nutrients
        )


        if selected is None:

            continue


        food_name = str(
            selected["food"]
        )


        # ----------------------------------------------------
        # NUTRIENT INFORMATION
        # ----------------------------------------------------

        nutrient_parts = []


        for nutrient, column in (
            NUTRIENT_COLUMNS.items()
        ):

            if column not in selected:

                continue


            value = selected[
                column
            ]


            if pd.notna(value):

                try:

                    nutrient_parts.append(
                        f"{nutrient}: "
                        f"{float(value):.2f}"
                    )

                except Exception:

                    pass


        rows.append({

            "age_months":
                age_months,

            "predicted_status":
                predicted_status,

            "priority_profile":
                profile,

            "priority_nutrients":
                ", ".join(
                    priority_nutrients
                ),

            "meal":
                meal_name,

            "food":
                food_name,

            "priority_score":
                float(
                    selected[
                        "final_meal_score"
                    ]
                ),

            "nutrient_information":
                "; ".join(
                    nutrient_parts
                )
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# ADD NUTRIENT INFORMATION
# ============================================================

def add_meal_nutrient_information(
    meal_plan_df
):

    return meal_plan_df.copy()


# ============================================================
# AGENT
# ============================================================

class MealPlanAgent:

    def generate(
        self,
        predicted_status,
        age_months,
        recommendations_df=None
    ):

        return generate_meal_plan(
            predicted_status,
            age_months,
            recommendations_df
        )


meal_plan_agent = MealPlanAgent()


# ============================================================
# SAVE
# ============================================================

meal_plan_template = generate_meal_plan(
    "normal",
    24
)

meal_plan_template.to_csv(
    REPORT_DIR /
    "meal_plan_template.csv",
    index=False
)


with open(
    MODEL_DIR /
    "meal_plan_information.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        {
            "agent":
                "Child Suitable Meal Plan Agent",

            "meal_slots": [
                "Breakfast",
                "Mid-Morning",
                "Lunch",
                "Evening Snack",
                "Dinner"
            ],

            "priority_profiles":
                MEAL_PRIORITY_PROFILES
        },

        f,
        indent=4
    )


print()
print("=" * 75)
print("PART 10 COMPLETED")
print("=" * 75)
print()
print(
    "Child-suitable meal plan agent created."
)
print(
    "Raw foods, desserts and unsuitable processed foods are excluded."
)
# ================= ORIGINAL NOTEBOOK CELL 32 =================
# ============================================================
# PART 11 — NUTRITION ADVICE / MALNUTRITION PREVENTION AGENT
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in [
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    OUTPUT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. NUTRIENT PRIORITY PROFILES
# ------------------------------------------------------------

NUTRIENT_PRIORITY_PROFILES = {

    "normal": {
        "energy": "Support age-appropriate energy intake",
        "protein": "Support normal growth and tissue development",
        "iron": "Support healthy blood and growth",
        "calcium": "Support bone and teeth development",
        "zinc": "Support growth and immune function",
        "vitamin_a": "Support normal growth and vision",
        "vitamin_c": "Support immune function and iron absorption",
        "fiber": "Support healthy digestion"
    },

    "moderate": {
        "energy": "Increase nutrient-dense energy sources",
        "protein": "Prioritize protein-rich foods for growth",
        "iron": "Prioritize iron-rich foods",
        "calcium": "Prioritize calcium-containing foods",
        "zinc": "Include zinc-rich foods",
        "vitamin_a": "Include vitamin-A-rich foods",
        "vitamin_c": "Include vitamin-C-rich foods to support iron absorption",
        "fiber": "Include suitable sources of dietary fiber"
    },

    "severe": {
        "energy": "Prioritize nutrient-dense energy sources",
        "protein": "Strongly prioritize protein-rich foods",
        "iron": "Prioritize iron-rich foods",
        "calcium": "Prioritize calcium-containing foods",
        "zinc": "Prioritize zinc-rich foods",
        "vitamin_a": "Prioritize vitamin-A-rich foods",
        "vitamin_c": "Include vitamin-C-rich foods",
        "vitamin_b12": "Include suitable vitamin-B12 food sources"
    }
}


# ------------------------------------------------------------
# 3. DETERMINE PRIORITY PROFILE
# ------------------------------------------------------------

def determine_advice_profile(predicted_status):

    status = str(
        predicted_status
    ).lower().strip()

    if any(
        word in status
        for word in [
            "severe",
            "high",
            "critical"
        ]
    ):
        return "severe"

    if any(
        word in status
        for word in [
            "moderate",
            "medium",
            "risk",
            "at risk",
            "underweight"
        ]
    ):
        return "moderate"

    return "normal"


# ------------------------------------------------------------
# 4. GENERAL AGE-BASED ADVICE
# ------------------------------------------------------------

def generate_age_based_advice(age_months):

    age_months = float(age_months)

    if age_months < 6:

        return [
            "Focus on age-appropriate feeding practices.",
            "Breast milk is an important source of nutrition during early infancy.",
            "Monitor growth measurements regularly."
        ]

    elif age_months < 12:

        return [
            "Provide age-appropriate complementary foods.",
            "Gradually introduce a variety of nutrient-dense foods.",
            "Include suitable sources of protein, iron and energy.",
            "Continue age-appropriate milk feeding."
        ]

    elif age_months < 24:

        return [
            "Provide a varied diet containing energy- and nutrient-dense foods.",
            "Include suitable protein-rich foods regularly.",
            "Include iron-rich foods together with vitamin-C-rich foods.",
            "Offer a variety of fruits and vegetables.",
            "Provide age-appropriate meals and snacks throughout the day."
        ]

    else:

        return [
            "Provide a varied and balanced diet.",
            "Include protein-rich foods regularly.",
            "Include a variety of vegetables and fruits.",
            "Include iron- and calcium-containing foods.",
            "Provide suitable meals and snacks throughout the day."
        ]


# ------------------------------------------------------------
# 5. NUTRIENT-SPECIFIC ADVICE
# ------------------------------------------------------------

def generate_nutrient_advice(
    priority_nutrients
):

    advice = []

    nutrient_advice_map = {

        "energy":
            "Include nutrient-dense energy sources such as suitable cereals, grains, pulses and other energy-containing foods.",

        "protein":
            "Include suitable protein-rich foods such as pulses, lentils, chickpeas, beans, dairy foods and other age-appropriate protein sources.",

        "iron":
            "Include iron-rich foods such as pulses, lentils, chickpeas, leafy greens and other suitable iron-containing foods.",

        "calcium":
            "Include suitable calcium-containing foods such as milk, curd, yogurt and other calcium-rich foods.",

        "zinc":
            "Include suitable zinc-containing foods such as pulses, legumes, dairy foods and other nutrient-dense foods.",

        "vitamin_a":
            "Include suitable vitamin-A-containing foods such as leafy greens and other colorful vegetables and fruits.",

        "vitamin_c":
            "Include vitamin-C-containing fruits and vegetables alongside iron-rich foods.",

        "vitamin_b12":
            "Include suitable vitamin-B12-containing foods according to the child's dietary pattern.",

        "fiber":
            "Include suitable fruits, vegetables, pulses and whole-food sources of dietary fiber."
    }

    for nutrient in priority_nutrients:

        if nutrient in nutrient_advice_map:

            advice.append(
                nutrient_advice_map[nutrient]
            )

    return advice


# ------------------------------------------------------------
# 6. MEAL-PLAN BASED ADVICE
# ------------------------------------------------------------

def generate_meal_plan_advice(
    meal_plan_df
):

    advice = []

    if meal_plan_df is None:
        return advice

    if not isinstance(
        meal_plan_df,
        pd.DataFrame
    ):
        return advice

    if meal_plan_df.empty:
        return advice

    if "meal" in meal_plan_df.columns:

        available_meals = (
            meal_plan_df["meal"]
            .astype(str)
            .tolist()
        )

        if "Breakfast" in available_meals:
            advice.append(
                "Include a nutrient-dense breakfast appropriate for the child's age."
            )

        if "Lunch" in available_meals:
            advice.append(
                "Include a balanced lunch containing suitable energy, protein and micronutrient sources."
            )

        if "Dinner" in available_meals:
            advice.append(
                "Provide a suitable nutrient-dense evening meal."
            )

    return advice


# ------------------------------------------------------------
# 7. PREVENTION ADVICE
# ------------------------------------------------------------

def generate_prevention_advice(
    predicted_status
):

    profile = determine_advice_profile(
        predicted_status
    )

    prevention = []

    if profile == "severe":

        prevention = [
            "Maintain consistent age-appropriate feeding routines.",
            "Prioritize nutrient-dense foods identified by the recommendation system.",
            "Monitor weight, height and MUAC regularly.",
            "Track changes in nutrition status over time."
        ]

    elif profile == "moderate":

        prevention = [
            "Maintain regular age-appropriate meals and snacks.",
            "Include a variety of nutrient-dense foods.",
            "Prioritize the nutrients identified by the recommendation system.",
            "Monitor growth measurements periodically."
        ]

    else:

        prevention = [
            "Maintain a varied and balanced diet.",
            "Continue regular age-appropriate meals and snacks.",
            "Include a variety of fruits, vegetables, grains and protein sources.",
            "Continue monitoring growth and nutrition status."
        ]

    return prevention


# ------------------------------------------------------------
# 8. FOOD RECOMMENDATION ADVICE
# ------------------------------------------------------------

def generate_food_based_advice(
    recommendations_df
):

    advice = []

    if recommendations_df is None:
        return advice

    if not isinstance(
        recommendations_df,
        pd.DataFrame
    ):
        return advice

    if recommendations_df.empty:
        return advice

    if "food" not in recommendations_df.columns:
        return advice

    foods = (
        recommendations_df["food"]
        .astype(str)
        .tolist()
    )

    foods = [
        food
        for food in foods
        if food.strip()
    ]

    if foods:

        display_foods = foods[:5]

        advice.append(
            "Prioritized foods include: "
            + ", ".join(display_foods)
            + "."
        )

    return advice


# ------------------------------------------------------------
# 9. MAIN NUTRITION ADVICE FUNCTION
# ------------------------------------------------------------

def generate_nutrition_advice(
    predicted_status,
    age_months,
    priority_nutrients=None,
    recommendations_df=None,
    meal_plan_df=None
):
    """
    Generate personalized nutrition and
    malnutrition-prevention advice.

    No user input is collected here.
    Part 15 supplies the actual child information.
    """

    age_months = float(age_months)

    if not 0 <= age_months <= 60:
        raise ValueError(
            "Age must be between 0 and 60 months."
        )

    profile = determine_advice_profile(
        predicted_status
    )

    # --------------------------------------------------------
    # Determine nutrient priorities
    # --------------------------------------------------------

    if priority_nutrients is None:

        priority_nutrients = list(
            NUTRIENT_PRIORITY_PROFILES[
                profile
            ].keys()
        )

    priority_nutrients = [
        str(nutrient)
        for nutrient in priority_nutrients
    ]

    # --------------------------------------------------------
    # Generate advice sections
    # --------------------------------------------------------

    age_advice = generate_age_based_advice(
        age_months
    )

    nutrient_advice = generate_nutrient_advice(
        priority_nutrients
    )

    food_advice = generate_food_based_advice(
        recommendations_df
    )

    meal_advice = generate_meal_plan_advice(
        meal_plan_df
    )

    prevention_advice = generate_prevention_advice(
        predicted_status
    )

    # --------------------------------------------------------
    # Combine all advice
    # --------------------------------------------------------

    all_advice = []

    all_advice.extend(
        age_advice
    )

    all_advice.extend(
        nutrient_advice
    )

    all_advice.extend(
        food_advice
    )

    all_advice.extend(
        meal_advice
    )

    all_advice.extend(
        prevention_advice
    )

    # Remove duplicate advice
    unique_advice = []

    for item in all_advice:

        if item not in unique_advice:
            unique_advice.append(item)

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "agent": "Nutrition Advice / Malnutrition Prevention Agent",

        "age_months": age_months,

        "predicted_status": str(
            predicted_status
        ),

        "priority_profile": profile,

        "priority_nutrients": priority_nutrients,

        "age_based_advice": age_advice,

        "nutrient_advice": nutrient_advice,

        "food_based_advice": food_advice,

        "meal_plan_advice": meal_advice,

        "prevention_advice": prevention_advice,

        "all_advice": unique_advice
    }


# ------------------------------------------------------------
# 10. AGENT CLASS
# ------------------------------------------------------------

class NutritionAdviceAgent:

    def __init__(self):

        self.agent_name = (
            "Nutrition Advice / Malnutrition Prevention Agent"
        )

    def generate(
        self,
        predicted_status,
        age_months,
        priority_nutrients=None,
        recommendations_df=None,
        meal_plan_df=None
    ):

        result = generate_nutrition_advice(

            predicted_status=predicted_status,

            age_months=age_months,

            priority_nutrients=priority_nutrients,

            recommendations_df=recommendations_df,

            meal_plan_df=meal_plan_df
        )

        result["agent"] = (
            self.agent_name
        )

        return result


# ------------------------------------------------------------
# 11. CREATE AGENT
# ------------------------------------------------------------

nutrition_advice_agent = (
    NutritionAdviceAgent()
)


# ------------------------------------------------------------
# 12. SAVE CONFIGURATION
# ------------------------------------------------------------

advice_configuration = {

    "agent":
        "Nutrition Advice / Malnutrition Prevention Agent",

    "age_range_months": [
        0,
        60
    ],

    "priority_profiles":
        list(
            NUTRIENT_PRIORITY_PROFILES.keys()
        ),

    "output_sections": [
        "age_based_advice",
        "nutrient_advice",
        "food_based_advice",
        "meal_plan_advice",
        "prevention_advice"
    ]
}

ADVICE_CONFIG_PATH = (
    MODEL_DIR /
    "nutrition_advice_agent_information.json"
)

with open(
    ADVICE_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        advice_configuration,
        file,
        indent=4
    )


# ------------------------------------------------------------
# 13. SAVE TEMPLATE
# ------------------------------------------------------------

advice_template = pd.DataFrame({

    "section": [
        "Age-based advice",
        "Nutrient advice",
        "Food-based advice",
        "Meal-plan advice",
        "Prevention advice"
    ],

    "advice": [
        "",
        "",
        "",
        "",
        ""
    ]
})

ADVICE_TEMPLATE_PATH = (
    REPORT_DIR /
    "nutrition_advice_template.csv"
)

advice_template.to_csv(
    ADVICE_TEMPLATE_PATH,
    index=False
)


# ------------------------------------------------------------
# 14. FINAL STATUS
# ------------------------------------------------------------

print("=" * 60)
print("PART 11 — NUTRITION ADVICE / MALNUTRITION PREVENTION")
print("=" * 60)

print(
    "Agent:",
    nutrition_advice_agent.agent_name
)

print(
    "Supported age range:",
    "0–60 months"
)

print(
    "Priority profiles:",
    list(
        NUTRIENT_PRIORITY_PROFILES.keys()
    )
)

print(
    "Output sections:"
)

for section in [
    "Age-based advice",
    "Nutrient advice",
    "Food-based advice",
    "Meal-plan advice",
    "Malnutrition-prevention advice"
]:

    print("  ✓", section)

print()
print(
    "Configuration saved:",
    ADVICE_CONFIG_PATH
)

print(
    "Template saved:",
    ADVICE_TEMPLATE_PATH
)

print()
print(
    "STATUS: ✓ PART 11 READY FOR PART 15"
)

print("=" * 60)
# ================= ORIGINAL NOTEBOOK CELL 33 =================
# ============================================================
# PART 12 — CHILD NUTRITION MONITORING AGENT
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in [
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    OUTPUT_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# 2. REQUIRED FEATURES
# ------------------------------------------------------------

MONITORING_FEATURES = [
    "age_months",
    "weight_kg",
    "height_cm",
    "muac_cm",
    "bmi"
]


# ------------------------------------------------------------
# 3. CALCULATE BMI
# ------------------------------------------------------------

def calculate_monitoring_bmi(
    weight_kg,
    height_cm
):

    weight_kg = float(weight_kg)
    height_cm = float(height_cm)

    if weight_kg <= 0:
        raise ValueError(
            "Weight must be greater than zero."
        )

    if height_cm <= 0:
        raise ValueError(
            "Height must be greater than zero."
        )

    height_m = height_cm / 100.0

    bmi = (
        weight_kg /
        (height_m ** 2)
    )

    return float(bmi)


# ------------------------------------------------------------
# 4. VALIDATE VISIT DATA
# ------------------------------------------------------------

def validate_monitoring_visit(
    visit
):

    required_fields = [
        "age_months",
        "weight_kg",
        "height_cm",
        "muac_cm"
    ]

    missing = [
        field
        for field in required_fields
        if field not in visit
    ]

    if missing:
        raise ValueError(
            f"Missing monitoring fields: {missing}"
        )

    age_months = float(
        visit["age_months"]
    )

    weight_kg = float(
        visit["weight_kg"]
    )

    height_cm = float(
        visit["height_cm"]
    )

    muac_cm = float(
        visit["muac_cm"]
    )

    if not 0 <= age_months <= 60:

        raise ValueError(
            "Age must be between 0 and 60 months."
        )

    if weight_kg <= 0:

        raise ValueError(
            "Weight must be greater than zero."
        )

    if height_cm <= 0:

        raise ValueError(
            "Height must be greater than zero."
        )

    if muac_cm <= 0:

        raise ValueError(
            "MUAC must be greater than zero."
        )

    bmi = calculate_monitoring_bmi(
        weight_kg,
        height_cm
    )

    validated_visit = {
        "age_months": age_months,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "muac_cm": muac_cm,
        "bmi": bmi
    }

    return validated_visit


# ------------------------------------------------------------
# 5. PREPARE VISIT RECORD
# ------------------------------------------------------------

def prepare_visit_record(
    visit,
    visit_number=None,
    visit_date=None,
    predicted_status=None
):

    validated_visit = (
        validate_monitoring_visit(
            visit
        )
    )

    record = validated_visit.copy()

    if visit_number is not None:

        record["visit_number"] = (
            visit_number
        )

    if visit_date is not None:

        record["visit_date"] = str(
            visit_date
        )

    if predicted_status is not None:

        record["predicted_status"] = str(
            predicted_status
        )

    return record


# ------------------------------------------------------------
# 6. CALCULATE CHANGE BETWEEN TWO VISITS
# ------------------------------------------------------------

def calculate_visit_change(
    previous_visit,
    current_visit
):

    previous = validate_monitoring_visit(
        previous_visit
    )

    current = validate_monitoring_visit(
        current_visit
    )

    changes = {}

    for feature in MONITORING_FEATURES:

        previous_value = float(
            previous[feature]
        )

        current_value = float(
            current[feature]
        )

        absolute_change = (
            current_value -
            previous_value
        )

        if previous_value != 0:

            percentage_change = (
                absolute_change /
                abs(previous_value)
            ) * 100

        else:

            percentage_change = np.nan

        changes[feature] = {
            "previous": previous_value,
            "current": current_value,
            "absolute_change": float(
                absolute_change
            ),
            "percentage_change": (
                float(percentage_change)
                if not np.isnan(
                    percentage_change
                )
                else None
            )
        }

    return changes


# ------------------------------------------------------------
# 7. DESCRIBE MEASUREMENT TREND
# ------------------------------------------------------------

def describe_change(
    change_value,
    feature
):

    change_value = float(
        change_value
    )

    if abs(change_value) < 1e-8:

        return "No meaningful change"

    if change_value > 0:

        return (
            f"{feature} increased"
        )

    return (
        f"{feature} decreased"
    )


# ------------------------------------------------------------
# 8. GENERATE VISIT COMPARISON TABLE
# ------------------------------------------------------------

def create_visit_comparison_table(
    previous_visit,
    current_visit
):

    changes = calculate_visit_change(
        previous_visit,
        current_visit
    )

    rows = []

    for feature in MONITORING_FEATURES:

        item = changes[feature]

        rows.append({

            "feature":
                feature,

            "previous_value":
                item["previous"],

            "current_value":
                item["current"],

            "absolute_change":
                item["absolute_change"],

            "percentage_change":
                item["percentage_change"],

            "trend":
                describe_change(
                    item["absolute_change"],
                    feature
                )
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 9. ANALYZE NUTRITION STATUS TREND
# ------------------------------------------------------------

def analyze_status_trend(
    previous_status,
    current_status
):

    if previous_status is None:

        return {
            "previous_status": None,
            "current_status": str(
                current_status
            ),
            "status_changed": False,
            "trend": "Initial assessment"
        }

    previous = str(
        previous_status
    ).strip()

    current = str(
        current_status
    ).strip()

    if previous.lower() == current.lower():

        trend = "Status unchanged"
        changed = False

    else:

        trend = (
            f"Status changed from "
            f"{previous} to {current}"
        )

        changed = True

    return {
        "previous_status": previous,
        "current_status": current,
        "status_changed": changed,
        "trend": trend
    }


# ------------------------------------------------------------
# 10. ANALYZE OVERALL GROWTH TREND
# ------------------------------------------------------------

def analyze_growth_trend(
    previous_visit,
    current_visit
):

    changes = calculate_visit_change(
        previous_visit,
        current_visit
    )

    weight_change = (
        changes["weight_kg"]
        ["absolute_change"]
    )

    height_change = (
        changes["height_cm"]
        ["absolute_change"]
    )

    muac_change = (
        changes["muac_cm"]
        ["absolute_change"]
    )

    bmi_change = (
        changes["bmi"]
        ["absolute_change"]
    )

    observations = []

    if weight_change > 0:
        observations.append(
            "Weight increased."
        )
    elif weight_change < 0:
        observations.append(
            "Weight decreased."
        )
    else:
        observations.append(
            "Weight remained unchanged."
        )

    if height_change > 0:
        observations.append(
            "Height increased."
        )
    elif height_change < 0:
        observations.append(
            "Height decreased."
        )
    else:
        observations.append(
            "Height remained unchanged."
        )

    if muac_change > 0:
        observations.append(
            "MUAC increased."
        )
    elif muac_change < 0:
        observations.append(
            "MUAC decreased."
        )
    else:
        observations.append(
            "MUAC remained unchanged."
        )

    if bmi_change > 0:
        observations.append(
            "BMI increased."
        )
    elif bmi_change < 0:
        observations.append(
            "BMI decreased."
        )
    else:
        observations.append(
            "BMI remained unchanged."
        )

    return observations


# ------------------------------------------------------------
# 11. MONITOR MULTIPLE VISITS
# ------------------------------------------------------------

def monitor_multiple_visits(
    visits
):
    """
    visits must be a list of dictionaries.

    Example structure:

    [
        {
            "age_months": ...,
            "weight_kg": ...,
            "height_cm": ...,
            "muac_cm": ...,
            "predicted_status": ...
        },
        ...
    ]
    """

    if visits is None or len(visits) == 0:

        raise ValueError(
            "At least one visit is required."
        )

    records = []

    for index, visit in enumerate(
        visits,
        start=1
    ):

        record = prepare_visit_record(
            visit=visit,
            visit_number=index,
            visit_date=visit.get(
                "visit_date"
            ),
            predicted_status=visit.get(
                "predicted_status"
            )
        )

        records.append(record)

    monitoring_df = pd.DataFrame(
        records
    )

    return monitoring_df


# ------------------------------------------------------------
# 12. GENERATE MONITORING SUMMARY
# ------------------------------------------------------------

def generate_monitoring_summary(
    visits
):

    monitoring_df = monitor_multiple_visits(
        visits
    )

    summary = {
        "number_of_visits":
            int(len(monitoring_df)),

        "first_visit":
            monitoring_df.iloc[0]
            .to_dict(),

        "latest_visit":
            monitoring_df.iloc[-1]
            .to_dict()
    }

    # --------------------------------------------------------
    # Compare first and latest visit
    # --------------------------------------------------------

    if len(monitoring_df) >= 2:

        first_visit = (
            monitoring_df.iloc[0]
            .to_dict()
        )

        latest_visit = (
            monitoring_df.iloc[-1]
            .to_dict()
        )

        summary[
            "overall_changes"
        ] = calculate_visit_change(
            first_visit,
            latest_visit
        )

        summary[
            "growth_observations"
        ] = analyze_growth_trend(
            first_visit,
            latest_visit
        )

        summary[
            "status_trend"
        ] = analyze_status_trend(
            first_visit.get(
                "predicted_status"
            ),
            latest_visit.get(
                "predicted_status"
            )
        )

    else:

        summary[
            "overall_changes"
        ] = {}

        summary[
            "growth_observations"
        ] = [
            "Only one visit is available.",
            "Additional visits are required to evaluate changes over time."
        ]

        summary[
            "status_trend"
        ] = analyze_status_trend(
            None,
            monitoring_df.iloc[0].get(
                "predicted_status"
            )
        )

    return summary


# ------------------------------------------------------------
# 13. CREATE MONITORING REPORT TABLE
# ------------------------------------------------------------

def create_monitoring_report(
    visits
):

    monitoring_df = monitor_multiple_visits(
        visits
    )

    report_rows = []

    for _, row in monitoring_df.iterrows():

        report_rows.append({

            "visit_number":
                row.get(
                    "visit_number"
                ),

            "visit_date":
                row.get(
                    "visit_date",
                    ""
                ),

            "age_months":
                row["age_months"],

            "weight_kg":
                row["weight_kg"],

            "height_cm":
                row["height_cm"],

            "muac_cm":
                row["muac_cm"],

            "bmi":
                row["bmi"],

            "predicted_status":
                row.get(
                    "predicted_status",
                    ""
                )
        })

    return pd.DataFrame(
        report_rows
    )


# ------------------------------------------------------------
# 14. MONITORING AGENT CLASS
# ------------------------------------------------------------

class ChildNutritionMonitoringAgent:

    def __init__(self):

        self.agent_name = (
            "Child Nutrition Monitoring Agent"
        )

    def monitor(
        self,
        visits
    ):

        monitoring_df = (
            create_monitoring_report(
                visits
            )
        )

        summary = (
            generate_monitoring_summary(
                visits
            )
        )

        return {
            "agent":
                self.agent_name,

            "monitoring_data":
                monitoring_df,

            "summary":
                summary
        }


# ------------------------------------------------------------
# 15. CREATE AGENT
# ------------------------------------------------------------

monitoring_agent = (
    ChildNutritionMonitoringAgent()
)


# ------------------------------------------------------------
# 16. SAVE CONFIGURATION
# ------------------------------------------------------------

monitoring_configuration = {

    "agent":
        "Child Nutrition Monitoring Agent",

    "monitored_features":
        MONITORING_FEATURES,

    "age_range_months": [
        0,
        60
    ],

    "outputs": [
        "visit comparison",
        "measurement changes",
        "growth observations",
        "nutrition status trend",
        "monitoring summary"
    ]
}

MONITORING_CONFIG_PATH = (
    MODEL_DIR /
    "monitoring_agent_information.json"
)

with open(
    MONITORING_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        monitoring_configuration,
        file,
        indent=4
    )


# ------------------------------------------------------------
# 17. SAVE EMPTY MONITORING TEMPLATE
# ------------------------------------------------------------

monitoring_template = pd.DataFrame({

    "visit_number": [],

    "visit_date": [],

    "age_months": [],

    "weight_kg": [],

    "height_cm": [],

    "muac_cm": [],

    "bmi": [],

    "predicted_status": []
})

MONITORING_TEMPLATE_PATH = (
    REPORT_DIR /
    "child_monitoring_template.csv"
)

monitoring_template.to_csv(
    MONITORING_TEMPLATE_PATH,
    index=False
)


# ------------------------------------------------------------
# 18. FINAL STATUS
# ------------------------------------------------------------

print("=" * 65)
print("PART 12 — CHILD NUTRITION MONITORING AGENT")
print("=" * 65)

print(
    "Agent:",
    monitoring_agent.agent_name
)

print(
    "Monitored features:"
)

for feature in MONITORING_FEATURES:

    print(
        "  ✓",
        feature
    )

print()
print(
    "Monitoring capabilities:"
)

for capability in [
    "BMI calculation",
    "Visit-to-visit comparison",
    "Weight trend",
    "Height trend",
    "MUAC trend",
    "BMI trend",
    "Nutrition status trend",
    "Growth observations",
    "Multiple-visit monitoring"
]:

    print(
        "  ✓",
        capability
    )

print()
print(
    "Configuration saved:",
    MONITORING_CONFIG_PATH
)

print(
    "Template saved:",
    MONITORING_TEMPLATE_PATH
)

print()
print(
    "STATUS: ✓ PART 12 READY FOR PART 15"
)

print("=" * 65)
# ================= ORIGINAL NOTEBOOK CELL 34 =================
# ============================================================
# PART 13 - DYNAMIC PDF REPORT GENERATION AGENT
# ============================================================

from pathlib import Path
from datetime import datetime
import ast
import json
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
    KeepTogether
)


# ============================================================
# REPORT DIRECTORIES
# ============================================================

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLES
# ============================================================

styles = getSampleStyleSheet()

PDF_TITLE_STYLE = ParagraphStyle(
    "PDFTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    alignment=TA_CENTER,
    spaceAfter=5
)

PDF_SUBTITLE_STYLE = ParagraphStyle(
    "PDFSubtitle",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"),
    spaceAfter=18
)

PDF_SECTION_STYLE = ParagraphStyle(
    "PDFSection",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=4,
    spaceAfter=8
)

PDF_SUBSECTION_STYLE = ParagraphStyle(
    "PDFSubsection",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=10.5,
    leading=13,
    textColor=colors.HexColor("#333333"),
    spaceBefore=5,
    spaceAfter=4
)

PDF_BODY_STYLE = ParagraphStyle(
    "PDFBody",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13,
    spaceAfter=4
)

PDF_SMALL_STYLE = ParagraphStyle(
    "PDFSmall",
    parent=styles["BodyText"],
    fontSize=8,
    leading=10
)

PDF_TABLE_HEADER_STYLE = ParagraphStyle(
    "PDFTableHeader",
    parent=styles["BodyText"],
    fontName="Helvetica-Bold",
    fontSize=7.8,
    leading=9.5,
    textColor=colors.white
)

PDF_TABLE_CELL_STYLE = ParagraphStyle(
    "PDFTableCell",
    parent=styles["BodyText"],
    fontSize=7.8,
    leading=9.8
)

PDF_BULLET_STYLE = ParagraphStyle(
    "PDFBullet",
    parent=styles["BodyText"],
    fontSize=8.8,
    leading=12,
    leftIndent=12,
    firstLineIndent=-7,
    spaceAfter=4
)


# ============================================================
# HELPER
# ============================================================

def pdf_paragraph(value, style=PDF_BODY_STYLE):

    if value is None:
        value = ""

    text = str(value)

    # Prevent problematic characters
    text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

    return Paragraph(text, style)


# ============================================================
# CONVERT VALUE TO CLEAN TEXT
# ============================================================

def clean_pdf_value(value):

    if value is None:
        return ""

    if isinstance(value, float):

        if math.isnan(value):
            return ""

        return f"{value:.2f}"

    if isinstance(value, dict):

        return ", ".join(
            f"{k}: {v}"
            for k, v in value.items()
        )

    if isinstance(value, list):

        return ", ".join(
            clean_pdf_value(x)
            for x in value
        )

    return str(value)


# ============================================================
# DYNAMIC TABLE
# ============================================================

def create_pdf_table(
    data,
    column_widths,
    header=True
):

    formatted_data = []

    for row_index, row in enumerate(data):

        formatted_row = []

        for value in row:

            if row_index == 0 and header:

                formatted_row.append(
                    pdf_paragraph(
                        clean_pdf_value(value),
                        PDF_TABLE_HEADER_STYLE
                    )
                )

            else:

                formatted_row.append(
                    pdf_paragraph(
                        clean_pdf_value(value),
                        PDF_TABLE_CELL_STYLE
                    )
                )

        formatted_data.append(formatted_row)

    table = Table(
        formatted_data,
        colWidths=column_widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT"
    )

    commands = [

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.35,
            colors.HexColor("#B7B7B7")
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        ),

        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            5
        ),

        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            5
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            5
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            5
        )
    ]

    if header:

        commands.append(
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#365F91")
            )
        )

        for row in range(1, len(formatted_data)):

            if row % 2 == 0:

                commands.append(
                    (
                        "BACKGROUND",
                        (0, row),
                        (-1, row),
                        colors.HexColor("#F6F8FB")
                    )
                )

    table.setStyle(
        TableStyle(commands)
    )

    return table


# ============================================================
# PAGE FOOTER
# ============================================================

def pdf_footer(canvas, doc):

    canvas.saveState()

    canvas.setStrokeColor(
        colors.HexColor("#D0D7DE")
    )

    canvas.line(
        18 * mm,
        13 * mm,
        192 * mm,
        13 * mm
    )

    canvas.setFont(
        "Helvetica",
        7.5
    )

    canvas.setFillColor(
        colors.HexColor("#666666")
    )

    canvas.drawString(
        18 * mm,
        8.5 * mm,
        "Child Nutrition Assessment Report"
    )

    canvas.drawRightString(
        192 * mm,
        8.5 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ============================================================
# EXTRACT PRIORITY NUTRIENTS
# ============================================================

def get_pdf_priority_rows(nutrient_result):

    rows = [
        [
            "Nutrient",
            "Priority Weight",
            "Priority"
        ]
    ]

    if nutrient_result is None:
        return rows

    priority_df = nutrient_result.get(
        "priority_df"
    )

    if priority_df is not None:

        try:

            for _, row in priority_df.iterrows():

                nutrient_name = row.get(
                    "nutrient_name",
                    row.get("nutrient", "")
                )

                weight = row.get(
                    "priority_weight",
                    0
                )

                rows.append(
                    [
                        nutrient_name,
                        f"{float(weight):.2f}",
                        "Prioritized"
                    ]
                )

            return rows

        except Exception:
            pass

    priority_list = nutrient_result.get(
        "priority_nutrients",
        []
    )

    for item in priority_list:

        if isinstance(item, dict):

            nutrient_name = item.get(
                "nutrient_name",
                item.get("nutrient", "")
            )

            weight = item.get(
                "priority_weight",
                0
            )

            rows.append(
                [
                    nutrient_name,
                    f"{float(weight):.2f}",
                    "Prioritized"
                ]
            )

    return rows


# ============================================================
# RECOMMENDATION TABLE
# ============================================================

def get_pdf_recommendation_rows(
    recommendations_df
):

    rows = [
        [
            "Recommended Food",
            "Image Food",
            "Image Relation"
        ]
    ]

    if recommendations_df is None:
        return rows

    try:

        for _, row in recommendations_df.iterrows():

            recommended_food = row.get(
                "recommended_food",
                row.get(
                    "food",
                    ""
                )
            )

            image_food = row.get(
                "image_food",
                "—"
            )

            image_relation = row.get(
                "image_relation",
                "none"
            )

            if not image_food:
                image_food = "—"

            rows.append(
                [
                    recommended_food,
                    image_food,
                    image_relation
                ]
            )

    except Exception as e:

        print(
            "Recommendation PDF error:",
            e
        )

    return rows


# ============================================================
# NUTRIENT TABLE
# ============================================================

def get_pdf_nutrient_rows(
    recommendations_df,
    food_df
):

    rows = [
        [
            "Food",
            "Protein",
            "Iron",
            "Calcium",
            "Zinc",
            "Vitamin A",
            "Vitamin C",
            "Fiber",
            "Energy"
        ]
    ]

    if recommendations_df is None:
        return rows

    if food_df is None:
        return rows

    nutrient_map = {
        "Protein": "Protein",
        "Iron": "Iron",
        "Calcium": "Calcium",
        "Zinc": "Zinc",
        "Vitamin A": "Vitamin A",
        "Vitamin C": "Vitamin C",
        "Fiber": "Dietary Fiber",
        "Energy": "Caloric Value"
    }

    try:

        for _, rec in recommendations_df.iterrows():

            food_name = rec.get(
                "recommended_food",
                rec.get("food", "")
            )

            if not food_name:
                continue

            matches = food_df[
                food_df["food"].astype(str).str.lower()
                == str(food_name).lower()
            ]

            if matches.empty:

                normalized = str(food_name).lower()

                matches = food_df[
                    food_df["food"]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        normalized,
                        regex=False,
                        na=False
                    )
                ]

            if matches.empty:
                continue

            food_row = matches.iloc[0]

            row_values = [
                food_name
            ]

            for _, column_name in nutrient_map.items():

                value = food_row.get(
                    column_name,
                    ""
                )

                try:
                    value = f"{float(value):.2f}"
                except Exception:
                    value = str(value)

                row_values.append(value)

            rows.append(row_values)

    except Exception as e:

        print(
            "Nutrient PDF error:",
            e
        )

    return rows


# ============================================================
# MEAL PLAN TABLE
# ============================================================

def get_pdf_meal_rows(
    meal_plan_df
):

    rows = []

    if meal_plan_df is None:
        return rows

    try:

        # Dynamically use the actual meal-plan columns
        available_columns = list(
            meal_plan_df.columns
        )

        preferred_columns = [
            "meal",
            "food",
            "priority_nutrients",
            "nutrient_information"
        ]

        selected_columns = [
            c for c in preferred_columns
            if c in available_columns
        ]

        if not selected_columns:

            selected_columns = [
                "food"
            ] if "food" in available_columns else available_columns[:2]

        header = [
            c.replace("_", " ").title()
            for c in selected_columns
        ]

        rows.append(header)

        for _, row in meal_plan_df.iterrows():

            rows.append(
                [
                    clean_pdf_value(
                        row.get(c, "")
                    )
                    for c in selected_columns
                ]
            )

    except Exception as e:

        print(
            "Meal plan PDF error:",
            e
        )

    return rows


# ============================================================
# ADVICE SECTION
# ============================================================

def add_advice_section(
    story,
    advice_result
):

    if not advice_result:
        return

    story.append(
        pdf_paragraph(
            "9. Nutrition Advice",
            PDF_SECTION_STYLE
        )
    )

    story.append(
        pdf_paragraph(
            "Personalized nutrition and prevention guidance",
            PDF_SUBSECTION_STYLE
        )
    )

    sections = [
        (
            "Age-Based Advice",
            "age_based_advice"
        ),
        (
            "Nutrient Advice",
            "nutrient_advice"
        ),
        (
            "Food-Based Advice",
            "food_based_advice"
        ),
        (
            "Meal-Plan Advice",
            "meal_plan_advice"
        ),
        (
            "Prevention Advice",
            "prevention_advice"
        )
    ]

    for title_text, key in sections:

        values = advice_result.get(
            key,
            []
        )

        if not values:
            continue

        story.append(
            pdf_paragraph(
                title_text,
                PDF_SUBSECTION_STYLE
            )
        )

        if isinstance(values, str):
            values = [values]

        for item in values:

            story.append(
                pdf_paragraph(
                    "• " + str(item),
                    PDF_BULLET_STYLE
                )
            )


# ============================================================
# MAIN PDF FUNCTION
# ============================================================

def generate_child_nutrition_pdf(
    report_data,
    output_path=None
):

    if output_path is None:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_path = (
            REPORT_DIR /
            f"Child_Nutrition_Report_{timestamp}.pdf"
        )

    output_path = Path(output_path)

    prediction = report_data.get(
        "prediction",
        {}
    )

    nutrient_result = report_data.get(
        "nutrient_analysis",
        {}
    )

    recommendations_df = report_data.get(
        "recommendations_df"
    )

    image_df = report_data.get(
        "image_results_df"
    )

    meal_plan_df = report_data.get(
        "meal_plan_df"
    )

    advice_result = report_data.get(
        "nutrition_advice",
        {}
    )

    food_df = report_data.get(
        "food_df"
    )

    shap_image = report_data.get(
        "shap_image"
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        pdf_paragraph(
            "Child Nutrition Assessment Report",
            PDF_TITLE_STYLE
        )
    )

    story.append(
        pdf_paragraph(
            "Agentic AI Nutrition Recommendation System",
            PDF_SUBTITLE_STYLE
        )
    )

    # ========================================================
    # 1 CHILD INFORMATION
    # ========================================================

    story.append(
        pdf_paragraph(
            "1. Child Information",
            PDF_SECTION_STYLE
        )
    )

    child_rows = [
        ["Measurement", "Value"],
        [
            "Age",
            f"{prediction.get('age_months', '')} months"
        ],
        [
            "Weight",
            f"{prediction.get('weight_kg', ''):.2f} kg"
            if prediction.get("weight_kg") is not None
            else ""
        ],
        [
            "Height",
            f"{prediction.get('height_cm', ''):.2f} cm"
            if prediction.get("height_cm") is not None
            else ""
        ],
        [
            "MUAC",
            f"{prediction.get('muac_cm', ''):.2f} cm"
            if prediction.get("muac_cm") is not None
            else ""
        ],
        [
            "BMI",
            f"{prediction.get('bmi', ''):.2f}"
            if prediction.get("bmi") is not None
            else ""
        ]
    ]

    story.append(
        create_pdf_table(
            child_rows,
            [75 * mm, 75 * mm]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # ========================================================
    # 2 PREDICTION
    # ========================================================

    story.append(
        pdf_paragraph(
            "2. Nutrition Risk Prediction",
            PDF_SECTION_STYLE
        )
    )

    prediction_rows = [
        ["Prediction Item", "Result"],
        [
            "Predicted Nutrition Status",
            prediction.get(
                "predicted_status",
                ""
            )
        ],
        [
            "Best Performing Model",
            prediction.get(
                "best_model",
                ""
            )
        ]
    ]

    story.append(
        create_pdf_table(
            prediction_rows,
            [75 * mm, 75 * mm]
        )
    )

    story.append(
        Spacer(1, 7)
    )

    story.append(
        pdf_paragraph(
            "Prediction Probabilities",
            PDF_SUBSECTION_STYLE
        )
    )

    probability_rows = [
        ["Class", "Probability"]
    ]

    probabilities = prediction.get(
        "probabilities",
        {}
    )

    for class_name, probability in probabilities.items():

        try:
            probability_text = (
                f"{float(probability):.2f}%"
            )
        except Exception:
            probability_text = str(probability)

        probability_rows.append(
            [
                class_name,
                probability_text
            ]
        )

    story.append(
        create_pdf_table(
            probability_rows,
            [75 * mm, 75 * mm]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # ========================================================
    # 3 NUTRIENT PRIORITIES
    # ========================================================

    story.append(
        pdf_paragraph(
            "3. Personalized Nutrient Priorities",
            PDF_SECTION_STYLE
        )
    )

    priority_rows = get_pdf_priority_rows(
        nutrient_result
    )

    story.append(
        create_pdf_table(
            priority_rows,
            [70 * mm, 38 * mm, 42 * mm]
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 4 RECOMMENDED FOODS
    # ========================================================

    story.append(
        pdf_paragraph(
            "4. Recommended Foods",
            PDF_SECTION_STYLE
        )
    )

    recommendation_rows = (
        get_pdf_recommendation_rows(
            recommendations_df
        )
    )

    story.append(
        create_pdf_table(
            recommendation_rows,
            [70 * mm, 55 * mm, 35 * mm]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # ========================================================
    # 5 NUTRIENTS
    # ========================================================

    story.append(
        pdf_paragraph(
            "5. Nutrients in Recommended Foods",
            PDF_SECTION_STYLE
        )
    )

    nutrient_rows = get_pdf_nutrient_rows(
        recommendations_df,
        food_df
    )

    story.append(
        create_pdf_table(
            nutrient_rows,
            [
                38 * mm,
                15 * mm,
                13 * mm,
                16 * mm,
                13 * mm,
                16 * mm,
                16 * mm,
                15 * mm,
                15 * mm
            ]
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 6 IMAGES
    # ========================================================

    story.append(
        pdf_paragraph(
            "6. Indian Food Images",
            PDF_SECTION_STYLE
        )
    )

    if image_df is not None:

        try:

            for _, row in image_df.iterrows():

                recommended_food = row.get(
                    "recommended_food",
                    ""
                )

                image_food = row.get(
                    "image_food",
                    ""
                )

                relation = row.get(
                    "image_relation",
                    "none"
                )

                image_path = row.get(
                    "image_path",
                    ""
                )

                if not image_path:
                    continue

                image_path = Path(
                    str(image_path)
                )

                if not image_path.exists():
                    continue

                story.append(
                    pdf_paragraph(
                        f"<b>Recommended food:</b> "
                        f"{recommended_food}",
                        PDF_BODY_STYLE
                    )
                )

                story.append(
                    pdf_paragraph(
                        f"<b>Image food:</b> "
                        f"{image_food}",
                        PDF_BODY_STYLE
                    )
                )

                story.append(
                    pdf_paragraph(
                        f"<b>Image relation:</b> "
                        f"{relation}",
                        PDF_BODY_STYLE
                    )
                )

                img = RLImage(
                    str(image_path)
                )

                max_width = 100 * mm
                max_height = 55 * mm

                scale = min(
                    max_width / img.imageWidth,
                    max_height / img.imageHeight
                )

                img.drawWidth = (
                    img.imageWidth * scale
                )

                img.drawHeight = (
                    img.imageHeight * scale
                )

                img.hAlign = "CENTER"

                story.append(img)

                story.append(
                    Spacer(1, 10)
                )

        except Exception as e:

            print(
                "Image PDF error:",
                e
            )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 7 MEAL PLAN
    # ========================================================

    story.append(
        pdf_paragraph(
            "7. Personalized Meal Plan",
            PDF_SECTION_STYLE
        )
    )

    meal_rows = get_pdf_meal_rows(
        meal_plan_df
    )

    if len(meal_rows) > 1:

        number_columns = len(
            meal_rows[0]
        )

        total_width = 150 * mm

        column_width = (
            total_width /
            number_columns
        )

        story.append(
            create_pdf_table(
                meal_rows,
                [column_width] *
                number_columns
            )
        )

    else:

        story.append(
            pdf_paragraph(
                "Meal plan information is not available.",
                PDF_BODY_STYLE
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 8 SHAP
    # ========================================================

    story.append(
        pdf_paragraph(
            "8. Explainable AI — SHAP",
            PDF_SECTION_STYLE
        )
    )

    if shap_image:

        shap_path = Path(
            str(shap_image)
        )

        if shap_path.exists():

            img = RLImage(
                str(shap_path)
            )

            max_width = 150 * mm
            max_height = 180 * mm

            scale = min(
                max_width / img.imageWidth,
                max_height / img.imageHeight
            )

            img.drawWidth = (
                img.imageWidth * scale
            )

            img.drawHeight = (
                img.imageHeight * scale
            )

            img.hAlign = "CENTER"

            story.append(img)

    else:

        story.append(
            pdf_paragraph(
                "SHAP explanation generated for the selected prediction.",
                PDF_BODY_STYLE
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 9 NUTRITION ADVICE
    # ========================================================

    add_advice_section(
        story,
        advice_result
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    pdf = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Child Nutrition Assessment Report",
        author="Agentic AI Nutrition Recommendation System"
    )

    pdf.build(
        story,
        onFirstPage=pdf_footer,
        onLaterPages=pdf_footer
    )

    print(
        f"PDF generated successfully:\n{output_path}"
    )

    return output_path


# ============================================================
# REPORT AGENT
# ============================================================

class NutritionReportAgent:

    def generate(
        self,
        report_data,
        output_path=None
    ):

        return generate_child_nutrition_pdf(
            report_data=report_data,
            output_path=output_path
        )


print("Part 13 - Dynamic PDF Report Generation Agent loaded successfully.")
# ================= ORIGINAL NOTEBOOK CELL 35 =================
# ============================================================
# PART 14 — AGENTIC AI ORCHESTRATOR
# ============================================================
#
# IMPORTANT:
# Part 14 does NOT redefine run_image_matching_agent().
# The real function belongs to Part 7.
#
# Agent flow:
#
# Prediction Agent
#       ↓
# Nutrient Analysis / Recommendation Agent
#       ↓
# Image Matching Agent
#       ↓
# Deep Learning Image Validation Agent
#       ↓
# SHAP Explanation Agent
#       ↓
# Meal Plan Agent
#       ↓
# Nutrition Advice Agent
#       ↓
# Report Agent
#
# No child input is requested in Part 14.
# Part 15 provides the final user input.
# ============================================================


import os
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


print("=" * 75)
print("PART 14 — AGENTIC AI ORCHESTRATOR")
print("=" * 75)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SAFE DATAFRAME FUNCTION
# ============================================================

def safe_dataframe(value):

    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        return pd.DataFrame(value)

    if isinstance(value, dict):
        return pd.DataFrame([value])

    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


# ============================================================
# 3. SAFE JSON CONVERSION
# ============================================================

def make_json_serializable(value):

    if isinstance(value, dict):

        return {
            str(k): make_json_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            make_json_serializable(v)
            for v in value
        ]

    if isinstance(value, tuple):

        return [
            make_json_serializable(v)
            for v in value
        ]

    if isinstance(value, pd.DataFrame):

        return value.to_dict(orient="records")

    if isinstance(value, pd.Series):

        return value.to_dict()

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (np.floating,)):

        return float(value)

    if isinstance(value, (np.bool_,)):

        return bool(value)

    if isinstance(value, Path):

        return str(value)

    return value


# ============================================================
# 4. CHECK REQUIRED AGENTS
# ============================================================

def check_required_agents():

    required_functions = {

        "Prediction Agent":
            "predict_child",

        "Image Matching Agent":
            "run_image_matching_agent",

        "SHAP Explanation Agent":
            "explain_child_prediction",

        "Meal Plan Agent":
            "generate_meal_plan",

        "Nutrition Advice Agent":
            "generate_nutrition_advice"
    }

    missing = []

    for agent_name, function_name in required_functions.items():

        if function_name not in globals():

            missing.append(
                f"{agent_name} → {function_name}"
            )

    if missing:

        print()
        print("WARNING — Some agent functions are not loaded:")

        for item in missing:
            print("   -", item)

        print()

    return missing


missing_agents = check_required_agents()


# ============================================================
# 5. IMAGE MATCHING WRAPPER
# ============================================================
#
# IMPORTANT:
# This function has a DIFFERENT name.
#
# We deliberately do NOT create:
#
# def run_image_matching_agent(...)
#
# because that function already belongs to Part 7.
# ============================================================

def execute_image_matching_agent(
    recommendations_df,
    nutrient_priority_result=None
):

    # --------------------------------------------------------
    # Make sure Part 7 function exists
    # --------------------------------------------------------

    if "run_image_matching_agent" not in globals():

        raise RuntimeError(
            "Part 7 function 'run_image_matching_agent' "
            "is not available.\n"
            "Run the corrected Part 7 before Part 14."
        )


    # --------------------------------------------------------
    # Call the REAL Part 7 function
    # --------------------------------------------------------

    matched_recommendations = run_image_matching_agent(

        recommendations_df=
            recommendations_df,

        nutrient_priority_result=
            nutrient_priority_result
    )


    return safe_dataframe(
        matched_recommendations
    )


# ============================================================
# 6. DEEP LEARNING IMAGE VALIDATION WRAPPER
# ============================================================

def execute_deep_learning_image_agent(
    recommendations_with_images,
    confidence_threshold=0.50
):

    # --------------------------------------------------------
    # If Part 8 function exists, use it
    # --------------------------------------------------------

    if "run_deep_learning_image_agent" in globals():

        try:

            validated = run_deep_learning_image_agent(

                recommendations_with_images=
                    recommendations_with_images,

                confidence_threshold=
                    confidence_threshold
            )

            return safe_dataframe(validated)

        except TypeError:

            # Compatibility with alternative Part 8 signature
            try:

                validated = run_deep_learning_image_agent(
                    recommendations_with_images
                )

                return safe_dataframe(validated)

            except Exception as error:

                print(
                    "Deep Learning image validation warning:",
                    error
                )


    # --------------------------------------------------------
    # If DL validation function is unavailable,
    # preserve the matched recommendations.
    # --------------------------------------------------------

    result = safe_dataframe(
        recommendations_with_images
    )

    if result.empty:
        return result


    if "dl_validation_status" not in result.columns:

        result["dl_validation_status"] = (
            "not_validated"
        )

    if "dl_prediction" not in result.columns:

        result["dl_prediction"] = ""

    if "dl_confidence" not in result.columns:

        result["dl_confidence"] = np.nan


    return result


# ============================================================
# 7. NUTRITION RECOMMENDATION EXECUTION
# ============================================================

def execute_nutrition_recommendation_agent(
    prediction_result,
    top_n=10
):

    # --------------------------------------------------------
    # Preferred Part 6 function
    # --------------------------------------------------------

    if "run_nutrition_recommendation_agent" in globals():

        function = globals()[
            "run_nutrition_recommendation_agent"
        ]

        try:

            result = function(

                prediction_result=
                    prediction_result,

                top_n=
                    top_n
            )

            return safe_dataframe(result)

        except TypeError:

            # Compatibility with a simpler signature
            try:

                result = function(
                    prediction_result,
                    top_n
                )

                return safe_dataframe(result)

            except TypeError:

                result = function(
                    prediction_result
                )

                result = safe_dataframe(result)

                return result.head(top_n)


    # --------------------------------------------------------
    # Alternative Part 6 function
    # --------------------------------------------------------

    if "generate_food_recommendations" in globals():

        function = globals()[
            "generate_food_recommendations"
        ]

        predicted_status = prediction_result[
            "predicted_status"
        ]

        age_months = prediction_result[
            "age_months"
        ]

        try:

            result = function(

                predicted_status=
                    predicted_status,

                age_months=
                    age_months,

                top_n=
                    top_n
            )

        except TypeError:

            try:

                result = function(

                    predicted_status=
                        predicted_status,

                    age_months=
                        age_months
                )

            except TypeError:

                result = function(
                    predicted_status
                )

        return safe_dataframe(result).head(top_n)


    raise RuntimeError(
        "No nutrition recommendation function is available. "
        "Run the corrected Part 6 before Part 14."
    )


# ============================================================
# 8. NUTRIENT PRIORITY ANALYSIS
# ============================================================

def execute_nutrient_priority_agent(
    prediction_result,
    recommendations_df
):

    # --------------------------------------------------------
    # Try Part 6 priority analysis
    # --------------------------------------------------------

    if "analyze_nutrient_priorities" in globals():

        function = globals()[
            "analyze_nutrient_priorities"
        ]

        predicted_status = prediction_result[
            "predicted_status"
        ]

        try:

            result = function(
                predicted_status
            )

            return result

        except TypeError:

            try:

                result = function(
                    prediction_result=
                        prediction_result
                )

                return result

            except Exception:
                pass


    # --------------------------------------------------------
    # Try priority profile if available
    # --------------------------------------------------------

    if "determine_priority_profile" in globals():

        try:

            return determine_priority_profile(
                prediction_result[
                    "predicted_status"
                ]
            )

        except Exception:
            pass


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    predicted_status = str(
        prediction_result[
            "predicted_status"
        ]
    ).lower().strip()


    priority_profiles = {

        "normal": [
            "protein",
            "calcium",
            "iron",
            "fiber",
            "vitamin_a",
            "vitamin_c"
        ],

        "moderate": [
            "energy",
            "protein",
            "iron",
            "calcium",
            "zinc",
            "vitamin_a",
            "vitamin_c"
        ],

        "severe": [
            "energy",
            "protein",
            "iron",
            "calcium",
            "zinc",
            "vitamin_a",
            "vitamin_c",
            "vitamin_b12"
        ]
    }


    return {

        "profile":
            predicted_status,

        "priority_nutrients":
            priority_profiles.get(
                predicted_status,
                priority_profiles["normal"]
            )
    }


# ============================================================
# 9. SHAP EXECUTION
# ============================================================

def execute_shap_agent(
    prediction_result
):

    if "explain_child_prediction" not in globals():

        return {

            "available": False,

            "message":
                "SHAP explanation function is not available."
        }


    try:

        result = explain_child_prediction(

            age_months=
                prediction_result[
                    "age_months"
                ],

            weight_kg=
                prediction_result[
                    "weight_kg"
                ],

            height_cm=
                prediction_result[
                    "height_cm"
                ],

            muac_cm=
                prediction_result[
                    "muac_cm"
                ],

            predicted_class_encoded=
                prediction_result[
                    "predicted_class"
                ]
        )

        return result

    except TypeError:

        # Compatibility with older SHAP function
        try:

            result = explain_child_prediction(

                prediction_result[
                    "age_months"
                ],

                prediction_result[
                    "weight_kg"
                ],

                prediction_result[
                    "height_cm"
                ],

                prediction_result[
                    "muac_cm"
                ]
            )

            return result

        except Exception as error:

            return {

                "available": False,

                "error":
                    str(error)
            }

    except Exception as error:

        return {

            "available": False,

            "error":
                str(error)
        }


# ============================================================
# 10. MEAL PLAN EXECUTION
# ============================================================

def execute_meal_plan_agent(
    prediction_result,
    recommendations_df
):

    if "generate_meal_plan" not in globals():

        raise RuntimeError(
            "Part 10 function 'generate_meal_plan' "
            "is not available."
        )


    function = globals()[
        "generate_meal_plan"
    ]


    predicted_status = prediction_result[
        "predicted_status"
    ]

    age_months = prediction_result[
        "age_months"
    ]


    try:

        result = function(

            predicted_status=
                predicted_status,

            age_months=
                age_months,

            recommendations_df=
                recommendations_df
        )

    except TypeError:

        try:

            result = function(

                predicted_status,
                age_months,
                recommendations_df
            )

        except TypeError:

            result = function(

                predicted_status=
                    predicted_status,

                age_months=
                    age_months
            )


    result = safe_dataframe(result)


    # --------------------------------------------------------
    # Add nutrient information if Part 10 provides function
    # --------------------------------------------------------

    if (
        not result.empty
        and
        "add_meal_nutrient_information" in globals()
    ):

        try:

            result = add_meal_nutrient_information(
                result
            )

        except Exception as error:

            print(
                "Meal nutrient enrichment warning:",
                error
            )


    return result


# ============================================================
# 11. NUTRITION ADVICE EXECUTION
# ============================================================

def execute_nutrition_advice_agent(
    prediction_result,
    nutrient_priority_result,
    recommendations_df,
    meal_plan_df
):

    if "generate_nutrition_advice" not in globals():

        return {

            "available": False,

            "message":
                "Nutrition advice function is not available."
        }


    function = globals()[
        "generate_nutrition_advice"
    ]


    predicted_status = prediction_result[
        "predicted_status"
    ]

    age_months = prediction_result[
        "age_months"
    ]


    try:

        result = function(

            predicted_status=
                predicted_status,

            age_months=
                age_months,

            priority_nutrients=
                nutrient_priority_result,

            recommendations_df=
                recommendations_df,

            meal_plan_df=
                meal_plan_df
        )

        return result


    except TypeError:

        try:

            result = function(

                predicted_status=
                    predicted_status,

                age_months=
                    age_months,

                recommendations_df=
                    recommendations_df,

                meal_plan_df=
                    meal_plan_df
            )

            return result

        except TypeError:

            try:

                result = function(

                    predicted_status=
                        predicted_status,

                    age_months=
                        age_months
                )

                return result

            except Exception as error:

                return {

                    "available": False,

                    "error":
                        str(error)
                }

    except Exception as error:

        return {

            "available": False,

            "error":
                str(error)
        }


# ============================================================
# 12. REPORT AGENT
# ============================================================

def execute_report_agent(
    final_result
):

    # --------------------------------------------------------
    # Part 13 can be called here if a report function exists.
    # --------------------------------------------------------

    possible_report_functions = [

        "generate_final_pdf_report",

        "generate_child_nutrition_report",

        "generate_nutrition_pdf_report",

        "create_pdf_report",

        "generate_pdf_report"
    ]


    for function_name in possible_report_functions:

        if function_name in globals():

            function = globals()[
                function_name
            ]

            try:

                return function(
                    final_result
                )

            except TypeError:

                try:

                    return function(
                        report_data=
                            final_result
                    )

                except Exception:
                    pass

            except Exception:
                pass


    # --------------------------------------------------------
    # Report agent unavailable.
    # Part 15 can still receive all results.
    # --------------------------------------------------------

    return None


# ============================================================
# 13. AGENTIC NUTRITION ORCHESTRATOR
# ============================================================

class AgenticNutritionOrchestrator:

    def __init__(self):

        self.status = {}

        self.execution_log = []


    # --------------------------------------------------------
    # UPDATE STATUS
    # --------------------------------------------------------

    def update_status(
        self,
        agent_name,
        status,
        message=""
    ):

        self.status[
            agent_name
        ] = {

            "status":
                status,

            "message":
                message
        }


        self.execution_log.append({

            "agent":
                agent_name,

            "status":
                status,

            "message":
                message
        })


    # --------------------------------------------------------
    # RUN COMPLETE PIPELINE
    # --------------------------------------------------------

    def run(
        self,
        age_months,
        weight_kg,
        height_cm,
        muac_cm,
        top_n_foods=10
    ):

        # ====================================================
        # INITIALIZE
        # ====================================================

        self.status = {}
        self.execution_log = []


        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        age_months = float(age_months)
        weight_kg = float(weight_kg)
        height_cm = float(height_cm)
        muac_cm = float(muac_cm)


        if not 0 <= age_months <= 60:

            raise ValueError(
                "Age must be between 0 and 60 months."
            )


        if weight_kg <= 0:

            raise ValueError(
                "Weight must be greater than zero."
            )


        if height_cm <= 0:

            raise ValueError(
                "Height must be greater than zero."
            )


        if muac_cm <= 0:

            raise ValueError(
                "MUAC must be greater than zero."
            )


        # ====================================================
        # AGENT 1 — PREDICTION
        # ====================================================

        self.update_status(

            "Prediction Agent",

            "running",

            "Predicting child nutrition status."
        )


        try:

            if "predict_child" not in globals():

                raise RuntimeError(
                    "Part 5 function "
                    "'predict_child' is not available."
                )


            prediction_result = predict_child(

                age_months=
                    age_months,

                weight_kg=
                    weight_kg,

                height_cm=
                    height_cm,

                muac_cm=
                    muac_cm
            )


            prediction_result = (
                make_json_serializable(
                    prediction_result
                )
            )


            self.update_status(

                "Prediction Agent",

                "completed",

                "Nutrition status predicted successfully."
            )


        except Exception as error:

            self.update_status(

                "Prediction Agent",

                "failed",

                str(error)
            )

            raise


        # ====================================================
        # AGENT 2 — NUTRIENT PRIORITY ANALYSIS
        # ====================================================

        self.update_status(

            "Nutrient Analysis Agent",

            "running",

            "Determining personalized nutrient priorities."
        )


        try:

            nutrient_analysis = (
                execute_nutrient_priority_agent(

                    prediction_result=
                        prediction_result,

                    recommendations_df=
                        pd.DataFrame()
                )
            )


            nutrient_analysis = (
                make_json_serializable(
                    nutrient_analysis
                )
            )


            self.update_status(

                "Nutrient Analysis Agent",

                "completed",

                "Personalized nutrient priorities generated."
            )


        except Exception as error:

            self.update_status(

                "Nutrient Analysis Agent",

                "failed",

                str(error)
            )

            raise


        # ====================================================
        # AGENT 3 — FOOD RECOMMENDATION
        # ====================================================

        self.update_status(

            "Nutrition Recommendation Agent",

            "running",

            "Generating nutrition-first food recommendations."
        )


        try:

            recommendations_df = (
                execute_nutrition_recommendation_agent(

                    prediction_result=
                        prediction_result,

                    top_n=
                        top_n_foods
                )
            )


            recommendations_df = safe_dataframe(
                recommendations_df
            )


            self.update_status(

                "Nutrition Recommendation Agent",

                "completed",

                f"{len(recommendations_df)} food recommendations generated."
            )


        except Exception as error:

            self.update_status(

                "Nutrition Recommendation Agent",

                "failed",

                str(error)
            )

            raise


        # ====================================================
        # UPDATE NUTRIENT ANALYSIS
        # ====================================================

        if (
            nutrient_analysis is None
            or
            (
                isinstance(
                    nutrient_analysis,
                    dict
                )
                and
                len(nutrient_analysis) == 0
            )
        ):

            try:

                nutrient_analysis = (
                    execute_nutrient_priority_agent(

                        prediction_result=
                            prediction_result,

                        recommendations_df=
                            recommendations_df
                    )
                )

            except Exception:
                pass


        # ====================================================
        # AGENT 4 — IMAGE MATCHING
        # ====================================================
        #
        # IMPORTANT:
        # We call execute_image_matching_agent().
        #
        # We DO NOT call a Part-14 function named
        # run_image_matching_agent().
        #
        # Part 7 owns run_image_matching_agent().
        # ====================================================

        self.update_status(

            "Image Matching Agent",

            "running",

            "Matching recommended foods with Indian food images."
        )


        try:

            recommendations_with_images = (
                execute_image_matching_agent(

                    recommendations_df=
                        recommendations_df,

                    nutrient_priority_result=
                        nutrient_analysis
                )
            )


            recommendations_with_images = safe_dataframe(
                recommendations_with_images
            )


            self.update_status(

                "Image Matching Agent",

                "completed",

                "Exact/related Indian food images matched."
            )


        except Exception as error:

            self.update_status(

                "Image Matching Agent",

                "failed",

                str(error)
            )

            raise


        # ====================================================
        # AGENT 5 — DEEP LEARNING IMAGE VALIDATION
        # ====================================================

        self.update_status(

            "Deep Learning Image Validation Agent",

            "running",

            "Validating selected food images using the trained DL model."
        )


        try:

            validated_recommendations = (
                execute_deep_learning_image_agent(

                    recommendations_with_images=
                        recommendations_with_images,

                    confidence_threshold=
                        0.50
                )
            )


            self.update_status(

                "Deep Learning Image Validation Agent",

                "completed",

                "Food image validation completed."
            )


        except Exception as error:

            self.update_status(

                "Deep Learning Image Validation Agent",

                "failed",

                str(error)
            )

            raise


        # ====================================================
        # AGENT 6 — SHAP EXPLANATION
        # ====================================================

        self.update_status(

            "SHAP Explanation Agent",

            "running",

            "Generating explanation for the predicted class."
        )


        try:

            shap_result = execute_shap_agent(
                prediction_result
            )


            self.update_status(

                "SHAP Explanation Agent",

                "completed",

                "SHAP explanation generated."
            )


        except Exception as error:

            shap_result = {

                "available": False,

                "error":
                    str(error)
            }


            self.update_status(

                "SHAP Explanation Agent",

                "failed",

                str(error)
            )


        # ====================================================
        # AGENT 7 — MEAL PLAN
        # ====================================================

        self.update_status(

            "Meal Plan Agent",

            "running",

            "Generating personalized meal plan."
        )


        try:

            meal_plan_df = (
                execute_meal_plan_agent(

                    prediction_result=
                        prediction_result,

                    recommendations_df=
                        validated_recommendations
                )
            )


            meal_plan_df = safe_dataframe(
                meal_plan_df
            )


            self.update_status(

                "Meal Plan Agent",

                "completed",

                "Meal plan generated successfully."
            )


        except Exception as error:

            self.update_status(

                "Meal Plan Agent",

                "failed",

                str(error)
            )

            raise


        # ====================================================
        # AGENT 8 — NUTRITION ADVICE
        # ====================================================

        self.update_status(

            "Nutrition Advice Agent",

            "running",

            "Generating nutrition and prevention guidance."
        )


        try:

            nutrition_advice = (
                execute_nutrition_advice_agent(

                    prediction_result=
                        prediction_result,

                    nutrient_priority_result=
                        nutrient_analysis,

                    recommendations_df=
                        validated_recommendations,

                    meal_plan_df=
                        meal_plan_df
                )
            )


            self.update_status(

                "Nutrition Advice Agent",

                "completed",

                "Nutrition advice generated."
            )


        except Exception as error:

            nutrition_advice = {

                "available": False,

                "error":
                    str(error)
            }


            self.update_status(

                "Nutrition Advice Agent",

                "failed",

                str(error)
            )


        # ====================================================
        # BUILD COMPLETE RESULT
        # ====================================================

        final_result = {

            "child_input": {

                "age_months":
                    age_months,

                "weight_kg":
                    weight_kg,

                "height_cm":
                    height_cm,

                "muac_cm":
                    muac_cm
            },


            "prediction":
                prediction_result,


            "nutrient_analysis":
                nutrient_analysis,


            "recommendations":
                recommendations_df,


            "recommendations_with_images":
                recommendations_with_images,


            "validated_recommendations":
                validated_recommendations,


            "shap":
                shap_result,


            "meal_plan":
                meal_plan_df,


            "nutrition_advice":
                nutrition_advice,


            "agent_status":
                self.status,


            "execution_log":
                self.execution_log
        }


        # ====================================================
        # SAVE ORCHESTRATOR RESULT
        # ====================================================

        serializable_result = (
            make_json_serializable(
                final_result
            )
        )


        with open(

            REPORT_DIR /
            "agentic_pipeline_execution.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                serializable_result,

                file,

                indent=4,

                ensure_ascii=False
            )


        # ====================================================
        # REPORT AGENT
        # ====================================================

        self.update_status(

            "Report Agent",

            "ready",

            "All report data assembled for Part 13/15."
        )


        return final_result


# ============================================================
# 14. CREATE ORCHESTRATOR
# ============================================================

nutrition_orchestrator = (
    AgenticNutritionOrchestrator()
)


# ============================================================
# 15. SAVE ORCHESTRATOR INFORMATION
# ============================================================

orchestrator_information = {

    "architecture":
        "Agentic AI Child Nutrition Recommendation and Malnutrition Prevention",

    "agents": [

        "Prediction Agent",

        "Nutrient Analysis Agent",

        "Nutrition Recommendation Agent",

        "Image Matching Agent",

        "Deep Learning Image Validation Agent",

        "SHAP Explanation Agent",

        "Meal Plan Agent",

        "Nutrition Advice Agent",

        "Report Agent"
    ],

    "user_inputs": [

        "age_months",

        "weight_kg",

        "height_cm",

        "muac_cm"
    ],

    "part_7_function_owner":
        "Part 7",

    "part_7_function":
        "run_image_matching_agent",

    "part_14_wrapper":
        "execute_image_matching_agent",

    "deep_learning_retraining":
        False,

    "part_15_provides_user_input":
        True
}


with open(

    MODEL_DIR /
    "agentic_orchestrator_information.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        orchestrator_information,

        file,

        indent=4
    )


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 75)
print("PART 14 LOADED SUCCESSFULLY")
print("=" * 75)

print()
print("Orchestrator:")
print("    nutrition_orchestrator")

print()
print("Image matching:")
print("    Part 7 → run_image_matching_agent()")
print("    Part 14 → execute_image_matching_agent()")

print()
print("Deep Learning:")
print("    Existing trained model will be used.")
print("    No retraining is performed by Part 14.")

print()
print("Final user input:")
print("    Part 15")

print()
print("Ready to run Part 15.")
print("=" * 75)
# ================= STREAMLIT COMPATIBILITY =================

# The notebook already creates these objects in Parts 5-14.
# These checks fail with a precise message instead of the generic Streamlit
# "pipeline could not be loaded" message.

_REQUIRED_PIPELINE_NAMES = [
    "predict_child",
    "analyze_nutrient_priorities",
    "nutrition_recommendation_agent",
    "run_image_matching_agent",
    "run_deep_learning_image_agent",
    "generate_meal_plan",
    "generate_nutrition_advice",
    "explain_child_prediction",
    "create_shap_waterfall_plot",
    "generate_shap_explanation_text",
    "nutrition_orchestrator",
]

_missing_pipeline_names = [
    name for name in _REQUIRED_PIPELINE_NAMES
    if name not in globals()
]

if _missing_pipeline_names:
    raise ImportError(
        "The existing Parts 1-14 logic loaded, but these expected objects "
        "are missing: " + ", ".join(_missing_pipeline_names)
    )

print("=" * 75)
print("CHILD NUTRITION AI PIPELINE READY")
print("Parts 1-14 loaded.")
print("Part 15 input() was NOT executed.")
print("Training cells are available as train_part4_models() and "
      "train_part8_mobilenetv2(), but are not run automatically.")
print("=" * 75)
