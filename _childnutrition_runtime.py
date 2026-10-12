# =====================================================
# AUTO-GENERATED CHILD NUTRITION RUNTIME
# MobileNetV2 training has been removed
# =====================================================

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# =====================================================
# NOTEBOOK CODE CELL 1
# =====================================================



# =====================================================
# NOTEBOOK CODE CELL 2
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 3
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 4
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 5
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 6
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 7
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 8
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 9
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 10
# =====================================================

import inspect

print("SIGNATURE:")
print(inspect.signature(run_image_matching_agent))

print("\nSOURCE:")
print(inspect.getsource(run_image_matching_agent))

# =====================================================
# NOTEBOOK CODE CELL 11
# =====================================================

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
# 8.15 TRAINING SKIPPED DURING STREAMLIT EXECUTION
# ============================================================

print("
" + "=" * 70)
print("SKIPPING DEEP LEARNING TRAINING")
print("=" * 70)

print("✓ Existing trained MobileNetV2 model will be loaded.")





# ============================================================
# 8.16 LOAD EXISTING TRAINED MODEL
# ============================================================

if DL_MODEL_PATH.exists():

    print("
Loading existing trained MobileNetV2 model...")

    food_model = tf.keras.models.load_model(
        DL_MODEL_PATH,
        compile=False
    )

    print("✓ Existing MobileNetV2 model loaded successfully.")
    print("✓ MobileNetV2 TRAINING: NOT EXECUTED")

else:

    raise FileNotFoundError(
        "Trained MobileNetV2 model was not found: "
        + str(DL_MODEL_PATH)
    )


# ============================================================
# 8.17 TRAINING HISTORY
# ============================================================

# No training is performed during Streamlit execution.
# Therefore no new training history is generated.

history = None

# ============================================================
# 8.18 TRAINING VISUALIZATION
# ============================================================

print("✓ Training-history plots skipped because no training was performed.")

# ============================================================
# 8.19 TRAINING LOSS VISUALIZATION
# ============================================================

print("✓ Training-loss plots skipped because no training was performed.")


# ============================================================
# CONTINUE WITH REMAINING PART 8 CODE
# ============================================================


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

# =====================================================
# NOTEBOOK CODE CELL 12
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 13
# =====================================================



# =====================================================
# NOTEBOOK CODE CELL 14
# =====================================================

import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# =====================================================
# NOTEBOOK CODE CELL 15
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 16
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 17
# =====================================================

PROCESSED_CHILD_DATASET = DATA_DIR / "processed_child_dataset.csv"

if not PROCESSED_CHILD_DATASET.exists():
    raise FileNotFoundError(
        f"Processed child dataset not found:\n{PROCESSED_CHILD_DATASET}\n"
        "Please run Part 3 first."
    )

processed_child_df = pd.read_csv(PROCESSED_CHILD_DATASET)

print("Processed child dataset shape:", processed_child_df.shape)

display(processed_child_df.head())

# =====================================================
# NOTEBOOK CODE CELL 18
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 19
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 20
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 21
# =====================================================

print("Calculating SHAP values...")

shap_background_values = shap_explainer(
    X_background_model
)

print("✓ SHAP values calculated.")

# =====================================================
# NOTEBOOK CODE CELL 22
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 23
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 24
# =====================================================

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

# =====================================================
# NOTEBOOK CODE CELL 25
# =====================================================

global_shap_report = REPORT_DIR / "shap_global_feature_importance.csv"

shap_importance_df.to_csv(
    global_shap_report,
    index=False
)

print("Saved:", global_shap_report)
