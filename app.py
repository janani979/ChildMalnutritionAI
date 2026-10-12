# ============================================================
# app.py
# CHILD NUTRITION AGENTIC AI - STREAMLIT UI
#
# This file is the UI layer for the existing project.
# It does NOT replace Parts 1-15.
#
# IMPORTANT:
# Your existing notebook contains the actual project functions.
# Before running this app, export/copy Parts 1-14 into a Python
# module named:
#
#     nutrition_pipeline.py
#
# Part 15 input() should NOT be copied into that module.
#
# The module must expose these existing functions/objects:
#     predict_child
#     analyze_nutrient_priorities
#     nutrition_recommendation_agent
#     run_image_matching_agent
#     run_deep_learning_image_agent
#     generate_meal_plan
#     explain_child_prediction
#     create_shap_waterfall_plot
#     generate_nutrition_advice
#
# It may also expose:
#     REPORT_DIR
#     OUTPUT_DIR
#     IMAGE_ROOT
# ============================================================

from pathlib import Path
import json
import traceback
import importlib
import sys

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Child Nutrition AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_ROOT = Path(
    r"C:\indian_food_images\Indian Food Images\Indian Food Images"
)

for folder in [
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    OUTPUT_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD EXISTING PROJECT PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Loads the existing project logic from nutrition_pipeline.py.

    Keep your Parts 1-14 logic in that module.
    Do not put Part 15's input() section inside it.
    """
    try:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))

        module = importlib.import_module(
            "nutrition_pipeline"
        )

       

        return module, None

    except Exception as exc:
        return None, exc


pipeline, pipeline_error = load_pipeline()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_probability(value):
    """
    Converts probability to percentage.
    Accepts either:
        0.85 -> 85%
        85   -> 85%
    """
    value = safe_float(value)

    if 0 <= value <= 1:
        return value * 100

    return value


def normalize_status(status):
    return str(status).strip().title()


def dataframe_from_any(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if value is None:
        return pd.DataFrame()

    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def get_column(df, possible_names):
    """
    Returns the first existing column from possible_names.
    """
    if not isinstance(df, pd.DataFrame):
        return None

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:
        key = str(name).strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def format_nutrient_priority_dataframe(priority_df):
    if not isinstance(priority_df, pd.DataFrame):
        return pd.DataFrame()

    df = priority_df.copy()

    return df


def extract_priority_names(priority_result):
    """
    Handles the structured result produced by the existing
    Nutrient Analysis Agent.
    """
    if not isinstance(priority_result, dict):
        return []

    priority_df = priority_result.get(
        "priority_dataframe"
    )

    if isinstance(priority_df, pd.DataFrame):
        name_col = get_column(
            priority_df,
            [
                "nutrient_name",
                "nutrient",
                "name"
            ]
        )

        if name_col:
            return [
                str(x)
                for x in priority_df[name_col].tolist()
            ]

    values = priority_result.get(
        "priority_nutrients",
        []
    )

    if isinstance(values, list):
        output = []

        for item in values:
            if isinstance(item, dict):
                name = (
                    item.get("nutrient_name")
                    or item.get("nutrient")
                    or item.get("name")
                )

                if name:
                    output.append(
                        str(name)
                    )

            elif isinstance(item, str):
                output.append(item)

        return output

    return []


def find_pdf_file():
    """
    Finds the most recently modified PDF generated by the
    existing project PDF generator.
    """
    candidates = []

    if REPORT_DIR.exists():
        candidates.extend(
            REPORT_DIR.glob("*.pdf")
        )

    if OUTPUT_DIR.exists():
        candidates.extend(
            OUTPUT_DIR.glob("*.pdf")
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return candidates[0]


def save_json_safe(data, path):
    def convert(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(
                orient="records"
            )

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(k): convert(v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [
                convert(v)
                for v in value
            ]

        return value

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            convert(data),
            file,
            indent=4
        )


def call_pipeline_function(name, *args, **kwargs):
    """
    Calls an existing function from nutrition_pipeline.py.
    """
    if pipeline is None:
        raise RuntimeError(
            "Existing project pipeline could not be loaded."
        )

    function = getattr(
        pipeline,
        name,
        None
    )

    if function is None:
        raise AttributeError(
            f"Function '{name}' was not found in "
            "nutrition_pipeline.py."
        )

    return function(
        *args,
        **kwargs
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 110px;
    }

    .result-label {
        font-size: 0.82rem;
        opacity: 0.7;
    }

    .result-value {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .food-card {
        padding: 0.75rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.75rem;
    }

    .small-text {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    .status-normal {
        font-size: 1.5rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🥗 Child Nutrition AI"
    )

    st.caption(
        "Agentic AI Framework for Explainable Child "
        "Nutrition Recommendation and Malnutrition Prevention"
    )

    st.divider()

    st.markdown(
        "### System Modules"
    )

    st.write("✓ ML Risk Prediction")
    st.write("✓ Nutrient Priority Analysis")
    st.write("✓ Indian Food Recommendation")
    st.write("✓ Food Image Matching")
    st.write("✓ MobileNetV2 Validation")
    st.write("✓ SHAP Explainability")
    st.write("✓ Personalized Meal Plan")
    st.write("✓ Nutrition Advice")
    st.write("✓ PDF Report Generation")

    st.divider()

    st.caption(
        "Input range: 0–60 months"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Child Nutrition AI Assessment System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Explainable nutritional-risk prediction, Indian food "
    "recommendation, meal planning and prevention guidance"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# PIPELINE ERROR
# ============================================================

if pipeline_error is not None:

    st.error(
        "The existing project pipeline could not be loaded."
    )

    st.markdown(
        """
        **Required file**

        Place your existing project logic in:

        `nutrition_pipeline.py`

        This module should contain Parts 1–14 logic and expose
        the existing agent functions. Do not put Part 15's
        `input()` section in this module.
        """
    )

    with st.expander(
        "Show technical error"
    ):
        st.code(
            traceback.format_exc()
        )

    st.stop()


# ============================================================
# CHILD INPUT
# ============================================================

st.markdown(
    '<div class="section-title">1. Child Information</div>',
    unsafe_allow_html=True
)

input_col1, input_col2 = st.columns(2)

with input_col1:

    age_months = st.number_input(
        "Age (months)",
        min_value=0.0,
        max_value=60.0,
        value=22.0,
        step=1.0,
        help="Child age must be between 0 and 60 months."
    )

    height_cm = st.number_input(
        "Height (cm)",
        min_value=0.1,
        value=82.0,
        step=0.1
    )


with input_col2:

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=0.1,
        value=10.0,
        step=0.1
    )

    muac_cm = st.number_input(
        "MUAC (cm)",
        min_value=0.1,
        value=13.5,
        step=0.1
    )


# ============================================================
# BMI PREVIEW
# ============================================================

bmi_preview = (
    weight_kg /
    ((height_cm / 100.0) ** 2)
)

st.info(
    f"Calculated BMI: **{bmi_preview:.2f}**"
)


# ============================================================
# RUN ASSESSMENT
# ============================================================

run_assessment = st.button(
    "🚀 Run Child Nutrition Assessment",
    type="primary",
    use_container_width=True
)


# ============================================================
# ASSESSMENT PIPELINE
# ============================================================

if run_assessment:

    if age_months < 0 or age_months > 60:
        st.error(
            "Age must be between 0 and 60 months."
        )
        st.stop()

    if weight_kg <= 0:
        st.error(
            "Weight must be greater than zero."
        )
        st.stop()

    if height_cm <= 0:
        st.error(
            "Height must be greater than zero."
        )
        st.stop()

    if muac_cm <= 0:
        st.error(
            "MUAC must be greater than zero."
        )
        st.stop()


    with st.spinner(
        "Running the Agentic AI nutrition pipeline..."
    ):

        try:

            # ==================================================
            # 1. PREDICTION
            # ==================================================

            prediction = call_pipeline_function(
                "predict_child",
                age_months,
                weight_kg,
                height_cm,
                muac_cm
            )

            predicted_status = str(
                prediction[
                    "predicted_status"
                ]
            )

            best_model = str(
                prediction[
                    "best_model"
                ]
            )

            probabilities = prediction.get(
                "probabilities",
                {}
            )


            # ==================================================
            # 2. NUTRIENT ANALYSIS
            # ==================================================

            nutrient_result = call_pipeline_function(
                "analyze_nutrient_priorities",
                predicted_status
            )

            priority_nutrients = nutrient_result.get(
                "priority_nutrients",
                []
            )

            priority_df = nutrient_result.get(
                "priority_dataframe",
                pd.DataFrame()
            )


            # ==================================================
            # 3. FOOD RECOMMENDATION
            # ==================================================

            recommendation_result = (
                pipeline.nutrition_recommendation_agent.run(
                    predicted_status=predicted_status,
                    age_months=age_months,
                    top_n=10
                )
            )

            recommendations_df = (
                recommendation_result[
                    "recommendations"
                ].copy()
            )


            # ==================================================
            # 4. IMAGE MATCHING
            # ==================================================

            try:

                recommendations_df = (
                    call_pipeline_function(
                        "run_image_matching_agent",
                        recommendations_df=
                            recommendations_df,
                        nutrient_priority_result=
                            nutrient_result
                    )
                )

            except Exception as image_error:

                st.warning(
                    "Food image matching could not be completed: "
                    + str(image_error)
                )


            # ==================================================
            # 5. DEEP LEARNING IMAGE VALIDATION
            # ==================================================

            try:

                validated = (
                    call_pipeline_function(
                        "run_deep_learning_image_agent",
                        recommendations_df,
                        confidence_threshold=0.50
                    )
                )

                if isinstance(
                    validated,
                    pd.DataFrame
                ):
                    recommendations_df = validated

            except Exception as dl_error:

                st.warning(
                    "Deep-learning image validation could not "
                    "be completed: "
                    + str(dl_error)
                )


            # ==================================================
            # 6. MEAL PLAN
            # ==================================================

            meal_plan_df = call_pipeline_function(
                "generate_meal_plan",
                predicted_status=predicted_status,
                age_months=age_months,
                recommendations_df=recommendations_df
            )


            # ==================================================
            # 7. SHAP
            # ==================================================

            try:

                shap_result = call_pipeline_function(
                    "explain_child_prediction",
                    age_months=age_months,
                    weight_kg=weight_kg,
                    height_cm=height_cm,
                    muac_cm=muac_cm
                )

            except Exception as shap_error:

                shap_result = {}

                st.warning(
                    "SHAP explanation could not be generated: "
                    + str(shap_error)
                )


            # ==================================================
            # 8. SHAP PLOT
            # ==================================================

            shap_plot_path = (
                REPORT_DIR /
                "final_child_shap_waterfall.png"
            )

            try:

                call_pipeline_function(
                    "create_shap_waterfall_plot",
                    shap_result,
                    output_path=shap_plot_path
                )

            except Exception:
                shap_plot_path = None


            # ==================================================
            # 9. NUTRITION ADVICE
            # ==================================================

            try:

                nutrition_advice = call_pipeline_function(
                    "generate_nutrition_advice",
                    predicted_status=predicted_status,
                    age_months=age_months,
                    priority_nutrients=priority_nutrients,
                    recommendations_df=recommendations_df,
                    meal_plan_df=meal_plan_df
                )

            except Exception as advice_error:

                nutrition_advice = {
                    "predicted_status":
                        predicted_status,
                    "priority_nutrients":
                        priority_nutrients,
                    "all_advice":
                        []
                }

                st.warning(
                    "Nutrition advice could not be generated: "
                    + str(advice_error)
                )


            # ==================================================
            # STORE SESSION RESULTS
            # ==================================================

            result_bundle = {

                "age_months":
                    float(age_months),

                "weight_kg":
                    float(weight_kg),

                "height_cm":
                    float(height_cm),

                "muac_cm":
                    float(muac_cm),

                "bmi":
                    float(bmi_preview),

                "prediction":
                    prediction,

                "predicted_status":
                    predicted_status,

                "best_model":
                    best_model,

                "probabilities":
                    probabilities,

                "nutrient_result":
                    nutrient_result,

                "priority_nutrients":
                    priority_nutrients,

                "priority_df":
                    priority_df,

                "recommendations_df":
                    recommendations_df,

                "meal_plan_df":
                    meal_plan_df,

                "shap_result":
                    shap_result,

                "shap_plot_path":
                    shap_plot_path,

                "nutrition_advice":
                    nutrition_advice
            }

            st.session_state[
                "child_nutrition_result"
            ] = result_bundle


            # ==================================================
            # SAVE UI RESULT JSON
            # ==================================================

            json_path = (
                OUTPUT_DIR /
                "ui_child_nutrition_result.json"
            )

            save_json_safe(
                result_bundle,
                json_path
            )

            st.session_state[
                "ui_json_path"
            ] = json_path


            st.success(
                "✓ Child nutrition assessment completed successfully."
            )


        except Exception as exc:

            st.error(
                "The assessment pipeline encountered an error."
            )

            with st.expander(
                "Show technical error"
            ):
                st.code(
                    traceback.format_exc()
                )

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if (
    "child_nutrition_result"
    not in st.session_state
):
    st.markdown(
        """
        ### Ready for assessment

        Enter the child's **age, weight, height and MUAC**, then
        select **Run Child Nutrition Assessment**.
        """
    )

    st.stop()


result = st.session_state[
    "child_nutrition_result"
]


predicted_status = result[
    "predicted_status"
]

best_model = result[
    "best_model"
]

probabilities = result[
    "probabilities"
]

priority_df = result[
    "priority_df"
]

recommendations_df = result[
    "recommendations_df"
]

meal_plan_df = result[
    "meal_plan_df"
]

shap_result = result[
    "shap_result"
]

nutrition_advice = result[
    "nutrition_advice"
]

shap_plot_path = result[
    "shap_plot_path"
]


# ============================================================
# PREDICTION RESULT
# ============================================================

st.markdown(
    '<div class="section-title">2. Nutrition Risk Prediction</div>',
    unsafe_allow_html=True
)

result_col1, result_col2, result_col3, result_col4 = (
    st.columns(4)
)

with result_col1:

    st.metric(
        "BMI",
        f"{result['bmi']:.2f}"
    )

with result_col2:

    st.metric(
        "Predicted Status",
        predicted_status
    )

with result_col3:

    st.metric(
        "Best ML Model",
        best_model
    )

with result_col4:

    st.metric(
        "Age",
        f"{result['age_months']:.0f} months"
    )


# ============================================================
# PROBABILITIES
# ============================================================

st.markdown(
    '<div class="section-title">3. Prediction Probabilities</div>',
    unsafe_allow_html=True
)

if isinstance(
    probabilities,
    dict
) and probabilities:

    probability_df = pd.DataFrame(
        {
            "Nutrition Status":
                list(probabilities.keys()),

            "Probability (%)":
                [
                    safe_probability(v)
                    for v in probabilities.values()
                ]
        }
    )

    st.dataframe(
        probability_df.style.format(
            {
                "Probability (%)":
                    "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    chart_df = (
        probability_df
        .set_index(
            "Nutrition Status"
        )
    )

    st.bar_chart(
        chart_df[
            "Probability (%)"
        ]
    )


# ============================================================
# NUTRIENT PRIORITIES
# ============================================================

st.markdown(
    '<div class="section-title">4. Personalized Nutrient Priorities</div>',
    unsafe_allow_html=True
)

if isinstance(
    priority_df,
    pd.DataFrame
) and not priority_df.empty:

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )

else:

    priority_names = extract_priority_names(
        result["nutrient_result"]
    )

    if priority_names:

        st.write(
            " • ".join(
                priority_names
            )
        )

    else:

        st.info(
            "No structured nutrient priority table was returned."
        )



# ============================================================
# FOOD RECOMMENDATIONS
# ============================================================

st.markdown(
    '<div class="section-title">5. Recommended Indian Foods</div>',
    unsafe_allow_html=True
)

if (
    isinstance(recommendations_df, pd.DataFrame)
    and not recommendations_df.empty
):

    food_col = get_column(
        recommendations_df,
        ["recommended_food", "food"]
    )

    if food_col:
        st.dataframe(
            recommendations_df[[food_col]].rename(
                columns={food_col: "recommended_food"}
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "The recommended food column was not found."
        )

else:
    st.info("No food recommendations are available.")

# ============================================================
# FOOD IMAGES — FIND AND DISPLAY DATASET IMAGES
# ============================================================

st.markdown(
    '<div class="section-title">6. Indian Food Images</div>',
    unsafe_allow_html=True
)

# Normalize food/folder names so spaces, hyphens and underscores
# do not prevent a match.
def _normalize_food_folder(value):
    import re
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# Keys are normalized recommendation names; values are possible
# dataset folder names. Add an alias here if your dataset uses a
# different folder name for a recommended food.
FOOD_FOLDER_ALIASES = {
    "lentils_cooked": ["lentil", "lentils", "dal", "daal"],
    "lentils": ["lentil", "lentils", "dal", "daal"],
    "dal": ["lentil", "dal", "daal"],
    "daal": ["lentil", "dal", "daal"],
    "mustard_spinach": ["mustard_greens", "mustard_spinach", "sarson_ka_saag"],
    "mustard_spinach_cooked": ["mustard_greens", "mustard_spinach", "sarson_ka_saag"],
    "mustard_greens_cooked": ["mustard_greens", "mustard_spinach", "sarson_ka_saag"],
    "okra_cooked": ["okra", "bhindi_masala", "bhindi"],
    "chickpeas_cooked": ["chickpeas", "chana_masala", "chana"],
    "rice_bran": ["rice_bran"],
    "chaptti_roti_indian_bread": ["chapati", "chaptti", "roti", "chaptti_roti_indian_bread"],
    "chapati": ["chapati", "chaptti", "roti"],
}

_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_recommended_food_image(food_name, existing_path=None):
    """Return a valid image path for a recommended food, or None."""
    # First use the image-matching agent's path when it is valid.
    if existing_path is not None and not pd.isna(existing_path):
        path_text = str(existing_path).strip()
        if path_text and path_text.lower() not in {"nan", "none", "null"}:
            candidate = Path(path_text)
            if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS:
                return candidate

    if not IMAGE_ROOT.is_dir():
        return None

    normalized = _normalize_food_folder(food_name)
    folder_names = [normalized]
    folder_names.extend(FOOD_FOLDER_ALIASES.get(normalized, []))

    # Commonly the recommendation includes words like "cooked" or
    # "indian bread", which are not part of the dataset folder name.
    for suffix in ("_cooked", "_indian_bread", "_cooked_indian_bread"):
        if normalized.endswith(suffix):
            folder_names.append(normalized[:-len(suffix)])

    # Try exact folder names first. Case-insensitive comparison also
    # works when folder capitalization differs.
    available_folders = {
        _normalize_food_folder(folder.name): folder
        for folder in IMAGE_ROOT.iterdir()
        if folder.is_dir()
    }

    checked = set()
    for name in folder_names:
        key = _normalize_food_folder(name)
        if not key or key in checked:
            continue
        checked.add(key)
        folder = available_folders.get(key)
        if folder is None:
            continue

        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
        )
        if images:
            return images[0]

    return None


if not IMAGE_ROOT.is_dir():
    st.error(f"Food image dataset folder was not found: {IMAGE_ROOT}")
elif isinstance(recommendations_df, pd.DataFrame) and not recommendations_df.empty:
    image_cards = []
    missing_foods = []

    for _, row in recommendations_df.iterrows():
        food_col = get_column(
            recommendations_df,
            ["recommended_food", "food", "food_name", "Food", "name"]
        )
        food_name = str(row[food_col]).strip() if food_col else "Recommended food"
        existing_path = row.get("image_path", None)
        image_path = _find_recommended_food_image(food_name, existing_path)

        if image_path is not None:
            image_food_col = get_column(
                recommendations_df,
                ["image_food", "image_food_class", "dl_predicted_class"]
            )
            relation_col = get_column(
                recommendations_df,
                ["image_relation", "image_matching_status"]
            )
            image_label = str(row[image_food_col]) if image_food_col else image_path.parent.name
            relation = str(row[relation_col]) if relation_col else "Dataset image"
            image_cards.append((food_name, image_path, image_label, relation))
        else:
            missing_foods.append(food_name)

    if image_cards:
        columns = st.columns(min(3, len(image_cards)))
        for index, (food_name, image_path, image_label, relation) in enumerate(image_cards):
            with columns[index % len(columns)]:
                st.image(
                    str(image_path),
                    caption=f"Recommended: {food_name}",
                    use_container_width=True
                )
                st.caption(f"Dataset folder: {image_label}")
                st.caption(f"Image source: {relation}")

    st.caption(f"Food images displayed: {len(image_cards)} / {len(recommendations_df)}")

    if missing_foods:
        st.warning(
            "No matching dataset image was found for: "
            + ", ".join(missing_foods)
            + ". Check the folder names under the configured image dataset path."
        )
else:
    st.info("No food recommendations are available to display.")


# # ============================================================
# # DEEP LEARNING VALIDATION
# # ============================================================

# st.markdown(
#     '<div class="section-title">7. Deep Learning Image Validation</div>',
#     unsafe_allow_html=True
# )

# validation_col = get_column(
#     recommendations_df,
#     [
#         "dl_validation_status",
#         "image_validation_status",
#         "validation_status"
#     ]
# )

# confidence_col = get_column(
#     recommendations_df,
#     [
#         "dl_confidence",
#         "validation_confidence",
#         "image_prediction_confidence"
#     ]
# )

# predicted_image_col = get_column(
#     recommendations_df,
#     [
#         "dl_predicted_class",
#         "predicted_image_class",
#         "image_predicted_class"
#     ]
# )

# if validation_col:

#     validation_display = recommendations_df.copy()

#     selected_columns = [
#         c for c in [
#             get_column(
#                 recommendations_df,
#                 [
#                     "recommended_food",
#                     "food"
#                 ]
#             ),
#             get_column(
#                 recommendations_df,
#                 [
#                     "image_food",
#                     "image_food_class"
#                 ]
#             ),
#             validation_col,
#             confidence_col,
#             predicted_image_col
#         ]
#         if c
#     ]

#     if selected_columns:

#         st.dataframe(
#             validation_display[
#                 selected_columns
#             ],
#             use_container_width=True,
#             hide_index=True
#         )

# else:

#     st.info(
#         "Deep-learning validation results are not available "
#         "in the returned recommendation table."
#     )


# ============================================================
# MEAL PLAN
# ============================================================

st.markdown(
    '<div class="section-title">8. Personalized Meal Plan</div>',
    unsafe_allow_html=True
)

if (
    isinstance(
        meal_plan_df,
        pd.DataFrame
    )
    and not meal_plan_df.empty
):

    st.dataframe(
        meal_plan_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No meal plan was returned."
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">9. Explainable AI — SHAP</div>',
    unsafe_allow_html=True
)

if shap_result:

    shap_info = {}

    if isinstance(
        shap_result,
        dict
    ):

        for key in [
            "predicted_status",
            "predicted_class",
            "base_value",
            "feature_values",
            "feature_names"
        ]:

            if key in shap_result:

                value = shap_result[key]

                if isinstance(
                    value,
                    (dict, list, str, int, float)
                ):
                    shap_info[key] = value


    if shap_info:

        st.json(
            shap_info
        )

else:

    st.info(
        "SHAP explanation data was not returned."
    )


if (
    shap_plot_path is not None
    and Path(
        shap_plot_path
    ).exists()
):

    st.image(
        str(shap_plot_path),
        caption="SHAP explanation for the current child",
        use_container_width=True
    )


# ============================================================
# NUTRITION ADVICE
# ============================================================

st.markdown(
    '<div class="section-title">10. Nutrition Advice & Prevention</div>',
    unsafe_allow_html=True
)

if isinstance(
    nutrition_advice,
    dict
):

    advice_items = []

    if "all_advice" in nutrition_advice:

        all_advice = nutrition_advice[
            "all_advice"
        ]

        if isinstance(
            all_advice,
            list
        ):

            advice_items.extend(
                all_advice
            )

    for key in [
        "age_based_advice",
        "nutrient_advice",
        "meal_plan_advice",
        "prevention_advice",
        "food_based_advice"
    ]:

        value = nutrition_advice.get(
            key
        )

        if isinstance(
            value,
            list
        ):

            advice_items.extend(
                value
            )

        elif isinstance(
            value,
            str
        ):

            advice_items.append(
                value
            )

    if advice_items:

        seen = set()

        for advice in advice_items:

            text = str(
                advice
            ).strip()

            if not text:
                continue

            if text in seen:
                continue

            seen.add(
                text
            )

            st.markdown(
                f"- {text}"
            )

    else:

        for key, value in nutrition_advice.items():

            if key == "priority_nutrients":
                continue

            if isinstance(
                value,
                (str, int, float)
            ):

                st.markdown(
                    f"**{key.replace('_', ' ').title()}:** "
                    f"{value}"
                )

else:

    st.write(
        nutrition_advice
    )


# ============================================================
# DOWNLOAD JSON
# ============================================================

st.markdown(
    '<div class="section-title">11. Results</div>',
    unsafe_allow_html=True
)

json_path = st.session_state.get(
    "ui_json_path"
)

if (
    json_path is not None
    and Path(
        json_path
    ).exists()
):

    with open(
        json_path,
        "rb"
    ) as file:

        st.download_button(
            label="⬇ Download Assessment JSON",
            data=file.read(),
            file_name="child_nutrition_assessment.json",
            mime="application/json",
            use_container_width=True
        )


# ============================================================
# GENERATE / DOWNLOAD PDF
# ============================================================

st.markdown(
    '<div class="section-title">12. PDF Report</div>',
    unsafe_allow_html=True
)

st.write(
    "The PDF report uses the existing project PDF generator "
    "and the current child's assessment results."
)

generate_pdf = st.button(
    "📄 Generate Final PDF Report",
    use_container_width=True
)

if generate_pdf:

    # --------------------------------------------------------
    # The existing PDF generator can be kept as a separate
    # function in nutrition_pipeline.py:
    #
    #     generate_final_pdf_report(...)
    #
    # If that function is available, call it.
    # Otherwise the UI checks for a PDF already generated by
    # the notebook's PDF generator.
    # --------------------------------------------------------

    pdf_function = getattr(
        pipeline,
        "generate_final_pdf_report",
        None
    )

    if pdf_function is not None:

        try:

            pdf_path = (
                REPORT_DIR /
                "Child_Nutrition_Final_Report.pdf"
            )

            generated = pdf_function(
                age_months=result["age_months"],
                weight_kg=result["weight_kg"],
                height_cm=result["height_cm"],
                muac_cm=result["muac_cm"],
                bmi=result["bmi"],
                predicted_status=result["predicted_status"],
                best_model=result["best_model"],
                probabilities=result["probabilities"],
                priority_df=result["priority_df"],
                recommendations_df=result["recommendations_df"],
                meal_plan_df=result["meal_plan_df"],
                shap_result=result["shap_result"],
                nutrition_advice=result["nutrition_advice"],
                output_file=pdf_path
            )

            if generated is not None:
                pdf_path = Path(
                    generated
                )

            if pdf_path.exists():

                st.session_state[
                    "pdf_path"
                ] = pdf_path

                st.success(
                    "✓ PDF report generated successfully."
                )

            else:

                st.warning(
                    "PDF generator completed but the expected "
                    "PDF file was not found."
                )

        except TypeError:

            # Compatibility fallback for a PDF generator that
            # uses the current notebook globals rather than
            # explicit parameters.

            try:

                generated = pdf_function()

                if generated is not None:
                    pdf_path = Path(
                        generated
                    )
                else:
                    pdf_path = find_pdf_file()

                if (
                    pdf_path is not None
                    and pdf_path.exists()
                ):

                    st.session_state[
                        "pdf_path"
                    ] = pdf_path

                    st.success(
                        "✓ PDF report generated successfully."
                    )

                else:

                    st.warning(
                        "PDF generator ran, but no PDF file "
                        "could be located."
                    )

            except Exception:

                st.error(
                    "The existing PDF generator could not "
                    "be executed."
                )

                with st.expander(
                    "Show PDF error"
                ):

                    st.code(
                        traceback.format_exc()
                    )

        except Exception:

            st.error(
                "The PDF report could not be generated."
            )

            with st.expander(
                "Show PDF error"
            ):

                st.code(
                    traceback.format_exc()
                )

    else:

        existing_pdf = find_pdf_file()

        if existing_pdf is not None:

            st.session_state[
                "pdf_path"
            ] = existing_pdf

            st.info(
                "Using the latest PDF generated by the "
                "existing project PDF generator."
            )

        else:

            st.warning(
                "No PDF generator function was found in "
                "nutrition_pipeline.py and no existing PDF "
                "was found in the reports/ or outputs/ folder."
            )


# ============================================================
# PDF DOWNLOAD
# ============================================================

pdf_path = st.session_state.get(
    "pdf_path"
)

if (
    pdf_path is not None
    and Path(
        pdf_path
    ).exists()
):

    with open(
        pdf_path,
        "rb"
    ) as file:

        st.download_button(
            label="⬇ Download Final Child Nutrition PDF",
            data=file.read(),
            file_name=Path(
                pdf_path
            ).name,
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agentic AI Framework for Explainable Child Nutrition "
    "Recommendation and Malnutrition Prevention"
)

st.caption(
    "ML Prediction • Nutrient Analysis • Food Recommendation • "
    "Image Matching • MobileNetV2 • SHAP • Meal Planning • Reporting"
)
