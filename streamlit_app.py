
import os
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Marine Ecosystem Habitability Prediction",
    page_icon="🌊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "models.zip"

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "maximum_pressure",
    "mean_temperature",
    "mean_salinity",
]

HYBRIDS = {
    "Hybrid 1 - LR + RF": ["LR", "RF"],
    "Hybrid 2 - RF + GBT": ["RF", "GBT"],
    "Hybrid 3 - DT + RF": ["DT", "RF"],
    "Hybrid 4 - LR + GBT": ["LR", "GBT"],
    "Hybrid 5 - SVM + RF": ["SVM", "RF"],
    "Hybrid 6 - LR + RF + GBT": ["LR", "RF", "GBT"],
}


def find_model_directory():
    """
    Extract models.zip and locate the actual Spark model folder.
    Handles ZIPs with or without a top-level models/ directory.
    """
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(
            f"models.zip was not found at {ZIP_PATH}"
        )

    extract_dir = ROOT / "_extracted_models"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(extract_dir)

    candidates = [
        extract_dir / "models",
        extract_dir,
    ]

    # Also handle an extra parent directory inside the ZIP.
    candidates.extend(
        path.parent
        for path in extract_dir.rglob("preprocessing")
        if path.is_dir()
    )

    for candidate in candidates:
        metadata = candidate / "preprocessing" / "metadata"

        if metadata.is_dir():
            return candidate

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        entries = archive.namelist()[:25]

    raise FileNotFoundError(
        "Could not find preprocessing/metadata inside models.zip. "
        f"First ZIP entries: {entries}"
    )


@st.cache_resource(show_spinner="Starting Apache Spark ML...")
def load_models():
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import (
        LogisticRegressionModel,
        DecisionTreeClassificationModel,
        RandomForestClassificationModel,
        GBTClassificationModel,
        LinearSVCModel,
    )

    model_dir = find_model_directory()

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("MarineHabitabilityStreamlit")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    preprocessing = PipelineModel.load(
        str(model_dir / "preprocessing")
    )

    model_classes = {
        "LR": LogisticRegressionModel,
        "DT": DecisionTreeClassificationModel,
        "RF": RandomForestClassificationModel,
        "GBT": GBTClassificationModel,
        "SVM": LinearSVCModel,
    }

    model_paths = {
        "LR": "base_lr",
        "DT": "base_dt",
        "RF": "base_rf",
        "GBT": "base_gbt",
        "SVM": "base_svm",
    }

    models = {}

    for name, model_class in model_classes.items():
        path = model_dir / model_paths[name]
        models[name] = model_class.load(str(path))

    calibrator_path = (
        model_dir / "svm_probability_calibrator.joblib"
    )

    calibrator = joblib.load(calibrator_path)

    return spark, preprocessing, models, calibrator


def predict(values):
    from pyspark.sql.types import (
        StructType,
        StructField,
        DoubleType,
    )
    from pyspark.ml.functions import vector_to_array

    spark, preprocessing, models, calibrator = load_models()

    schema = StructType([
        StructField(column, DoubleType(), False)
        for column in FEATURE_COLUMNS
    ])

    input_data = spark.createDataFrame(
        [tuple(float(value) for value in values)],
        schema=schema,
    )

    transformed = preprocessing.transform(input_data)

    probabilities = {}

    for name, model in models.items():
        prediction = model.transform(transformed)

        if name == "SVM":
            margin = float(
                prediction.select(
                    vector_to_array("rawPrediction")[1].alias(
                        "margin"
                    )
                ).first()["margin"]
            )

            probability = float(
                calibrator.predict_proba(
                    np.array([[margin]])
                )[0, 1]
            )
        else:
            probability = float(
                prediction.select(
                    vector_to_array("probability")[1].alias(
                        "probability_value"
                    )
                ).first()["probability_value"]
            )

        probabilities[name] = probability

    results = []

    for hybrid_name, members in HYBRIDS.items():
        score = float(
            np.mean([
                probabilities[member]
                for member in members
            ])
        )

        results.append({
            "Hybrid Algorithm": hybrid_name,
            "Oxygen Stress Probability (%)": round(
                score * 100, 2
            ),
            "Prediction": (
                "Severe oxygen stress"
                if score >= 0.5
                else "No severe oxygen stress"
            ),
        })

    return pd.DataFrame(results)


# ------------------------------
# STREAMLIT USER INTERFACE
# ------------------------------

st.title("🌊 Marine Ecosystem Habitability Prediction")

st.write(
    "### Apache Spark ML | Six Hybrid Ensemble Models"
)

st.info(
    "This dashboard estimates severe oxygen depletion "
    "risk using ocean measurements. It does not directly "
    "predict species-specific marine habitability."
)

st.divider()

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=8.8342,
            format="%.4f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=134.0848,
            format="%.4f",
        )

        maximum_pressure = st.number_input(
            "Maximum Pressure (dbar)",
            min_value=0.0,
            value=4339.0,
        )

    with right:
        mean_temperature = st.number_input(
            "Mean Temperature (°C)",
            value=4.0279,
            format="%.4f",
        )

        mean_salinity = st.number_input(
            "Mean Salinity",
            min_value=0.0,
            value=34.6102,
            format="%.4f",
        )

    submitted = st.form_submit_button(
        "🔍 Predict with Apache Spark",
        type="primary",
        use_container_width=True,
    )


if submitted:
    values = [
        latitude,
        longitude,
        maximum_pressure,
        mean_temperature,
        mean_salinity,
    ]

    try:
        with st.spinner(
            "Running six hybrid models using Apache Spark ML..."
        ):
            results = predict(values)

        st.success("Apache Spark ML predictions completed!")

        st.subheader("Six Hybrid Model Predictions")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Prediction Comparison")

        figure = px.bar(
            results,
            x="Hybrid Algorithm",
            y="Oxygen Stress Probability (%)",
            color="Prediction",
            title="Severe Oxygen Stress Probability by Hybrid",
            range_y=[0, 100],
            text="Oxygen Stress Probability (%)",
        )

        figure.update_layout(
            xaxis_title="Hybrid Algorithm",
            yaxis_title="Oxygen Stress Probability (%)",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        st.caption(
            "All predictions are calculated using the "
            "saved Apache Spark ML classification models."
        )

    except Exception as error:
        st.error(
            f"Spark prediction failed: {error}"
        )
        st.exception(error)
