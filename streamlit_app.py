
import os
import zipfile
from functools import lru_cache

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

@st.cache_resource
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

    if not os.path.exists("models"):
        with zipfile.ZipFile("models.zip") as archive:
            archive.extractall(".")

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("MarineHabitabilityStreamlit")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    preprocessing = PipelineModel.load("models/preprocessing")

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

    models = {
        name: cls.load("models/" + model_paths[name])
        for name, cls in model_classes.items()
    }

    calibrator = joblib.load(
        "models/svm_probability_calibrator.joblib"
    )

    return spark, preprocessing, models, calibrator


def predict(values):
    from pyspark.sql.types import (
        StructType, StructField, DoubleType
    )
    from pyspark.ml.functions import vector_to_array

    spark, preprocessing, models, calibrator = load_models()

    schema = StructType([
        StructField(name, DoubleType(), False)
        for name in FEATURE_COLUMNS
    ])

    frame = spark.createDataFrame(
        [tuple(float(v) for v in values)],
        schema=schema,
    )

    transformed = preprocessing.transform(frame)

    probabilities = {}

    for name, model in models.items():
        result = model.transform(transformed)

        if name == "SVM":
            margin = float(
                result.select(
                    vector_to_array("rawPrediction")[1]
                    .alias("margin")
                ).first()["margin"]
            )
            probabilities[name] = float(
                calibrator.predict_proba(
                    np.array([[margin]])
                )[0, 1]
            )
        else:
            probabilities[name] = float(
                result.select(
                    vector_to_array("probability")[1]
                    .alias("prob")
                ).first()["prob"]
            )

    rows = []

    for hybrid_name, members in HYBRIDS.items():
        score = float(np.mean([
            probabilities[name] for name in members
        ]))

        rows.append({
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

    return pd.DataFrame(rows)


st.title("🌊 Marine Ecosystem Habitability Prediction")
st.write(
    "Apache Spark ML | Six Hybrid Ensemble Models"
)
st.info(
    "Predictions estimate severe oxygen depletion risk "
    "from ocean profile measurements. They do not directly "
    "measure species-specific habitability."
)

with st.form("marine_form"):
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            value=8.8342, format="%.4f"
        )
        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=134.0848, format="%.4f"
        )
        pressure = st.number_input(
            "Maximum Pressure (dbar)",
            min_value=0.0, value=4339.0
        )

    with col2:
        temperature = st.number_input(
            "Mean Temperature (°C)",
            value=4.0279, format="%.4f"
        )
        salinity = st.number_input(
            "Mean Salinity",
            min_value=0.0, value=34.6102,
            format="%.4f"
        )

    submitted = st.form_submit_button(
        "Predict with Apache Spark",
        use_container_width=True,
        type="primary",
    )

if submitted:
    with st.spinner(
        "Running six hybrid models using Apache Spark ML..."
    ):
        try:
            results = predict([
                latitude,
                longitude,
                pressure,
                temperature,
                salinity,
            ])

            st.success("Spark ML predictions completed!")
            st.subheader("Six Hybrid Model Predictions")
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

            fig = px.bar(
                results,
                x="Hybrid Algorithm",
                y="Oxygen Stress Probability (%)",
                color="Prediction",
                title="Severe Oxygen Stress Probability",
                range_y=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as error:
            st.error(
                "Spark prediction failed: " + str(error)
            )
            st.exception(error)

