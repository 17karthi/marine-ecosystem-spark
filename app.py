import os
from pathlib import Path
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    LogisticRegressionModel,
    DecisionTreeClassificationModel,
    RandomForestClassificationModel,
    GBTClassificationModel,
    LinearSVCModel
)

MODELS = Path(__file__).parent / "models"

HYBRIDS = {
    "Hybrid 1 - LR + RF": ["LR", "RF"],
    "Hybrid 2 - RF + GBT": ["RF", "GBT"],
    "Hybrid 3 - DT + RF": ["DT", "RF"],
    "Hybrid 4 - LR + GBT": ["LR", "GBT"],
    "Hybrid 5 - SVM + RF": ["SVM", "RF"],
    "Hybrid 6 - LR + RF + GBT": ["LR", "RF", "GBT"]
}

@lru_cache(maxsize=1)
def load_spark_models():
    spark = (
        SparkSession.builder
        .appName("Marine_Ecosystem_Spark_Prediction")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )

    preprocessing = PipelineModel.load(
        str(MODELS / "preprocessing")
    )

    models = {
        "LR": LogisticRegressionModel.load(
            str(MODELS / "base_lr")
        ),
        "DT": DecisionTreeClassificationModel.load(
            str(MODELS / "base_dt")
        ),
        "RF": RandomForestClassificationModel.load(
            str(MODELS / "base_rf")
        ),
        "GBT": GBTClassificationModel.load(
            str(MODELS / "base_gbt")
        ),
        "SVM": LinearSVCModel.load(
            str(MODELS / "base_svm")
        )
    }

    calibrator = joblib.load(
        MODELS / "svm_probability_calibrator.joblib"
    )

    return spark, preprocessing, models, calibrator


def predict(lat, lon, pressure, temperature, salinity):
    try:
        values = {
            "latitude": float(lat),
            "longitude": float(lon),
            "maximum_pressure": float(pressure),
            "mean_temperature": float(temperature),
            "mean_salinity": float(salinity)
        }
    except (ValueError, TypeError):
        raise gr.Error("Enter all five numeric measurements.")

    if not all(np.isfinite(v) for v in values.values()):
        raise gr.Error("Measurements must be finite.")

    if not -90 <= values["latitude"] <= 90:
        raise gr.Error("Latitude must be between -90 and 90.")

    if not -180 <= values["longitude"] <= 180:
        raise gr.Error("Longitude must be between -180 and 180.")

    if values["maximum_pressure"] < 0:
        raise gr.Error("Pressure must not be negative.")

    spark, preprocessing, models, calibrator = (
        load_spark_models()
    )

    input_df = spark.createDataFrame([values])
    features = preprocessing.transform(input_df)

    scores = {}

    for name, model in models.items():
        output = model.transform(features)

        if name == "SVM":
            margin = output.select(
                vector_to_array("rawPrediction")[1].alias("v")
            ).first()["v"]

            score = calibrator.predict_proba(
                np.array([[float(margin)]])
            )[0, 1]
        else:
            score = output.select(
                vector_to_array("probability")[1].alias("v")
            ).first()["v"]

        scores[name] = float(score)

    rows = []
    for hybrid, members in HYBRIDS.items():
        score = float(np.mean([scores[m] for m in members]))
        rows.append({
            "Hybrid Algorithm": hybrid,
            "Stress Score (%)": round(score * 100, 2),
            "Prediction": (
                "Severe oxygen-related stress"
                if score >= 0.5
                else "No severe oxygen-related stress predicted"
            )
        })

    table = pd.DataFrame(rows)

    reference_score = float(np.mean([
        scores["LR"], scores["RF"], scores["GBT"]
    ]))

    prediction = (
        "Severe oxygen-related stress predicted"
        if reference_score >= 0.5
        else "No severe oxygen-related stress predicted"
    )

    summary = (
        "### Reference Hybrid: LR + RF + GBT\n\n"
        f"**Stress score:** {reference_score:.2%}\n\n"
        f"**Prediction:** {prediction}\n\n"
        "*The model estimates severe oxygen-related stress, "
        "not species-specific habitability.*"
    )

    fig = go.Figure(go.Bar(
        x=table["Stress Score (%)"],
        y=table["Hybrid Algorithm"],
        orientation="h",
        text=[
            f"{v:.2f}%"
            for v in table["Stress Score (%)"]
        ],
        textposition="auto"
    ))

    fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="red"
    )

    fig.update_layout(
        title="Six Hybrid Predictions — Apache Spark ML",
        xaxis_title="Severe oxygen-stress score (%)",
        xaxis_range=[0, 100],
        template="plotly_white",
        height=480
    )

    return summary, table, fig


with gr.Blocks(
    title="Marine Ecosystem — Apache Spark ML"
) as demo:

    gr.Markdown(
        "# 🌊 Marine Ecosystem Prediction\n"
        "### Powered by Apache Spark ML\n"
        "Enter five ocean measurements to get "
        "six hybrid predictions."
    )

    with gr.Row():
        with gr.Column():
            lat = gr.Number(label="Latitude", value=8.8342)
            lon = gr.Number(label="Longitude", value=134.0848)
            pressure = gr.Number(
                label="Maximum Pressure (dbar)",
                value=4339
            )
            temperature = gr.Number(
                label="Mean Temperature (°C)",
                value=4.0279
            )
            salinity = gr.Number(
                label="Mean Salinity",
                value=34.6102
            )

            button = gr.Button(
                "Predict with Apache Spark",
                variant="primary"
            )

        with gr.Column():
            summary = gr.Markdown()
            table = gr.Dataframe(
                label="Six Hybrid Predictions",
                interactive=False
            )

    chart = gr.Plot(label="Hybrid Comparison")

    button.click(
        fn=predict,
        inputs=[
            lat, lon, pressure, temperature, salinity
        ],
        outputs=[summary, table, chart],
        concurrency_limit=1
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "10000")),
        share=False,
        show_error=True
    )
