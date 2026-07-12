from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler

from heart_failure_pyspark import (
    DEFAULT_INPUT_PATH,
    FEATURE_COLUMNS,
    TARGET,
    create_spark,
    load_data,
    log_step,
)


DEFAULT_MODEL_PATH = "models/logistic_regression_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train va luu model Logistic Regression cho app du doan."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Duong dan file CSV tren HDFS.",
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_PATH,
        help="Thu muc luu Spark ML PipelineModel.",
    )
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    return Pipeline(
        stages=[
            VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw"),
            StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withMean=True,
                withStd=True,
            ),
            LogisticRegression(
                featuresCol="features",
                labelCol=TARGET,
                maxIter=100,
                regParam=0.05,
                elasticNetParam=0.0,
            ),
        ]
    )


def write_metadata(model_path: Path, row_count: int) -> None:
    metadata = {
        "model_type": "PySpark PipelineModel",
        "algorithm": "LogisticRegression",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "row_count": row_count,
        "binary_fields": {
            "anaemia": {"0": "No", "1": "Yes"},
            "diabetes": {"0": "No", "1": "Yes"},
            "high_blood_pressure": {"0": "No", "1": "Yes"},
            "sex": {"0": "Female", "1": "Male"},
            "smoking": {"0": "No", "1": "Yes"},
        },
        "sample_input": {
            "age": 60,
            "anaemia": 0,
            "creatinine_phosphokinase": 250,
            "diabetes": 0,
            "ejection_fraction": 38,
            "high_blood_pressure": 0,
            "platelets": 265000,
            "serum_creatinine": 1.1,
            "serum_sodium": 136,
            "sex": 1,
            "smoking": 0,
            "time": 120,
        },
    }
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_output)

    log_step(f"Doc du lieu train tu: {args.input}")
    spark = create_spark()
    try:
        df = load_data(spark, args.input).cache()
        row_count = df.count()
        log_step(f"Train Logistic Regression voi {row_count} dong.")

        model = build_pipeline().fit(df)
        model.write().overwrite().save(str(model_path))
        write_metadata(model_path, row_count)

        log_step(f"Da luu model tai: {model_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
