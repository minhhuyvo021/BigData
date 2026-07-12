from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    ClusteringEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType


TARGET = "DEATH_EVENT"
DEFAULT_INPUT_PATH = (
    "hdfs://localhost:9000/user/tony/input/dataset/"
    "heart_failure_clinical_records_dataset.csv"
)

FEATURE_COLUMNS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phan tich phan 4 de bai Big Data bang PySpark."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Duong dan file CSV dau vao tren HDFS.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Thu muc ghi ket qua.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed chia train/test.")
    return parser.parse_args()


def create_spark() -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (
        SparkSession.builder.appName("HeartFailurePart4")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def log_step(message: str) -> None:
    print(f"[HeartFailurePart4] {message}", flush=True)


def load_data(spark: SparkSession, input_path: str) -> DataFrame:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )

    numeric_casts = {
        "age": "double",
        "anaemia": "integer",
        "creatinine_phosphokinase": "double",
        "diabetes": "integer",
        "ejection_fraction": "double",
        "high_blood_pressure": "integer",
        "platelets": "double",
        "serum_creatinine": "double",
        "serum_sodium": "double",
        "sex": "integer",
        "smoking": "integer",
        "time": "double",
        TARGET: "integer",
    }

    for column, dtype in numeric_casts.items():
        df = df.withColumn(column, F.col(column).cast(dtype))

    return df.dropna(subset=FEATURE_COLUMNS + [TARGET])


def write_csv(df: DataFrame, output_path: Path) -> None:
    for field in df.schema.fields:
        if isinstance(field.dataType, ArrayType):
            df = df.withColumn(field.name, F.concat_ws(", ", F.col(field.name)))

    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(output_path))
    )


def build_scaled_features(df: DataFrame, output_col: str = "scaled_features") -> DataFrame:
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol=output_col,
        withMean=True,
        withStd=True,
    )
    return scaler.fit(assembler.transform(df)).transform(assembler.transform(df))


def evaluate_logistic_regression(
    df: DataFrame, output_dir: Path, seed: int
) -> dict[str, float]:
    train_df, test_df = df.randomSplit([0.75, 0.25], seed=seed)

    pipeline = Pipeline(
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

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df).cache()

    binary_roc = BinaryClassificationEvaluator(
        labelCol=TARGET, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    binary_pr = BinaryClassificationEvaluator(
        labelCol=TARGET, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    multi = MulticlassClassificationEvaluator(
        labelCol=TARGET, predictionCol="prediction"
    )

    confusion = predictions.select(
        F.sum(((F.col(TARGET) == 1) & (F.col("prediction") == 1)).cast("int")).alias(
            "tp"
        ),
        F.sum(((F.col(TARGET) == 0) & (F.col("prediction") == 1)).cast("int")).alias(
            "fp"
        ),
        F.sum(((F.col(TARGET) == 1) & (F.col("prediction") == 0)).cast("int")).alias(
            "fn"
        ),
        F.sum(((F.col(TARGET) == 0) & (F.col("prediction") == 0)).cast("int")).alias(
            "tn"
        ),
    ).first()

    tp = float(confusion["tp"] or 0)
    fp = float(confusion["fp"] or 0)
    fn = float(confusion["fn"] or 0)
    tn = float(confusion["tn"] or 0)
    precision_death = tp / (tp + fp) if tp + fp else 0.0
    recall_death = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    metrics = {
        "algorithm": "LogisticRegression",
        "train_rows": float(train_df.count()),
        "test_rows": float(test_df.count()),
        "accuracy": multi.setMetricName("accuracy").evaluate(predictions),
        "weighted_precision": multi.setMetricName("weightedPrecision").evaluate(
            predictions
        ),
        "weighted_recall": multi.setMetricName("weightedRecall").evaluate(predictions),
        "f1": multi.setMetricName("f1").evaluate(predictions),
        "precision_death_class": precision_death,
        "recall_death_class": recall_death,
        "specificity_survival_class": specificity,
        "area_under_roc": binary_roc.evaluate(predictions),
        "area_under_pr": binary_pr.evaluate(predictions),
    }

    metrics_df = df.sparkSession.createDataFrame(
        [(key, float(value) if key != "algorithm" else value) for key, value in metrics.items()],
        ["metric", "value"],
    )
    write_csv(metrics_df, output_dir / "classification_metrics")

    prediction_sample = predictions.select(
        "age",
        "age_group",
        TARGET,
        "prediction",
        F.round(vector_to_array("probability")[1], 4).alias("death_probability"),
    ).orderBy(F.desc("death_probability"))
    write_csv(prediction_sample, output_dir / "classification_predictions")

    predictions.unpersist()
    return {k: v for k, v in metrics.items() if isinstance(v, float)}


def evaluate_kmeans(df: DataFrame, output_dir: Path, seed: int) -> dict[str, float]:
    scaled_df = build_scaled_features(df, output_col="features").cache()
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    k_rows = []
    best_model = None
    best_score = None
    best_k = None

    for k in range(2, 7):
        model = KMeans(
            featuresCol="features",
            predictionCol="prediction",
            k=k,
            seed=seed,
            maxIter=50,
        ).fit(scaled_df)
        transformed = model.transform(scaled_df)
        score = evaluator.evaluate(transformed)
        k_rows.append((k, float(score)))

        if best_score is None or score > best_score:
            best_model = model
            best_score = score
            best_k = k

    k_selection = df.sparkSession.createDataFrame(k_rows, ["k", "silhouette"])
    write_csv(k_selection.orderBy(F.desc("silhouette")), output_dir / "kmeans_k_selection")

    clustered = best_model.transform(scaled_df).cache()

    cluster_label_counts = clustered.groupBy("prediction", TARGET).count()
    majority_per_cluster = cluster_label_counts.groupBy("prediction").agg(
        F.max("count").alias("majority_count"),
        F.sum("count").alias("cluster_count"),
    )
    purity_row = majority_per_cluster.select(
        (F.sum("majority_count") / F.sum("cluster_count")).alias("purity")
    ).first()
    purity = float(purity_row["purity"])

    cluster_metrics = df.sparkSession.createDataFrame(
        [
            ("KMeans", "best_k", float(best_k)),
            ("KMeans", "silhouette", float(best_score)),
            ("KMeans", "purity_by_death_event", purity),
        ],
        ["algorithm", "metric", "value"],
    )
    write_csv(cluster_metrics, output_dir / "kmeans_cluster_metrics")

    profile_exprs = [
        F.count("*").alias("rows"),
        F.round(F.avg(TARGET), 4).alias("death_rate"),
    ]
    profile_exprs.extend(F.round(F.avg(c), 4).alias(f"avg_{c}") for c in FEATURE_COLUMNS)
    cluster_profile = clustered.groupBy("prediction").agg(*profile_exprs).orderBy(
        "prediction"
    )
    write_csv(cluster_profile, output_dir / "kmeans_cluster_profile")

    clustered.unpersist()
    scaled_df.unpersist()
    return {
        "best_k": float(best_k),
        "silhouette": float(best_score),
        "purity": purity,
    }


def quantile_bins(df: DataFrame, column: str) -> tuple[float, float]:
    q1, q2 = df.approxQuantile(column, [0.33, 0.66], 0.01)
    return float(q1), float(q2)


def binned_item(column: str, q1: float, q2: float) -> F.Column:
    return F.when(F.col(column) <= q1, F.lit(f"{column}=low")).when(
        F.col(column) <= q2, F.lit(f"{column}=medium")
    ).otherwise(F.lit(f"{column}=high"))


def build_transactions(df: DataFrame) -> DataFrame:
    quantile_columns = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
    ]
    quantiles = {column: quantile_bins(df, column) for column in quantile_columns}

    item_columns = [
        F.concat(F.lit("age_group="), F.col("age_group")),
        F.when(F.col("anaemia") == 1, F.lit("anaemia=yes")).otherwise(
            F.lit("anaemia=no")
        ),
        F.when(F.col("diabetes") == 1, F.lit("diabetes=yes")).otherwise(
            F.lit("diabetes=no")
        ),
        F.when(F.col("high_blood_pressure") == 1, F.lit("high_bp=yes")).otherwise(
            F.lit("high_bp=no")
        ),
        F.when(F.col("sex") == 1, F.lit("sex=male")).otherwise(F.lit("sex=female")),
        F.when(F.col("smoking") == 1, F.lit("smoking=yes")).otherwise(
            F.lit("smoking=no")
        ),
        F.when(F.col(TARGET) == 1, F.lit("death=yes")).otherwise(F.lit("death=no")),
    ]

    item_columns.extend(
        binned_item(column, *quantiles[column]) for column in quantile_columns
    )

    return df.select(F.array_distinct(F.array(*item_columns)).alias("items"))


def add_itemset_metrics(rules: DataFrame, itemsets: DataFrame, total_rows: int) -> DataFrame:
    itemset_support = itemsets.select(
        F.concat_ws("||", F.sort_array("items")).alias("itemset_key"),
        (F.col("freq") / F.lit(total_rows)).alias("itemset_support"),
    )

    rule_keys = rules.withColumn(
        "antecedent_key", F.concat_ws("||", F.sort_array("antecedent"))
    ).withColumn(
        "consequent_key", F.concat_ws("||", F.sort_array("consequent"))
    ).withColumn(
        "union_key",
        F.concat_ws("||", F.sort_array(F.array_union("antecedent", "consequent"))),
    )

    ant = itemset_support.select(
        F.col("itemset_key").alias("antecedent_key"),
        F.col("itemset_support").alias("antecedent_support"),
    )
    cons = itemset_support.select(
        F.col("itemset_key").alias("consequent_key"),
        F.col("itemset_support").alias("consequent_support"),
    )
    union = itemset_support.select(
        F.col("itemset_key").alias("union_key"),
        F.col("itemset_support").alias("rule_support"),
    )

    return (
        rule_keys.join(ant, "antecedent_key", "left")
        .join(cons, "consequent_key", "left")
        .join(union, "union_key", "left")
        .withColumn(
            "all_confidence",
            F.col("rule_support")
            / F.greatest(F.col("antecedent_support"), F.col("consequent_support")),
        )
        .withColumn(
            "kulczynski",
            F.lit(0.5)
            * (
                F.col("rule_support") / F.col("antecedent_support")
                + F.col("rule_support") / F.col("consequent_support")
            ),
        )
        .withColumn(
            "cosine",
            F.col("rule_support")
            / F.sqrt(F.col("antecedent_support") * F.col("consequent_support")),
        )
    )


def evaluate_fpgrowth(df: DataFrame, output_dir: Path) -> dict[str, float]:
    transactions = build_transactions(df).cache()
    total_rows = transactions.count()

    model = FPGrowth(
        itemsCol="items",
        minSupport=0.08,
        minConfidence=0.45,
    ).fit(transactions)

    itemsets = model.freqItemsets.withColumn(
        "support", F.round(F.col("freq") / F.lit(total_rows), 6)
    )
    top_itemsets = itemsets.orderBy(F.desc("freq"), F.desc(F.size("items"))).limit(50)
    write_csv(top_itemsets, output_dir / "fpgrowth_itemsets")

    rules = add_itemset_metrics(model.associationRules, model.freqItemsets, total_rows)
    columns = [
        "antecedent",
        "consequent",
        "confidence",
        "lift",
        "rule_support",
        "all_confidence",
        "kulczynski",
        "cosine",
    ]
    top_rules = (
        rules.select(*columns)
        .orderBy(F.desc("lift"), F.desc("confidence"))
        .limit(50)
    )
    write_csv(top_rules, output_dir / "fpgrowth_rules")

    death_rules = rules.filter(F.array_contains("consequent", "death=yes"))
    best_death_rule = death_rules.orderBy(F.desc("lift"), F.desc("confidence")).first()
    all_rules_count = rules.count()
    death_rules_count = death_rules.count()

    transactions.unpersist()
    return {
        "transactions": float(total_rows),
        "frequent_itemsets": float(model.freqItemsets.count()),
        "association_rules": float(all_rules_count),
        "death_rules": float(death_rules_count),
        "best_death_rule_lift": float(best_death_rule["lift"]) if best_death_rule else 0.0,
        "best_death_rule_confidence": float(best_death_rule["confidence"])
        if best_death_rule
        else 0.0,
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def write_summary(
    output_dir: Path,
    row_count: int,
    classification: dict[str, float],
    clustering: dict[str, float],
    association: dict[str, float],
) -> None:
    lines: Iterable[str] = [
        "# Tom tat danh gia phan 4",
        "",
        f"Dataset sau khi lam sach co {row_count} dong.",
        "",
        "## 1. Logistic Regression - phan loai",
        "",
        "- Bai toan: du doan `DEATH_EVENT` tu cac chi so lam sang.",
        f"- Accuracy: {format_metric(classification['accuracy'])}.",
        f"- F1-score: {format_metric(classification['f1'])}.",
        f"- ROC AUC: {format_metric(classification['area_under_roc'])}.",
        f"- Recall lop tu vong: {format_metric(classification['recall_death_class'])}.",
        "",
        "Nhan xet: day la mo hinh co giam sat, danh gia truc tiep bang tap test theo holdout.",
        "",
        "## 2. KMeans - phan cum",
        "",
        "- Bai toan: gom nhom benh nhan theo do tuong dong cua cac thuoc tinh lam sang.",
        f"- So cum tot nhat theo silhouette: k = {int(clustering['best_k'])}.",
        f"- Silhouette: {format_metric(clustering['silhouette'])}.",
        f"- Purity theo nhan `DEATH_EVENT`: {format_metric(clustering['purity'])}.",
        "",
        "Nhan xet: silhouette la danh gia khong giam sat, purity la danh gia co giam sat dua tren nhan that.",
        "",
        "## 3. FPGrowth - tap pho bien va luat ket hop",
        "",
        "- Bai toan: tim cac mau dieu kien thuong xuat hien cung nhau sau khi roi rac hoa du lieu.",
        f"- So tap pho bien: {int(association['frequent_itemsets'])}.",
        f"- So luat ket hop: {int(association['association_rules'])}.",
        f"- So luat co ket qua `death=yes`: {int(association['death_rules'])}.",
        f"- Lift tot nhat cua nhom luat `death=yes`: {format_metric(association['best_death_rule_lift'])}.",
        f"- Confidence tuong ung: {format_metric(association['best_death_rule_confidence'])}.",
        "",
        "Nhan xet: lift, all-confidence, Kulczynski va cosine giup danh gia do manh cua luat thay vi chi nhin support.",
        "",
        "## 4. So sanh 3 phuong phap",
        "",
        "- Logistic Regression phu hop nhat khi muc tieu la du doan nguy co tu vong vi co nhan va co metrics ro rang.",
        "- KMeans huu ich de phat hien nhom benh nhan tuong dong, nhung khong toi uu truc tiep cho viec du doan nhan.",
        "- FPGrowth de giai thich cac mau dieu kien ket hop, phu hop de dua ra nhan xet mo ta hon la du doan tung ca nhan.",
        "",
        "Ket luan: neu can mo hinh du doan, uu tien Logistic Regression; neu can phan nhom, dung KMeans; neu can tim mau ket hop de giai thich du lieu, dung FPGrowth.",
        "",
    ]
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"Doc du lieu tu: {args.input}")
    spark = create_spark()
    try:
        df = load_data(spark, args.input).cache()
        row_count = df.count()
        log_step(f"Da load {row_count} dong sau khi lam sach.")

        log_step("Dang chay Logistic Regression...")
        classification = evaluate_logistic_regression(df, output_dir, args.seed)

        log_step("Dang chay KMeans...")
        clustering = evaluate_kmeans(df, output_dir, args.seed)

        log_step("Dang chay FPGrowth...")
        association = evaluate_fpgrowth(df, output_dir)

        write_summary(output_dir, row_count, classification, clustering, association)
        log_step(f"Hoan tat. Xem tom tat tai: {output_dir / 'comparison_summary.md'}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
