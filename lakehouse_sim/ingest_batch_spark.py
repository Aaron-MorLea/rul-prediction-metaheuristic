"""
PySpark Batch Ingestion for RUL Data Pipeline

Simulates a Databricks-style lakehouse pipeline for processing
turbofan engine sensor data.

This script demonstrates:
- Reading raw sensor data
- ETL transformations
- Writing to Delta-style tables (Parquet)
- Ready for Databricks migration
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pathlib import Path
import os


def create_spark_session(app_name: str = "RUL-DataPipeline") -> SparkSession:
    """Create and configure Spark session."""
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def define_schema() -> StructType:
    """Define schema for C-MAPSS data."""
    
    return StructType([
        StructField("unit_number", IntegerType(), False),
        StructField("time_cycles", IntegerType(), False),
        StructField("op_setting_1", DoubleType(), True),
        StructField("op_setting_2", DoubleType(), True),
        StructField("op_setting_3", DoubleType(), True),
    ] + [
        StructField(f"sensor_{i}", DoubleType(), True)
        for i in range(1, 22)
    ])


def ingest_raw_data(spark: SparkSession, input_path: str, subset: str = "FD001"):
    """
    Ingest raw sensor data.
    
    In Databricks: Read from ADLS/S3
    """
    
    print(f"Ingesting data from {input_path}")
    
    schema = define_schema()
    
    train_path = f"{input_path}/train_{subset}.txt"
    test_path = f"{input_path}/test_{subset}.txt"
    
    train_df = spark.read.csv(
        train_path,
        schema=schema,
        sep=r"\s+",
        header=False
    )
    
    test_df = spark.read.csv(
        test_path,
        schema=schema,
        sep=r"\s+",
        header=False
    )
    
    print(f"Train records: {train_df.count()}")
    print(f"Test records: {test_df.count()}")
    
    return train_df, test_df


def transform_data(df):
    """
    Apply transformations to raw data.
    
    - Add ingestion timestamp
    - Add data quality flags
    - Normalize sensor values
    """
    
    df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    
    for col_name in [f"sensor_{i}" for i in range(1, 22)]:
        df = df.withColumn(
            f"{col_name}_is_null",
            F.col(col_name).isNull()
        )
    
    null_counts = df.select([
        F.sum(F.col(f"sensor_{i}_is_null").cast("int")).alias(f"sensor_{i}_nulls")
        for i in range(1, 22)
    ])
    
    print("Null value analysis:")
    null_counts.show()
    
    return df


def compute_rul(spark: SparkSession, df, max_rul: int = 125):
    """
    Compute Remaining Useful Life for each engine unit.
    
    This is equivalent to the feature_engineering.py logic
    but in PySpark for distributed processing.
    """
    
    from pyspark.window import Window
    
    window_spec = Window.partitionBy("unit_number").orderBy("time_cycles")
    
    max_cycles = df.groupBy("unit_number").agg(
        F.max("time_cycles").alias("max_cycle")
    )
    
    df = df.join(max_cycles, on="unit_number")
    
    df = df.withColumn(
        "RUL",
        F.col("max_cycle") - F.col("time_cycles")
    )
    
    df = df.withColumn(
        "RUL_capped",
        F.least(F.col("RUL"), F.lit(max_rul))
    )
    
    df = df.drop("max_cycle", "RUL")
    df = df.withColumnRenamed("RUL_capped", "RUL")
    
    return df


def add_rolling_features(spark: SparkSession, df, windows=[5, 10]):
    """
    Add rolling window features using PySpark.
    
    In production: Use Spark's window functions for efficiency
    """
    
    from pyspark.window import Window
    
    window_spec = Window.partitionBy("unit_number").orderBy("time_cycles")
    
    for window in windows:
        for sensor in [f"sensor_{i}" for i in range(1, 6)]:
            df = df.withColumn(
                f"{sensor}_roll_mean_{window}",
                F.avg(sensor).over(window_spec.rowsBetween(-window, 0))
            )
            df = df.withColumn(
                f"{sensor}_roll_std_{window}",
                F.stddev(sensor).over(window_spec.rowsBetween(-window, 0))
            )
    
    return df


def write_to_lakehouse(df, output_path: str, partition_col: str = None):
    """
    Write processed data to lakehouse format.
    
    In Databricks: Write to Delta tables
    Local simulation: Write to Parquet
    """
    
    print(f"Writing to {output_path}")
    
    df = df.withColumn("processed_date", F.current_date())
    
    if partition_col:
        df.write \
            .mode("overwrite") \
            .partitionBy(partition_col) \
            .parquet(output_path)
    else:
        df.write \
            .mode("overwrite") \
            .parquet(output_path)
    
    print(f"Successfully wrote {df.count()} records")


def run_batch_pipeline(
    input_path: str = "data/raw",
    output_path: str = "data/processed",
    subset: str = "FD001"
):
    """
    Run complete batch ingestion pipeline.
    """
    
    print("=" * 60)
    print("Starting Batch Ingestion Pipeline")
    print("=" * 60)
    
    spark = create_spark_session("RUL-BatchIngestion")
    
    train_df, test_df = ingest_raw_data(spark, input_path, subset)
    
    train_df = transform_data(train_df)
    train_df = compute_rul(spark, train_df)
    train_df = add_rolling_features(spark, train_df)
    
    test_df = transform_data(test_df)
    test_df = compute_rul(spark, test_df)
    test_df = add_rolling_features(spark, test_df)
    
    train_output = f"{output_path}/{subset}_train"
    test_output = f"{output_path}/{subset}_test"
    
    write_to_lakehouse(train_df, train_output, partition_col="unit_number")
    write_to_lakehouse(test_df, test_output, partition_col="unit_number")
    
    print("\nPipeline completed successfully!")
    
    spark.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="Input data path")
    parser.add_argument("--output", default="data/processed", help="Output data path")
    parser.add_argument("--subset", default="FD001", help="C-MAPSS subset")
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    run_batch_pipeline(args.input, args.output, args.subset)