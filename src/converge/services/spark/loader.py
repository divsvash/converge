"""
Dataset loading.

Kept separate from profiler.py so the profiler stays a pure function
(DataFrame in, typed objects out) that's trivial to unit test with
in-memory DataFrames, independent of file I/O.
"""
from pyspark.sql import DataFrame, SparkSession


def load_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Loads a CSV with header + schema inference. P0 uses schema
    inference rather than a fixed schema registry since datasets are
    user-uploaded and arbitrary; a hardened version could validate
    against an expected schema later if needed.
    """
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )
