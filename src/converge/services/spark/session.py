"""
Spark session management for P0.

P0 runs a single local Spark session (local[*]) -- no cluster, no
resource manager config. This module exists as its own file (rather
than inlining SparkSession.builder calls everywhere) so that later
phases can swap in cluster configuration without touching profiler /
executor code.
"""
from pyspark.sql import SparkSession

from converge.core.logging import get_logger

logger = get_logger(__name__)

_spark_session: SparkSession | None = None


def get_spark_session(app_name: str = "genai-pipeline-p0") -> SparkSession:
    """
    Returns a process-wide local SparkSession, creating it on first call.
    Reused across calls to avoid the cost of repeated JVM startup.
    """
    global _spark_session
    if _spark_session is None:
        logger.info("Starting local Spark session: %s", app_name)
        _spark_session = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        _spark_session.sparkContext.setLogLevel("WARN")
    return _spark_session


def stop_spark_session() -> None:
    """Stops the shared session, if one exists. Mainly for test teardown."""
    global _spark_session
    if _spark_session is not None:
        logger.info("Stopping Spark session")
        _spark_session.stop()
        _spark_session = None
