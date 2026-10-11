import pytest

from converge.core.config import RAW_DATA_DIR
from converge.services.spark.loader import load_csv
from converge.services.spark.profiler import profile_dataset
from converge.services.spark.session import get_spark_session, stop_spark_session


@pytest.fixture(scope="session")
def spark():
    session = get_spark_session(app_name="genai-pipeline-p0-tests")
    yield session
    stop_spark_session()


@pytest.fixture(scope="session")
def sample_hr_df(spark):
    path = str(RAW_DATA_DIR / "sample_hr.csv")
    return load_csv(spark, path)


@pytest.fixture(scope="session")
def sample_profile_result(sample_hr_df):
    """The Step-1 profiler output for the sample dataset, reused across
    validator tests so every test validates against the exact same
    deterministic evidence."""
    return profile_dataset(sample_hr_df, dataset_id="sample_hr")

