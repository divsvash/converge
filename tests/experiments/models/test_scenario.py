import hashlib

import pytest
from pydantic import ValidationError

from experiments.models.scenario import Dataset, Scenario


@pytest.fixture()
def tmp_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    return path


# ---------------------------------------------------------------------------
# Dataset checksum behavior
# ---------------------------------------------------------------------------

def test_dataset_from_path_computes_correct_sha256(tmp_csv):
    ds = Dataset.from_path("ds1", str(tmp_csv))
    expected = hashlib.sha256(tmp_csv.read_bytes()).hexdigest()
    assert ds.checksum == expected


def test_same_content_produces_same_checksum_regardless_of_path_or_id(tmp_path):
    content = "x,y\n1,2\n"
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    path_a.write_text(content)
    path_b.write_text(content)

    ds_a = Dataset.from_path("dataset_a", str(path_a))
    ds_b = Dataset.from_path("dataset_b", str(path_b))
    assert ds_a.checksum == ds_b.checksum


def test_different_content_produces_different_checksum_for_drift_detection(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    ds_before = Dataset.from_path("ds1", str(path))

    path.write_text("a,b\n1,2\n3,4\n")  # simulate the file changing on disk
    ds_after = Dataset.from_path("ds1", str(path))

    assert ds_before.checksum != ds_after.checksum
    assert ds_before.dataset_id == ds_after.dataset_id  # same id -- this is the drift the checksum must catch


def test_checksum_must_be_valid_64_char_hex():
    with pytest.raises(ValidationError):
        Dataset(dataset_id="ds1", path="/tmp/x.csv", checksum="not-a-valid-checksum")


def test_checksum_too_short_is_rejected():
    with pytest.raises(ValidationError):
        Dataset(dataset_id="ds1", path="/tmp/x.csv", checksum="abc123")


def test_checksum_is_lowercased():
    digest = "A" * 64
    ds = Dataset(dataset_id="ds1", path="/tmp/x.csv", checksum=digest)
    assert ds.checksum == "a" * 64


def test_dataset_is_frozen(tmp_csv):
    ds = Dataset.from_path("ds1", str(tmp_csv))
    with pytest.raises(ValidationError):
        ds.path = "/somewhere/else.csv"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

def test_scenario_references_a_dataset(tmp_csv):
    ds = Dataset.from_path("ds1", str(tmp_csv))
    sc = Scenario(scenario_id="s1", name="Sample HR scenario", dataset=ds)
    assert sc.dataset.dataset_id == "ds1"
    assert sc.description is None


def test_scenario_accepts_optional_description(tmp_csv):
    ds = Dataset.from_path("ds1", str(tmp_csv))
    sc = Scenario(scenario_id="s1", name="Sample HR scenario", dataset=ds, description="20-row HR dataset")
    assert sc.description == "20-row HR dataset"


def test_scenario_is_frozen(tmp_csv):
    ds = Dataset.from_path("ds1", str(tmp_csv))
    sc = Scenario(scenario_id="s1", name="Sample HR scenario", dataset=ds)
    with pytest.raises(ValidationError):
        sc.name = "renamed"


def test_scenario_requires_dataset(tmp_csv):
    with pytest.raises(ValidationError):
        Scenario(scenario_id="s1", name="Sample HR scenario")


def test_scenario_serialization_round_trip(tmp_csv):
    ds = Dataset.from_path("ds1", str(tmp_csv))
    sc = Scenario(scenario_id="s1", name="Sample HR scenario", dataset=ds, description="desc")
    reloaded = Scenario.model_validate_json(sc.model_dump_json())
    assert reloaded == sc
