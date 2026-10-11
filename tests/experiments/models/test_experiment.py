import pytest
from pydantic import ValidationError

from experiments.models.configuration import Configuration, StrategyName
from experiments.models.experiment import Experiment
from experiments.models.scenario import Dataset, Scenario


@pytest.fixture()
def scenario(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    ds = Dataset.from_path("ds1", str(path))
    return Scenario(scenario_id="s1", name="Sample scenario", dataset=ds)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_experiment_requires_at_least_one_configuration(scenario):
    with pytest.raises(ValidationError):
        Experiment(name="exp", scenario=scenario, configurations=[], repetition_count=1)


def test_experiment_with_one_configuration_is_valid(scenario):
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    exp = Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=1)
    assert len(exp.configurations) == 1


def test_experiment_id_is_generated_and_unique_per_instance(scenario):
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    exp_a = Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=1)
    exp_b = Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=1)
    assert exp_a.experiment_id != exp_b.experiment_id


# ---------------------------------------------------------------------------
# repetition_count validation
# ---------------------------------------------------------------------------

def test_repetition_count_must_be_at_least_one(scenario):
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    with pytest.raises(ValidationError):
        Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=0)


def test_repetition_count_rejects_negative(scenario):
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    with pytest.raises(ValidationError):
        Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=-1)


def test_repetition_count_of_one_is_valid(scenario):
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    exp = Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=1)
    assert exp.repetition_count == 1


# ---------------------------------------------------------------------------
# Duplicate configurations (by content) are rejected
# ---------------------------------------------------------------------------

def test_duplicate_configurations_by_content_are_rejected(scenario):
    cfg_a = Configuration(strategy=StrategyName.SPARK_ONLY)
    cfg_b = Configuration(strategy=StrategyName.SPARK_ONLY)  # same content, different instance
    with pytest.raises(ValidationError):
        Experiment(name="exp", scenario=scenario, configurations=[cfg_a, cfg_b], repetition_count=1)


def test_distinct_configurations_are_accepted(scenario):
    cfg_spark = Configuration(strategy=StrategyName.SPARK_ONLY)
    cfg_converge = Configuration(
        strategy=StrategyName.CONVERGE, provider="gemini", model="gemini-2.0-flash",
        model_class="llm", deployment_mode="remote", ai_placement="during", ai_role="recommendation",
    )
    exp = Experiment(
        name="exp", scenario=scenario, configurations=[cfg_spark, cfg_converge], repetition_count=2
    )
    assert len(exp.configurations) == 2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_experiment_serialization_round_trip(scenario):
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    exp = Experiment(name="exp", scenario=scenario, configurations=[cfg], repetition_count=3)
    reloaded = Experiment.model_validate_json(exp.model_dump_json())
    assert reloaded.experiment_id == exp.experiment_id
    assert reloaded.repetition_count == 3
    assert reloaded.scenario == exp.scenario
