import pytest

from converge.core.config import RAW_DATA_DIR
from converge.services.ai.base import AIProvider
from experiments.models.configuration import AIPlacement, AIRole, Configuration, DeploymentMode, ModelClass, StrategyName
from experiments.models.scenario import Dataset, Scenario


class FakeAIProvider(AIProvider):
    def __init__(self, recommendations=None, raise_exc=None):
        self._recommendations = recommendations or []
        self._raise_exc = raise_exc

    @property
    def provider_name(self) -> str:
        return "fake-provider"

    @property
    def model_name(self) -> str:
        return "fake-model-v0"

    @property
    def prompt_version(self) -> str:
        return "fake-prompt-v0"

    def generate_recommendations(self, profile_result):
        if self._raise_exc is not None:
            raise self._raise_exc
        return list(self._recommendations)


@pytest.fixture()
def sample_dataset():
    return Dataset.from_path("sample_hr", str(RAW_DATA_DIR / "sample_hr.csv"))


@pytest.fixture()
def sample_scenario(sample_dataset):
    return Scenario(scenario_id="sample_hr_scenario", name="Sample HR", dataset=sample_dataset)


@pytest.fixture()
def spark_only_configuration():
    return Configuration(strategy=StrategyName.SPARK_ONLY)


@pytest.fixture()
def ai_only_configuration():
    return Configuration(
        strategy=StrategyName.AI_ONLY, provider="fake-provider", model="fake-model-v0",
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )


@pytest.fixture()
def converge_configuration():
    return Configuration(
        strategy=StrategyName.CONVERGE, provider="fake-provider", model="fake-model-v0",
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
