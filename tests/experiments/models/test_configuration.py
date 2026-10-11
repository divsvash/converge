import pytest
from pydantic import ValidationError

from experiments.models.configuration import (
    _VALID_STRATEGY_PLACEMENT_ROLE,
    AIPlacement,
    AIRole,
    Configuration,
    DeploymentMode,
    ModelClass,
    StrategyName,
)


def _valid_ai_kwargs(strategy: StrategyName) -> dict:
    return dict(
        strategy=strategy,
        provider="gemini",
        model="gemini-2.0-flash",
        model_class=ModelClass.LLM,
        deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING,
        ai_role=AIRole.RECOMMENDATION,
    )


# ---------------------------------------------------------------------------
# SPARK_ONLY rejects AI fields
# ---------------------------------------------------------------------------

def test_spark_only_with_no_ai_fields_is_valid():
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    assert cfg.provider is None
    assert cfg.ai_role is None


@pytest.mark.parametrize("field,value", [
    ("provider", "gemini"),
    ("model", "gemini-2.0-flash"),
    ("model_class", ModelClass.LLM),
    ("deployment_mode", DeploymentMode.REMOTE),
    ("ai_placement", AIPlacement.DURING),
    ("ai_role", AIRole.RECOMMENDATION),
])
def test_spark_only_rejects_each_ai_field_individually(field, value):
    with pytest.raises(ValidationError):
        Configuration(strategy=StrategyName.SPARK_ONLY, **{field: value})


# ---------------------------------------------------------------------------
# AI configurations require their necessary fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [StrategyName.AI_ONLY, StrategyName.CONVERGE])
def test_ai_strategy_with_all_fields_is_valid(strategy):
    cfg = Configuration(**_valid_ai_kwargs(strategy))
    assert cfg.strategy == strategy


@pytest.mark.parametrize("strategy", [StrategyName.AI_ONLY, StrategyName.CONVERGE])
@pytest.mark.parametrize("missing_field", [
    "provider", "model", "model_class", "deployment_mode", "ai_placement", "ai_role",
])
def test_ai_strategy_missing_any_required_field_is_rejected(strategy, missing_field):
    kwargs = _valid_ai_kwargs(strategy)
    kwargs[missing_field] = None
    with pytest.raises(ValidationError):
        Configuration(**kwargs)


@pytest.mark.parametrize("strategy", [StrategyName.AI_ONLY, StrategyName.CONVERGE])
def test_ai_strategy_rejects_empty_provider_string(strategy):
    kwargs = _valid_ai_kwargs(strategy)
    kwargs["provider"] = "   "
    with pytest.raises(ValidationError):
        Configuration(**kwargs)


@pytest.mark.parametrize("strategy", [StrategyName.AI_ONLY, StrategyName.CONVERGE])
def test_ai_strategy_rejects_empty_model_string(strategy):
    kwargs = _valid_ai_kwargs(strategy)
    kwargs["model"] = ""
    with pytest.raises(ValidationError):
        Configuration(**kwargs)


# ---------------------------------------------------------------------------
# Invalid strategy/placement/role combinations are rejected
# ---------------------------------------------------------------------------

def test_only_two_combinations_are_currently_valid():
    """
    White-box check on the closed combination set itself: with exactly
    one AIPlacement value and one AIRole value defined right now, every
    combination that passes the required-fields check is necessarily
    the only valid one per strategy -- there is no way to construct an
    'invalid combination' through the public API today without an
    invalid enum value (which Pydantic itself already rejects at the
    type level, tested separately). This test exists so the narrowness
    of the combination set is itself verified, not just assumed.
    """
    assert _VALID_STRATEGY_PLACEMENT_ROLE == frozenset({
        (StrategyName.CONVERGE, AIPlacement.DURING, AIRole.RECOMMENDATION),
        (StrategyName.AI_ONLY, AIPlacement.DURING, AIRole.RECOMMENDATION),
    })


def test_spark_only_never_appears_in_the_valid_combination_set():
    assert all(s != StrategyName.SPARK_ONLY for s, _, _ in _VALID_STRATEGY_PLACEMENT_ROLE)


def test_unknown_ai_placement_value_is_rejected_at_the_type_level():
    kwargs = _valid_ai_kwargs(StrategyName.CONVERGE)
    kwargs["ai_placement"] = "before"  # not a real AIPlacement member
    with pytest.raises(ValidationError):
        Configuration(**kwargs)


def test_unknown_ai_role_value_is_rejected_at_the_type_level():
    kwargs = _valid_ai_kwargs(StrategyName.CONVERGE)
    kwargs["ai_role"] = "validation"  # not a real AIRole member
    with pytest.raises(ValidationError):
        Configuration(**kwargs)


# ---------------------------------------------------------------------------
# Stable, content-derived Configuration identity
# ---------------------------------------------------------------------------

def test_configuration_id_is_stable_for_identical_fields():
    cfg_a = Configuration(**_valid_ai_kwargs(StrategyName.CONVERGE))
    cfg_b = Configuration(**_valid_ai_kwargs(StrategyName.CONVERGE))
    assert cfg_a is not cfg_b
    assert cfg_a.configuration_id == cfg_b.configuration_id


def test_configuration_id_differs_for_different_strategy():
    cfg_converge = Configuration(**_valid_ai_kwargs(StrategyName.CONVERGE))
    cfg_ai_only = Configuration(**_valid_ai_kwargs(StrategyName.AI_ONLY))
    assert cfg_converge.configuration_id != cfg_ai_only.configuration_id


def test_configuration_id_differs_for_different_model():
    kwargs_a = _valid_ai_kwargs(StrategyName.CONVERGE)
    kwargs_b = _valid_ai_kwargs(StrategyName.CONVERGE)
    kwargs_b["model"] = "gemini-1.5-pro"
    assert Configuration(**kwargs_a).configuration_id != Configuration(**kwargs_b).configuration_id


def test_configuration_id_is_not_a_random_uuid():
    """Confirms configuration_id is deterministic, not generated fresh per instance."""
    ids = {Configuration(strategy=StrategyName.SPARK_ONLY).configuration_id for _ in range(5)}
    assert len(ids) == 1


def test_configuration_is_frozen():
    cfg = Configuration(strategy=StrategyName.SPARK_ONLY)
    with pytest.raises(ValidationError):
        cfg.strategy = StrategyName.AI_ONLY


def test_configuration_serialization_round_trip():
    cfg = Configuration(**_valid_ai_kwargs(StrategyName.CONVERGE))
    reloaded = Configuration.model_validate_json(cfg.model_dump_json())
    assert reloaded == cfg
    assert reloaded.configuration_id == cfg.configuration_id
