from experiments.metrics.cost import compute_cost_metrics
from experiments.models.configuration import StrategyName


# ---------------------------------------------------------------------------
# Spark-only reports zero AI usage where semantically defined
# ---------------------------------------------------------------------------

def test_spark_only_reports_zero_not_none():
    cost = compute_cost_metrics(StrategyName.SPARK_ONLY)
    assert cost.monetary_cost == 0.0
    assert cost.input_tokens == 0
    assert cost.output_tokens == 0
    assert cost.total_tokens == 0


# ---------------------------------------------------------------------------
# Unknown AI usage remains None rather than zero
# ---------------------------------------------------------------------------

def test_ai_only_reports_none_not_zero():
    cost = compute_cost_metrics(StrategyName.AI_ONLY)
    assert cost.monetary_cost is None
    assert cost.input_tokens is None
    assert cost.output_tokens is None
    assert cost.total_tokens is None


def test_converge_reports_none_not_zero():
    cost = compute_cost_metrics(StrategyName.CONVERGE)
    assert cost.monetary_cost is None
    assert cost.input_tokens is None
    assert cost.output_tokens is None
    assert cost.total_tokens is None


def test_none_and_zero_are_distinguishable():
    """The actual point of this module: None (unknown) must never be
    conflated with 0 (genuinely zero) anywhere."""
    spark_cost = compute_cost_metrics(StrategyName.SPARK_ONLY)
    ai_cost = compute_cost_metrics(StrategyName.AI_ONLY)
    assert spark_cost.monetary_cost == 0.0
    assert ai_cost.monetary_cost is None
    assert spark_cost.monetary_cost != ai_cost.monetary_cost


# ---------------------------------------------------------------------------
# No pricing inference
# ---------------------------------------------------------------------------

def test_no_hardcoded_pricing_table_in_module_source():
    import inspect
    from experiments.metrics import cost as cost_module
    source = inspect.getsource(cost_module)
    # No dollar-amount-looking literals or per-token rate constants
    assert "0.00" not in source.replace("monetary_cost=0.0", "")
    assert "price" not in source.lower()
    assert "PRICING" not in source


def test_cost_never_derived_from_model_identity():
    """compute_cost_metrics takes only a StrategyName -- it has no
    parameter through which a model/provider string could influence
    the returned values, which is what rules out inferring cost from
    model identity."""
    import inspect
    sig = inspect.signature(compute_cost_metrics)
    assert list(sig.parameters.keys()) == ["strategy"]
