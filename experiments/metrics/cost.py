"""
AI cost metrics.

VERIFIED (not assumed) before writing this module: converge's
AIProvider interface (converge.services.ai.base.AIProvider) exposes
exactly three identity properties -- provider_name, model_name,
prompt_version -- and one method, generate_recommendations(), which
returns only list[Recommendation]. Neither the interface nor
GeminiProvider's implementation surfaces token usage or monetary cost
anywhere. The underlying google-genai SDK's response object does carry
a `usage_metadata` field internally, but converge's GeminiProvider
does not read or expose it, and this phase does not modify converge to
add that (out of scope -- "do not modify src/converge unless
absolutely necessary", and it is not necessary for building the
measurement layer around the EXISTING interface).

CONSEQUENCE: under the current AIProvider interface, there is no
source of real token/cost data for AI_ONLY or CONVERGE runs. Every
field is None for those strategies -- NOT zero, NOT estimated, NOT
inferred from model identity, NOT looked up from a hardcoded pricing
table. None means "unknown/unavailable", which is the honest state of
affairs here.

For SPARK_ONLY, zero IS the semantically correct value (no AI call is
ever made -- see experiments/strategies/spark_only.py, which has no
import of converge.services.ai at all), so this module reports 0 for
every field there, not None. This distinction -- None (unknown) vs. 0
(genuinely zero) -- is the entire point of this module and is
verified directly by its tests.
"""
from __future__ import annotations

from experiments.models.configuration import StrategyName
from experiments.models.result import CostMetrics


def compute_cost_metrics(strategy: StrategyName) -> CostMetrics:
    if strategy == StrategyName.SPARK_ONLY:
        return CostMetrics(monetary_cost=0.0, currency=None, input_tokens=0, output_tokens=0, total_tokens=0)

    # AI_ONLY / CONVERGE: no source of real usage data exists under
    # the current AIProvider interface. Every field stays None.
    return CostMetrics(monetary_cost=None, currency=None, input_tokens=None, output_tokens=None, total_tokens=None)
