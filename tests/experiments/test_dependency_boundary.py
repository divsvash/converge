"""
Structural dependency boundary checks for experiments/models/.

These are AST-based (not just "no test failed to import"), matching
the verification technique already used in prior phases (see the
repository migration's import-graph checks) -- scanning the actual
source for import statements is a stronger guarantee than inferring
it from the fact that tests happened to pass.
"""
import ast
from pathlib import Path

import experiments.models

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_MODELS_DIR = _REPO_ROOT / "experiments" / "models"
_CONVERGE_SRC_DIR = _REPO_ROOT / "src" / "converge"


def _imported_top_level_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
    return modules


def test_experiments_models_never_import_converge():
    """
    experiments/models/ is designed to be independent of converge
    entirely -- nothing it represents (Dataset, Scenario, Configuration,
    Experiment, Run, Result) requires importing converge's code.
    """
    offending = {}
    for py_file in _EXPERIMENTS_MODELS_DIR.glob("*.py"):
        modules = _imported_top_level_modules(py_file)
        if "converge" in modules:
            offending[str(py_file)] = modules
    assert offending == {}, f"experiments/models files importing converge: {offending}"


def test_converge_never_imports_experiments():
    """
    The hard, structural rule: converge's core (models, services,
    storage) must never import experiments. The CLI (converge.cli) is
    allowed to — it's a composition root that wires both together.
    """
    offending = {}
    for py_file in _CONVERGE_SRC_DIR.rglob("*.py"):
        if "/cli/" in str(py_file):
            continue  # CLI is a composition root, allowed to wire experiments
        modules = _imported_top_level_modules(py_file)
        if "experiments" in modules:
            offending[str(py_file)] = modules
    assert offending == {}, f"converge files importing experiments: {offending}"


def test_experiments_models_public_api_is_exactly_what_is_exported():
    expected = {
        "AIPlacement", "AIRole", "Configuration", "CostMetrics", "Dataset",
        "DeploymentMode", "Experiment", "GroundTruthResult", "ModelClass",
        "PerformanceMetrics", "QualityMetrics", "ReproducibilityMetadata",
        "Result", "Run", "RunStatus", "Scenario", "StrategyName",
    }
    assert set(experiments.models.__all__) == expected
    for name in expected:
        assert hasattr(experiments.models, name), f"{name} not actually importable from experiments.models"
