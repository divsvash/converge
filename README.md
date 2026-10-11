# CONVERGE

**Validation-Guided Generative AI for Traditional Big Data Pipelines**

Research prototype for *"Future Directions: Integrating Generative AI with Traditional Big Data Pipelines."*

## Research Problem

Can a language model improve data-cleaning outcomes in a traditional Spark pipeline — and if so, does a validation boundary between AI recommendations and Spark execution reduce hallucination-driven errors compared to unvalidated AI?

### Research Objectives

1. Integrate a language model for automated data-cleaning suggestions and insight generation.
2. Design a safe architecture combining generative AI with Hadoop/Spark while controlling hallucination risks.

## Architecture

```
  Dataset (CSV)
       │
  ┌────▼────┐
  │  Spark   │  deterministic profiling: null counts, duplicates,
  │ Profiler │  categorical variants, whitespace, date formats, outliers
  └────┬────┘
       │  ProfileResult (bounded evidence)
  ┌────▼────┐
  │    AI    │  LLM/SLM proposes structured Recommendation objects
  │ Provider │  (Gemini API or local Ollama) — sees ONLY profile/evidence
  └────┬────┘
       │  list[Recommendation]
  ┌────▼──────┐
  │ Validator  │  independent, deterministic, evidence-based checks
  │            │  (approved_auto / needs_review / rejected / not_executable)
  └────┬──────┘
       │  user approval boundary
  ┌────▼──────┐
  │  Spark     │  executes ONLY approved, validated operations
  │ Executor   │  immutable ExecutionPlan, fail-fast, no hidden reordering
  └────┬──────┘
       │  cleaned dataset
  ┌────▼────────┐
  │ Comparator   │  deterministic before/after quality measurement
  │ + Insights   │  (no AI — purely deterministic)
  └─────────────┘
```

**Safety model:** AI recommends → Validator decides → Spark executes. The AI never sees raw data, never executes code, and never bypasses the validator. The validator is independent of both AI and Spark execution.

## Experimental Design

Three strategies compared under controlled conditions:

| Strategy | AI | Validator | Description |
|---|---|---|---|
| **SPARK_ONLY** | None | None | Deterministic rule-based cleaning (whitespace trim + duplicate removal only) |
| **AI_ONLY** | Yes | **No** | AI recommends, structural filter only, no evidence validation |
| **CONVERGE** | Yes | **Yes** | Full pipeline: AI recommends, validator gates, Spark executes |

AI strategies are further varied by:
- **Model class**: LLM / SLM (explicit, not inferred)
- **Deployment mode**: REMOTE (Gemini API) / LOCAL (Ollama)
- **Provider/model**: concrete identity (e.g., `gemini-2.0-flash`, `phi3:mini`)

### Controlled Scenarios

| Scenario | Corruption | Count | Target Column |
|---|---|---|---|
| S1 | Whitespace injection | 3 cells | `name` |
| S2 | Exact duplicate rows | 2 rows | (whole row) |
| S3 | Categorical variants | 3 cells | `department` |
| S4 | Missing values | 2 cells | `salary` |
| S5 | Mixed date formats | 3 cells | `hire_date` |
| S6 | All five combined | 13 total | mixed |

All scenarios derive from the same 15-row clean reference dataset (`data/research/clean.csv`) with deterministic corruption (fixed seed). Ground-truth manifests record every injected corruption for repair-correctness evaluation.

### Metrics

- **Deterministic quality**: before/after issue counts via the same Spark profiler/comparator for all strategies
- **Ground-truth correctness**: `repair_recall`, `false_modification_rate`, per-corruption repair accuracy
- **Performance**: wall-clock timing (strategy execution / evaluation / total)
- **Cost**: token usage where the provider exposes it (None for unknown, 0 for SPARK_ONLY)
- **Reliability**: success rate + recommendation consistency across repeated runs

No overall score. No ranking. No significance tests in the software — those belong to the paper.

## Repository Structure

```
src/converge/          production pipeline (installed package)
  core/                config, exceptions, logging
  models/              domain models + operation catalog
  services/
    spark/             loader, profiler, executor
    validation/        evidence-based validator
    ai/                AIProvider, GeminiProvider, OllamaProvider
    comparison/        deterministic quality comparator
    insights/          deterministic insight generator
    pipeline/          orchestrator (state machine)
  storage/             RunStore (JSON persistence)
  cli/                 CLI entry point

experiments/           research harness (NOT part of installed package)
  models/              experiment domain: Dataset, Scenario, Configuration, Run, Result
  strategies/          spark_only, ai_only, converge adapters
  metrics/             quality, performance, cost, reliability
  scenarios/           corruption generator + ground-truth evaluator
  runner/              ExperimentRunner, export, experiment matrix
  providers.py         provider resolution factory

tests/                 mirrors src/converge/ and experiments/
data/
  raw/                 sample_hr.csv (development fixture)
  research/            clean reference + S1-S6 corrupted fixtures + manifests
  runs/                per-run pipeline artifacts
docs/                  architecture, decisions, research notes
paper/                 manuscript assets (figures, tables, results)
```

## Installation

**Requirements:** Python ≥ 3.10, Java 11+ (for PySpark).

```bash
pip install -e ".[dev]"
```

## Running a Single Pipeline

```bash
# With Gemini
export GEMINI_API_KEY="your-key"
converge run-pipeline --csv data/raw/sample_hr.csv --provider gemini --model gemini-2.0-flash

# With Ollama (requires a running Ollama instance)
ollama pull phi3:mini
converge run-pipeline --csv data/raw/sample_hr.csv --provider ollama --model phi3:mini

# Auto-approve NEEDS_REVIEW recommendations (non-interactive)
converge run-pipeline --csv data.csv --provider gemini --model gemini-2.0-flash --auto-approve
```

## Running Controlled Experiments

```bash
# Gemini only, scenarios S1-S3, 3 repetitions
converge run-experiment --experiment-id exp001 \
  --gemini-model gemini-2.0-flash \
  --scenarios S1_whitespace S2_duplicates S3_categorical \
  --repetitions 3

# Gemini + Ollama, all scenarios
converge run-experiment --experiment-id exp002 \
  --gemini-model gemini-2.0-flash \
  --ollama-model phi3:mini \
  --repetitions 3

# Results appear in data/research/results/<experiment_id>/
#   runs.json, summary.csv, aggregate.csv, metadata.json
```

## Gemini Setup

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. `export GEMINI_API_KEY="your-key"`

## Ollama Setup

1. Install Ollama: https://ollama.com/download
2. Pull a model: `ollama pull phi3:mini`
3. Ensure Ollama is running: `ollama serve` (default: `http://localhost:11434`)

## Tests

```bash
pytest                    # runs all tests (~472)
pytest tests/core/        # just core tests
pytest tests/experiments/ # just experiment tests
```

All tests use fake/mock AI providers — no API keys or running Ollama required.

## Known Limitations

- **AIProvider exposes no token/cost data**: GeminiProvider does not surface the underlying SDK's `usage_metadata`; all cost fields are `None` for AI strategies. Extending this requires modifying `src/converge`, which is frozen for this research phase.
- **No ground-truth for dtype corruption or outliers**: the profiler doesn't emit `TYPE_INCONSISTENCY` evidence, and `FLAG_OUTLIER` is finding-only with no executable remediation.
- **Single-machine Spark**: runs as a local SparkSession, not a cluster. Scale experiments are a separate concern.
- **No sampling parameter capture**: `AIProvider` exposes no temperature/seed property; GeminiProvider hardcodes `temperature=0.0` internally but this isn't reportable through the interface.
- **Clean fixture is 15 rows**: deliberately small for ground-truth correctness establishment; scale testing comes later.

### Research CLI packaging boundary

`converge run-pipeline` works with the installed package. `converge run-experiment`
requires running from the **repository root** because `experiments/` is a
source-checkout research harness, intentionally excluded from the installed
`converge` package. Running it elsewhere gives an explicit error.

Ground-truth comparison is exact for CSV field values (including spaces and
case). `injected_duplicates_removed` measures disappearance of injected IDs;
`correctly_removed_duplicates` additionally requires preservation of the
original source ID. `all_original_rows_lost` counts missing clean-reference
row IDs across all scenarios. False-modification rate covers surviving
original rows only, so always inspect `all_original_rows_lost` alongside it.
