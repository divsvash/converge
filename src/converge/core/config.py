"""
Minimal application configuration for P0.

Deliberately small: no settings management framework, no env-driven
service discovery, no infrastructure config. Just the constants the
profiler (and later stages) need.

Project / system name: CONVERGE
    "CONVERGE: Validation-Guided Generative AI for Traditional Big
    Data Pipelines" is the name of this prototype system. It does not
    change the assigned academic project title
    ("Generative AI Assisted Big Data Pipeline Prototype") or research
    paper title ("Future Directions: Integrating Generative AI with
    Traditional Big Data Pipelines"), which remain fixed.
"""
from pathlib import Path

PROJECT_NAME = "CONVERGE"
PROJECT_TAGLINE = "Validation-Guided Generative AI for Traditional Big Data Pipelines"

# Project root: this file lives at src/converge/core/config.py, so the
# repository root is 4 directories up (core -> converge -> src -> root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RUNS_DIR = DATA_DIR / "runs"

# --- Profiler thresholds (tunable constants, not user-facing config) ---

# A string column is treated as "categorical" (eligible for case/variant
# grouping) if its distinct-value ratio is below this threshold.
# High-cardinality string columns (e.g. names, free text) are skipped
# for categorical-variant analysis to avoid meaningless noise.
CATEGORICAL_DISTINCT_RATIO_THRESHOLD = 0.5

# Minimum distinct value count still considered for categorical analysis
# even if the dataset is small (avoids skipping tiny datasets outright).
CATEGORICAL_MIN_ROWS_FOR_RATIO_CHECK = 5

# IQR multiplier for numeric outlier detection (1.5 is the standard
# Tukey convention).
OUTLIER_IQR_MULTIPLIER = 1.5

# Max number of representative sample rows Spark will attach as evidence
# for any single issue (bounds what the AI stage will later receive).
MAX_EVIDENCE_SAMPLE_ROWS = 10

# Max number of distinct value/count pairs attached as evidence for a
# categorical column's value distribution.
MAX_EVIDENCE_DISTINCT_VALUES = 30

# Column name substrings used to heuristically detect identifier columns
# (excluded from duplicate-detection criteria and categorical analysis).
ID_COLUMN_HINTS = ("id",)

# Column name substrings used to heuristically detect date-like string
# columns for date-format profiling.
DATE_COLUMN_HINTS = ("date", "_at", "_on")
