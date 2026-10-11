"""
Regression test for core.config's path resolution.

This exists specifically because PROJECT_ROOT is computed from
`__file__`'s nesting depth (see core/config.py's comment) -- a
repository restructuring that changes how many directories deep
config.py lives (e.g. the backend/ -> src/converge/ migration, which
added one level of nesting) can silently break DATA_DIR/RAW_DATA_DIR/
RUNS_DIR resolution without any import error, since Path objects are
happily constructed even if they don't point at anything real. This
test would have caught the depth change during that migration.
"""
from converge.core import config


def test_data_directory_resolution_under_the_src_layout():
    """
    PROJECT_ROOT must resolve to the actual repository root (the
    directory containing pyproject.toml and src/), and DATA_DIR /
    RAW_DATA_DIR / RUNS_DIR must all resolve to real, existing paths
    under it -- not just well-formed Path objects.
    """
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    assert (config.PROJECT_ROOT / "src" / "converge").is_dir()

    assert config.DATA_DIR.exists()
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"

    assert config.RAW_DATA_DIR.exists()
    assert (config.RAW_DATA_DIR / "sample_hr.csv").exists()

    assert config.RUNS_DIR.parent == config.DATA_DIR
