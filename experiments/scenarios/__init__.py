"""
Controlled corruption scenarios and ground-truth evaluation.

    models.py       -- CorruptionType, CellCorruption, DuplicateCorruption,
                        CorruptionSpec, GroundTruthManifest
    generator.py    -- write_clean_fixture(), corrupt_dataset(),
                        generate_all_scenarios(), SCENARIO_SPECS
    ground_truth.py -- evaluate() -> GroundTruthMetrics

No dependency on converge -- these are pure experiment-side
data/logic. The evaluator is strategy-agnostic: it reads a clean
reference, a corruption manifest, and an output CSV, with zero
knowledge of which strategy produced the output.
"""
