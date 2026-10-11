# Results

Final experiment results should be placed here after running the complete experiment matrix.

Expected source:
```
data/research/results/<experiment_id>/
├── runs.json        — complete per-run records
├── summary.csv      — one row per attempted run
├── aggregate.csv    — grouped by scenario × strategy × provider × model
└── metadata.json    — experiment identity and reproducibility info
```

Copy the final experiment's result directory here for paper-writing stability.

Do not commit intermediate/pilot results as final.
