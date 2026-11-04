# Draft

Paper manuscript for "Future Directions: Integrating Generative AI with
Traditional Big Data Pipelines" (IEEE conference format, 11 pages).

- `converge_ieee.tex` — main file (IEEEtran, conference mode)
- `sections/` — abstract, intro/related/problem (`method.tex`), architecture,
  methodology (`eval.tex`), results (`results_e1/e2/e3.tex`), discussion
- `refs.bib` — bibliography
- `converge_ieee.pdf` — compiled paper

## Reproducing every number in the paper

Spark on Windows needs Hadoop `winutils.exe`; the experiments were run in
WSL2 Ubuntu 24.04 (Python 3.12, OpenJDK 17) against Ollama on the Windows
host (`qwen2.5-coder:3b`), from the repository root:

```bash
python3 -m venv ~/converge-venv
~/converge-venv/bin/pip install "pyspark==3.5.1" "pydantic==2.13.4" "google-genai==2.17.0" "pytest==8.3.3" matplotlib
export PYTHONPATH=src:.

# E1: 6 scenarios x 3 arms x 3 repetitions (~2 h on a 4-core CPU)
~/converge-venv/bin/python -m converge.cli.main run-experiment \
  --experiment-id exp_20261104_qwen3b --ollama-model qwen2.5-coder:3b --repetitions 3

# E1 recommendation audit (+ offline any-match validator ablation)
~/converge-venv/bin/python paper/scripts/audit_recommendations.py exp_20261104_qwen3b

# E2: hallucination injection (no LLM)
~/converge-venv/bin/python paper/scripts/hallucination_injection.py

# E3: scalability 1e2..1e6 rows
~/converge-venv/bin/python paper/scripts/scalability.py 100 1000 10000 100000 1000000

# Figures + table fragments (paper/figures, paper/tables)
~/converge-venv/bin/python paper/scripts/make_figures.py exp_20261104_qwen3b
```

Then build: `pdflatex converge_ieee && bibtex converge_ieee && pdflatex converge_ieee && pdflatex converge_ieee`.

Raw outputs: `data/research/results/exp_20261104_qwen3b/`, `paper/results/`.
