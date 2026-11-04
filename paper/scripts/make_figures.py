"""
Generate all paper figures and LaTeX table fragments from REAL results.

Inputs:
    data/research/results/<EXP>/summary.csv      (main matrix, E1)
    paper/results/hallucination_injection.json    (E2)
    paper/results/scalability.json                (E3)
    paper/results/recommendation_audit.json       (E1 recommendation-level audit)
Outputs: paper/figures/*.pdf, paper/tables/*.tex
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})
COL = {"spark_only": "#7a7a7a", "ai_only": "#d1495b", "converge": "#2e86ab"}
LBL = {"spark_only": "SPARK_ONLY", "ai_only": "AI_ONLY", "converge": "CONVERGE"}
TLBL = {"spark_only": r"\textsc{Spark\_Only}", "ai_only": r"\textsc{AI\_Only}", "converge": r"\textsc{Converge}"}
SCEN = ["S1_whitespace", "S2_duplicates", "S3_categorical", "S4_missing", "S5_dates", "S6_mixed"]
SHORT = {s: s.split("_")[0] for s in SCEN}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_summary(exp: str):
    rows = list(csv.DictReader(open(ROOT / "data/research/results" / exp / "summary.csv", encoding="utf-8")))
    g = defaultdict(list)
    for r in rows:
        g[(r["scenario_id"], r["strategy"])].append(r)
    return rows, g


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def _vals(rs, key):
    return [_f(r[key]) for r in rs if r["success"] == "True" and _f(r[key]) is not None]


def _quality(rs, sc):
    """Primary correctness metric: cell repair recall, except S2 (no cell
    corruptions) where it is the fraction of injected duplicates removed."""
    if sc.startswith("S2"):
        return [v / 2.0 for v in _vals(rs, "correctly_removed_duplicates")]
    return _vals(rs, "repair_recall")


def fig_main(g, exp):
    strategies = ["spark_only", "ai_only", "converge"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95))
    panels = [("q", "(a) Repair recall (S2: dup. removal rate)"),
              ("originally_clean_values_modified", "(b) Clean cells wrongly modified"),
              ("strategy_execution_wall_seconds", "(c) Strategy wall time (s)")]
    w = 0.26
    for ax, (m, title) in zip(axes, panels):
        for i, s in enumerate(strategies):
            means, sds = [], []
            for sc in SCEN:
                rs = g.get((sc, s), [])
                vals = _quality(rs, sc) if m == "q" else _vals(rs, m)
                mu, sd = mean_sd(vals)
                means.append(mu or 0)
                sds.append(sd or 0)
            xs = [j + (i - 1) * w for j in range(len(SCEN))]
            lo = [min(sd, mu) for mu, sd in zip(means, sds)]
            hi = [min(sd, 1 - mu) if m == "q" else sd for mu, sd in zip(means, sds)]
            ax.bar(xs, means, w, yerr=[lo, hi], color=COL[s], label=LBL[s], capsize=1.5,
                   error_kw={"lw": 0.6}, zorder=3)
        ax.set_xticks(range(len(SCEN)))
        ax.set_xticklabels([SHORT[s] for s in SCEN])
        ax.set_title(title)
        ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
        if m == "q":
            ax.set_ylim(0, 1.08)
        if m.startswith("strategy"):
            ax.set_yscale("log")
    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "main_results.pdf")
    plt.close(fig)


def table_main(g, path):
    fmt = lambda v, p=2: "--" if v is None else f"{v:.{p}f}"
    lines = []
    for sc in SCEN:
        for s in ["spark_only", "ai_only", "converge"]:
            rs = g.get((sc, s), [])
            if not rs:
                continue
            q, qsd = mean_sd(_quality(rs, sc))
            c, _ = mean_sd(_vals(rs, "correctly_repaired"))
            ic, _ = mean_sd(_vals(rs, "incorrectly_repaired"))
            u, _ = mean_sd(_vals(rs, "unrepaired"))
            cm, cmsd = mean_sd(_vals(rs, "originally_clean_values_modified"))
            fmr, _ = mean_sd(_vals(rs, "false_modification_rate"))
            ib, _ = mean_sd(_vals(rs, "issues_before"))
            ia, _ = mean_sd(_vals(rs, "issues_after"))
            t, _ = mean_sd(_vals(rs, "strategy_execution_wall_seconds"))
            crep = "--" if sc.startswith("S2") else f"{fmt(c,1)}/{fmt(ic,1)}/{fmt(u,1)}"
            name = sc.split("_")[0] if s == "spark_only" else ""
            lines.append(f"{name} & {TLBL[s]} & {fmt(q)}$\\pm${fmt(qsd)} & {crep} & {fmt(cm,1)} & {fmt(fmr,3)} & "
                         f"{fmt(ib,1)}$\\to${fmt(ia,1)} & {fmt(t,0)} \\\\")
        lines.append("\\midrule")
    path.write_text("\n".join(lines[:-1]) + "\n")


RULE_ABBR = {"operation_supported": "unknown op", "forbidden_operation_rejected": "forbidden",
             "target_columns_exist": "no such column", "dtype_compatible": "dtype",
             "parameters_valid": "parameters", "evidence_exists": "no evidence",
             "evidence_supports_claim": "claim contradicted"}


def _ordered(inj):
    key = lambda c: (-1 if c["case"] == "N0" else int(c["case"][1:]))
    return sorted(inj["cases"], key=key)


def fig_injection(inj):
    cases = _ordered(inj)
    names = [c["case"] for c in cases]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.45), sharex=True)
    for ax, m, title in [(axes[0], "repair_recall", "(a) Repair recall on S6"),
                         (axes[1], "damage", "(b) Clean cells modified + original rows lost")]:
        for i, s in enumerate(["ai_only", "converge"]):
            vals, failed = [], []
            for c in cases:
                r = c[s]
                failed.append(not r["success"])
                if not r["success"]:
                    vals.append(0)
                else:
                    vals.append(r["repair_recall"] if m == "repair_recall"
                                else r["originally_clean_values_modified"] + r["all_original_rows_lost"])
            xs = [j + (i - 0.5) * 0.38 for j in range(len(cases))]
            ax.bar(xs, vals, 0.38, color=COL[s], label=LBL[s], zorder=3)
            for x, f in zip(xs, failed):
                if f:
                    ax.text(x, 0, "x", ha="center", va="bottom", color=COL[s], fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(names, rotation=0, fontsize=5.5)
        ax.set_title(title)
        ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
    axes[0].set_ylim(0, 0.85)
    axes[0].legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "injection.pdf")
    plt.close(fig)


def table_injection(inj, path):
    out = []
    for c in _ordered(inj):
        v = c["validator"]
        if v is None:
            verdict = "--"
        elif v["status"] == "rejected":
            rules = [r for r in v["failed_rules"] if r != "evidence_supports_claim" or len(v["failed_rules"]) == 1]
            verdict = r"rej.\ (" + RULE_ABBR.get(rules[-1], rules[-1]) + ")"
        else:
            verdict = r"\textbf{" + {"approved_auto": "approved", "needs_review": "review"}[v["status"]] + "}"

        def cell(r):
            if not r["success"]:
                return r"\multicolumn{2}{c}{\textit{run aborted}}"
            dmg = r["originally_clean_values_modified"] + r["all_original_rows_lost"]
            return f"{r['repair_recall']:.2f} & {dmg}"
        mode = {"control": "", "append": "+", "substitute": r"$\leftrightarrow$"}[c["mode"]]
        out.append(f"{c['case']} & {c['class']} & {mode} & {verdict} & {cell(c['ai_only'])} & {cell(c['converge'])} \\\\")
    path.write_text("\n".join(out) + "\n")


def fig_scale(sc):
    rows = [r["rows"] for r in sc]
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))
    ax.loglog(rows, [r["profile_s"] for r in sc], "o-", color="#2e86ab", label="Spark profiling (s)", ms=3)
    ax.loglog(rows, [r["execute_s"] for r in sc], "s-", color="#7a7a7a", label="Spark execution (s)", ms=3)
    ax.loglog(rows, [r["validate_ms"] / 1000 for r in sc], "^-", color="#d1495b", label="Validation (s)", ms=3)
    ax2 = ax.twinx()
    ax2.semilogx(rows, [r["approx_prompt_tokens"] for r in sc], "d--", color="#edae49", ms=3,
                 label="Prompt size (tokens)")
    ax2.set_ylim(0, max(r["approx_prompt_tokens"] for r in sc) * 1.4)
    ax2.set_ylabel("Prompt tokens (approx.)")
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Dataset rows")
    ax.set_ylabel("Wall time (s)")
    ax.set_ylim(1e-4, 1e3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", ncol=2, fontsize=6,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(FIG / "scalability.pdf")
    plt.close(fig)


def fig_audit(audit):
    from collections import Counter
    c = Counter()
    for r in audit["recommendations"]:
        c[r["outcome"]] += 1
    items = sorted(c.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#2e86ab")
    ax.set_xlabel("LLM recommendations (all CONVERGE runs)")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "audit.pdf")
    plt.close(fig)


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    if exp:
        rows, g = load_summary(exp)
        fig_main(g, exp)
        table_main(g, TAB / "main_results.tex")
    p = ROOT / "paper/results/hallucination_injection.json"
    if p.exists():
        inj = json.loads(p.read_text())
        fig_injection(inj)
        table_injection(inj, TAB / "injection.tex")
    p = ROOT / "paper/results/scalability.json"
    if p.exists():
        sc = json.loads(p.read_text())
        fig_scale(sc)
        (TAB / "scalability.tex").write_text("\n".join(
            f"$10^{{{len(str(r['rows'])) - 1}}}$ & {r['profile_s']:.1f} & {r['execute_s']:.1f} & "
            f"{r['prompt_chars']:,} & {r['evidence_items']} & {r['validate_ms']:.2f} \\\\" for r in sc) + "\n")
    print("figures ->", FIG)
