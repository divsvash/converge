"""
Statistical analysis across E1 (seed 42) + E4 (seeds 7, 123), the replay
study (E5), and the LLM call log. Standard library only.

  - percentile bootstrap 95% CIs (10,000 resamples, fixed RNG seed)
  - paired sign-flip permutation test (exact if n <= 20, else 100k Monte Carlo)

Writes paper/results/stats.json and LaTeX fragments into paper/tables/.
"""
from __future__ import annotations

import csv
import itertools
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables"
RES = ROOT / "paper" / "results"
RNG = random.Random(20261107)
SCEN = ["S3_categorical", "S4_missing", "S6_mixed"]
EXPS = {42: "exp_20261104_qwen3b", 7: "exp_seed7_qwen3b", 123: "exp_seed123_qwen3b"}


def boot_ci(xs, n=10000):
    xs = list(xs)
    if not xs:
        return (None, None)
    if len(set(xs)) == 1:
        return (xs[0], xs[0])
    ms = sorted(st.mean(RNG.choices(xs, k=len(xs))) for _ in range(n))
    return (ms[int(0.025 * n)], ms[int(0.975 * n) - 1])


def perm_test(diffs, mc=100000):
    """Two-sided paired sign-flip test of mean(diff) == 0."""
    d = [x for x in diffs]
    n = len(d)
    if n == 0 or all(x == 0 for x in d):
        return 1.0
    obs = abs(sum(d))
    if n <= 20:
        cnt = sum(1 for signs in itertools.product((1, -1), repeat=n)
                  if abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12)
        return cnt / 2 ** n
    cnt = sum(1 for _ in range(mc) if abs(sum(x if RNG.random() < .5 else -x for x in d)) >= obs - 1e-12)
    return (cnt + 1) / (mc + 1)


def f(x, p=2):
    return "--" if x is None else f"{x:.{p}f}"


def load_live():
    rows = []
    for seed, exp in EXPS.items():
        p = ROOT / "data/research/results" / exp / "summary.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["scenario_id"] in SCEN:
                r["seed"] = seed
                rows.append(r)
    return rows


def live_analysis(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["scenario_id"], r["strategy"])].append(r)
    num = lambda r, k: float(r[k]) if r[k] not in ("", None) else None
    out, lines = {}, []
    for sc in SCEN:
        for arm in ["spark_only", "ai_only", "converge"]:
            rs = [r for r in g[(sc, arm)] if r["success"] == "True"]
            rec = [num(r, "repair_recall") for r in rs]
            dmg = [num(r, "originally_clean_values_modified") + num(r, "all_original_rows_lost") for r in rs]
            inc = [num(r, "incorrectly_repaired") for r in rs]
            out[f"{sc}|{arm}"] = {"n": len(g[(sc, arm)]), "ok": len(rs),
                                 "recall": st.mean(rec) if rec else None, "recall_ci": boot_ci(rec),
                                 "damage": st.mean(dmg) if dmg else None, "damage_ci": boot_ci(dmg),
                                 "incorrect": st.mean(inc) if inc else None,
                                 "seeds": sorted({r["seed"] for r in rs})}
        # pairing on (seed, rep): independent model calls, same input
        key = lambda r: (r["seed"], r["repetition_index"])
        a = {key(r): r for r in g[(sc, "ai_only")] if r["success"] == "True"}
        c = {key(r): r for r in g[(sc, "converge")] if r["success"] == "True"}
        ks = sorted(set(a) & set(c))
        dd = [(num(a[k], "originally_clean_values_modified") + num(a[k], "all_original_rows_lost"))
              - (num(c[k], "originally_clean_values_modified") + num(c[k], "all_original_rows_lost")) for k in ks]
        dr = [num(c[k], "repair_recall") - num(a[k], "repair_recall") for k in ks]
        out[f"{sc}|paired"] = {"n": len(ks), "damage_reduction": st.mean(dd) if dd else None,
                              "damage_reduction_ci": boot_ci(dd), "p_damage": perm_test(dd),
                              "recall_diff": st.mean(dr) if dr else None, "recall_diff_ci": boot_ci(dr),
                              "p_recall": perm_test(dr)}
    lab = {"spark_only": r"\textsc{Spark\_Only}", "ai_only": r"\textsc{AI\_Only}", "converge": r"\textsc{Converge}"}
    for sc in SCEN:
        for arm in ["spark_only", "ai_only", "converge"]:
            o = out[f"{sc}|{arm}"]
            lines.append(f"{sc.split('_')[0] if arm == 'spark_only' else ''} & {lab[arm]} & {o['ok']}/{o['n']} & "
                         f"{f(o['recall'])} [{f(o['recall_ci'][0])}, {f(o['recall_ci'][1])}] & "
                         f"{f(o['damage'], 1)} [{f(o['damage_ci'][0], 1)}, {f(o['damage_ci'][1], 1)}] & "
                         f"{f(o['incorrect'], 1)} \\\\")
        pr = out[f"{sc}|paired"]
        lines.append(f" & \\multicolumn{{5}}{{l}}{{\\textit{{paired (n={pr['n']}): damage "
                     f"$\\Delta$={f(pr['damage_reduction'], 1)} [{f(pr['damage_reduction_ci'][0], 1)}, "
                     f"{f(pr['damage_reduction_ci'][1], 1)}], $p$={pr['p_damage']:.3g}; recall $\\Delta$="
                     f"{f(pr['recall_diff'])}, $p$={pr['p_recall']:.3g}}}}} \\\\")
        lines.append("\\midrule")
    (TAB / "seeds.tex").write_text("\n".join(lines[:-1]) + "\n")
    return out


REPLAY_ORDER = ["AI_ONLY", "AI_ONLY+prec", "V0", "V0+any", "V0+conf", "V0+coerce", "V0+null", "V0+sem",
                "Vfix", "Vfix+sem", "Vfix+null", "V0-evidence", "V0-claims", "V0-params", "V0-dtype"]
REPLAY_LABEL = {"AI_ONLY": "AI\\_Only (model order)", "AI_ONLY+prec": "AI\\_Only + precedence",
                "V0": "Converge (shipped, V0)", "V0+any": "V0 + any-match claims",
                "V0+conf": "V0 + no conflation", "V0+coerce": "V0 + coercibility",
                "V0+null": "V0 + null-tolerant params", "V0+sem": "V0 + semantic guards",
                "Vfix": "V0 + all three fixes", "Vfix+sem": "\\textbf{Fixes + semantic guards}",
                "Vfix+null": "Fixes + null-tolerant", "V0-evidence": "V0 $-$ evidence rules",
                "V0-claims": "V0 $-$ claim cross-check", "V0-params": "V0 $-$ parameter rule",
                "V0-dtype": "V0 $-$ dtype rule"}


def replay_analysis():
    p = RES / "replay.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    items = [i for i in d["items"] if "pipelines" in i]
    groups = {"live_seed42": [i for i in items if i["source"].startswith("E1")],
              "live_heldout": [i for i in items if i["source"].startswith("E4")],
              "injected": [i for i in items if i["source"].startswith("E2")]}
    res = {"unique_executions": d["unique_executions"],
           "parse_errors": sum(1 for i in d["items"] if "parse_error" in i),
           "n": {k: len(v) for k, v in groups.items()}}

    def metrics(its, pipe):
        ok = [i["pipelines"][pipe] for i in its if i["pipelines"][pipe]["success"]]
        abort = sum(1 for i in its if not i["pipelines"][pipe]["success"])
        rec = [x["repair_recall"] for x in ok]
        dmg = [x["originally_clean_values_modified"] + x["all_original_rows_lost"] for x in ok]
        # an aborted run delivers nothing: score recall 0, damage 0 for the "delivered" view
        rec_all = [i["pipelines"][pipe]["repair_recall"] if i["pipelines"][pipe]["success"] else 0.0 for i in its]
        harm = sum(1 for x in dmg if x > 0)
        return {"n": len(its), "aborts": abort, "recall": st.mean(rec_all) if rec_all else None,
                "recall_ci": boot_ci(rec_all), "damage": st.mean(dmg) if dmg else None,
                "damage_ci": boot_ci(dmg), "harmful_runs": harm,
                "incorrect": st.mean([x["incorrectly_repaired"] for x in ok]) if ok else None}

    for gname, its in groups.items():
        res[gname] = {pipe: metrics(its, pipe) for pipe in REPLAY_ORDER}
        # paired tests vs V0 on identical model outputs
        for pipe in REPLAY_ORDER:
            def val(i, q, key):
                x = i["pipelines"][q]
                if not x["success"]:
                    return 0.0
                return x["repair_recall"] if key == "r" else x["originally_clean_values_modified"] + x["all_original_rows_lost"]
            dr = [val(i, pipe, "r") - val(i, "V0", "r") for i in its]
            dd = [val(i, pipe, "d") - val(i, "V0", "d") for i in its]
            res[gname][pipe].update({"d_recall_vs_V0": st.mean(dr) if dr else None, "p_recall_vs_V0": perm_test(dr),
                                     "d_damage_vs_V0": st.mean(dd) if dd else None, "p_damage_vs_V0": perm_test(dd)})
    lines = []
    for pipe in REPLAY_ORDER:
        cells = []
        for gname in ["live_seed42", "live_heldout", "injected"]:
            m = res[gname][pipe]
            cells.append(f"{f(m['recall'])} & {f(m['damage'], 1)} & {m['aborts']}")
        lines.append(f"{REPLAY_LABEL[pipe]} & " + " & ".join(cells) + " \\\\")
        if pipe in ("AI_ONLY+prec", "Vfix+null"):
            lines.append("\\midrule")
    (TAB / "replay.tex").write_text("\n".join(lines) + "\n")
    return res


def token_analysis():
    p = RES / "llm_calls.jsonl"
    if not p.exists():
        return None
    calls = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    by = defaultdict(list)
    for c in calls:
        by[c["dataset_id"].split("__")[0]].append(c)
    out = {"calls": len(calls)}
    for sc, cs in sorted(by.items()):
        out[sc] = {"prompt_tokens": st.mean(c["prompt_tokens"] for c in cs),
                   "output_tokens": st.mean(c["output_tokens"] for c in cs),
                   "output_tokens_sd": st.pstdev([c["output_tokens"] for c in cs]),
                   "wall_s": st.mean(c["wall_s"] for c in cs), "wall_sd": st.pstdev([c["wall_s"] for c in cs]),
                   "prefill_tok_s": st.mean(c["prompt_tokens"] / c["prompt_eval_s"] for c in cs if c["prompt_eval_s"]),
                   "decode_tok_s": st.mean(c["output_tokens"] / c["eval_s"] for c in cs if c["eval_s"]),
                   "prefill_share": st.mean(c["prompt_eval_s"] / c["wall_s"] for c in cs)}
    xs = [c["output_tokens"] for c in calls]
    ys = [c["wall_s"] for c in calls]
    if len(xs) > 2:
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        out["r_outtokens_wall"] = cov / ((sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** .5)
    per_input = defaultdict(list)  # identical prompt <=> same seed + scenario
    for c in calls:
        per_input[(c["tag"].split(":")[0], c["dataset_id"].split("__")[0])].append(c["raw_output"])
    out["distinct_outputs_per_input"] = {f"{k[0]}|{k[1]}": f"{len(set(v))}/{len(v)}"
                                         for k, v in sorted(per_input.items())}
    lines = [f"{sc.split('_')[0]} & {o['prompt_tokens']:.0f} & {o['output_tokens']:.0f}$\\pm${o['output_tokens_sd']:.0f} "
             f"& {o['prefill_tok_s']:.1f} & {o['decode_tok_s']:.1f} & {100 * o['prefill_share']:.0f}\\% "
             f"& {o['wall_s']:.0f}$\\pm${o['wall_sd']:.0f} \\\\" for sc, o in out.items() if isinstance(o, dict) and "wall_s" in o]
    (TAB / "tokens.tex").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    stats = {"live": live_analysis(load_live()), "replay": replay_analysis(), "tokens": token_analysis()}
    (RES / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1)[:6000])
