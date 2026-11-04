"""Probe: profile each scenario once, cache ProfileResult JSON, time one LLM call."""
import json, sys, time
from pathlib import Path
from converge.services.spark.session import get_spark_session, stop_spark_session
from converge.services.spark.loader import load_csv
from converge.services.spark.profiler import profile_dataset
from converge.services.ai.prompts import build_user_prompt, SYSTEM_PROMPT
from converge.services.ai.ollama_provider import OllamaProvider

out = Path("paper/results/profiles"); out.mkdir(parents=True, exist_ok=True)
spark = get_spark_session("probe")
for sid in ["S1_whitespace","S2_duplicates","S3_categorical","S4_missing","S5_dates","S6_mixed"]:
    t=time.perf_counter(); pr = profile_dataset(load_csv(spark, f"data/research/{sid}.csv"), dataset_id=sid)
    dt=time.perf_counter()-t
    (out/f"{sid}.json").write_text(pr.model_dump_json(indent=1))
    up = build_user_prompt(pr)
    print(sid, "profile_s=%.1f"%dt, "evidence=",[ (e.issue_type.value,e.columns) for e in pr.evidence], "prompt_chars=", len(SYSTEM_PROMPT)+len(up), flush=True)
if len(sys.argv)>1:
    from converge.models.profile import ProfileResult
    pr = ProfileResult.model_validate_json((out/"S1_whitespace.json").read_text())
    p = OllamaProvider(sys.argv[1])
    t=time.perf_counter(); recs = p.generate_recommendations(pr); print("llm_s=%.1f"%(time.perf_counter()-t))
    for r in recs: print(r.model_dump_json())
stop_spark_session()
