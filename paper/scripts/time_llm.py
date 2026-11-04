import json, sys, time
sys.path.insert(0, ".")
from converge.models.profile import ProfileResult
from converge.services.ai.ollama_provider import OllamaProvider
run = json.load(open(sys.argv[1]))
pr = ProfileResult.model_validate(run["profile_result"])
p = OllamaProvider(sys.argv[2])
t = time.perf_counter(); recs = p.generate_recommendations(pr)
print("llm_seconds=%.1f n=%d" % (time.perf_counter() - t, len(recs)))
for r in recs: print(r.model_dump_json(exclude={"recommendation_id"}))
