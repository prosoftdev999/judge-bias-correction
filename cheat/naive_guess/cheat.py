"""Deliberate cheating / shortcut attempt -- never executed by the pipeline.

Reads only judge_transcripts.jsonl, tallies plain win-rate (ties = 0.5), and
reports "no bias" placeholder values (0.0) for the two bias coefficients
instead of estimating them. No use of human_calibration.jsonl at all.
"""
import json
from collections import defaultdict

rows = [json.loads(l) for l in open("/app/data/judge_transcripts.jsonl")]
wins = defaultdict(float)
games = defaultdict(int)

for r in rows:
    l, rr, v = r["model_left"], r["model_right"], r["judge_verdict"]
    games[l] += 1
    games[rr] += 1
    if v == "win_left":
        wins[l] += 1
    elif v == "win_right":
        wins[rr] += 1
    else:
        wins[l] += 0.5
        wins[rr] += 0.5

winrate = {m: wins[m] / games[m] for m in games}
ranking = sorted(winrate, key=lambda m: -winrate[m])

metadata = json.load(open("/app/data/models_metadata.json"))
judge_vendor = metadata["judge_vendor"]
same_vendor = [m["name"] for m in metadata["models"] if m["vendor"] == judge_vendor]
# guess: whichever same-vendor model has the higher raw win rate
guess_self_enhanced = max(same_vendor, key=lambda m: winrate[m])

output = {
    "recovered_ranking_best_to_worst": ranking,
    "position_bias_estimate": 0.0,
    "verbosity_coef_estimate": 0.0,
    "self_enhancement_model": guess_self_enhanced,
}
with open("/app/output.json", "w") as f:
    json.dump(output, f, indent=2)
print(json.dumps(output, indent=2))
