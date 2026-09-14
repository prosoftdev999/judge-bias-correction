# cheat/naive_guess

A deliberate cheating/shortcut attempt. Per the harbor task-authoring guide,
`cheat/` is kept in the bundle but never executed by the pipeline -- it's
here so the verifier can be checked against it during authoring.

**What it does** (`cheat.py`): tallies plain judge win-rate (ties = 0.5)
over `judge_transcripts.jsonl` only, with no correction for position,
verbosity, vendor/self-enhancement bias, the second `session_tag` mixed
into the file, or the swap_id label-consistency defect, and reports "no
bias" placeholder values (`0.0`) for `position_bias_estimate` /
`verbosity_coef_estimate` instead of actually estimating them. Never reads
`human_calibration.jsonl`. This is the shape both of a lazy/incomplete
attempt and of a shortcut-seeking one: read the transcripts, skip the
modeling.

**Why it must fail** (numbers below are from actually running `cheat.py`
against the shipped data, not estimated): plain win-rate tallying gets the
ranking wrong -- `Halcyon-mini` (self-enhanced, and written verbosely) is
inflated to rank #3 instead of its true 5th-of-9 place, which the
verifier's Spearman-rho check on `recovered_ranking_best_to_worst` catches
(rho = 0.950, just under the 0.97 floor -- not a wide margin, but still a
fail). The fabricated `0.0` bias estimates miss the true values (0.30 /
0.45) by far more than the verifier's tolerances (0.08 / 0.07). This cheat's
`self_enhancement_model` guess ("pick whichever same-vendor model has the
highest raw win rate") also gets it wrong here: it reports `"Aurora-7B"`,
not `"Halcyon-mini"` -- the swap_id corruption (concentrated on
Halcyon-mini's own matchups in the sealed data) drags its raw win rate
down enough that Aurora-7B's genuinely higher true skill wins the naive
heuristic instead. So this baseline now fails all four substantive checks,
not just some of them -- no single check is relied on alone to gate the
task.

**Verified locally** (harbor has no generic "run an arbitrary directory as
an agent" shorthand, so this was checked by running the two bundle
containers directly rather than through `harbor run -a`): built
`environment/Dockerfile`, ran `cheat.py` in it against a mounted `/app`,
then built `tests/Dockerfile` and ran `test.sh` against that same `/app`
mount. Result: `reward.txt` = `0`; all four of
`test_ranking_recovers_true_skill_order`, `test_position_bias_estimate`,
`test_verbosity_coef_estimate`, and `test_self_enhancement_model_identified`
fail (only the two well-formedness checks pass).

Three more dangerous near-misses were also checked the same way (not
shipped as `cheat/` entries since none of them are shortcuts -- each is a
fully-correct pipeline minus exactly one omission), by patching
`solution/solve.py` to skip one hygiene step and re-running it against the
live data:

- **Skips the swap_id consistency check** (session_tag filtering and
  malformed-row cleaning both still correct): still recovers the right
  ranking (rho=1.0) and self-enhancement label, but
  `verbosity_coef_estimate` lands 0.088 away from ground truth -- outside
  the 0.07 tolerance.
- **Skips session_tag filtering** (swap consistency and malformed-row
  cleaning both still correct, so it pools the primary and audit judges):
  `position_bias_estimate` and `verbosity_coef_estimate` land 0.227 and
  0.217 away from ground truth respectively -- the audit judge's
  opposite-signed bias profile drags both estimates the wrong way, well
  outside both tolerances.
- **Mishandles ties** (treats every tie as a left win instead of a 0.5/0.5
  split, everything else correct): `position_bias_estimate` lands 0.292
  away from ground truth -- 3.6x the tolerance.

In every case `reward.txt` is still `0`. The one variant that *is*
tolerance-safe -- dropping tie rows entirely instead of half-weighting them
-- lands 0.035 / 0.041 away from ground truth, comfortably inside both
0.08 / 0.07 bounds, confirming the tolerances aren't so tight that
legitimate estimator variation gets penalized.
