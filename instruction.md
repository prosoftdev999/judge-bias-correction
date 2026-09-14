# Judge Bias Correction

Nine candidate language models were evaluated in pairwise comparisons by the primary evaluation judge, an LLM. Judges of this kind are known to sometimes favor response position, response length, or a model associated with their own vendor. Multiple candidates share the judge's vendor. An independent human comparison set covering the same matchups is also provided.

## Inputs

Read these files from `/app/data`:

- `judge_transcripts.jsonl`: pairwise LLM-judge records with `prompt_id`, `model_left`, `model_right`, `response_left_tokens`, `response_right_tokens`, `judge_verdict`, `swap_id`, and `session_tag`.
- `human_calibration.jsonl`: independent human comparisons with `prompt_id`, `model_a`, `model_b`, and `human_verdict`.
- `models_metadata.json`: candidate model/vendor metadata, the primary judge's vendor, and token-length normalization parameters.

## Required analysis

Produce, for the primary evaluation judge specifically:

1. all candidate models ranked from best to worst by true underlying quality;
2. its left/right position-bias coefficient;
3. its verbosity coefficient, using the normalization values in `models_metadata.json`; and
4. the candidate model (if any) supported by the data as receiving self-enhancement from it.

## Output

Write exactly one required artifact: `/app/output.json`.

It must be valid JSON with these keys:

```json
{
  "recovered_ranking_best_to_worst": ["model-1", "model-2"],
  "position_bias_estimate": 0.0,
  "verbosity_coef_estimate": 0.0,
  "self_enhancement_model": "model-name"
}
```

`recovered_ranking_best_to_worst` must contain every candidate model exactly once, using the names in the input data. The two bias estimates must be JSON numbers, and `self_enhancement_model` must use an exact candidate-model name.

You have 7200 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
