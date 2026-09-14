# Judge Bias Correction

A machine-learning evaluation and statistical calibration challenge focused on separating underlying item quality from systematic differences between judges.

Real evaluation pipelines often aggregate scores from multiple reviewers, automated evaluators, or model-based judges. Those judges may not use the scoring scale identically. Some can be consistently stricter or more lenient, while others may exhibit systematic preferences that distort naive aggregate scores.

This project focuses on reconstructing a calibrated view of evaluation outcomes rather than treating every observed judgment as directly comparable.

## Overview

Suppose several judges evaluate overlapping sets of items.

A raw score may contain contributions from both:

```text
observed judgment
        =
underlying item signal
        +
judge-specific effects
        +
observation noise
```

Simply averaging judgments can therefore produce biased rankings when the assignment of judges to items is uneven.

The objective of the task is to use the supplied evaluation evidence to infer the relevant calibration structure and produce the requested corrected results.

## Repository Structure

```text
judge-bias-correction/
├── cheat/
│   └── naive_guess/
├── environment/
├── solution/
├── tests/
├── instruction.md
└── task.toml
```

### `instruction.md`

Defines the evaluation-reconstruction problem, available evidence, required output, and scoring conventions.

### `environment/`

Contains the reproducible task environment and solver-visible data.

### `solution/`

Contains the reference implementation used to validate the task.

### `tests/`

Contains the independent task verifier.

### `cheat/naive_guess/`

Contains a deliberately simple baseline useful for demonstrating why uncalibrated or naive aggregation is insufficient.

### `task.toml`

Defines task metadata and execution configuration.

## Motivation

Multi-judge evaluation appears in many settings:

- model benchmarking
- human preference studies
- pairwise comparison systems
- rubric-based grading
- quality review
- ranking experiments
- LLM-as-a-judge evaluation

A common failure mode is to assume that scores from different judges share exactly the same interpretation.

For example, one evaluator might systematically score items lower than another even when they agree on relative quality.

If different items are evaluated by different mixtures of judges, raw averages may confound item quality with evaluator behavior.

## Calibration Problem

The project treats judge behavior as something that must be inferred from overlapping observations.

Useful evidence can come from judges evaluating common items or from comparison structures that connect otherwise separate parts of the dataset.

Conceptually:

```text
raw evaluations
       ↓
recover judge relationships
       ↓
estimate systematic judge effects
       ↓
separate evaluator effects from item signal
       ↓
produce calibrated evaluation results
```

The exact mathematical model and required output are defined by `instruction.md`.

## Why Naive Aggregation Can Fail

Consider two items:

```text
Item A → evaluated mostly by strict judges
Item B → evaluated mostly by lenient judges
```

Even if Item A is objectively stronger, its raw average could be lower.

Therefore:

```text
mean(raw scores)
```

is not necessarily equivalent to:

```text
estimated underlying quality
```

The challenge is to use shared evidence to determine which differences are attributable to the items and which are attributable to the judges.

## Key Technical Themes

This project involves reasoning about:

- evaluator calibration
- systematic bias
- noisy measurements
- overlapping observations
- ranking consistency
- statistical identifiability
- normalization
- latent-variable inference
- robust evaluation
- deterministic data processing

## Baseline

The `cheat/naive_guess/` directory represents a simple baseline approach.

Such a baseline can be useful for comparison because it illustrates how apparently reasonable heuristics can fail when judge assignment and evaluator behavior are not uniform.

A successful solution should derive the requested quantities from the supplied evidence rather than rely on a global average or fixed assumptions about individual judges.

## Reproducibility

The repository includes a self-contained task environment.

The intended workflow is:

```text
input evidence
     ↓
calibration / reconstruction
     ↓
required artifact
     ↓
independent verification
```

The solver should produce deterministic output from the supplied data.

## Validation

The `tests/` directory contains the verifier used to check the generated artifact.

Correctness should depend on the requested evaluation semantics rather than implementation-specific details of the solver.

For the authoritative schema, field names, numerical conventions, and output location, see:

```text
instruction.md
```

## Applications

The underlying ideas are relevant to:

- LLM-as-a-judge calibration
- benchmark reliability
- preference-model evaluation
- human annotation studies
- ranking systems
- multi-rater scoring
- quality-control workflows
- experimental measurement systems

## Goal

The goal is to produce an evaluation that reflects the evidence about the items themselves while accounting for systematic differences between the judges that produced that evidence.

The task emphasizes reconstruction and calibration over naive score aggregation.

## License

No license is currently specified.
