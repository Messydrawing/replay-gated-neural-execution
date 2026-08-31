# Replay-Gated Neural Execution

## Decoupling Persistent Behavioral Specifications from Neural Realizations in Frozen Language Models

This pre-release repository is a **reference implementation and aggregate
frozen-evidence slice** for Paper 1. It is not a mirror of the full
Cutting-LLM/NCO workspace and is not yet a full actual-model reproduction
package.

The runtime receives a typed Z64 behavioral predicate `U` and a neural context.
It proposes a context-specific 9,216-dimensional physical action, evaluates
each candidate by isolated FP32/BF16 replay, and permits a commit only when both
the action-level certificate and the run-level audit pass. Paper 1 does not
claim capability creation, capability installation, or unseen-Utility late
binding.

The public abstraction separates three roles:

1. `ItemCertificate`: observed behavior, physical contract, exact-zero, and
   candidate-permutation checks for one action;
2. `RunAudit`: snapshot restoration, state isolation, information-path
   integrity, evidence binding, and action/Utility/protocol identities;
3. `Authorization`: the conjunction that alone may permit final commit.

After selection, the exact same action is replayed again from the original
context, recertified, identity-checked, and only then committed. Any failure
causes abstention or fail-closed rejection before commit.

No model weights, private prompts, raw context payloads, witness tensors, SSH
details, or future UnseenUtility experiments are included.

## Frozen system identity

- Backbone: `Qwen/Qwen3-0.6B`, fully frozen, snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Semantic registry: 64 public sequence-score coordinates (`Z64`).
- Physical action: `3 layers x 3 write positions x 1024 hidden = 9216` values.
- Write layers: 7, 14, and 21.
- Dual-precision item certificate: Top-1, target-effect peak, and robust margin
  `>= 0.2` must pass in FP32 and BF16.
- Energy: population P99 `<= 0.018`; per-action hard lease `<= 0.021`, plus
  bridge, answer, and per-write caps.
- Candidate permutation error `<= 1e-6`; hard, non-finite, and
  information-path violations must be zero.
- The model is never trained or updated by these experiments.

The 9,216 dimension is a property of this concrete port layout, not a universal
ABI constant. Under the same layout it scales with selected layers, write sites,
and hidden width; a fixed-width receptor or factorized port could change that
scaling, but is not established here.

Exact target-machine and model-file identities are recorded in
[`environment-lock.txt`](environment-lock.txt) and
[`frozen/configs/model_identity.json`](frozen/configs/model_identity.json).

## Frozen aggregate results

| Experiment | Cascade / replay coverage | Full-search coverage | Key cost result | Authorization / contract audit |
|---|---:|---:|---:|---:|
| Reference Stability | 767/767 frozen witnesses replay | 22 cells: 3/3; 1 cell: 0/3 | slow-path stability, not a speed test | protocol-valid closeout |
| Cascade Pilot | 256/256 | 256/256 | median normalized-cost reduction: 90.30% at `m=1`, 93.84% at `m=4` | 0 observed authorization bypasses; 0 hard/information-path/nonfinite violations |
| Frozen Fresh | 256/256 | 256/256 | median normalized-cost reduction: 89.45% at `m=1`, 92.91% at `m=4` | 0 observed authorization bypasses; 0 hard/information-path/nonfinite violations |

Cost reductions are relative to the frozen full operational search, not a bare
model forward. Frozen Fresh `m=1` mean reduction is 76.06% and its P95 cost
ratio is 1.35, so the mechanism is not uniformly faster on every request.

The frozen adjudication schema retains a legacy field named `unsafe_commits`.
In this release it records observed commit-authority violations under the
registered protocol; it is **not** evidence of general model or system safety.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[test,figures]"
pytest
python scripts/verify_release.py
python scripts/summarize_results.py
python scripts/make_figures.py
```

Unit tests use deterministic fake backends and do not download a model. The
figure script writes to ignored `build/figures/`; it never overwrites the
archived frozen figure.

Without the omitted raw contexts, action witnesses, and target-GPU runners, a
reader can verify file/semantic integrity, test the public abstractions, and
recompute summaries and figures from aggregate closeouts. A reader cannot
rerun the reported Qwen experiments from this repository alone. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Evidence/source separation

- `frozen/` contains immutable aggregate results, frozen configuration
  identities, archived figures, source hashes, and the read-only execution-path
  audit. These objects identify the experiments reported in the paper.
- `reference/` contains a cleaned implementation of the paper's runtime
  semantics. Several dependency modules are byte-identical frozen copies and
  are labeled as such in provenance; the authorization/replay interfaces are a
  later reference reimplementation and **did not generate the archived
  results**.

Scientific evidence identity is never reassigned to the later reference code.
See [`frozen/SOURCE_PROVENANCE.md`](frozen/SOURCE_PROVENANCE.md) and
[`frozen/FROZEN_CODE_PATH_AUDIT.md`](frozen/FROZEN_CODE_PATH_AUDIT.md).

## Repository map

- `reference/replay_gated_execution/authorization.py`: formalized item/audit/authorization reference types.
- `reference/replay_gated_execution/replay.py`: reference isolated replay and final same-action commit interface.
- `reference/replay_gated_execution/cascade.py`: cleaned fail-closed controller reference.
- `reference/replay_gated_execution/semantic.py`: byte-identical typed-predicate source copy.
- `reference/replay_gated_execution/physical_abi.py`: byte-identical physical-contract source copy.
- `frozen/aggregate_results/`: immutable aggregate scientific closeouts.
- `frozen/manifests/`: source identities and public-copy provenance.

## Claim boundary

The frozen evidence supports context-indexed, replay-certified neural
realization and lower operational search cost on the registered public
benchmarks. It does not establish unseen-Utility late binding, natural-language
capability synthesis, cross-backbone generality, general safety, or lower
end-to-end cost than bare generation.

## Release status

This repository is publicly available as a source-visible research artifact.
The current `LICENSE` remains All Rights Reserved and grants no open-source,
redistribution, or reuse rights; contact the authors for permission unless a
separate license is adopted later.
