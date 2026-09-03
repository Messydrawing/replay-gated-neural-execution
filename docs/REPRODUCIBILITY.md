# Reproducibility and evidence boundary

## What this artifact is

This repository is a reference implementation plus aggregate evidence from
protocol-valid frozen runs. It excludes historical variants, raw 9,216-D
witnesses, context payloads, target-server runners, model weights, and future
ABI stages.

Consequently, the repository supports four reproducibility levels:

1. **Integrity:** recompute the file manifest and embedded JSON semantic hashes.
2. **Abstraction behavior:** run deterministic tests for the behavioral
   predicate, physical contract, certificate/audit separation, transactional
   replay, final same-action recertification, and fail-closed cascade.
3. **Aggregate analysis:** recompute reported tables and figures from included
   adjudication and closeout JSON.
4. **Executable miniature:** run actual candidate search, physical projection,
   isolated FP32/BF16 replay certification, and cross-state replay on two
   retired public contexts with pinned SmolLM2-360M-Instruct.

It does **not** reproduce the historical Qwen candidate search or original
closeouts. Those require omitted formal context commitments, raw formal
witness/action payloads, exact target runners, and their authorization chain.
The SmolLM2 minisuite is a separate executable miniature and is explicitly not
assigned the identity of those archived results.

## Public real-model minisuite

`minisuite/` freezes two retired DesignPublic contexts and four Utility
ordinals. Its single command acquires a context chart, runs the same
analytic-plus-fallback action search used by the second-backbone evaluation,
certifies every selected action in FP32 and BF16, checks the physical ABI and
candidate permutation, and replays each action in both contexts. The model is
fully frozen. See `minisuite/README.md` and the target reference output under
`minisuite/expected/target_3090_v1/`.

## Runtime lifecycle represented by the code

For one context, a conforming runtime captures a candidate-free pre-write
snapshot. Each behavioral predicate then follows this lifecycle:

1. a tier proposes an action and trajectory;
2. the action is replayed from the same original state in FP32 and BF16;
3. `ItemCertificate` records only observed behavior and physical legality;
4. `RunAudit` separately records restoration, no state leakage,
   information-path integrity, evidence binding, and identities;
5. a finite item miss may escalate, while an invalid item or run audit fails
   closed;
6. the selected exact action is replayed once more from the original state,
   recertified, checked against its action/Utility/protocol hashes, and only
   then committed; exhaustion produces explicit abstention.

No failed tier may pass its action, optimizer state, hidden state, KV cache, or
RNG state into a later tier. The aggregate cost ledger charges every entered
tier, including misses.

## What can be checked locally

- Utility encoding and absence of capability labels in numerical input;
- physical energy and per-write constraints;
- reduced pullback operations on synthetic fixtures;
- candidate isolation and final-commit transaction semantics;
- fail-closed cascade control and explicit abstention;
- aggregate cost windows, bootstrap intervals, and registered gates;
- included frozen headline numbers and evidence hashes.

## Exact original target environment

The frozen target was an NVIDIA GeForce RTX 3090 under Linux with Python
3.12.7, PyTorch 2.12.1+cu126, Transformers 4.57.6, CUDA 12.6, NumPy 2.5.1,
and SciPy 1.18.0. Deterministic algorithms and cuDNN deterministic mode were
not enabled; this is recorded rather than silently strengthened. The exact
Qwen snapshot and model/tokenizer file hashes are in
`frozen/configs/model_identity.json`.

## Cost interpretation

The primary unit is target-GPU paired-replay-equivalent cost. Context-shared
acquisition and per-predicate costs are recomposed into deterministic cyclic
windows for `m = 1, 2, 4, 8, 64`. Model loading, evidence hashing, and offline
packaging are excluded. Cascade is compared with full operational search, not
with a bare forward pass.

Bootstrap bounds use seed `924117`, 10,000 resamples, and the frozen window unit
in `frozen/configs/cascade_public.json`.
