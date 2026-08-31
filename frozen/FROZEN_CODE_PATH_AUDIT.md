# Read-only frozen execution-path audit

Date: 2026-08-31

This audit was performed against the unchanged source identities listed in
`manifests/frozen_source_identity.json`. It does not modify or re-adjudicate the
archived experiments.

## Exact-zero admission

The helper `cascade_core.py::Certificate.valid_for_commit()` did not include
`exact_zero_pass`. That helper was bound for clean-room/controller tests, but the
formal scientific runner `run_cascade_context.py` did not import or call it.

The formal runner imported
`cascade_adjudication.py::row_certificate_pass()` and `row_invalid()`.
Both paths explicitly required:

```text
exact_zero_pass is True
```

`row_invalid()` also treated exact-zero failure as an invalid run rather than a
finite miss. Therefore the helper inconsistency did not provide an admission
path for formal CascadePilot or Frozen Fresh rows. No historical result is
silently repaired by the public reference implementation.

## Final dual-precision replay

The frozen runtime adapter defined `_final_transactional_admission()` as an
FP32 and BF16 replay of the exact selected float32 action through
`replay_transactional()`, followed by a fresh behavioral/physical admission
calculation. This function was called for terminal reduced-tier candidates and
full-search candidates before their rows were returned to the formal runner.

The archived experiment records certification/selection of an action; it is not
a deployed stateful serving service. The explicit `backend.commit()` transaction
in the public reference layer is a clarified runtime interface, not a claim that
the cleaned code generated the archived rows.

## Audit conclusion

```text
FROZEN_EXPERIMENT_SOURCE_MODIFIED = false
FORMAL_EXACT_ZERO_GATE_PRESENT = true
FINAL_SELECTED_ACTION_FP32_BF16_REPLAY_PRESENT = true
REFERENCE_REIMPLEMENTATION_GENERATED_ARCHIVED_RESULTS = false
HISTORICAL_READJUDICATION_REQUIRED_BY_THIS_AUDIT = false
```
