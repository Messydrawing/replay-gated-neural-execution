# Source provenance

The archived scientific evidence and the later public reference implementation
have different identities.

## Frozen layer

`aggregate_results/`, `configs/`, `figures/`, and `manifests/` identify the
protocol-valid runs reported by the paper. The public release does not edit the
private frozen sources and then reassign their results to new code. Exact source
hashes for the formal runner path are retained in
`manifests/frozen_source_identity.json`.

Some small dependency modules under `../reference/replay_gated_execution/` are
byte-identical copies of frozen source. They carry the label
`FROZEN-SOURCE EXACT COPY` and their bytes are checked by
`scripts/verify_release.py` against `manifests/source_provenance.json`.

## Reference layer

The following are later cleaned interfaces or reimplementations:

- `authorization.py`
- `cascade.py`
- `certificate.py`
- `replay.py`
- `witness_search.py`

Their formal label is:

```text
REFERENCE REIMPLEMENTATION
NOT USED TO GENERATE ARCHIVED RESULTS
```

They expose the paper's `ItemCertificate` / `RunAudit` / `Authorization`
decomposition and an explicit final commit interface for inspection and unit
testing. They do not replace the archived code identity and do not retroactively
generate the included results.

Historical modules with obsolete contracts remain in the private research
workspace. Their absence here is scope filtering, not deletion of research
history.
