# Reference implementation boundary

`replay_gated_execution/` is the inspectable Python package shipped with this
artifact.

- Files marked `FROZEN-SOURCE EXACT COPY` in
  `../frozen/manifests/source_provenance.json` are byte-identical dependencies.
- Files listed under `reference_reimplementations` are later cleaned interfaces
  and did **not** generate the archived results.

The package exists to test and explain the runtime semantics. The scientific
evidence remains bound to the source identities under `../frozen/`.
