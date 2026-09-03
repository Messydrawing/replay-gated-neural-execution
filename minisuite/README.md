# Public executable minisuite

This directory is a small, real-model companion to the aggregate frozen-evidence slice. It runs the replay-gated execution path on **two retired public contexts × four typed Utilities** using the pinned `SmolLM2-360M-Instruct` revision.

It includes the public prompts, the 8,640-dimensional action ABI, the analytic-plus-fallback finder, independent FP32/BF16 `ReplayCert`, candidate-permutation audit, physical-budget audit, and a 2×2 cross-state transfer check. It does not train or modify the model.

Linux/CUDA one-command run (the model is downloaded at the pinned revision if `--model` is omitted):

```bash
python minisuite/run_public_minisuite.py --output minisuite/output
python minisuite/verify_output.py minisuite/output
```

To reuse an existing local model directory:

```bash
python minisuite/run_public_minisuite.py --model /path/to/pinned/model --output minisuite/output
python minisuite/verify_output.py minisuite/output
```

The command writes `result.json`, `actions.npz`, and per-context optimization traces. A successful run must report 8/8 native dual-precision certificates, zero unsafe commits, zero candidate-permutation error (within `1e-6`), and all energies below the frozen hard contract. Off-diagonal transfer is descriptive: failures support state-indexed realization, but it is not a required pass condition for this miniature.

This is deliberately not presented as a byte-identical re-execution of the historical Qwen formal experiment. The exact historical evidence remains under `frozen/`; this directory is an inspectable, executable miniature on an open second backbone.
