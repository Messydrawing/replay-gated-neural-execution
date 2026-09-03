from __future__ import annotations

import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
INCLUDE = ("config", "data", "runtime")


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = {}
    for directory in INCLUDE:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    files["run_public_minisuite.py"] = sha(ROOT / "run_public_minisuite.py")
    files["verify_output.py"] = sha(ROOT / "verify_output.py")
    value = {"schema": "REPLAY_GATED_NEURAL_EXECUTION_PUBLIC_MINISUITE_HASHES_V1", "files": files}
    (ROOT / "SHA256SUMS.json").write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "manifest_sha256": sha(ROOT / "SHA256SUMS.json")}, indent=2))


if __name__ == "__main__":
    main()
