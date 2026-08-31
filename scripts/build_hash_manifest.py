from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "build"}


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix == ".pyc":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative.as_posix()}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
