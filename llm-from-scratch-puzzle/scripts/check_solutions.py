from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = (
    "chapter01_cs336_transformer_e2e",
    "chapter02_classification",
    "chapter03_instruction_finetuning",
)


def main() -> int:
    failures: list[str] = []
    for chapter in CHAPTERS:
        script = ROOT / chapter / "solution.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"PASS {chapter}: {result.stdout.strip()}")
        else:
            failures.append(chapter)
            print(f"FAIL {chapter}\n{result.stdout}{result.stderr}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} chapter(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"All {len(CHAPTERS)} chapter solutions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
