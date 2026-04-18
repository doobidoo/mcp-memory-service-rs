#!/usr/bin/env python3
"""
Embedding parity check: Python upstream vs Rust port.

Runs the Rust `embed <text>` subcommand for each fixture, runs the same
text through the upstream Python ONNXEmbeddingModel, and asserts the
first-5 coefficients agree to 4 decimals and the L2 norm is 1.

Exit 0 on full parity, 1 on any mismatch. Intended as the embedding-
level gate before the full M4 parity suite lands.

Prereqs:
  - The Rust binary has been built (release or debug) — script auto-
    detects and will run `cargo build` if neither is present.
  - A Python venv with `onnxruntime` and `tokenizers` installed, with
    the upstream repo at UPSTREAM_PATH importable. Adjust paths via
    env vars `MMS_UPSTREAM_PATH` and `MMS_PY` (Python interpreter).

Usage:
    python scripts/parity_check.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

FIXTURES = [
    "hello world",
    "Rust is a systems programming language",
    "Python is popular for data science",
    "Bananas are yellow fruits",
]

TOLERANCE = 5e-4  # 4-decimal agreement after Rust's %.4f rounding
REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PATH = Path(
    os.environ.get("MMS_UPSTREAM_PATH",
                   str(REPO_ROOT.parent / "mcp-memory-service"))
)


def find_rust_binary() -> Path:
    for variant in ("release", "debug"):
        p = REPO_ROOT / "target" / variant / "mcp-memory-service-rs"
        if p.exists():
            return p
    print("no built binary found — running `cargo build`", file=sys.stderr)
    subprocess.run(["cargo", "build"], cwd=REPO_ROOT, check=True)
    p = REPO_ROOT / "target" / "debug" / "mcp-memory-service-rs"
    if not p.exists():
        raise RuntimeError(f"build produced no binary at {p}")
    return p


def run_rust(binary: Path, text: str) -> tuple[list[float], float]:
    result = subprocess.run(
        [str(binary), "embed", text],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    first_match = re.search(r"first 5\s*:\s*\[([^\]]+)\]", out)
    l2_match = re.search(r"L2 norm\s*:\s*([0-9.]+)", out)
    if not first_match or not l2_match:
        raise RuntimeError(f"could not parse rust output:\n{out}")
    first = [float(x.strip()) for x in first_match.group(1).split(",")]
    return first, float(l2_match.group(1))


def run_python(text: str):
    sys.path.insert(0, str(UPSTREAM_PATH / "src"))
    from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel

    if not hasattr(run_python, "_model"):
        run_python._model = ONNXEmbeddingModel()
    v = run_python._model.encode(text)[0]
    first = [round(float(x), 4) for x in v[:5]]
    l2 = float((v * v).sum()) ** 0.5
    return first, l2


def main() -> int:
    binary = find_rust_binary()
    print(f"rust    : {binary.relative_to(REPO_ROOT)}")
    print(f"python  : upstream at {UPSTREAM_PATH}")
    print()

    header = f"  {'text':<40s} {'match':<6s} {'l2(py)':<9s} {'l2(rs)':<9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    failures = 0
    for text in FIXTURES:
        rs_first, rs_l2 = run_rust(binary, text)
        py_first, py_l2 = run_python(text)
        first_ok = all(abs(a - b) < TOLERANCE for a, b in zip(py_first, rs_first))
        l2_ok = abs(py_l2 - 1.0) < 1e-4 and abs(rs_l2 - 1.0) < 1e-4
        ok = first_ok and l2_ok
        if not ok:
            failures += 1
        label = text if len(text) <= 40 else text[:37] + "..."
        print(f"  {label:<40s} {'OK' if ok else 'FAIL':<6s} "
              f"{py_l2:<9.6f} {rs_l2:<9.6f}")
        if not ok:
            print(f"    py first5 : {py_first}")
            print(f"    rs first5 : {rs_first}")

    print()
    print(f"result  : {len(FIXTURES) - failures}/{len(FIXTURES)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
