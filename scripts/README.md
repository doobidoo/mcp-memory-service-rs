# scripts

Lightweight helpers that sit outside the Rust build.

## `parity_check.py`

Embedding-level parity test against the Python upstream. For each fixture text, runs `target/*/mcp-memory-service-rs embed <text>` and compares its first-5 coefficients + L2 norm against the same text pushed through `ONNXEmbeddingModel` in the upstream Python package.

Exit code 0 on full parity, 1 on any mismatch.

```bash
# Uses the upstream repo next to this one by default:
#   ../mcp-memory-service/   (clone of doobidoo/mcp-memory-service)
# and its venv's Python interpreter.
../mcp-memory-service/.venv/bin/python scripts/parity_check.py

# Override paths if needed:
MMS_UPSTREAM_PATH=/path/to/mcp-memory-service \
  /path/to/python scripts/parity_check.py
```

The Python venv needs `onnxruntime` and `tokenizers` installed. `uv pip install onnxruntime tokenizers` inside the upstream venv is enough.

When M4 lands, this script stays as the fast embedding gate; the M4 suite will layer full storage-level parity on top (store in Python, retrieve in Rust against the shared SQLite file, assert identical ranking).
