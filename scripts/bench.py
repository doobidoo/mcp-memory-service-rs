#!/usr/bin/env python3
"""
Performance comparison: Python upstream MCP server vs Rust port.

Measures cold-start (fork → first successful tool response), warm
per-operation latency (100 sequential ops), and peak RSS. Prints a
markdown table so results can be pasted into release notes.

Both servers run over the same stdio MCP protocol, against a fresh
temp DB, against the same pre-cached ONNX model so embedding load
time is the same order of magnitude on both sides.

Prereqs:
  - Rust release binary built (`cargo build --release`).
  - Upstream Python repo + venv with onnxruntime + tokenizers.

Usage:
    ../mcp-memory-service/.venv/bin/python scripts/bench.py
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = Path(
    os.environ.get(
        "MMS_UPSTREAM_PATH",
        str(REPO_ROOT.parent / "mcp-memory-service"),
    )
)
RUST_BIN = REPO_ROOT / "target" / "release" / "mcp-memory-service-rs"
PY_BIN = UPSTREAM / ".venv" / "bin" / "python"

NUM_STORES = 100
NUM_RETRIEVES = 100


def _frame(msg: dict) -> bytes:
    return (json.dumps(msg) + "\n").encode()


def _init_msgs():
    return [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "bench", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]


def _tool_call(id_: int, name: str, args: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call",
            "params": {"name": name, "arguments": args}}


class Server:
    """Thin stdio MCP client that can send/receive JSON-RPC frames and
    measure wall-clock time for each one."""

    def __init__(self, label: str, argv: list[str], env: dict):
        self.label = label
        self.argv = argv
        self.env = env
        self.proc: subprocess.Popen | None = None

    def start(self) -> float:
        """Fork, handshake, return seconds until initialize response arrives."""
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self.env,
        )
        for m in _init_msgs():
            self.proc.stdin.write(_frame(m))
        self.proc.stdin.flush()
        # Read until we see the initialize response.
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} died during handshake")
            r = json.loads(line)
            if r.get("id") == 1:
                break
        return time.perf_counter() - t0

    def call(self, name: str, args: dict, id_: int) -> tuple[dict, float]:
        t0 = time.perf_counter()
        self.proc.stdin.write(_frame(_tool_call(id_, name, args)))
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} died during call")
            r = json.loads(line)
            if r.get("id") == id_:
                dt = time.perf_counter() - t0
                if "error" in r:
                    raise RuntimeError(f"{self.label} {name} error: {r['error']}")
                return r["result"], dt

    def peak_rss_mb(self) -> float:
        """Peak RSS of the child in MB (via ps, best-effort)."""
        if not self.proc:
            return 0.0
        try:
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(self.proc.pid)],
                text=True,
            ).strip()
            kb = int(out)
            return kb / 1024.0
        except Exception:
            return 0.0

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def bench_server(label: str, argv: list[str], env: dict, db_path: Path):
    print(f"\n=== {label} ===")
    print(f"  binary: {argv[0]}")
    print(f"  db    : {db_path}")

    # Cold-start = fork until initialize response.
    server = Server(label, argv, env)
    cold_s = server.start()
    print(f"  cold-start      : {cold_s * 1000:7.1f} ms")

    # Warm store latency.
    store_lats = []
    base_id = 100
    for i in range(NUM_STORES):
        content = f"bench memory number {i}: Rust port timing harness"
        _, dt = server.call(
            "store_memory",
            {"content": content, "tags": [f"bench", f"i{i % 10}"]},
            base_id + i,
        )
        store_lats.append(dt)

    # Warm retrieve latency.
    retrieve_lats = []
    base_id = 10_000
    for i in range(NUM_RETRIEVES):
        _, dt = server.call(
            "retrieve_memory",
            {"query": "timing harness memory", "n_results": 5},
            base_id + i,
        )
        retrieve_lats.append(dt)

    rss = server.peak_rss_mb()
    server.close()

    def fmt(xs):
        return (
            f"p50={percentile(xs, 0.5) * 1000:5.1f}ms "
            f"p95={percentile(xs, 0.95) * 1000:5.1f}ms "
            f"min={min(xs) * 1000:5.1f}ms"
        )

    print(f"  store  x{NUM_STORES:<4}    : {fmt(store_lats)}")
    print(f"  retriev x{NUM_RETRIEVES:<4}    : {fmt(retrieve_lats)}")
    print(f"  peak RSS (live) : {rss:7.1f} MB")

    return {
        "cold_ms": cold_s * 1000,
        "store_p50_ms": percentile(store_lats, 0.5) * 1000,
        "store_p95_ms": percentile(store_lats, 0.95) * 1000,
        "retrieve_p50_ms": percentile(retrieve_lats, 0.5) * 1000,
        "retrieve_p95_ms": percentile(retrieve_lats, 0.95) * 1000,
        "rss_mb": rss,
    }


def main() -> int:
    if not RUST_BIN.exists():
        print("release binary missing — running `cargo build --release`",
              file=sys.stderr)
        subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT, check=True)

    with tempfile.TemporaryDirectory(prefix="mms-bench-") as d:
        rust_db = Path(d) / "rust.db"
        py_db = Path(d) / "py.db"

        rust_env = os.environ.copy()
        rust_env["MCP_MEMORY_DB_PATH"] = str(rust_db)

        py_env = os.environ.copy()
        py_env["MCP_MEMORY_DB_PATH"] = str(py_db)
        py_env["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
        py_env["PYTHONPATH"] = str(UPSTREAM / "src")

        rust = bench_server(
            "Rust port (release)",
            [str(RUST_BIN), "serve"],
            rust_env,
            rust_db,
        )
        py = bench_server(
            "Python upstream",
            [str(PY_BIN), "-m", "mcp_memory_service.server"],
            py_env,
            py_db,
        )

    def ratio(py_val, rs_val):
        if rs_val == 0:
            return float("inf")
        return py_val / rs_val

    print()
    print("| Metric            | Python          | Rust            | Ratio       |")
    print("|-------------------|-----------------|-----------------|-------------|")
    rows = [
        ("cold-start",      f"{py['cold_ms']:.1f} ms", f"{rust['cold_ms']:.1f} ms", f"{ratio(py['cold_ms'], rust['cold_ms']):.1f}x faster"),
        ("store p50",       f"{py['store_p50_ms']:.1f} ms", f"{rust['store_p50_ms']:.1f} ms", f"{ratio(py['store_p50_ms'], rust['store_p50_ms']):.1f}x"),
        ("store p95",       f"{py['store_p95_ms']:.1f} ms", f"{rust['store_p95_ms']:.1f} ms", f"{ratio(py['store_p95_ms'], rust['store_p95_ms']):.1f}x"),
        ("retrieve p50",    f"{py['retrieve_p50_ms']:.1f} ms", f"{rust['retrieve_p50_ms']:.1f} ms", f"{ratio(py['retrieve_p50_ms'], rust['retrieve_p50_ms']):.1f}x"),
        ("retrieve p95",    f"{py['retrieve_p95_ms']:.1f} ms", f"{rust['retrieve_p95_ms']:.1f} ms", f"{ratio(py['retrieve_p95_ms'], rust['retrieve_p95_ms']):.1f}x"),
        ("RSS (live)",      f"{py['rss_mb']:.1f} MB", f"{rust['rss_mb']:.1f} MB", f"{ratio(py['rss_mb'], rust['rss_mb']):.1f}x smaller"),
    ]
    for name, a, b, r in rows:
        print(f"| {name:<17} | {a:<15} | {b:<15} | {r:<11} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
