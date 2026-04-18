# mcp-memory-service-rs

Schema-compatible Rust drop-in for the 7 core tools of [doobidoo/mcp-memory-service](https://github.com/doobidoo/mcp-memory-service).

**Status:** M0 scaffold landed. Boots over stdio MCP, opens the SQLite DB, applies the shared schema, and answers `ping`. Real memory tools (`store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory`, `list_memories`, `check_database_health`, `get_cache_stats`) arrive in M1–M3.

## Goal

Replace the Python server with a Rust binary that:
- Shares the same SQLite file (swap the binary, keep the data).
- Starts in ~50 ms instead of ~500 ms.
- Runs in ~50 MB RAM instead of ~300 MB.
- Ships as a single file — no venv, no pip.

## What's in / what's out

**In:** the 7 MCP tools listed above, ONNX embeddings (`all-MiniLM-L6-v2`), external-API embedding fallback, soft-delete, tag GLOB search, pagination.

**Out (Python-only, by design):** consolidation, harvest, quality scoring, graph storage, Cloudflare backend, hybrid sync, web dashboard, mDNS discovery, backup scheduler, HTTP/SSE transport.

Full design rationale + wire-level behavior contract: [SPEC.md](SPEC.md).

## Try it

```bash
# Build (first run pulls ~235 crates)
cargo build --release

# Smoke-test DB open + schema + sqlite-vec load
./target/release/mcp-memory-service-rs verify --db /tmp/mms.db

# Run the MCP server over stdio (plug into your MCP client config)
./target/release/mcp-memory-service-rs serve
```

## Benchmarks

Sequential stdio MCP workload, 100 stores + 100 retrieves each, fresh DB per run, ONNX model already cached on disk. Measured on an Apple Silicon Mac, single client, no concurrency.

| Metric        | Python upstream | Rust port   | Ratio            |
|---------------|-----------------|-------------|------------------|
| cold-start    | 4475 ms         | **123 ms**  | **36× faster**   |
| RSS (live)    | 576 MB          | **242 MB**  | **2.4× smaller** |
| store p50     | 13.7 ms         | 12.5 ms     | 1.1× faster      |
| store p95     | 43.0 ms         | 29.5 ms     | 1.5× faster      |
| retrieve p50  | **6.8 ms**      | 22.6 ms     | 0.3× (Rust slower) |
| retrieve p95  | **8.5 ms**      | 36.4 ms     | 0.2× (Rust slower) |

The retrieve regression is tracked — current suspect is the 3× KNN oversampling we do to guarantee `n_results` alive rows even when some matches are soft-deleted. Reproduce locally with `scripts/bench.py`.

## Parity with upstream

Two test suites verify Rust ↔ Python compatibility:

- `scripts/parity_check.py` — **4/4** embedding fixtures agree to 4 decimals, L2 norm = 1.0 on both sides.
- `scripts/m4_parity.py` — **23/23** storage assertions pass: list / search / retrieve ordering / pagination / tag case sensitivity / delete dry-run / soft-delete filter / health fields against a fresh DB populated by the Python writer.

Both suites run on every push via GitHub Actions (`.github/workflows/ci.yml`).

## License

This Rust port is dual-licensed:

- **Noncommercial use** — free under the [PolyForm Noncommercial License 1.0.0](LICENSE). Personal projects, research, academic and hobby work, and use by noncommercial organizations are covered. No further permission needed.
- **Commercial use** — requires a separate commercial license. See [COMMERCIAL.md](COMMERCIAL.md) for scope and contact. Email `henry.krupp@gmail.com`.

The upstream Python project [`doobidoo/mcp-memory-service`](https://github.com/doobidoo/mcp-memory-service) remains under Apache-2.0 — this port's license choice does not affect it.
