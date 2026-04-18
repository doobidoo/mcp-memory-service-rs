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

Sequential stdio MCP workload, 100 stores + 100 retrieves each, fresh DB per run, ONNX model + weights already warm on disk. Measured on an Apple Silicon Mac, single client, no concurrency.

| Metric        | Python upstream | Rust port    | Ratio             |
|---------------|-----------------|--------------|-------------------|
| cold-start    | 3619 ms         | **68 ms**    | **53× faster**    |
| RSS (live)    | 561 MB          | **241 MB**   | **2.3× smaller**  |
| store p50     | 10.3 ms         | **8.5 ms**   | 1.2× faster       |
| store p95     | 11.5 ms         | **8.7 ms**   | 1.3× faster       |
| retrieve p50  | **7.0 ms**      | 8.8 ms       | 0.8× (20% slower) |
| retrieve p95  | **7.7 ms**      | 9.2 ms       | 0.8×              |

Rust wins decisively on cold-start and memory footprint and matches or beats Python on writes. Retrieve is ~20% slower per-call but p95 is tighter (more predictable). Reproduce locally with `scripts/bench.py`.

Two key tunings got us here:

- **Conditional KNN oversampling.** The original code always asked sqlite-vec for `3 × n_results` rows to protect against tombstoned matches. That doubled per-call latency on fresh DBs. We now probe `memories.deleted_at IS NOT NULL LIMIT 1` (index-backed, microseconds) and only oversample when soft-deletes exist.
- **`intra_threads = 4`** for the ONNX Runtime session. The default `available_parallelism()` spawns one worker per core, which on 10-core Apple Silicon spends more time dispatching than computing for single-sentence embeddings. Forcing 4 threads matches typical Python defaults and cut retrieve p50 from 22 ms to 8 ms.

## Parity with upstream

Two test suites verify Rust ↔ Python compatibility:

- `scripts/parity_check.py` — **4/4** embedding fixtures agree to 4 decimals, L2 norm = 1.0 on both sides.
- `scripts/m4_parity.py` — **23/23** storage assertions pass: list / search / retrieve ordering / pagination / tag case sensitivity / delete dry-run / soft-delete filter / health fields against a fresh DB populated by the Python writer.

Both suites run on every push via GitHub Actions (`.github/workflows/ci.yml`).

## Versioning and releases

This project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). What counts as what is defined in [CHANGELOG.md](CHANGELOG.md) — briefly: MAJOR on schema or wire-shape breakage, MINOR on new tools or optional fields, PATCH on fixes and perf work. Every release has a corresponding entry in the changelog and a GitHub release with the same notes.

## License

This Rust port is dual-licensed:

- **Noncommercial use** — free under the [PolyForm Noncommercial License 1.0.0](LICENSE). Personal projects, research, academic and hobby work, and use by noncommercial organizations are covered. No further permission needed.
- **Commercial use** — requires a separate commercial license. See [COMMERCIAL.md](COMMERCIAL.md) for scope and contact. Email `henry.krupp@gmail.com`.

The upstream Python project [`doobidoo/mcp-memory-service`](https://github.com/doobidoo/mcp-memory-service) remains under Apache-2.0 — this port's license choice does not affect it.
