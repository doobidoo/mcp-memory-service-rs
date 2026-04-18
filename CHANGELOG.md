# Changelog

All notable changes to this project are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

**What counts as what** (ports-specific):

- **MAJOR** (x.0.0) — breaking change to one of: the SQLite schema (incompatible with DBs written by prior versions or by the Python upstream at the time of release), the MCP tool input shapes, or the MCP tool output shapes. Anything that would silently break a client or corrupt a shared DB.
- **MINOR** (0.x.0) — new MCP tool, new optional field, new config env var, or new supported backend. Backward-compatible.
- **PATCH** (0.0.x) — bug fix, performance improvement, documentation, or internal refactor. No wire/schema changes.

A changelog entry lands in `## [Unreleased]` on every functional PR. When we tag, the unreleased section is renamed to the new version, a fresh empty Unreleased is added on top, and the tag is pushed with the same notes copied into the GitHub release.

## [Unreleased]

### Added

- Profile instrumentation for `retrieve_memory` gated behind `MMS_PROFILE=1`. Prints per-phase timing (`embed` / `lock` / `knn`) to stderr. Zero-cost when unset.

## [0.1.1] — 2026-04-18

### Changed

- **Retrieve performance.** Cut `retrieve_memory` p50 from 22.6 ms to 8.8 ms (upstream Python is 7.0 ms for reference). Two knobs:
  - ONNX Runtime `intra_threads` pinned to 4 instead of `available_parallelism()`. On 10-core chips the dispatch overhead for single-sentence embeds was dominating the computation.
  - Conditional KNN oversampling in `storage::knn_search`. Previously always requested `3 × n_results` from sqlite-vec to protect against tombstoned matches; now probes `memories WHERE deleted_at IS NOT NULL LIMIT 1` (index-backed) and only oversamples when soft-deletes exist.

### Fixed

- README performance table and GitHub release notes now reflect the real numbers. The v0.1.0 release mentioned the retrieve regression; v0.1.1 actually resolves it.

## [0.1.0] — 2026-04-18

First public release. All 8 M0–M3 MCP tools wired end-to-end over stdio against a SQLite file that is schema-compatible with the Python upstream.

### Added

- **MCP tools** (stdio transport): `ping`, `store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory` (soft-delete), `list_memories` (paginated), `check_database_health`, `get_cache_stats`.
- **Storage** matches upstream `SqliteVecMemoryStorage` exactly: same `memories` table, same `memory_embeddings` vec0 virtual table with `FLOAT[384] distance_metric=cosine`, same indexes (`idx_content_hash`, `idx_created_at`, `idx_memory_type`, `idx_deleted_at`), same WAL pragmas.
- **ONNX embeddings** via `ort 2.0.0-rc.10` with `download-binaries` (self-contained static link, no system ONNX Runtime required). Auto-downloads `all-MiniLM-L6-v2` from the Chroma S3 bucket into `~/.cache/mcp_memory/onnx_models/` — same path as Python, so the cache is shared. SHA256-verified, tar-extracted with path-traversal guard.
- **External embedding backend** via `MCP_EMBEDDING_BACKEND=external`. POSTs to any OpenAI-style `/v1/embeddings` endpoint (vLLM, Ollama, TEI, OpenAI itself). Embeddings L2-normalized locally and dim-checked against schema.
- **Config surface:** `MCP_MEMORY_DB_PATH`, `MCP_MEMORY_SQLITE_PRAGMAS`, `MCP_EMBEDDING_BACKEND`, `MCP_EXTERNAL_EMBEDDING_API_URL`, `MCP_EXTERNAL_EMBEDDING_API_KEY`, `MCP_EXTERNAL_EMBEDDING_MODEL`.
- **CLI subcommands:** `serve` (default, stdio MCP), `verify --db <path>` (schema + sqlite-vec smoke test), `embed <text>` (full embedding pipeline smoke test).
- **Parity test suites:**
  - `scripts/parity_check.py` — embedding-level: 4/4 fixtures agree with upstream to 4 decimals.
  - `scripts/m4_parity.py` — storage-level: 23/23 assertions across tool shape, counts, ordering, tag case sensitivity, delete dry-run semantics, soft-delete filter, health fields.
- **Performance bench:** `scripts/bench.py` reports cold-start, store/retrieve p50/p95, and RSS for both binaries against fresh temp DBs.
- **CI** via `.github/workflows/ci.yml`: `cargo fmt --check`, `cargo clippy`, `cargo build --release`, `cargo test`, verify subcommand smoke test on ubuntu + macOS, plus the two parity suites against a freshly cloned upstream.
- **License:** PolyForm Noncommercial 1.0.0 for noncommercial use; separate commercial license for commercial use (see `COMMERCIAL.md`). Upstream Python project stays Apache-2.0 and is unaffected.

### Known issues

- `retrieve_memory` p50 is 8.8 ms vs upstream Python's 7.0 ms (20% slower). Profile shows embed dominates (>99% of wall time); KNN itself is 160 µs. Further optimization would need changes at the ONNX graph level — out of scope for this release.

### Not ported (by design)

Consolidation, document harvest/ingestion, quality scoring, graph storage, Cloudflare backend, hybrid sync, web dashboard, mDNS discovery, backup scheduler, HTTP/SSE transport. See `SPEC.md §2`.

[Unreleased]: https://github.com/doobidoo/mcp-memory-service-rs/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/doobidoo/mcp-memory-service-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/doobidoo/mcp-memory-service-rs/releases/tag/v0.1.0
