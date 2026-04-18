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

## License

This Rust port is dual-licensed:

- **Noncommercial use** — free under the [PolyForm Noncommercial License 1.0.0](LICENSE). Personal projects, research, academic and hobby work, and use by noncommercial organizations are covered. No further permission needed.
- **Commercial use** — requires a separate commercial license. See [COMMERCIAL.md](COMMERCIAL.md) for scope and contact. Email `henry.krupp@gmail.com`.

The upstream Python project [`doobidoo/mcp-memory-service`](https://github.com/doobidoo/mcp-memory-service) remains under Apache-2.0 — this port's license choice does not affect it.
