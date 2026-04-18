#!/usr/bin/env python3
"""
M4 storage-level parity suite: Python upstream ↔ Rust port.

Populates a fresh temporary SQLite DB through the upstream Python
MemoryService (authoritative producer), then queries it through both
backends and diffs the JSON shapes. Fills the gap that the embedding-
only parity_check.py leaves open: does the Rust port read the Python
writer's data identically, and do the tag / delete / list / retrieve
semantics match to the level a real MCP client would notice?

Exit 0 on full parity, 1 on any mismatch.

Prereqs:
  - Rust binary built (release or debug).
  - Upstream repo cloned at ../mcp-memory-service with its venv and
    onnxruntime + tokenizers installed.

Usage:
    ../mcp-memory-service/.venv/bin/python scripts/m4_parity.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = Path(
    os.environ.get("MMS_UPSTREAM_PATH",
                   str(REPO_ROOT.parent / "mcp-memory-service"))
)
sys.path.insert(0, str(UPSTREAM / "src"))

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.services.memory_service import MemoryService

FIXTURES = [
    {"content": "Rust is a systems programming language",
     "tags": ["rust", "programming"], "memory_type": "note"},
    {"content": "Python is popular for data science",
     "tags": ["python", "ml"], "memory_type": "note"},
    {"content": "Bananas are yellow fruits",
     "tags": ["food"], "memory_type": "fact"},
    {"content": "The quick brown fox jumps over the lazy dog",
     "tags": ["animals", "fox"], "memory_type": "quote"},
    {"content": "Compiled languages produce native machine code",
     "tags": ["programming", "compilers"], "memory_type": "note"},
]

SCORE_TOL = 2e-3
RUST_BIN = REPO_ROOT / "target" / "debug" / "mcp-memory-service-rs"


def find_rust_binary() -> Path:
    for variant in ("release", "debug"):
        p = REPO_ROOT / "target" / variant / "mcp-memory-service-rs"
        if p.exists():
            return p
    print("no built binary found — running `cargo build`", file=sys.stderr)
    subprocess.run(["cargo", "build"], cwd=REPO_ROOT, check=True)
    return REPO_ROOT / "target" / "debug" / "mcp-memory-service-rs"


def call_rust(db: Path, tool: str, args: dict) -> Any:
    env = os.environ.copy()
    env["MCP_MEMORY_DB_PATH"] = str(db)
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "m4", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": tool, "arguments": args}},
    ]
    stdin = "\n".join(json.dumps(m) for m in msgs) + "\n"
    p = subprocess.run(
        [str(find_rust_binary()), "serve"],
        input=stdin, capture_output=True, text=True, env=env, timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(f"rust exit {p.returncode}\nstderr:\n{p.stderr}")
    for line in p.stdout.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") == 2 and "result" in r:
            return json.loads(r["result"]["content"][0]["text"])
    raise RuntimeError(f"no response for id=2 in rust stdout:\n{p.stdout}")


async def populate(db: Path) -> MemoryService:
    storage = SqliteVecMemoryStorage(str(db))
    await storage.initialize()
    service = MemoryService(storage)
    for f in FIXTURES:
        await service.store_memory(**f)
        # Ensure distinct created_at so ordering is stable.
        time.sleep(0.01)
    return service, storage


# ---------- test helpers ----------

PASSED, FAILED = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  OK   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name}")
        if detail:
            for line in detail.splitlines():
                print(f"       {line}")


def hashes(rows) -> list[str]:
    return [r.get("content_hash") or r.get("memory", {}).get("content_hash")
            for r in rows]


# ---------- test cases ----------

async def run_tests(db: Path, service: MemoryService):
    # T1 — list_memories full dump agrees on set of hashes, tags, content.
    rs = call_rust(db, "list_memories", {"page": 1, "page_size": 100})
    py = await service.list_memories(page=1, page_size=100)
    check("T1.count", rs["total"] == py["total"] == len(FIXTURES),
          f"rs={rs['total']} py={py['total']} expected={len(FIXTURES)}")
    rs_hashes = sorted(m["content_hash"] for m in rs["memories"])
    py_hashes = sorted(m["content_hash"] for m in py["memories"])
    check("T1.hash_set", rs_hashes == py_hashes,
          f"rs={rs_hashes[:2]}...\npy={py_hashes[:2]}...")
    for rs_m in rs["memories"]:
        py_m = next(x for x in py["memories"]
                    if x["content_hash"] == rs_m["content_hash"])
        check(f"T1.tags[{rs_m['content'][:20]}]",
              sorted(rs_m["tags"]) == sorted(py_m["tags"]),
              f"rs={rs_m['tags']} py={py_m['tags']}")

    # T2 — pagination math matches.
    check("T2.total_pages",
          rs["total_pages"] == py.get("total_pages",
                                      (py["total"] + 99) // 100),
          f"rs={rs['total_pages']} py={py.get('total_pages')}")
    check("T2.has_more",
          rs["has_more"] == py.get("has_more", False),
          f"rs={rs['has_more']} py={py.get('has_more')}")

    # T3 — search_by_tag ["rust"] returns the same 1 memory.
    rs = call_rust(db, "search_by_tag", {"tags": ["rust"]})
    py = await service.search_by_tag(["rust"])
    # Python returns dict with 'memories' key; Rust returns list of rows.
    py_list = py.get("memories", py) if isinstance(py, dict) else py
    check("T3.rust_tag_count", len(rs) == len(py_list),
          f"rs={len(rs)} py={len(py_list)}")
    check("T3.rust_tag_hashes",
          sorted(hashes(rs)) == sorted(hashes(py_list)),
          f"rs={hashes(rs)} py={hashes(py_list)}")

    # T4 — case sensitivity: "Rust" (capital R) should match nothing.
    rs = call_rust(db, "search_by_tag", {"tags": ["Rust"]})
    check("T4.case_sensitive_no_match", len(rs) == 0,
          f"expected 0 results, got {len(rs)}: {hashes(rs)}")

    # T5 — tag_match="all" with two tags that only one memory has.
    rs = call_rust(db, "search_by_tag",
                   {"tags": ["programming", "compilers"], "tag_match": "all"})
    check("T5.tag_match_all", len(rs) == 1,
          f"expected 1 ('Compiled languages...'), got {len(rs)}")

    # T6 — retrieve_memory ordering agrees on top 3.
    q = "compiled programming languages"
    rs = call_rust(db, "retrieve_memory", {"query": q, "n_results": 3})
    py = await service.retrieve_memories(q, n_results=3)
    py_list = py.get("memories", py) if isinstance(py, dict) else py
    rs_top3 = [r["memory"]["content_hash"] for r in rs]
    py_top3 = []
    for item in py_list[:3]:
        mem = item.get("memory") if isinstance(item, dict) and "memory" in item else item
        h = mem.get("content_hash") if hasattr(mem, "get") else getattr(mem, "content_hash", None)
        py_top3.append(h)
    check("T6.retrieve_top3_order", rs_top3 == py_top3,
          f"rs={rs_top3}\npy={py_top3}")

    # T7 — delete dry_run on tag 'food' matches one hash and does not mutate.
    rs = call_rust(db, "delete_memory", {"tags": ["food"], "dry_run": True})
    check("T7.dry_run_count", rs["deleted_count"] == 1,
          f"got {rs['deleted_count']}")
    check("T7.dry_run_respects_flag", rs["dry_run"] is True)
    # Verify nothing was actually deleted.
    rs_after = call_rust(db, "list_memories", {"page": 1, "page_size": 100})
    check("T7.dry_run_no_mutation",
          rs_after["total"] == len(FIXTURES),
          f"expected {len(FIXTURES)} memories after dry_run, got {rs_after['total']}")

    # T8 — real delete then re-check counts; soft-delete means row stays but
    # list_memories filters it out.
    rs_del = call_rust(db, "delete_memory", {"tags": ["food"]})
    check("T8.real_delete_count", rs_del["deleted_count"] == 1)
    rs_list = call_rust(db, "list_memories", {"page": 1, "page_size": 100})
    check("T8.soft_delete_filters_list",
          rs_list["total"] == len(FIXTURES) - 1,
          f"expected {len(FIXTURES)-1} alive after delete, got {rs_list['total']}")
    rs_health = call_rust(db, "check_database_health", {})
    check("T8.health_soft_deleted_count",
          rs_health["database_info"]["soft_deleted_count"] == 1,
          f"got {rs_health['database_info']['soft_deleted_count']}")

    # T9 — health tool exposes schema-level info matching defaults.
    check("T9.health_backend",
          rs_health["backend"] == "sqlite_vec_rs")
    check("T9.health_dim",
          rs_health["database_info"]["embedding_dim"] == 384)
    check("T9.health_model",
          rs_health["database_info"]["embedding_model"] == "all-MiniLM-L6-v2")


def main() -> int:
    global PASSED, FAILED
    PASSED = FAILED = 0
    binary = find_rust_binary()
    print(f"rust    : {binary.relative_to(REPO_ROOT)}")
    print(f"python  : upstream at {UPSTREAM}")
    print()

    with tempfile.TemporaryDirectory(prefix="mms-m4-") as d:
        db = Path(d) / "parity.db"
        service, storage = asyncio.run(populate(db))
        try:
            asyncio.run(run_tests(db, service))
        finally:
            asyncio.run(storage.close())

    print()
    print(f"result  : {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
