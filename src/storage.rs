use std::path::Path;
use std::sync::Once;

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::embeddings::EMBEDDING_DIM;
use crate::error::{AppError, Result};

/// Row-level representation of a stored memory. Matches SPEC §5 1:1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRow {
    pub content: String,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: f64,
    pub created_at_iso: String,
    pub updated_at: f64,
    pub updated_at_iso: String,
}

static VEC_INIT: Once = Once::new();

/// Register the sqlite-vec extension as an auto-extension so every new
/// connection loads vec0 transparently. Safe to call more than once —
/// the Once guard enforces single registration.
pub fn register_sqlite_vec() {
    VEC_INIT.call_once(|| {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

pub fn open(config: &Config) -> Result<Connection> {
    register_sqlite_vec();

    if let Some(parent) = config.db_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let conn = Connection::open(&config.db_path)?;
    apply_pragmas(&conn, config)?;
    apply_schema(&conn)?;
    Ok(conn)
}

fn apply_pragmas(conn: &Connection, config: &Config) -> Result<()> {
    // Defaults per SPEC §4. Env override wins for each key.
    let mut pragmas: Vec<(&str, String)> = vec![
        ("journal_mode", "WAL".to_string()),
        ("busy_timeout", "5000".to_string()),
        ("synchronous", "NORMAL".to_string()),
        ("cache_size", "10000".to_string()),
        ("temp_store", "MEMORY".to_string()),
    ];

    for (k, v) in &config.sqlite_pragmas {
        if let Some(slot) = pragmas.iter_mut().find(|(kk, _)| kk.eq_ignore_ascii_case(k)) {
            slot.1 = v.clone();
        } else {
            pragmas.push((Box::leak(k.clone().into_boxed_str()), v.clone()));
        }
    }

    for (k, v) in &pragmas {
        let sql = format!("PRAGMA {k} = {v};");
        conn.execute_batch(&sql)?;
    }
    Ok(())
}

// Schema MUST match the upstream Python `SqliteVecMemoryStorage` exactly so
// both backends can share a DB file. Column names, virtual-table shape, and
// indexes are all copied verbatim from
// src/mcp_memory_service/storage/sqlite_vec.py.
const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash    TEXT UNIQUE NOT NULL,
    content         TEXT NOT NULL,
    tags            TEXT,
    memory_type     TEXT,
    metadata        TEXT,
    created_at      REAL,
    updated_at      REAL,
    created_at_iso  TEXT,
    updated_at_iso  TEXT,
    deleted_at      REAL DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_created_at  ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_deleted_at  ON memories(deleted_at);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    content_embedding FLOAT[384] distance_metric=cosine
);
"#;

fn apply_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(SCHEMA_SQL)
        .map_err(|e| AppError::Schema(format!("failed to apply schema: {e}")))?;
    Ok(())
}

/// Count undeleted memories. Cheap sanity query used by `verify`.
pub fn count_memories(conn: &Connection) -> Result<i64> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL",
        [],
        |row| row.get(0),
    )?;
    Ok(n)
}

/// Verify the sqlite-vec extension is loaded and functional.
pub fn vec_version(conn: &Connection) -> Result<String> {
    let v: String = conn.query_row("SELECT vec_version()", [], |row| row.get(0))?;
    Ok(v)
}

pub fn db_size_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Size of the `-wal` sidecar file, if present. Returns 0 when WAL mode is
/// inactive or the file has been checkpointed away.
pub fn wal_size_bytes(db_path: &Path) -> u64 {
    let mut wal = db_path.as_os_str().to_os_string();
    wal.push("-wal");
    std::fs::metadata(Path::new(&wal))
        .map(|m| m.len())
        .unwrap_or(0)
}

pub fn total_memories_alive(conn: &Connection) -> Result<i64> {
    count_memories(conn)
}

/// Row count including soft-deleted entries — useful for debugging.
pub fn total_memories_including_deleted(conn: &Connection) -> Result<i64> {
    let n: i64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
    Ok(n)
}

/// SHA256 hex of content — deterministic across Python/Rust.
pub fn content_hash(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(content.as_bytes());
    hex::encode(h.finalize())
}

fn now_epoch_iso() -> (f64, String) {
    let now = chrono::Utc::now();
    let epoch = now.timestamp_millis() as f64 / 1000.0;
    let iso = now.to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
    (epoch, iso)
}

/// Insert a memory + its embedding. Idempotent: if the content_hash already
/// exists (alive or soft-deleted), returns `Ok(None)` without mutating.
pub fn insert_memory(
    conn: &Connection,
    row: &MemoryRow,
    embedding: &[f32],
) -> Result<Option<i64>> {
    if embedding.len() != EMBEDDING_DIM {
        return Err(AppError::Schema(format!(
            "embedding dim mismatch: got {}, expected {EMBEDDING_DIM}",
            embedding.len()
        )));
    }

    // Dedup check matches Python UNIQUE-constraint semantics: if a row with
    // this content_hash exists (alive or tombstoned), we do nothing.
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM memories WHERE content_hash = ?1",
            params![row.content_hash],
            |r| r.get(0),
        )
        .ok();
    if existing.is_some() {
        return Ok(None);
    }

    let tags_joined = row.tags.join(",");
    let metadata_json = serde_json::to_string(&row.metadata)
        .map_err(|e| AppError::Schema(format!("serialize metadata: {e}")))?;

    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "INSERT INTO memories (
            content_hash, content, tags, memory_type, metadata,
            created_at, updated_at, created_at_iso, updated_at_iso, deleted_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, NULL)",
        params![
            row.content_hash,
            row.content,
            tags_joined,
            row.memory_type,
            metadata_json,
            row.created_at,
            row.updated_at,
            row.created_at_iso,
            row.updated_at_iso,
        ],
    )?;
    let mem_id: i64 = tx.last_insert_rowid();

    let embedding_bytes = bytemuck_f32_slice_as_bytes(embedding);
    // Python stores vectors under `memories.id` as `rowid` — we must do the
    // same or the JOIN in knn_search won't find anything. `memories.id` is an
    // INTEGER PRIMARY KEY AUTOINCREMENT, which SQLite aliases to rowid.
    tx.execute(
        "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?1, ?2)",
        params![mem_id, embedding_bytes],
    )?;
    tx.commit()?;
    Ok(Some(mem_id))
}

fn bytemuck_f32_slice_as_bytes(v: &[f32]) -> Vec<u8> {
    // f32 → little-endian bytes. sqlite-vec expects packed f32[].
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Build a MemoryRow with fresh timestamps. `content_hash` derived from content.
pub fn new_memory_row(
    content: String,
    tags: Vec<String>,
    memory_type: Option<String>,
    metadata: serde_json::Value,
) -> MemoryRow {
    let (now_epoch, now_iso) = now_epoch_iso();
    let hash = content_hash(&content);
    MemoryRow {
        content,
        content_hash: hash,
        tags,
        memory_type,
        metadata,
        created_at: now_epoch,
        created_at_iso: now_iso.clone(),
        updated_at: now_epoch,
        updated_at_iso: now_iso,
    }
}

fn row_from_sqlite(r: &rusqlite::Row) -> rusqlite::Result<MemoryRow> {
    let tags_raw: Option<String> = r.get("tags")?;
    let tags: Vec<String> = tags_raw
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let metadata_raw: Option<String> = r.get("metadata")?;
    let metadata = match metadata_raw {
        Some(s) if !s.is_empty() => {
            serde_json::from_str(&s).unwrap_or(serde_json::Value::Object(Default::default()))
        }
        _ => serde_json::Value::Object(Default::default()),
    };
    Ok(MemoryRow {
        content: r.get("content")?,
        content_hash: r.get("content_hash")?,
        tags,
        memory_type: r.get("memory_type")?,
        metadata,
        created_at: r.get::<_, Option<f64>>("created_at")?.unwrap_or(0.0),
        created_at_iso: r.get::<_, Option<String>>("created_at_iso")?.unwrap_or_default(),
        updated_at: r.get::<_, Option<f64>>("updated_at")?.unwrap_or(0.0),
        updated_at_iso: r.get::<_, Option<String>>("updated_at_iso")?.unwrap_or_default(),
    })
}

/// KNN query: return top-N memories by cosine similarity to the query vector.
/// Cosine similarity = 1 - (cosine distance / 2) is NOT the right formula for
/// sqlite-vec default — it returns L2 distance for unit vectors. Since our
/// embeddings are L2-normalized, cosine_similarity = 1 - (l2_dist^2) / 2.
/// We return the similarity mapped to [0, 1] as Python does.
pub fn knn_search(
    conn: &Connection,
    query_embedding: &[f32],
    n_results: usize,
) -> Result<Vec<(MemoryRow, f32)>> {
    if query_embedding.len() != EMBEDDING_DIM {
        return Err(AppError::Schema(format!(
            "query embedding dim mismatch: got {}, expected {EMBEDDING_DIM}",
            query_embedding.len()
        )));
    }
    let q_bytes = bytemuck_f32_slice_as_bytes(query_embedding);
    // We declared the vec0 table with `distance_metric=cosine`, so the
    // returned `distance` is cosine distance (1 - cos) directly. Similarity
    // is simply `1 - distance`. Oversample 3x then truncate after the
    // soft-delete filter so KNN never gives us fewer than n_results alive
    // rows when some matches are tombstoned.
    let mut stmt = conn.prepare(
        "SELECT m.content, m.content_hash, m.tags, m.memory_type, m.metadata,
                m.created_at, m.created_at_iso, m.updated_at, m.updated_at_iso,
                me.distance
           FROM memory_embeddings me
           JOIN memories m ON m.rowid = me.rowid
          WHERE me.content_embedding MATCH ?1
            AND m.deleted_at IS NULL
            AND k = ?2
          ORDER BY me.distance",
    )?;
    let oversample = (n_results as i64).saturating_mul(3).max(n_results as i64);
    let rows = stmt
        .query_map(params![q_bytes, oversample], |r| {
            let distance: f32 = r.get("distance")?;
            let mem = row_from_sqlite(r)?;
            Ok((mem, distance))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let out = rows
        .into_iter()
        .take(n_results)
        .map(|(m, d)| (m, 1.0 - d))
        .collect();
    Ok(out)
}

/// How multi-tag queries combine.
#[derive(Debug, Clone, Copy)]
pub enum TagMatch {
    Any,
    All,
}

impl TagMatch {
    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("all") {
            TagMatch::All
        } else {
            TagMatch::Any
        }
    }
}

/// Build a `(...)` SQL fragment that realizes Python's tag-GLOB semantics:
///   (',' || REPLACE(tags, ' ', '') || ',') GLOB '*,<tag>,*'
/// ANDed or ORed per `mode`. Returns the fragment and bind values. The
/// caller is responsible for splicing the fragment into its own WHERE.
fn tag_where_clause(tags: &[String], mode: TagMatch) -> (String, Vec<String>) {
    if tags.is_empty() {
        return ("1=1".into(), Vec::new());
    }
    let op = match mode {
        TagMatch::Any => " OR ",
        TagMatch::All => " AND ",
    };
    let clauses: Vec<&str> = tags
        .iter()
        .map(|_| "(',' || REPLACE(tags, ' ', '') || ',') GLOB ?")
        .collect();
    let patterns: Vec<String> = tags.iter().map(|t| format!("*,{},*", t.trim())).collect();
    (format!("({})", clauses.join(op)), patterns)
}

/// Tag-filtered search, soft-delete aware. Case-sensitive exact-tag matching.
pub fn search_by_tag(
    conn: &Connection,
    tags: &[String],
    mode: TagMatch,
    memory_type: Option<&str>,
) -> Result<Vec<MemoryRow>> {
    let (tag_clause, tag_binds) = tag_where_clause(tags, mode);
    let mut sql = format!(
        "SELECT content, content_hash, tags, memory_type, metadata,
                created_at, created_at_iso, updated_at, updated_at_iso
           FROM memories
          WHERE deleted_at IS NULL
            AND {tag_clause}"
    );
    if memory_type.is_some() {
        sql.push_str(" AND memory_type = ?");
    }
    sql.push_str(" ORDER BY created_at DESC");

    let mut stmt = conn.prepare(&sql)?;
    let mut binds: Vec<Box<dyn rusqlite::ToSql>> = tag_binds
        .into_iter()
        .map(|s| Box::new(s) as Box<dyn rusqlite::ToSql>)
        .collect();
    if let Some(mt) = memory_type {
        binds.push(Box::new(mt.to_string()));
    }
    let bind_refs: Vec<&dyn rusqlite::ToSql> = binds.iter().map(|b| b.as_ref()).collect();
    let rows = stmt
        .query_map(bind_refs.as_slice(), row_from_sqlite)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Accepts an ISO8601 string or a float-as-string and returns the epoch. Used
/// for the `before` / `after` filters on delete + retrieval queries so that
/// clients can pass either format, matching Python's permissive behavior.
pub fn parse_time(raw: &str) -> Result<f64> {
    if let Ok(v) = raw.parse::<f64>() {
        return Ok(v);
    }
    chrono::DateTime::parse_from_rfc3339(raw)
        .map(|d| d.timestamp_millis() as f64 / 1000.0)
        .map_err(|e| AppError::Schema(format!("could not parse timestamp {raw:?}: {e}")))
}

/// Filter spec for `delete_memory` — mirrors the Python tool signature.
#[derive(Debug, Default)]
pub struct DeleteFilter {
    pub content_hash: Option<String>,
    pub tags: Vec<String>,
    pub tag_match: TagMatch,
    pub before: Option<f64>,
    pub after: Option<f64>,
    pub memory_type: Option<String>,
}

impl Default for TagMatch {
    fn default() -> Self {
        TagMatch::Any
    }
}

fn build_delete_where(filter: &DeleteFilter) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
    let mut parts: Vec<String> = vec!["deleted_at IS NULL".into()];
    let mut binds: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(hash) = &filter.content_hash {
        parts.push("content_hash = ?".into());
        binds.push(Box::new(hash.clone()));
    }
    if !filter.tags.is_empty() {
        let (clause, tag_binds) = tag_where_clause(&filter.tags, filter.tag_match);
        parts.push(clause);
        for t in tag_binds {
            binds.push(Box::new(t));
        }
    }
    if let Some(b) = filter.before {
        parts.push("created_at < ?".into());
        binds.push(Box::new(b));
    }
    if let Some(a) = filter.after {
        parts.push("created_at > ?".into());
        binds.push(Box::new(a));
    }
    if let Some(mt) = &filter.memory_type {
        parts.push("memory_type = ?".into());
        binds.push(Box::new(mt.clone()));
    }
    (parts.join(" AND "), binds)
}

/// Soft-delete every matching memory (unless `dry_run`). Returns the content
/// hashes that were (would have been) affected. Setting `deleted_at` — no
/// physical `DELETE FROM` — so data is recoverable.
pub fn soft_delete(conn: &Connection, filter: &DeleteFilter, dry_run: bool) -> Result<Vec<String>> {
    let (where_clause, binds) = build_delete_where(filter);
    let bind_refs: Vec<&dyn rusqlite::ToSql> = binds.iter().map(|b| b.as_ref()).collect();

    let select_sql = format!("SELECT content_hash FROM memories WHERE {where_clause}");
    let mut stmt = conn.prepare(&select_sql)?;
    let hashes: Vec<String> = stmt
        .query_map(bind_refs.as_slice(), |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    if dry_run || hashes.is_empty() {
        return Ok(hashes);
    }

    let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
    let update_sql = format!("UPDATE memories SET deleted_at = ? WHERE {where_clause}");
    let mut update_binds: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(now)];
    update_binds.extend(binds);
    let update_refs: Vec<&dyn rusqlite::ToSql> =
        update_binds.iter().map(|b| b.as_ref()).collect();
    conn.execute(&update_sql, update_refs.as_slice())?;
    Ok(hashes)
}

/// Paginated listing with optional tag + memory_type filters. The `tag`
/// argument is singular and — per SPEC §11-Q4 — reuses the exact GLOB
/// semantics of `search_by_tag` with a single-element vec in `any` mode.
pub fn list_memories(
    conn: &Connection,
    page: i64,
    page_size: i64,
    tag: Option<&str>,
    memory_type: Option<&str>,
) -> Result<(Vec<MemoryRow>, i64)> {
    let page = page.max(1);
    let page_size = page_size.max(1);

    let mut where_parts: Vec<String> = vec!["deleted_at IS NULL".into()];
    let mut binds: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(t) = tag {
        let (clause, tag_binds) = tag_where_clause(&[t.to_string()], TagMatch::Any);
        where_parts.push(clause);
        for p in tag_binds {
            binds.push(Box::new(p));
        }
    }
    if let Some(mt) = memory_type {
        where_parts.push("memory_type = ?".into());
        binds.push(Box::new(mt.to_string()));
    }
    let where_clause = where_parts.join(" AND ");

    let count_sql = format!("SELECT COUNT(*) FROM memories WHERE {where_clause}");
    let bind_refs: Vec<&dyn rusqlite::ToSql> = binds.iter().map(|b| b.as_ref()).collect();
    let total: i64 = conn.query_row(&count_sql, bind_refs.as_slice(), |r| r.get(0))?;

    let list_sql = format!(
        "SELECT content, content_hash, tags, memory_type, metadata,
                created_at, created_at_iso, updated_at, updated_at_iso
           FROM memories
          WHERE {where_clause}
          ORDER BY created_at DESC
          LIMIT ? OFFSET ?"
    );
    let offset = (page - 1) * page_size;
    let mut all_binds: Vec<Box<dyn rusqlite::ToSql>> = binds;
    all_binds.push(Box::new(page_size));
    all_binds.push(Box::new(offset));
    let all_refs: Vec<&dyn rusqlite::ToSql> = all_binds.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&list_sql)?;
    let rows = stmt
        .query_map(all_refs.as_slice(), row_from_sqlite)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok((rows, total))
}

