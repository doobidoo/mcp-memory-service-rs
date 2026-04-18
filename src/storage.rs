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

CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    content_hash TEXT PRIMARY KEY,
    embedding FLOAT[384]
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
    tx.execute(
        "INSERT INTO memory_embeddings (content_hash, embedding) VALUES (?1, ?2)",
        params![row.content_hash, embedding_bytes],
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
    let mut stmt = conn.prepare(
        "SELECT m.content, m.content_hash, m.tags, m.memory_type, m.metadata,
                m.created_at, m.created_at_iso, m.updated_at, m.updated_at_iso,
                e.distance
           FROM memory_embeddings e
           JOIN memories m ON m.content_hash = e.content_hash
          WHERE e.embedding MATCH ?1
            AND m.deleted_at IS NULL
            AND k = ?2
          ORDER BY e.distance",
    )?;
    let rows = stmt
        .query_map(params![q_bytes, n_results as i64], |r| {
            let distance: f32 = r.get("distance")?;
            let mem = row_from_sqlite(r)?;
            Ok((mem, distance))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    // sqlite-vec returns L2 distance on packed f32[]. For L2-normalized unit
    // vectors, |a - b|^2 = 2 - 2*cos(a,b), so cos = 1 - d^2/2. Range is
    // [-1, 1]. Python returns raw cosine (no clamp); match that so parity
    // tests against the upstream Python backend agree to 3-4 decimals.
    let out = rows
        .into_iter()
        .map(|(m, d)| (m, 1.0 - (d * d) / 2.0))
        .collect();
    Ok(out)
}

