use std::path::Path;
use std::sync::Once;

use rusqlite::Connection;

use crate::config::Config;
use crate::error::{AppError, Result};

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
