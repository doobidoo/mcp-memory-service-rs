//! Lightweight process-local counters exposed via `get_cache_stats`.
//!
//! The Python server's cache_stats shape is built around a module-level
//! cache that survives between stateless HTTP calls. Rust has no such
//! cache because the process *is* the singleton — but we still emit the
//! same JSON shape (see SPEC §11-Q5) filled with honest Rust-native
//! numbers so clients that grep the Python field names keep working.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct CacheStats {
    /// Process start time in unix ms. Set once at construction.
    pub started_at_ms: AtomicI64,
    /// Time spent loading the embedding model, in milliseconds.
    pub embed_load_ms: AtomicU64,
    /// Time spent opening the DB + applying schema, in milliseconds.
    pub db_init_ms: AtomicU64,

    /// Total tool invocations.
    pub total_calls: AtomicU64,
    pub store_count: AtomicU64,
    pub retrieve_count: AtomicU64,
    pub search_count: AtomicU64,
    pub delete_count: AtomicU64,
    pub list_count: AtomicU64,
    pub health_count: AtomicU64,
    pub stats_count: AtomicU64,
    pub ping_count: AtomicU64,
}

impl CacheStats {
    pub fn new() -> Self {
        let s = Self::default();
        s.started_at_ms.store(
            chrono::Utc::now().timestamp_millis(),
            Ordering::Relaxed,
        );
        s
    }

    pub fn set_embed_load_ms(&self, ms: u64) {
        self.embed_load_ms.store(ms, Ordering::Relaxed);
    }

    pub fn set_db_init_ms(&self, ms: u64) {
        self.db_init_ms.store(ms, Ordering::Relaxed);
    }

    pub fn record<F: Fn(&Self) -> &AtomicU64>(&self, field: F) {
        field(self).fetch_add(1, Ordering::Relaxed);
        self.total_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn uptime_seconds(&self) -> i64 {
        let now = chrono::Utc::now().timestamp_millis();
        let started = self.started_at_ms.load(Ordering::Relaxed);
        ((now - started) / 1000).max(0)
    }
}
