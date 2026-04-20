use std::sync::Arc;

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::io::stdio,
    ErrorData, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use std::sync::atomic::Ordering;

use crate::config::Config;
use crate::embeddings::Embedder;
use crate::stats::CacheStats;
use crate::storage::{self, DeleteFilter, MemoryRow, TagMatch};

pub struct MemoryServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    config: Arc<Config>,
    conn: Arc<Mutex<rusqlite::Connection>>,
    embedder: Arc<Mutex<Embedder>>,
    stats: Arc<CacheStats>,
}

// ---------- ping ----------

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct PingParams {}

#[derive(Debug, Serialize)]
pub struct PingResult {
    pub status: &'static str,
    pub backend: &'static str,
    pub vec_version: String,
    pub memory_count: i64,
}

// ---------- store_memory ----------

fn default_memory_type() -> String {
    "note".into()
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreMemoryParams {
    /// The memory content to store.
    pub content: String,
    /// Tags: either a comma-separated string or an array of strings.
    #[serde(default)]
    pub tags: Option<serde_json::Value>,
    #[serde(default = "default_memory_type")]
    pub memory_type: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct StoreMemorySuccess {
    pub success: bool,
    pub content_hash: String,
    pub duplicate: bool,
}

// ---------- retrieve_memory ----------

fn default_n_results() -> usize {
    10
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RetrieveMemoryParams {
    pub query: String,
    #[serde(default = "default_n_results")]
    pub n_results: usize,
}

#[derive(Debug, Serialize)]
pub struct RetrievedMemory {
    pub memory: MemoryRow,
    pub relevance_score: f32,
    pub debug_info: serde_json::Value,
}

// ---------- search_by_tag ----------

fn default_tag_match() -> String {
    "any".into()
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchByTagParams {
    pub tags: Vec<String>,
    #[serde(default = "default_tag_match")]
    pub tag_match: String,
    #[serde(default)]
    pub memory_type: Option<String>,
}

// ---------- delete_memory ----------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DeleteMemoryParams {
    #[serde(default)]
    pub content_hash: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default = "default_tag_match")]
    pub tag_match: String,
    #[serde(default)]
    pub before: Option<String>,
    #[serde(default)]
    pub after: Option<String>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
pub struct DeleteMemoryResult {
    pub success: bool,
    pub deleted_count: usize,
    pub deleted_hashes: Vec<String>,
    pub dry_run: bool,
}

// ---------- list_memories ----------

fn default_page() -> i64 {
    1
}
fn default_page_size() -> i64 {
    10
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListMemoriesParams {
    #[serde(default = "default_page")]
    pub page: i64,
    #[serde(default = "default_page_size")]
    pub page_size: i64,
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub memory_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListMemoriesResult {
    pub memories: Vec<MemoryRow>,
    pub total: i64,
    pub page: i64,
    pub page_size: i64,
    pub total_pages: i64,
    pub has_more: bool,
}

// ---------- check_database_health ----------

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct CheckHealthParams {}

// ---------- get_cache_stats ----------

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct GetCacheStatsParams {}

// ---------- impl ----------

#[tool_router]
impl MemoryServer {
    pub fn new(
        config: Arc<Config>,
        conn: rusqlite::Connection,
        embedder: Embedder,
        stats: Arc<CacheStats>,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            config,
            conn: Arc::new(Mutex::new(conn)),
            embedder: Arc::new(Mutex::new(embedder)),
            stats,
        }
    }

    #[tool(
        name = "ping",
        description = "Health probe: returns backend state and sqlite-vec version."
    )]
    async fn ping(
        &self,
        Parameters(_): Parameters<PingParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.ping_count);
        let conn = self.conn.lock().await;
        let vec_version = storage::vec_version(&conn).map_err(internal_err)?;
        let memory_count = storage::count_memories(&conn).map_err(internal_err)?;
        let payload = PingResult {
            status: "ok",
            backend: "sqlite_vec_rs",
            vec_version,
            memory_count,
        };
        json_result(&payload)
    }

    #[tool(
        name = "store_memory",
        description = "Store a memory with semantic indexing. Dedup by SHA256 content hash."
    )]
    async fn store_memory(
        &self,
        Parameters(p): Parameters<StoreMemoryParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.store_count);
        let tags = normalize_tags(p.tags.as_ref(), p.metadata.as_ref());
        let metadata = strip_tags_from_metadata(p.metadata.unwrap_or(serde_json::Value::Null));

        let row = storage::new_memory_row(p.content.clone(), tags, Some(p.memory_type), metadata);
        let embedding = {
            let mut emb = self.embedder.lock().await;
            emb.embed(&p.content).await.map_err(internal_err)?
        };

        let conn = self.conn.lock().await;
        let inserted = storage::insert_memory(&conn, &row, &embedding).map_err(internal_err)?;
        let payload = StoreMemorySuccess {
            success: true,
            content_hash: row.content_hash,
            duplicate: inserted.is_none(),
        };
        json_result(&payload)
    }

    #[tool(
        name = "retrieve_memory",
        description = "Semantic search: return top N memories by cosine similarity to query."
    )]
    async fn retrieve_memory(
        &self,
        Parameters(p): Parameters<RetrieveMemoryParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.retrieve_count);
        // Per-phase timing when MMS_PROFILE is set. Zero-cost when unset
        // (skipped by the Option branch). Left in because retrieve is the
        // metric we're most sensitive about vs the Python upstream — any
        // regression in this tool is a real concern.
        let profile_t0 = std::env::var("MMS_PROFILE")
            .is_ok()
            .then(std::time::Instant::now);
        let query_emb = {
            let mut emb = self.embedder.lock().await;
            emb.embed(&p.query).await.map_err(internal_err)?
        };
        let profile_t1 = profile_t0.as_ref().map(|_| std::time::Instant::now());
        let conn = self.conn.lock().await;
        let profile_t2 = profile_t0.as_ref().map(|_| std::time::Instant::now());
        let hits =
            storage::knn_search(&conn, &query_emb, p.n_results.max(1)).map_err(internal_err)?;
        if let (Some(t0), Some(t1), Some(t2)) = (profile_t0, profile_t1, profile_t2) {
            let t3 = std::time::Instant::now();
            eprintln!(
                "retrieve profile: embed={:?} lock={:?} knn={:?} total={:?}",
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t3 - t0
            );
        }
        let results: Vec<RetrievedMemory> = hits
            .into_iter()
            .map(|(mem, score)| RetrievedMemory {
                memory: mem,
                relevance_score: score,
                debug_info: serde_json::json!({"backend": "sqlite_vec_rs", "metric": "cosine"}),
            })
            .collect();
        json_result(&results)
    }

    #[tool(
        name = "search_by_tag",
        description = "Return memories whose tags match. Tag matching is case-sensitive; \
                       multi-tag queries use 'any' (OR) or 'all' (AND)."
    )]
    async fn search_by_tag(
        &self,
        Parameters(p): Parameters<SearchByTagParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.search_count);
        let mode = TagMatch::parse(&p.tag_match);
        let conn = self.conn.lock().await;
        let rows = storage::search_by_tag(&conn, &p.tags, mode, p.memory_type.as_deref())
            .map_err(internal_err)?;
        json_result(&rows)
    }

    #[tool(
        name = "delete_memory",
        description = "Soft-delete memories matching the given filter. Pass dry_run=true to \
                       preview. Supports content_hash, tags, memory_type, and before/after \
                       ISO8601 or epoch timestamps."
    )]
    async fn delete_memory(
        &self,
        Parameters(p): Parameters<DeleteMemoryParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.delete_count);
        let mode = TagMatch::parse(&p.tag_match);
        let before = p
            .before
            .as_deref()
            .map(storage::parse_time)
            .transpose()
            .map_err(internal_err)?;
        let after = p
            .after
            .as_deref()
            .map(storage::parse_time)
            .transpose()
            .map_err(internal_err)?;

        // If no selectors were provided, refuse — matches Python's safety
        // behavior of not mass-deleting on an empty filter.
        if p.content_hash.is_none()
            && p.tags.as_ref().map(|v| v.is_empty()).unwrap_or(true)
            && before.is_none()
            && after.is_none()
            && p.memory_type.is_none()
        {
            return Err(internal_err(
                "delete_memory requires at least one selector (content_hash, tags, \
                 memory_type, before, or after)",
            ));
        }

        let filter = DeleteFilter {
            content_hash: p.content_hash,
            tags: p.tags.unwrap_or_default(),
            tag_match: mode,
            before,
            after,
            memory_type: p.memory_type,
        };
        let conn = self.conn.lock().await;
        let hashes = storage::soft_delete(&conn, &filter, p.dry_run).map_err(internal_err)?;
        let result = DeleteMemoryResult {
            success: true,
            deleted_count: hashes.len(),
            deleted_hashes: hashes,
            dry_run: p.dry_run,
        };
        json_result(&result)
    }

    #[tool(
        name = "list_memories",
        description = "Paginated listing of memories, ordered newest-first, with optional \
                       single-tag filter and memory_type filter."
    )]
    async fn list_memories(
        &self,
        Parameters(p): Parameters<ListMemoriesParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.list_count);
        let conn = self.conn.lock().await;
        let (rows, total) = storage::list_memories(
            &conn,
            p.page,
            p.page_size,
            p.tag.as_deref(),
            p.memory_type.as_deref(),
        )
        .map_err(internal_err)?;
        let page = p.page.max(1);
        let page_size = p.page_size.max(1);
        let total_pages = if page_size > 0 {
            (total as f64 / page_size as f64).ceil() as i64
        } else {
            0
        };
        let has_more = page * page_size < total;
        let result = ListMemoriesResult {
            memories: rows,
            total,
            page,
            page_size,
            total_pages,
            has_more,
        };
        json_result(&result)
    }

    #[tool(
        name = "check_database_health",
        description = "Database connectivity + configuration snapshot: backend type, memory \
                       counts, file sizes, sqlite-vec version."
    )]
    async fn check_database_health(
        &self,
        Parameters(_): Parameters<CheckHealthParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.health_count);
        let conn = self.conn.lock().await;
        let alive = storage::total_memories_alive(&conn).map_err(internal_err)?;
        let total = storage::total_memories_including_deleted(&conn).map_err(internal_err)?;
        let vec_version = storage::vec_version(&conn).map_err(internal_err)?;
        let size = storage::db_size_bytes(&self.config.db_path);
        let wal_size = storage::wal_size_bytes(&self.config.db_path);

        let payload = serde_json::json!({
            "status": "healthy",
            "backend": "sqlite_vec_rs",
            "total_memories": alive,
            "database_info": {
                "path": self.config.db_path.display().to_string(),
                "size_bytes": size,
                "wal_size_bytes": wal_size,
                "soft_deleted_count": total - alive,
                "vec_version": vec_version,
                "embedding_model": crate::embeddings::MODEL_NAME,
                "embedding_dim": crate::embeddings::EMBEDDING_DIM,
            },
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        json_result(&payload)
    }

    #[tool(
        name = "get_cache_stats",
        description = "Return call counters and init timings, wrapped in a shape compatible \
                       with the upstream Python tool so clients don't need to branch."
    )]
    async fn get_cache_stats(
        &self,
        Parameters(_): Parameters<GetCacheStatsParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.stats.record(|s| &s.stats_count);
        let s = &self.stats;
        let total_calls = s.total_calls.load(Ordering::Relaxed);
        let embed_ms = s.embed_load_ms.load(Ordering::Relaxed);
        let db_ms = s.db_init_ms.load(Ordering::Relaxed);
        let uptime = s.uptime_seconds();

        // Rust has no cache-miss path — every call hits the live in-process
        // singleton. We fake the nested `storage_cache` / `service_cache`
        // buckets with hits == total_calls so scripts that grep those
        // fields keep working; documented in SPEC §11-Q5.
        let payload = serde_json::json!({
            "total_calls": total_calls,
            "hit_rate": 1.0,
            "storage_cache": {"hits": total_calls, "misses": 0, "size": 1},
            "service_cache": {"hits": total_calls, "misses": 0, "size": 1},
            "performance": {
                "avg_init_ms": embed_ms,
                "min_init_ms": db_ms,
                "max_init_ms": embed_ms.max(db_ms),
            },
            "per_tool": {
                "ping": s.ping_count.load(Ordering::Relaxed),
                "store_memory": s.store_count.load(Ordering::Relaxed),
                "retrieve_memory": s.retrieve_count.load(Ordering::Relaxed),
                "search_by_tag": s.search_count.load(Ordering::Relaxed),
                "delete_memory": s.delete_count.load(Ordering::Relaxed),
                "list_memories": s.list_count.load(Ordering::Relaxed),
                "check_database_health": s.health_count.load(Ordering::Relaxed),
                "get_cache_stats": s.stats_count.load(Ordering::Relaxed),
            },
            "uptime_seconds": uptime,
            "backend_info": {
                "backend": "sqlite_vec_rs",
                "db_path": self.config.db_path.display().to_string(),
                "model": crate::embeddings::MODEL_NAME,
                "embedding_dim": crate::embeddings::EMBEDDING_DIM,
            },
        });
        json_result(&payload)
    }
}

fn normalize_tags(
    tags: Option<&serde_json::Value>,
    metadata: Option<&serde_json::Value>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    out.extend(coerce_tag_field(tags));
    if let Some(meta) = metadata {
        if let Some(inner) = meta.get("tags") {
            out.extend(coerce_tag_field(Some(inner)));
        }
    }
    // Dedup while preserving order.
    let mut seen = std::collections::HashSet::new();
    out.retain(|t| seen.insert(t.clone()));
    out
}

fn coerce_tag_field(v: Option<&serde_json::Value>) -> Vec<String> {
    match v {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::String(s)) => s
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|e| e.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

fn strip_tags_from_metadata(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        obj.remove("tags");
    }
    if v.is_null() {
        return serde_json::Value::Object(Default::default());
    }
    v
}

fn json_result<T: Serialize>(payload: &T) -> std::result::Result<CallToolResult, ErrorData> {
    let json = serde_json::to_string(payload).map_err(|e| internal_err(e.to_string()))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.server_info = Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "Rust port of mcp-memory-service. All 8 M0–M3 tools wired: ping, store_memory, \
             retrieve_memory, search_by_tag, delete_memory, list_memories, \
             check_database_health, get_cache_stats."
                .into(),
        );
        info
    }
}

fn internal_err(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

pub async fn run_stdio(config: Config) -> crate::error::Result<()> {
    let stats = Arc::new(CacheStats::new());

    tracing::info!("loading embedding model (first run downloads ~80MB)");
    let t0 = std::time::Instant::now();
    let embedder = Embedder::load(&config).await?;
    stats.set_embed_load_ms(t0.elapsed().as_millis() as u64);
    tracing::info!(
        embed_load_ms = stats.embed_load_ms.load(Ordering::Relaxed),
        "embedding model ready"
    );

    let t1 = std::time::Instant::now();
    let conn = storage::open(&config)?;
    stats.set_db_init_ms(t1.elapsed().as_millis() as u64);

    let server = MemoryServer::new(Arc::new(config), conn, embedder, stats);
    let service = server
        .serve(stdio())
        .await
        .map_err(|e| crate::error::AppError::Config(format!("mcp init failed: {e}")))?;
    service.waiting().await.ok();
    Ok(())
}
