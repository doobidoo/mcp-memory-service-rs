use std::sync::Arc;

use rmcp::{
    ErrorData, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::io::stdio,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::config::Config;
use crate::embeddings::Embedder;
use crate::storage::{self, MemoryRow};

pub struct MemoryServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    #[allow(dead_code)]
    config: Arc<Config>,
    conn: Arc<Mutex<rusqlite::Connection>>,
    embedder: Arc<Mutex<Embedder>>,
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

// ---------- impl ----------

#[tool_router]
impl MemoryServer {
    pub fn new(
        config: Arc<Config>,
        conn: rusqlite::Connection,
        embedder: Embedder,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            config,
            conn: Arc::new(Mutex::new(conn)),
            embedder: Arc::new(Mutex::new(embedder)),
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
        let tags = normalize_tags(p.tags.as_ref(), p.metadata.as_ref());
        let metadata = strip_tags_from_metadata(p.metadata.unwrap_or(serde_json::Value::Null));

        let row = storage::new_memory_row(p.content.clone(), tags, Some(p.memory_type), metadata);
        let embedding = {
            let mut emb = self.embedder.lock().await;
            emb.embed(&p.content).map_err(internal_err)?
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
        let query_emb = {
            let mut emb = self.embedder.lock().await;
            emb.embed(&p.query).map_err(internal_err)?
        };
        let conn = self.conn.lock().await;
        let hits = storage::knn_search(&conn, &query_emb, p.n_results.max(1))
            .map_err(internal_err)?;
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
    let json =
        serde_json::to_string(payload).map_err(|e| internal_err(e.to_string()))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.server_info =
            Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "Rust port of mcp-memory-service. M1 wired: store_memory, retrieve_memory, ping. \
             Remaining tools (search_by_tag, delete_memory, list_memories, check_database_health, \
             get_cache_stats) arrive in M2/M3."
                .into(),
        );
        info
    }
}

fn internal_err(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

pub async fn run_stdio(config: Config) -> crate::error::Result<()> {
    tracing::info!("loading embedding model (first run downloads ~80MB)");
    let embedder = Embedder::load().await?;
    tracing::info!("embedding model ready");

    let conn = storage::open(&config)?;
    let server = MemoryServer::new(Arc::new(config), conn, embedder);
    let service = server
        .serve(stdio())
        .await
        .map_err(|e| crate::error::AppError::Config(format!("mcp init failed: {e}")))?;
    service.waiting().await.ok();
    Ok(())
}
