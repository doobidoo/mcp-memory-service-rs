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
use crate::storage;

/// M0 scaffold: one stub tool (`ping`) proves transport + router are wired.
/// Real memory tools land in M1+.
pub struct MemoryServer {
    #[allow(dead_code)] // tool_router is read via the #[tool_handler] macro
    tool_router: ToolRouter<Self>,
    #[allow(dead_code)]
    config: Arc<Config>,
    conn: Arc<Mutex<rusqlite::Connection>>,
}

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct PingParams {}

#[derive(Debug, Serialize)]
pub struct PingResult {
    pub status: &'static str,
    pub backend: &'static str,
    pub vec_version: String,
    pub memory_count: i64,
}

#[tool_router]
impl MemoryServer {
    pub fn new(config: Arc<Config>, conn: rusqlite::Connection) -> Self {
        Self {
            tool_router: Self::tool_router(),
            config,
            conn: Arc::new(Mutex::new(conn)),
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
        let json =
            serde_json::to_string(&payload).map_err(|e| internal_err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.server_info =
            Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "Rust port of mcp-memory-service (M0 scaffold). Only the `ping` tool is wired; \
             memory tools arrive in M1."
                .into(),
        );
        info
    }
}

fn internal_err(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

pub async fn run_stdio(config: Config) -> crate::error::Result<()> {
    let conn = storage::open(&config)?;
    let server = MemoryServer::new(Arc::new(config), conn);
    let service = server
        .serve(stdio())
        .await
        .map_err(|e| crate::error::AppError::Config(format!("mcp init failed: {e}")))?;
    service.waiting().await.ok();
    Ok(())
}
