use std::path::PathBuf;

use crate::error::{AppError, Result};

#[derive(Debug, Clone)]
#[allow(dead_code)] // embedding_* fields consumed in M1 when embeddings wire up
pub struct Config {
    pub db_path: PathBuf,
    pub sqlite_pragmas: Vec<(String, String)>,
    pub embedding_backend: EmbeddingBackend,
    pub external_api_url: Option<String>,
    pub external_api_key: Option<String>,
    pub external_model: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingBackend {
    Onnx,
    External,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let db_path = match std::env::var("MCP_MEMORY_DB_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => default_db_path(),
        };

        let sqlite_pragmas = parse_pragmas(std::env::var("MCP_MEMORY_SQLITE_PRAGMAS").ok());

        let backend = match std::env::var("MCP_EMBEDDING_BACKEND").as_deref() {
            Ok("external") => EmbeddingBackend::External,
            Ok("onnx") | Err(_) => EmbeddingBackend::Onnx,
            Ok(other) => {
                return Err(AppError::Config(format!(
                    "MCP_EMBEDDING_BACKEND must be 'onnx' or 'external', got '{other}'"
                )));
            }
        };

        let external_api_url = std::env::var("MCP_EXTERNAL_EMBEDDING_API_URL").ok();
        let external_api_key = std::env::var("MCP_EXTERNAL_EMBEDDING_API_KEY").ok();
        let external_model = std::env::var("MCP_EXTERNAL_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());

        Ok(Self {
            db_path,
            sqlite_pragmas,
            embedding_backend: backend,
            external_api_url,
            external_api_key,
            external_model,
        })
    }
}

fn default_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mcp_memory_service.db")
}

fn parse_pragmas(raw: Option<String>) -> Vec<(String, String)> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    raw.split(',')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            let k = k.trim();
            let v = v.trim();
            if k.is_empty() || v.is_empty() {
                None
            } else {
                Some((k.to_string(), v.to_string()))
            }
        })
        .collect()
}
