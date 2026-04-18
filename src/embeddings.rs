//! Embedding generation for mcp-memory-service.
//!
//! Two backends, selected via `Config::embedding_backend`:
//!   * ONNX (default): local `all-MiniLM-L6-v2` via `ort`. Port of
//!     `src/mcp_memory_service/embeddings/onnx_embeddings.py` with
//!     bitwise-identical output (verified by `scripts/parity_check.py`).
//!   * External: POST to any OpenAI-compatible `/v1/embeddings` endpoint
//!     (vLLM, Ollama, TEI, OpenAI itself). Response embeddings are
//!     L2-normalized here so downstream cosine math stays correct even
//!     if the remote model doesn't normalize.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ndarray::{Array, Array2, Axis};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{Tensor, Value},
};
use serde::Deserialize;

fn build_session(model_path: &Path) -> Result<Session> {
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(num_cpus_one()))
        .and_then(|b| b.commit_from_file(model_path))
        .map_err(|e| AppError::Config(format!("ort session: {e}")))
}

fn num_cpus_one() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::config::{Config, EmbeddingBackend};
use crate::error::{AppError, Result};

pub const MODEL_NAME: &str = "all-MiniLM-L6-v2";
pub const EMBEDDING_DIM: usize = 384;
const MODEL_URL: &str =
    "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz";
const MODEL_SHA256: &str =
    "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3";

pub enum Embedder {
    Onnx(OnnxEmbedder),
    External(ExternalEmbedder),
}

impl Embedder {
    /// Dispatch based on `config.embedding_backend`. The ONNX path downloads
    /// the model on first call; the external path just builds an HTTP client.
    pub async fn load(config: &Config) -> Result<Self> {
        match config.embedding_backend {
            EmbeddingBackend::Onnx => Ok(Self::Onnx(OnnxEmbedder::load().await?)),
            EmbeddingBackend::External => Ok(Self::External(ExternalEmbedder::new(config)?)),
        }
    }

    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Onnx(e) => e.embed(text),
            Self::External(e) => e.embed(text).await,
        }
    }

    #[allow(dead_code)] // reserved for future batch-store path
    pub async fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Onnx(e) => e.embed_batch(texts),
            Self::External(e) => e.embed_batch(texts).await,
        }
    }
}

pub struct OnnxEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl OnnxEmbedder {
    /// Load the model, downloading and extracting it on first use.
    /// The cache path mirrors the Python implementation exactly, so both
    /// servers share the same on-disk cache.
    pub async fn load() -> Result<Self> {
        let paths = ensure_model().await?;
        let session = build_session(&paths.model_file)?;
        let tokenizer = Tokenizer::from_file(&paths.tokenizer_file)
            .map_err(|e| AppError::Config(format!("tokenizer load failed: {e}")))?;
        Ok(Self { session, tokenizer })
    }

    /// Embed a single string. Returns 384 f32 values, L2-normalized.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text])?;
        Ok(out.pop().expect("non-empty batch"))
    }

    /// Embed a batch of strings. Each row is L2-normalized.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| AppError::Config(format!("tokenize failed: {e}")))?;

        let batch = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if max_len == 0 {
            return Err(AppError::Config("tokenizer produced empty batch".into()));
        }

        let mut input_ids = Array2::<i64>::zeros((batch, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch, max_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            for (j, &v) in ids.iter().enumerate() {
                input_ids[[i, j]] = v as i64;
            }
            for (j, &v) in mask.iter().enumerate() {
                attention_mask[[i, j]] = v as i64;
            }
            for (j, &v) in types.iter().enumerate() {
                token_type_ids[[i, j]] = v as i64;
            }
        }

        let input_ids_tensor: Value = Tensor::from_array(input_ids)
            .map_err(|e| AppError::Config(format!("input_ids tensor: {e}")))?
            .into_dyn();
        let attention_mask_tensor: Value = Tensor::from_array(attention_mask.clone())
            .map_err(|e| AppError::Config(format!("attention_mask tensor: {e}")))?
            .into_dyn();
        let token_type_ids_tensor: Value = Tensor::from_array(token_type_ids)
            .map_err(|e| AppError::Config(format!("token_type_ids tensor: {e}")))?
            .into_dyn();

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| AppError::Config(format!("inference: {e}")))?;

        // Output[0] is last_hidden_state of shape [batch, seq_len, hidden].
        let last_hidden = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AppError::Config(format!("extract output: {e}")))?;
        let last_hidden = last_hidden
            .view()
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| AppError::Config(format!("unexpected output shape: {e}")))?;

        // Mean-pool with attention mask, then L2-normalize.
        let mask_f32 = attention_mask.mapv(|v| v as f32);
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch);

        for b in 0..batch {
            let hidden_slice = last_hidden.index_axis(Axis(0), b); // [seq_len, hidden]
            let mask_slice = mask_f32.index_axis(Axis(0), b); // [seq_len]

            let mut pooled = Array::<f32, _>::zeros(hidden_slice.shape()[1]);
            let mut mask_sum: f32 = 0.0;
            for t in 0..hidden_slice.shape()[0] {
                let m = mask_slice[t];
                if m == 0.0 {
                    continue;
                }
                mask_sum += m;
                for h in 0..hidden_slice.shape()[1] {
                    pooled[h] += hidden_slice[[t, h]] * m;
                }
            }
            let denom = mask_sum.max(1e-9);
            pooled.mapv_inplace(|v| v / denom);

            let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            pooled.mapv_inplace(|v| v / norm);

            embeddings.push(pooled.to_vec());
        }

        Ok(embeddings)
    }
}

struct ModelPaths {
    model_file: PathBuf,
    tokenizer_file: PathBuf,
}

fn cache_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("mcp_memory")
        .join("onnx_models")
        .join(MODEL_NAME)
}

async fn ensure_model() -> Result<ModelPaths> {
    let root = cache_root();
    let extracted = root.join("onnx");
    let model_file = extracted.join("model.onnx");
    let tokenizer_file = extracted.join("tokenizer.json");

    if model_file.exists() && tokenizer_file.exists() {
        tracing::debug!(path = ?extracted, "onnx model cache hit");
        return Ok(ModelPaths {
            model_file,
            tokenizer_file,
        });
    }

    fs::create_dir_all(&root)?;
    let archive_path = root.join("onnx.tar.gz");

    let archive_ok = archive_path.exists() && verify_sha256(&archive_path, MODEL_SHA256).unwrap_or(false);
    if !archive_ok {
        tracing::info!(url = MODEL_URL, "downloading onnx model");
        let bytes = reqwest::get(MODEL_URL)
            .await
            .map_err(|e| AppError::Config(format!("model download: {e}")))?
            .error_for_status()
            .map_err(|e| AppError::Config(format!("model download status: {e}")))?
            .bytes()
            .await
            .map_err(|e| AppError::Config(format!("model download body: {e}")))?;
        let mut f = fs::File::create(&archive_path)?;
        f.write_all(&bytes)?;
        if !verify_sha256(&archive_path, MODEL_SHA256)? {
            let _ = fs::remove_file(&archive_path);
            return Err(AppError::Config("model archive sha256 mismatch".into()));
        }
    }

    tracing::info!(path = ?extracted, "extracting onnx model");
    extract_tarball(&archive_path, &root)?;

    if !model_file.exists() || !tokenizer_file.exists() {
        return Err(AppError::Config(format!(
            "extraction incomplete: expected {} and {}",
            model_file.display(),
            tokenizer_file.display()
        )));
    }

    Ok(ModelPaths {
        model_file,
        tokenizer_file,
    })
}

fn verify_sha256(path: &Path, expected_hex: &str) -> Result<bool> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let got = hex::encode(hasher.finalize());
    Ok(got.eq_ignore_ascii_case(expected_hex))
}

pub struct ExternalEmbedder {
    client: reqwest::Client,
    url: String,
    api_key: Option<String>,
    model: String,
}

#[derive(Deserialize)]
struct ExternalResponse {
    data: Vec<ExternalEmbedding>,
}

#[derive(Deserialize)]
struct ExternalEmbedding {
    embedding: Vec<f32>,
}

impl ExternalEmbedder {
    pub fn new(config: &Config) -> Result<Self> {
        let url = config
            .external_api_url
            .clone()
            .ok_or_else(|| AppError::Config(
                "MCP_EXTERNAL_EMBEDDING_API_URL is required when MCP_EMBEDDING_BACKEND=external"
                    .into(),
            ))?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| AppError::Config(format!("http client: {e}")))?;
        Ok(Self {
            client,
            url,
            api_key: config.external_api_key.clone(),
            model: config.external_model.clone(),
        })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text]).await?;
        Ok(out.pop().expect("non-empty batch"))
    }

    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let mut req = self.client.post(&self.url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AppError::Config(format!("external embedding request: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Config(format!(
                "external embedding HTTP {status}: {}",
                body.chars().take(400).collect::<String>()
            )));
        }

        let parsed: ExternalResponse = resp
            .json()
            .await
            .map_err(|e| AppError::Config(format!("external embedding parse: {e}")))?;

        if parsed.data.len() != texts.len() {
            return Err(AppError::Config(format!(
                "external embedding returned {} rows, expected {}",
                parsed.data.len(),
                texts.len()
            )));
        }

        let out: Vec<Vec<f32>> = parsed
            .data
            .into_iter()
            .map(|e| {
                if e.embedding.len() != EMBEDDING_DIM {
                    return Err(AppError::Config(format!(
                        "external embedding dim {} != schema dim {EMBEDDING_DIM}. \
                         Pick a model that returns {EMBEDDING_DIM}-d vectors or migrate \
                         the schema (not yet supported).",
                        e.embedding.len()
                    )));
                }
                // Remote model may not normalize — ensure unit length here so
                // the cosine conversion in storage::knn_search stays valid.
                let mut v = e.embedding;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in &mut v {
                    *x /= norm;
                }
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(out)
    }
}

fn extract_tarball(archive: &Path, dest: &Path) -> Result<()> {
    let tar_gz = fs::File::open(archive)?;
    let decoded = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(decoded);
    // Guard against path traversal: reject any entry that escapes dest after
    // normalizing. We check the raw path; tar crate already resolves ".." in
    // `unpack_in` but we defense-in-depth here.
    let dest_canonical = dest
        .canonicalize()
        .map_err(|e| AppError::Config(format!("canonicalize dest: {e}")))?;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let entry_path = entry.path()?.to_path_buf();
        let full = dest_canonical.join(&entry_path);
        if !full.starts_with(&dest_canonical) {
            return Err(AppError::Config(format!(
                "tarball entry escapes dest: {}",
                entry_path.display()
            )));
        }
        entry.unpack_in(&dest_canonical)?;
    }
    Ok(())
}
