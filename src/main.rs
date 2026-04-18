use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod config;
mod embeddings;
mod error;
mod server;
mod stats;
mod storage;

use config::Config;

#[derive(Parser, Debug)]
#[command(name = "mcp-memory-service-rs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the MCP server over stdio (default when no subcommand is given).
    Serve,

    /// Open the database, apply schema, verify sqlite-vec, print stats.
    /// Does not read/write any memories. Safe to run against a live DB.
    Verify {
        /// Override MCP_MEMORY_DB_PATH for this invocation.
        #[arg(long)]
        db: Option<PathBuf>,
    },

    /// Load the ONNX model and embed a single string. Smoke-tests the
    /// full embedding pipeline (download + tokenize + inference + pool).
    Embed {
        /// Text to embed.
        text: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // stdout is reserved for JSON-RPC. All logs go to stderr.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let config = Config::from_env().context("config")?;
            tracing::info!(db = ?config.db_path, "starting stdio server");
            server::run_stdio(config).await.context("server")?;
        }
        Command::Verify { db } => {
            let mut config = Config::from_env().context("config")?;
            if let Some(override_path) = db {
                config.db_path = override_path;
            }
            run_verify(&config)?;
        }
        Command::Embed { text } => {
            run_embed(&text).await?;
        }
    }

    Ok(())
}

async fn run_embed(text: &str) -> Result<()> {
    let config = Config::from_env().context("config")?;
    let t0 = std::time::Instant::now();
    let mut embedder = embeddings::Embedder::load(&config)
        .await
        .context("load model")?;
    let load_ms = t0.elapsed().as_millis();

    let t1 = std::time::Instant::now();
    let vec = embedder.embed(text).await.context("embed")?;
    let embed_ms = t1.elapsed().as_millis();

    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head: Vec<String> = vec.iter().take(5).map(|v| format!("{v:.4}")).collect();

    println!("mcp-memory-service-rs embed ✓");
    println!("  text        : {text:?}");
    println!("  load model  : {load_ms} ms");
    println!("  embed time  : {embed_ms} ms");
    println!("  dim         : {}", vec.len());
    println!("  L2 norm     : {norm:.6}");
    println!("  first 5     : [{}]", head.join(", "));
    Ok(())
}

fn run_verify(config: &Config) -> Result<()> {
    let t0 = std::time::Instant::now();
    let conn = storage::open(config).context("open db")?;
    let open_ms = t0.elapsed().as_millis();

    let vec_version = storage::vec_version(&conn).context("vec_version")?;
    let count = storage::count_memories(&conn).context("count")?;
    let size = storage::db_size_bytes(&config.db_path);

    // stdout in the verify subcommand is not a protocol channel — print a
    // plain report the user can read directly.
    println!("mcp-memory-service-rs verify ✓");
    println!("  db path       : {}", config.db_path.display());
    println!("  db size       : {size} bytes");
    println!("  open + schema : {open_ms} ms");
    println!("  vec_version   : {vec_version}");
    println!("  memories      : {count} (undeleted)");
    Ok(())
}
