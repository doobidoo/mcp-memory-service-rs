use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod config;
mod error;
mod server;
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
}

#[tokio::main]
async fn main() -> Result<()> {
    // stdout is reserved for JSON-RPC. All logs go to stderr.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
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
    }

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
