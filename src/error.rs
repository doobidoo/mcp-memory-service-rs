use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("schema error: {0}")]
    Schema(String),
}

pub type Result<T> = std::result::Result<T, AppError>;
