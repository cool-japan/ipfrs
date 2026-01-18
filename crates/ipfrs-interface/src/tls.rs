//! TLS/SSL Configuration for HTTPS Support
//!
//! Provides TLS certificate and key loading for secure HTTPS connections.

use axum_server::tls_rustls::RustlsConfig;
use std::io;
use std::path::{Path, PathBuf};

/// TLS configuration errors
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Failed to load certificate: {0}")]
    CertificateError(String),

    #[error("Failed to load private key: {0}")]
    PrivateKeyError(String),

    #[error("TLS configuration error: {0}")]
    ConfigError(String),
}

pub type TlsResult<T> = Result<T, TlsError>;

/// TLS configuration for HTTPS server
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to PEM-encoded certificate file
    pub cert_path: PathBuf,
    /// Path to PEM-encoded private key file
    pub key_path: PathBuf,
}

impl TlsConfig {
    /// Create a new TLS configuration
    pub fn new(cert_path: impl AsRef<Path>, key_path: impl AsRef<Path>) -> Self {
        Self {
            cert_path: cert_path.as_ref().to_path_buf(),
            key_path: key_path.as_ref().to_path_buf(),
        }
    }

    /// Build axum-server RustlsConfig from this TLS configuration
    ///
    /// This is an async method that loads the certificates and private key.
    pub async fn build_server_config(&self) -> TlsResult<RustlsConfig> {
        RustlsConfig::from_pem_file(&self.cert_path, &self.key_path)
            .await
            .map_err(|e| TlsError::ConfigError(format!("Failed to load TLS configuration: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_config_creation() {
        let config = TlsConfig::new("cert.pem", "key.pem");
        assert_eq!(config.cert_path, PathBuf::from("cert.pem"));
        assert_eq!(config.key_path, PathBuf::from("key.pem"));
    }

    // Note: Actual certificate loading tests would require test certificates
    // and are better done in integration tests
}
