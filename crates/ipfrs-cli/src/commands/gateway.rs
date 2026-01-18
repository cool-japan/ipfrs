//! HTTP Gateway command
//!
//! This module provides the HTTP gateway functionality with optional TLS support.

use anyhow::{Context, Result};

/// Run HTTP gateway server
///
/// # Arguments
///
/// * `listen` - Address and port to listen on
/// * `data_dir` - Data directory containing the IPFRS repository
/// * `tls_cert` - Optional path to TLS certificate file (PEM format)
/// * `tls_key` - Optional path to TLS private key file (PEM format)
///
/// # Errors
///
/// Returns an error if:
/// - Storage configuration is invalid
/// - TLS certificate or key files cannot be read
/// - Only one of cert/key is provided (both are required for TLS)
/// - Gateway fails to start
pub async fn run_gateway(
    listen: String,
    data_dir: String,
    tls_cert: Option<String>,
    tls_key: Option<String>,
) -> Result<()> {
    use ipfrs_interface::tls::TlsConfig;
    use ipfrs_interface::{Gateway, GatewayConfig};
    use ipfrs_storage::BlockStoreConfig;

    let storage_config = BlockStoreConfig {
        path: std::path::PathBuf::from(&data_dir).join("blocks"),
        cache_size: 100 * 1024 * 1024, // 100MB
    };

    // Validate TLS configuration
    let tls_config = match (tls_cert, tls_key) {
        (Some(cert), Some(key)) => {
            // Verify files exist
            if !std::path::Path::new(&cert).exists() {
                anyhow::bail!("TLS certificate file not found: {}", cert);
            }
            if !std::path::Path::new(&key).exists() {
                anyhow::bail!("TLS private key file not found: {}", key);
            }

            Some(TlsConfig {
                cert_path: cert.into(),
                key_path: key.into(),
            })
        }
        (None, None) => None,
        (Some(_), None) => {
            anyhow::bail!("TLS certificate provided but private key is missing (use --tls-key)");
        }
        (None, Some(_)) => {
            anyhow::bail!("TLS private key provided but certificate is missing (use --tls-cert)");
        }
    };

    let config = GatewayConfig {
        listen_addr: listen.clone(),
        storage_config,
        tls_config: tls_config.clone(),
        compression_config: Default::default(),
    };

    let protocol = if tls_config.is_some() {
        "HTTPS"
    } else {
        "HTTP"
    };
    eprintln!("Starting {} gateway on {}", protocol, listen);
    if tls_config.is_some() {
        eprintln!("TLS enabled - using secure connections");
    }

    let gateway = Gateway::new(config).context("Failed to create gateway")?;
    gateway
        .start()
        .await
        .context("Failed to start gateway server")?;

    Ok(())
}
