//! TLS/SSL Support for IPFRS
//!
//! Provides:
//! - TLS configuration and certificate management
//! - Self-signed certificate generation for development
//! - Production certificate loading from files
//! - HTTPS server configuration
//! - Certificate validation options

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS
    pub enabled: bool,
    /// Path to certificate file (PEM format)
    pub cert_path: Option<PathBuf>,
    /// Path to private key file (PEM format)
    pub key_path: Option<PathBuf>,
    /// Path to CA certificate for client verification
    pub ca_cert_path: Option<PathBuf>,
    /// Require client certificates
    pub require_client_cert: bool,
    /// TLS version (default: TLS 1.3)
    pub min_version: TlsVersion,
    /// Allowed cipher suites (None = use secure defaults)
    pub cipher_suites: Option<Vec<String>>,
    /// Certificate reload interval in seconds (for auto-renewal)
    pub reload_interval_secs: Option<u64>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cert_path: None,
            key_path: None,
            ca_cert_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls13,
            cipher_suites: None,
            reload_interval_secs: Some(3600), // Reload every hour
        }
    }
}

impl TlsConfig {
    /// Create a new TLS configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable TLS with certificate and key files
    pub fn with_cert_and_key(mut self, cert_path: PathBuf, key_path: PathBuf) -> Self {
        self.enabled = true;
        self.cert_path = Some(cert_path);
        self.key_path = Some(key_path);
        self
    }

    /// Enable client certificate verification
    pub fn with_client_auth(mut self, ca_cert_path: PathBuf) -> Self {
        self.ca_cert_path = Some(ca_cert_path);
        self.require_client_cert = true;
        self
    }

    /// Set minimum TLS version
    pub fn with_min_version(mut self, version: TlsVersion) -> Self {
        self.min_version = version;
        self
    }

    /// Set custom cipher suites
    pub fn with_cipher_suites(mut self, suites: Vec<String>) -> Self {
        self.cipher_suites = Some(suites);
        self
    }

    /// Set certificate reload interval
    pub fn with_reload_interval(mut self, secs: u64) -> Self {
        self.reload_interval_secs = Some(secs);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), TlsError> {
        if !self.enabled {
            return Ok(());
        }

        // Check certificate and key files exist
        if let Some(ref cert) = self.cert_path {
            if !cert.exists() {
                return Err(TlsError::CertificateNotFound(cert.clone()));
            }
        } else {
            return Err(TlsError::ConfigurationError(
                "Certificate path required when TLS is enabled".to_string(),
            ));
        }

        if let Some(ref key) = self.key_path {
            if !key.exists() {
                return Err(TlsError::KeyNotFound(key.clone()));
            }
        } else {
            return Err(TlsError::ConfigurationError(
                "Key path required when TLS is enabled".to_string(),
            ));
        }

        // Check CA certificate if client auth is enabled
        if self.require_client_cert {
            if let Some(ref ca) = self.ca_cert_path {
                if !ca.exists() {
                    return Err(TlsError::CaCertNotFound(ca.clone()));
                }
            } else {
                return Err(TlsError::ConfigurationError(
                    "CA certificate required when client auth is enabled".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// TLS protocol version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsVersion {
    /// TLS 1.2
    Tls12,
    /// TLS 1.3 (recommended)
    Tls13,
}

impl std::fmt::Display for TlsVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TlsVersion::Tls12 => write!(f, "TLS 1.2"),
            TlsVersion::Tls13 => write!(f, "TLS 1.3"),
        }
    }
}

/// Certificate information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    /// Subject common name
    pub common_name: String,
    /// Subject organization
    pub organization: Option<String>,
    /// Issuer common name
    pub issuer: String,
    /// Valid from (Unix timestamp)
    pub not_before: i64,
    /// Valid until (Unix timestamp)
    pub not_after: i64,
    /// Serial number
    pub serial: String,
    /// Key algorithm (e.g., "RSA", "ECDSA")
    pub key_algorithm: String,
    /// Is self-signed
    pub self_signed: bool,
}

impl CertificateInfo {
    /// Check if certificate is expired
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        now > self.not_after
    }

    /// Check if certificate is not yet valid
    pub fn is_not_yet_valid(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        now < self.not_before
    }

    /// Check if certificate is valid (not expired and not before valid time)
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && !self.is_not_yet_valid()
    }

    /// Days until expiration
    pub fn days_until_expiration(&self) -> i64 {
        let now = chrono::Utc::now().timestamp();
        (self.not_after - now) / 86400
    }
}

/// TLS certificate manager
#[derive(Clone)]
pub struct TlsManager {
    config: Arc<TlsConfig>,
}

impl TlsManager {
    /// Create a new TLS manager
    pub fn new(config: TlsConfig) -> Result<Self, TlsError> {
        config.validate()?;
        Ok(Self {
            config: Arc::new(config),
        })
    }

    /// Load certificate from file
    pub fn load_certificate(&self) -> Result<Vec<u8>, TlsError> {
        let path = self
            .config
            .cert_path
            .as_ref()
            .ok_or_else(|| TlsError::ConfigurationError("No certificate path".to_string()))?;

        std::fs::read(path)
            .with_context(|| format!("Failed to read certificate from {}", path.display()))
            .map_err(|e| TlsError::IoError(e.to_string()))
    }

    /// Load private key from file
    pub fn load_key(&self) -> Result<Vec<u8>, TlsError> {
        let path = self
            .config
            .key_path
            .as_ref()
            .ok_or_else(|| TlsError::ConfigurationError("No key path".to_string()))?;

        std::fs::read(path)
            .with_context(|| format!("Failed to read key from {}", path.display()))
            .map_err(|e| TlsError::IoError(e.to_string()))
    }

    /// Load CA certificate for client verification
    pub fn load_ca_cert(&self) -> Result<Vec<u8>, TlsError> {
        let path =
            self.config.ca_cert_path.as_ref().ok_or_else(|| {
                TlsError::ConfigurationError("No CA certificate path".to_string())
            })?;

        std::fs::read(path)
            .with_context(|| format!("Failed to read CA certificate from {}", path.display()))
            .map_err(|e| TlsError::IoError(e.to_string()))
    }

    /// Get configuration
    pub fn config(&self) -> &TlsConfig {
        &self.config
    }

    /// Check if TLS is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get certificate info (simplified - in production would parse actual cert)
    pub fn get_certificate_info(&self) -> Result<CertificateInfo, TlsError> {
        // In a real implementation, this would parse the certificate using
        // x509-parser or similar. For now, we return mock data.
        Ok(CertificateInfo {
            common_name: "ipfrs.local".to_string(),
            organization: Some("IPFRS".to_string()),
            issuer: "IPFRS CA".to_string(),
            not_before: chrono::Utc::now().timestamp(),
            not_after: chrono::Utc::now().timestamp() + (365 * 86400), // 1 year
            serial: "01".to_string(),
            key_algorithm: "RSA-2048".to_string(),
            self_signed: true,
        })
    }
}

/// Self-signed certificate generator for development
pub struct SelfSignedCertGenerator {
    common_name: String,
    organization: Option<String>,
    validity_days: u32,
}

impl SelfSignedCertGenerator {
    /// Create a new certificate generator
    pub fn new(common_name: String) -> Self {
        Self {
            common_name,
            organization: None,
            validity_days: 365,
        }
    }

    /// Set organization name
    pub fn with_organization(mut self, org: String) -> Self {
        self.organization = Some(org);
        self
    }

    /// Set validity period in days
    pub fn with_validity_days(mut self, days: u32) -> Self {
        self.validity_days = days;
        self
    }

    /// Generate self-signed certificate and private key
    ///
    /// Returns (certificate_pem, private_key_pem)
    ///
    /// Note: In a real implementation, this would use rcgen or similar.
    /// This is a placeholder that creates files for demonstration.
    pub fn generate(&self) -> Result<(String, String), TlsError> {
        // In production, use rcgen crate:
        // let cert = rcgen::generate_simple_self_signed(vec![self.common_name.clone()])?;
        // let cert_pem = cert.serialize_pem()?;
        // let key_pem = cert.serialize_private_key_pem();

        // For now, return placeholder PEM format
        let cert_pem = format!(
            "-----BEGIN CERTIFICATE-----\n\
             (Self-signed certificate for {})\n\
             (Generated for development purposes only)\n\
             (In production, use proper CA-signed certificates)\n\
             -----END CERTIFICATE-----\n",
            self.common_name
        );

        let key_pem = format!(
            "-----BEGIN PRIVATE KEY-----\n\
             (Private key for {})\n\
             (Keep this secure and never commit to version control)\n\
             -----END PRIVATE KEY-----\n",
            self.common_name
        );

        Ok((cert_pem, key_pem))
    }

    /// Generate and save to files
    pub fn generate_to_files(
        &self,
        cert_path: impl AsRef<Path>,
        key_path: impl AsRef<Path>,
    ) -> Result<(), TlsError> {
        let (cert_pem, key_pem) = self.generate()?;

        std::fs::write(cert_path.as_ref(), cert_pem.as_bytes())
            .with_context(|| {
                format!(
                    "Failed to write certificate to {}",
                    cert_path.as_ref().display()
                )
            })
            .map_err(|e| TlsError::IoError(e.to_string()))?;

        std::fs::write(key_path.as_ref(), key_pem.as_bytes())
            .with_context(|| format!("Failed to write key to {}", key_path.as_ref().display()))
            .map_err(|e| TlsError::IoError(e.to_string()))?;

        // Set restrictive permissions on key file (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(key_path.as_ref())
                .map_err(|e| TlsError::IoError(e.to_string()))?
                .permissions();
            perms.set_mode(0o600); // rw-------
            std::fs::set_permissions(key_path.as_ref(), perms)
                .map_err(|e| TlsError::IoError(e.to_string()))?;
        }

        Ok(())
    }
}

/// TLS errors
#[derive(Debug, Error)]
pub enum TlsError {
    #[error("Certificate not found: {0}")]
    CertificateNotFound(PathBuf),

    #[error("Private key not found: {0}")]
    KeyNotFound(PathBuf),

    #[error("CA certificate not found: {0}")]
    CaCertNotFound(PathBuf),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Certificate parse error: {0}")]
    ParseError(String),

    #[error("Certificate expired")]
    CertificateExpired,

    #[error("Certificate not yet valid")]
    CertificateNotYetValid,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tls_config_default() {
        let config = TlsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.min_version, TlsVersion::Tls13);
        assert!(!config.require_client_cert);
    }

    #[test]
    fn test_tls_config_builder() {
        let config = TlsConfig::new()
            .with_cert_and_key(PathBuf::from("cert.pem"), PathBuf::from("key.pem"))
            .with_min_version(TlsVersion::Tls12);

        assert!(config.enabled);
        assert_eq!(config.cert_path, Some(PathBuf::from("cert.pem")));
        assert_eq!(config.key_path, Some(PathBuf::from("key.pem")));
        assert_eq!(config.min_version, TlsVersion::Tls12);
    }

    #[test]
    fn test_tls_config_validation_disabled() {
        let config = TlsConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tls_config_validation_missing_cert() {
        let config = TlsConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tls_config_validation_missing_files() {
        let config = TlsConfig::new().with_cert_and_key(
            PathBuf::from("nonexistent.pem"),
            PathBuf::from("nonexistent.key"),
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tls_config_validation_success() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        // Create dummy files
        std::fs::write(&cert_path, b"cert").expect("test: write cert should succeed");
        std::fs::write(&key_path, b"key").expect("test: write key should succeed");

        let config = TlsConfig::new().with_cert_and_key(cert_path, key_path);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tls_version_display() {
        assert_eq!(TlsVersion::Tls12.to_string(), "TLS 1.2");
        assert_eq!(TlsVersion::Tls13.to_string(), "TLS 1.3");
    }

    #[test]
    fn test_certificate_info_validity() {
        let now = chrono::Utc::now().timestamp();

        // Valid certificate
        let valid_cert = CertificateInfo {
            common_name: "test.local".to_string(),
            organization: None,
            issuer: "Test CA".to_string(),
            not_before: now - 86400,     // 1 day ago
            not_after: now + 86400 * 30, // 30 days from now
            serial: "01".to_string(),
            key_algorithm: "RSA".to_string(),
            self_signed: false,
        };

        assert!(valid_cert.is_valid());
        assert!(!valid_cert.is_expired());
        assert!(!valid_cert.is_not_yet_valid());
        assert!(valid_cert.days_until_expiration() > 0);

        // Expired certificate
        let expired_cert = CertificateInfo {
            not_before: now - 86400 * 365,
            not_after: now - 86400,
            ..valid_cert.clone()
        };

        assert!(!expired_cert.is_valid());
        assert!(expired_cert.is_expired());
    }

    #[test]
    fn test_self_signed_cert_generator() {
        let gen = SelfSignedCertGenerator::new("localhost".to_string())
            .with_organization("Test Org".to_string())
            .with_validity_days(90);

        let result = gen.generate();
        assert!(result.is_ok());

        let (cert, key) = result.expect("test: self-signed cert generation should succeed");
        assert!(cert.contains("BEGIN CERTIFICATE"));
        assert!(key.contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn test_self_signed_cert_to_files() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        let gen = SelfSignedCertGenerator::new("localhost".to_string());
        let result = gen.generate_to_files(&cert_path, &key_path);
        assert!(result.is_ok());

        assert!(cert_path.exists());
        assert!(key_path.exists());

        // Check key file permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::metadata(&key_path)
                .expect("test: key file metadata should be readable")
                .permissions();
            assert_eq!(perms.mode() & 0o777, 0o600);
        }
    }

    #[test]
    fn test_tls_manager_creation() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        std::fs::write(&cert_path, b"cert data").expect("test: write cert should succeed");
        std::fs::write(&key_path, b"key data").expect("test: write key should succeed");

        let config = TlsConfig::new().with_cert_and_key(cert_path, key_path);
        let manager = TlsManager::new(config);
        assert!(manager.is_ok());

        let mgr = manager.expect("test: TLS manager creation should succeed");
        assert!(mgr.is_enabled());
    }

    #[test]
    fn test_tls_manager_load_files() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        let cert_data = b"certificate data";
        let key_data = b"private key data";

        std::fs::write(&cert_path, cert_data).expect("test: write cert should succeed");
        std::fs::write(&key_path, key_data).expect("test: write key should succeed");

        let config = TlsConfig::new().with_cert_and_key(cert_path, key_path);
        let manager = TlsManager::new(config).expect("test: TLS manager creation should succeed");

        let loaded_cert = manager
            .load_certificate()
            .expect("test: load_certificate should succeed");
        let loaded_key = manager.load_key().expect("test: load_key should succeed");

        assert_eq!(loaded_cert, cert_data);
        assert_eq!(loaded_key, key_data);
    }

    #[test]
    fn test_tls_manager_get_certificate_info() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        std::fs::write(&cert_path, b"cert").expect("test: write cert should succeed");
        std::fs::write(&key_path, b"key").expect("test: write key should succeed");

        let config = TlsConfig::new().with_cert_and_key(cert_path, key_path);
        let manager = TlsManager::new(config).expect("test: TLS manager creation should succeed");

        let info = manager
            .get_certificate_info()
            .expect("test: get_certificate_info should succeed");
        assert_eq!(info.common_name, "ipfrs.local");
        assert!(info.is_valid());
    }
}
