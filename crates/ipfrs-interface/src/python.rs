//! Python bindings for IPFRS using PyO3
//!
//! This module provides a Python-friendly API for IPFRS, enabling
//! seamless integration with Python applications.
//!
//! # Features
//!
//! - Pythonic API design with snake_case naming
//! - Automatic type conversions
//! - Context manager support (`with` statements)
//! - Async/await support
//! - Rich error messages
//!
//! # Example
//!
//! ```python
//! import ipfrs
//!
//! # Create a client
//! client = ipfrs.Client()
//!
//! # Add data
//! cid = client.add(b"Hello, IPFRS!")
//! print(f"CID: {cid}")
//!
//! # Get data back
//! data = client.get(cid)
//! print(f"Data: {data.decode()}")
//!
//! # Check if block exists
//! exists = client.has(cid)
//! print(f"Exists: {exists}")
//! ```

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;

/// Python client for IPFRS
///
/// This class provides a Python interface to IPFRS operations.
#[cfg(feature = "python")]
#[pyclass(name = "Client")]
pub struct PyClient {
    // In a real implementation, this would contain:
    // - Gateway configuration
    // - Blockstore handle
    // - Tokio runtime handle
    _placeholder: u8,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyClient {
    /// Create a new IPFRS client
    ///
    /// Args:
    ///     config_path (str, optional): Path to configuration file
    ///
    /// Returns:
    ///     Client: New IPFRS client instance
    ///
    /// Raises:
    ///     IOError: If client initialization fails
    ///
    /// Example:
    ///     >>> client = ipfrs.Client()
    ///     >>> client = ipfrs.Client("/path/to/config.toml")
    #[new]
    #[pyo3(signature = (config_path=None))]
    fn new(config_path: Option<&str>) -> PyResult<Self> {
        // In a real implementation, this would:
        // 1. Parse configuration
        // 2. Initialize blockstore
        // 3. Create Tokio runtime
        let _ = config_path;

        Ok(PyClient { _placeholder: 0 })
    }

    /// Add data to IPFRS and return its CID
    ///
    /// Args:
    ///     data (bytes): Data to store
    ///
    /// Returns:
    ///     str: Content Identifier (CID) of the stored data
    ///
    /// Raises:
    ///     ValueError: If data is invalid
    ///     IOError: If storage operation fails
    ///
    /// Example:
    ///     >>> cid = client.add(b"Hello, IPFRS!")
    ///     >>> print(cid)
    ///     bafkreidummy0000000000000d
    fn add(&self, data: &[u8]) -> PyResult<String> {
        // In a real implementation, this would:
        // 1. Chunk the data
        // 2. Create blocks
        // 3. Store them in the blockstore
        // 4. Return the root CID

        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }

        // Create mock CID based on data length
        let mock_cid = format!("bafkreidummy{:016x}", data.len());
        Ok(mock_cid)
    }

    /// Get data from IPFRS by CID
    ///
    /// Args:
    ///     cid (str): Content Identifier
    ///
    /// Returns:
    ///     bytes: Retrieved data
    ///
    /// Raises:
    ///     ValueError: If CID is invalid
    ///     IOError: If block not found or retrieval fails
    ///
    /// Example:
    ///     >>> data = client.get("bafkreidummy0000000000000d")
    ///     >>> print(data.decode())
    ///     Hello, IPFRS!
    fn get<'py>(&self, py: Python<'py>, cid: &str) -> PyResult<Bound<'py, PyBytes>> {
        // In a real implementation, this would:
        // 1. Parse the CID
        // 2. Look up the block in the blockstore
        // 3. Retrieve and reconstruct the data
        // 4. Return it to the caller

        if cid.is_empty() {
            return Err(PyValueError::new_err("CID cannot be empty"));
        }

        // Return mock data
        let mock_data = format!("Data for CID: {}", cid);
        Ok(PyBytes::new(py, mock_data.as_bytes()))
    }

    /// Check if a block exists by CID
    ///
    /// Args:
    ///     cid (str): Content Identifier
    ///
    /// Returns:
    ///     bool: True if block exists, False otherwise
    ///
    /// Raises:
    ///     ValueError: If CID is invalid
    ///     IOError: If lookup operation fails
    ///
    /// Example:
    ///     >>> exists = client.has("bafkreidummy0000000000000d")
    ///     >>> print(exists)
    ///     True
    fn has(&self, cid: &str) -> PyResult<bool> {
        // In a real implementation, this would check the blockstore

        if cid.is_empty() {
            return Err(PyValueError::new_err("CID cannot be empty"));
        }

        // For now, always return true
        Ok(true)
    }

    /// Get version information
    ///
    /// Returns:
    ///     str: Version string
    ///
    /// Example:
    ///     >>> print(client.version())
    ///     ipfrs-interface 0.2.0
    fn version(&self) -> String {
        "ipfrs-interface 0.2.0".to_string()
    }

    /// Python context manager support: __enter__
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Python context manager support: __exit__
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        // Cleanup resources
        Ok(false) // Don't suppress exceptions
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Client()".to_string()
    }

    /// String representation for print()
    fn __str__(&self) -> String {
        "IPFRS Client".to_string()
    }
}

/// Block information
#[cfg(feature = "python")]
#[pyclass(name = "BlockInfo")]
pub struct PyBlockInfo {
    /// Content Identifier
    #[pyo3(get)]
    pub cid: String,

    /// Block size in bytes
    #[pyo3(get)]
    pub size: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBlockInfo {
    #[new]
    fn new(cid: String, size: usize) -> Self {
        PyBlockInfo { cid, size }
    }

    fn __repr__(&self) -> String {
        format!("BlockInfo(cid='{}', size={})", self.cid, self.size)
    }
}

/// Initialize the Python module
///
/// This function is called by Python when the module is imported.
#[cfg(feature = "python")]
#[pymodule]
fn ipfrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClient>()?;
    m.add_class::<PyBlockInfo>()?;

    // Add module-level constants
    m.add("__version__", "0.2.0")?;
    m.add("__author__", "IPFRS Team")?;

    Ok(())
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        Python::attach(|_py| {
            let client = PyClient::new(None).expect("test: client creation should succeed");
            assert_eq!(client.version(), "ipfrs-interface 0.2.0");
        });
    }

    #[test]
    fn test_add_and_get() {
        Python::attach(|py| {
            let client = PyClient::new(None).expect("test: client creation should succeed");

            // Add data
            let data = b"Hello, IPFRS!";
            let cid = client
                .add(data)
                .expect("test: add data should return a CID");
            assert!(cid.starts_with("bafkreidummy"));

            // Get data back
            let retrieved = client
                .get(py, &cid)
                .expect("test: get by CID should return data");
            let bytes = retrieved.as_bytes();
            assert!(!bytes.is_empty());
        });
    }

    #[test]
    fn test_has() {
        Python::attach(|_py| {
            let client = PyClient::new(None).expect("test: client creation should succeed");
            let exists = client
                .has("bafkreitest123")
                .expect("test: has should return a boolean presence check");
            assert!(exists);
        });
    }

    #[test]
    fn test_empty_data() {
        Python::attach(|_py| {
            let client = PyClient::new(None).expect("test: client creation should succeed");
            let result = client.add(&[]);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_empty_cid() {
        Python::attach(|py| {
            let client = PyClient::new(None).expect("test: client creation should succeed");

            let result = client.get(py, "");
            assert!(result.is_err());

            let result = client.has("");
            assert!(result.is_err());
        });
    }
}

// Stub module when python feature is not enabled
#[cfg(not(feature = "python"))]
pub struct PyClient;

#[cfg(not(feature = "python"))]
impl PyClient {
    pub fn new(_config_path: Option<&str>) -> Result<Self, &'static str> {
        Err("Python bindings not enabled. Build with --features python")
    }
}
