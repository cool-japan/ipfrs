//! Memory-Mapped Tensor Server Example
//!
//! This example demonstrates how to use memory-mapped I/O for zero-copy
//! tensor serving with IPFRS. It shows:
//!
//! - Setting up an mmap cache for efficient file serving
//! - Serving tensors with zero-copy via mmap
//! - Handling partial tensor requests (slicing)
//! - Performance optimization with platform-specific hints
//!
//! # Usage
//!
//! ```bash
//! cargo run --example mmap_tensor_server
//! ```
//!
//! Then in another terminal:
//! ```bash
//! # Create a test tensor file
//! echo "test tensor data" > /tmp/test.tensor
//!
//! # Request the full tensor
//! curl http://localhost:8080/tensor/test
//!
//! # Request a byte range
//! curl http://localhost:8080/tensor/test?start=0&end=100
//! ```

use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use ipfrs_interface::mmap::{MmapCache, MmapConfig, MmapError};
use serde::Deserialize;
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;
use tokio::net::TcpListener;

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
struct AppState {
    /// Memory-mapped file cache
    mmap_cache: Arc<MmapCache>,
    /// Base directory for tensor storage
    storage_path: PathBuf,
    /// Mmap configuration
    _mmap_config: MmapConfig,
}

impl AppState {
    fn new(storage_path: PathBuf, max_cache_entries: usize) -> Self {
        Self {
            mmap_cache: Arc::new(MmapCache::new(max_cache_entries)),
            storage_path,
            _mmap_config: MmapConfig::sequential(), // Optimize for sequential reads
        }
    }
}

// ============================================================================
// Request Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct TensorQuery {
    /// Start byte offset (for partial requests)
    start: Option<usize>,
    /// End byte offset (for partial requests)
    end: Option<usize>,
}

// ============================================================================
// Handlers
// ============================================================================

/// Get tensor with zero-copy mmap serving
async fn get_tensor(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<TensorQuery>,
) -> Result<Response, MmapErrorResponse> {
    // Construct file path
    let file_path = state.storage_path.join(format!("{}.tensor", name));

    // Get or create memory-mapped file
    let mmap_file = state
        .mmap_cache
        .get_or_create(&file_path)
        .map_err(MmapErrorResponse::from)?;

    // Handle partial or full request
    let (data, is_partial) = if let (Some(start), Some(end)) = (query.start, query.end) {
        // Partial request (HTTP 206)
        let range_data = mmap_file
            .range(start..end)
            .map_err(MmapErrorResponse::from)?;
        (range_data, true)
    } else {
        // Full file request (HTTP 200)
        (mmap_file.bytes(), false)
    };

    // Build response
    let mut response = Response::builder()
        .status(if is_partial {
            StatusCode::PARTIAL_CONTENT
        } else {
            StatusCode::OK
        })
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, data.len())
        .header("X-Served-By", "mmap")
        .header("X-File-Size", mmap_file.size());

    // Add Content-Range header for partial requests
    if is_partial {
        if let (Some(start), Some(end)) = (query.start, query.end) {
            response = response.header(
                header::CONTENT_RANGE,
                format!("bytes {}-{}/{}", start, end - 1, mmap_file.size()),
            );
        }
    }

    Ok(response.body(data.into()).unwrap())
}

/// Get cache statistics
async fn get_cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    use axum::body::Body;

    let stats = format!(
        "{{\"cache_size\": {}, \"is_empty\": {}}}",
        state.mmap_cache.len(),
        state.mmap_cache.is_empty()
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(stats))
        .unwrap()
}

/// Clear the mmap cache
async fn clear_cache(State(state): State<AppState>) -> impl IntoResponse {
    use axum::body::Body;

    state.mmap_cache.clear();

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{\"status\": \"cache cleared\"}"))
        .unwrap()
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    use axum::body::Body;

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{\"status\": \"ok\"}"))
        .unwrap()
}

// ============================================================================
// Error Handling
// ============================================================================

struct MmapErrorResponse(MmapError);

impl From<MmapError> for MmapErrorResponse {
    fn from(err: MmapError) -> Self {
        MmapErrorResponse(err)
    }
}

impl IntoResponse for MmapErrorResponse {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            MmapError::FileNotFound(path) => {
                (StatusCode::NOT_FOUND, format!("File not found: {}", path))
            }
            MmapError::InvalidRange(msg) => {
                (StatusCode::BAD_REQUEST, format!("Invalid range: {}", msg))
            }
            MmapError::FileOpen(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to open file: {}", err),
            ),
            MmapError::MmapCreation(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create memory map: {}", msg),
            ),
        };

        Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .body(format!("{{\"error\": \"{}\"}}", message).into())
            .unwrap()
    }
}

// ============================================================================
// Main Server
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create storage directory if it doesn't exist
    let storage_path = PathBuf::from("/tmp/ipfrs-tensors");
    std::fs::create_dir_all(&storage_path)?;

    println!("📁 Storage path: {}", storage_path.display());
    println!("💾 Creating memory-mapped file cache (max 100 entries)...");

    // Create application state
    let state = AppState::new(storage_path.clone(), 100);

    // Create some example tensor files
    create_example_tensors(&storage_path)?;

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/tensor/:name", get(get_tensor))
        .route("/cache/stats", get(get_cache_stats))
        .route("/cache/clear", get(clear_cache))
        .with_state(state);

    // Start server
    let addr = "127.0.0.1:8080";
    println!("🚀 Starting mmap tensor server on http://{}", addr);
    println!("\n📖 API Endpoints:");
    println!("  GET /health              - Health check");
    println!("  GET /tensor/:name        - Get full tensor (zero-copy)");
    println!("  GET /tensor/:name?start=0&end=100 - Get tensor range");
    println!("  GET /cache/stats         - Get cache statistics");
    println!("  GET /cache/clear         - Clear mmap cache");
    println!("\n💡 Example requests:");
    println!("  curl http://localhost:8080/tensor/example1");
    println!("  curl http://localhost:8080/tensor/example1?start=0&end=100");
    println!("  curl http://localhost:8080/cache/stats");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create example tensor files for testing
fn create_example_tensors(storage_path: &StdPath) -> std::io::Result<()> {
    use std::io::Write;

    // Example 1: Small tensor (1KB)
    let example1_path = storage_path.join("example1.tensor");
    let mut file1 = std::fs::File::create(&example1_path)?;
    let data1: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    file1.write_all(&data1)?;
    println!("✅ Created example1.tensor (1 KB)");

    // Example 2: Medium tensor (1MB)
    let example2_path = storage_path.join("example2.tensor");
    let mut file2 = std::fs::File::create(&example2_path)?;
    let data2: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
    file2.write_all(&data2)?;
    println!("✅ Created example2.tensor (1 MB)");

    // Example 3: Large tensor (10MB)
    let example3_path = storage_path.join("example3.tensor");
    let mut file3 = std::fs::File::create(&example3_path)?;
    let chunk: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    for _ in 0..10240 {
        file3.write_all(&chunk)?;
    }
    println!("✅ Created example3.tensor (10 MB)");

    Ok(())
}
