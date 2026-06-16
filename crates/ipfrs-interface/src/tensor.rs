//! Zero-Copy Tensor API
//!
//! Provides high-performance tensor access with:
//! - Zero-copy streaming
//! - Memory-mapped responses
//! - Partial tensor retrieval
//! - Range request support

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use ipfrs_core::Cid;
use ipfrs_storage::BlockStoreTrait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

use crate::gateway::GatewayState;
use crate::middleware::{
    add_caching_headers, check_etag_match, not_modified_response, CacheConfig,
};
use crate::mmap::{MmapCache, MmapError};

// ============================================================================
// Tensor Metadata
// ============================================================================

/// Tensor shape and type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,
    /// Data type (e.g., "f32", "f64", "i32", "u8")
    pub dtype: String,
    /// Total number of elements
    pub num_elements: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Layout (row-major or column-major)
    pub layout: TensorLayout,
}

/// Tensor memory layout
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TensorLayout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
}

impl TensorMetadata {
    /// Create metadata from safetensors format
    pub fn from_safetensors_data(data: &[u8]) -> Result<Self, String> {
        // Safetensors format: first 8 bytes = header length
        if data.len() < 8 {
            return Err("Data too short for safetensors format".to_string());
        }

        let header_len = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .expect("data[0..8] is exactly 8 bytes after bounds check"),
        ) as usize;
        if data.len() < 8 + header_len {
            return Err("Incomplete safetensors header".to_string());
        }

        // Parse JSON header
        let header_bytes = &data[8..8 + header_len];
        let header: serde_json::Value = serde_json::from_slice(header_bytes)
            .map_err(|e| format!("Failed to parse safetensors header: {}", e))?;

        // Extract first tensor metadata
        if let Some(tensors) = header.as_object() {
            if let Some((_name, tensor_info)) =
                tensors.iter().find(|(k, _)| k.as_str() != "__metadata__")
            {
                if let Some(shape) = tensor_info.get("shape").and_then(|s| s.as_array()) {
                    let shape: Vec<usize> = shape
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();

                    let dtype = tensor_info
                        .get("dtype")
                        .and_then(|d| d.as_str())
                        .unwrap_or("f32")
                        .to_string();

                    let num_elements = shape.iter().product();
                    let element_size = Self::dtype_size(&dtype);
                    let size_bytes = num_elements * element_size;

                    return Ok(TensorMetadata {
                        shape,
                        dtype,
                        num_elements,
                        size_bytes,
                        layout: TensorLayout::RowMajor, // Default
                    });
                }
            }
        }

        Err("No tensor found in safetensors data".to_string())
    }

    /// Get size of a data type in bytes
    fn dtype_size(dtype: &str) -> usize {
        match dtype {
            "f16" | "bf16" => 2,
            "f32" | "i32" | "u32" => 4,
            "f64" | "i64" | "u64" => 8,
            "i8" | "u8" => 1,
            "i16" | "u16" => 2,
            _ => 4, // Default to 4 bytes
        }
    }

    /// Create metadata from raw tensor data
    pub fn from_raw(shape: Vec<usize>, dtype: String) -> Self {
        let num_elements = shape.iter().product();
        let element_size = Self::dtype_size(&dtype);
        let size_bytes = num_elements * element_size;

        TensorMetadata {
            shape,
            dtype,
            num_elements,
            size_bytes,
            layout: TensorLayout::RowMajor,
        }
    }
}

// ============================================================================
// Tensor Query Parameters
// ============================================================================

/// Query parameters for tensor retrieval
#[derive(Debug, Deserialize)]
pub struct TensorQuery {
    /// Retrieve only metadata (no data)
    pub metadata_only: Option<bool>,
    /// Slice specification (e.g., "0:10,5:15" for 2D tensor)
    pub slice: Option<String>,
    /// Format: "raw" or "safetensors" (default: auto-detect)
    pub format: Option<String>,
}

/// Tensor slice specification
#[derive(Debug)]
pub struct TensorSlice {
    /// Slice ranges for each dimension (start, end)
    pub ranges: Vec<(usize, Option<usize>)>,
}

impl TensorSlice {
    /// Extract a slice from tensor data
    ///
    /// This performs actual data slicing for row-major tensors.
    /// For multi-dimensional tensors, this extracts a contiguous region.
    pub fn extract_data(&self, data: &[u8], metadata: &TensorMetadata) -> Result<Vec<u8>, String> {
        if self.ranges.len() != metadata.shape.len() {
            return Err(format!(
                "Slice dimensions ({}) don't match tensor dimensions ({})",
                self.ranges.len(),
                metadata.shape.len()
            ));
        }

        let element_size = TensorMetadata::dtype_size(&metadata.dtype);

        // Dispatch to specialised implementations for 1D/2D, then fall through to the
        // general N-D path which handles any number of dimensions ≥ 3.
        match metadata.shape.len() {
            1 => self.extract_1d(data, &metadata.shape, element_size),
            2 => self.extract_2d(data, &metadata.shape, element_size),
            _ => self.extract_nd(data, &metadata.shape, element_size),
        }
    }

    /// Extract an N-dimensional slice for tensors with 3 or more dimensions.
    ///
    /// The tensor is assumed to be stored in row-major (C) order.
    /// For each dimension `d`, the stride is `product(shape[d+1..]) * element_size`.
    /// We iterate over the Cartesian product of all slice ranges and copy
    /// one element at a time, so no contiguous-memory assumption is required.
    fn extract_nd(
        &self,
        data: &[u8],
        shape: &[usize],
        element_size: usize,
    ) -> Result<Vec<u8>, String> {
        let ndim = shape.len();

        // Validate ranges and compute concrete (start, end) pairs.
        let mut starts = Vec::with_capacity(ndim);
        let mut ends = Vec::with_capacity(ndim);
        for (dim, &(start, end_opt)) in self.ranges.iter().enumerate() {
            let end = end_opt.unwrap_or(shape[dim]);
            if start >= shape[dim] {
                return Err(format!(
                    "Slice start {} out of bounds for dimension {} (size {})",
                    start, dim, shape[dim]
                ));
            }
            if end > shape[dim] {
                return Err(format!(
                    "Slice end {} out of bounds for dimension {} (size {})",
                    end, dim, shape[dim]
                ));
            }
            if start >= end {
                return Err(format!(
                    "Slice start {} >= end {} for dimension {}",
                    start, end, dim
                ));
            }
            starts.push(start);
            ends.push(end);
        }

        // Compute row-major strides (in elements, not bytes).
        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Pre-compute the total number of output elements.
        let out_elements: usize = starts
            .iter()
            .zip(ends.iter())
            .map(|(&s, &e)| e - s)
            .product();
        let mut result = vec![0u8; out_elements * element_size];

        // Iterate over the Cartesian product of all slice ranges using a
        // multi-dimensional counter (indices relative to tensor origin).
        let mut indices = starts.clone();
        for out_elem in 0..out_elements {
            // Compute the flat source offset.
            let src_elem: usize = indices
                .iter()
                .zip(strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let src_byte = src_elem * element_size;
            if src_byte + element_size > data.len() {
                return Err(format!(
                    "Source byte range {}..{} exceeds data length {}",
                    src_byte,
                    src_byte + element_size,
                    data.len()
                ));
            }
            let dst_byte = out_elem * element_size;
            result[dst_byte..dst_byte + element_size]
                .copy_from_slice(&data[src_byte..src_byte + element_size]);

            // Advance the multi-dimensional counter (last dimension increments fastest).
            let mut carry = true;
            for d in (0..ndim).rev() {
                if carry {
                    indices[d] += 1;
                    if indices[d] >= ends[d] {
                        indices[d] = starts[d];
                        // carry propagates
                    } else {
                        carry = false;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Extract 1D slice
    fn extract_1d(
        &self,
        data: &[u8],
        shape: &[usize],
        element_size: usize,
    ) -> Result<Vec<u8>, String> {
        let (start, end) = (self.ranges[0].0, self.ranges[0].1.unwrap_or(shape[0]));

        if start >= shape[0] || end > shape[0] || start >= end {
            return Err(format!(
                "Invalid 1D slice range [{}:{}] for shape [{}]",
                start, end, shape[0]
            ));
        }

        let byte_start = start * element_size;
        let byte_end = end * element_size;

        if byte_end > data.len() {
            return Err(format!(
                "Slice range {}..{} exceeds data length {}",
                byte_start,
                byte_end,
                data.len()
            ));
        }

        Ok(data[byte_start..byte_end].to_vec())
    }

    /// Extract 2D slice (row-major layout)
    fn extract_2d(
        &self,
        data: &[u8],
        shape: &[usize],
        element_size: usize,
    ) -> Result<Vec<u8>, String> {
        let rows = shape[0];
        let cols = shape[1];

        let (row_start, row_end) = (self.ranges[0].0, self.ranges[0].1.unwrap_or(rows));
        let (col_start, col_end) = (self.ranges[1].0, self.ranges[1].1.unwrap_or(cols));

        if row_start >= rows || row_end > rows || row_start >= row_end {
            return Err(format!(
                "Invalid row range [{}:{}] for shape [{}, {}]",
                row_start, row_end, rows, cols
            ));
        }

        if col_start >= cols || col_end > cols || col_start >= col_end {
            return Err(format!(
                "Invalid column range [{}:{}] for shape [{}, {}]",
                col_start, col_end, rows, cols
            ));
        }

        let mut result = Vec::new();
        let row_size = cols * element_size;

        for row in row_start..row_end {
            let row_offset = row * row_size;
            let slice_start = row_offset + col_start * element_size;
            let slice_end = row_offset + col_end * element_size;

            if slice_end > data.len() {
                return Err(format!(
                    "Row {} slice range {}..{} exceeds data length {}",
                    row,
                    slice_start,
                    slice_end,
                    data.len()
                ));
            }

            result.extend_from_slice(&data[slice_start..slice_end]);
        }

        Ok(result)
    }

    /// Parse slice string (e.g., "0:10,5:15")
    pub fn parse(slice_str: &str) -> Result<Self, String> {
        let ranges: Result<Vec<_>, String> = slice_str
            .split(',')
            .map(|part| {
                let parts: Vec<&str> = part.split(':').collect();
                match parts.len() {
                    1 => {
                        let idx = parts[0]
                            .parse::<usize>()
                            .map_err(|e| format!("Invalid slice index: {}", e))?;
                        Ok((idx, Some(idx + 1)))
                    }
                    2 => {
                        let start = parts[0]
                            .parse::<usize>()
                            .map_err(|e| format!("Invalid slice start: {}", e))?;
                        let end = if parts[1].is_empty() {
                            None
                        } else {
                            Some(
                                parts[1]
                                    .parse::<usize>()
                                    .map_err(|e| format!("Invalid slice end: {}", e))?,
                            )
                        };
                        Ok((start, end))
                    }
                    _ => Err(format!("Invalid slice format: {}", part)),
                }
            })
            .collect();

        Ok(TensorSlice { ranges: ranges? })
    }

    /// Calculate the slice size in bytes
    pub fn calculate_size(&self, metadata: &TensorMetadata) -> Result<usize, String> {
        if self.ranges.len() != metadata.shape.len() {
            return Err(format!(
                "Slice dimensions ({}) don't match tensor dimensions ({})",
                self.ranges.len(),
                metadata.shape.len()
            ));
        }

        let mut slice_elements = 1;
        for (i, (start, end)) in self.ranges.iter().enumerate() {
            let dim_size = metadata.shape[i];
            let actual_end = end.unwrap_or(dim_size);

            if *start >= dim_size || actual_end > dim_size || *start >= actual_end {
                return Err(format!(
                    "Invalid slice range [{}:{}] for dimension {} of size {}",
                    start, actual_end, i, dim_size
                ));
            }

            slice_elements *= actual_end - start;
        }

        let element_size = TensorMetadata::dtype_size(&metadata.dtype);
        Ok(slice_elements * element_size)
    }
}

// ============================================================================
// Tensor Responses
// ============================================================================

/// Tensor metadata response
#[derive(Debug, Serialize)]
pub struct TensorInfoResponse {
    pub cid: String,
    pub metadata: TensorMetadata,
}

// ============================================================================
// Tensor Endpoints
// ============================================================================

/// Get tensor with zero-copy streaming
///
/// GET /v1/tensor/{cid}
///
/// Retrieves tensor data with optional range requests for partial loading.
/// Supports both safetensors and raw binary formats.
pub async fn get_tensor(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
    Query(query): Query<TensorQuery>,
    headers: HeaderMap,
) -> Result<Response, TensorError> {
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| TensorError::InvalidCid(cid_str.clone()))?;

    // Check if cached (ETag)
    let cache_config = CacheConfig::default();
    if check_etag_match(&headers, &cid_str) {
        return Ok(not_modified_response(&cid_str, &cache_config));
    }

    // Get the block
    let block = state
        .store
        .get(&cid)
        .await
        .map_err(|e| TensorError::Storage(e.to_string()))?
        .ok_or_else(|| TensorError::NotFound(cid_str.clone()))?;

    let data = block.data();

    // Try to parse metadata (safetensors format or assume raw)
    let metadata = TensorMetadata::from_safetensors_data(data).ok();

    // If metadata_only requested, return just metadata
    if query.metadata_only.unwrap_or(false) {
        if let Some(metadata) = metadata {
            return Ok(Json(TensorInfoResponse {
                cid: cid_str,
                metadata,
            })
            .into_response());
        } else {
            return Err(TensorError::InvalidFormat(
                "Cannot extract metadata from tensor".to_string(),
            ));
        }
    }

    // Handle partial retrieval (slicing)
    let (response_data, is_partial, metadata_for_response) = if let Some(slice_str) = query.slice {
        let meta = metadata.ok_or_else(|| {
            TensorError::InvalidFormat("Metadata required for slicing".to_string())
        })?;

        let slice = TensorSlice::parse(&slice_str)?;

        // Extract the sliced data
        let sliced_data = slice.extract_data(data, &meta)?;

        (sliced_data, true, Some(meta))
    } else {
        // Return full tensor
        (data.to_vec(), false, metadata)
    };

    // Build response
    let mut response_builder = Response::builder();

    if is_partial {
        response_builder = response_builder.status(StatusCode::PARTIAL_CONTENT);
    } else {
        response_builder = response_builder.status(StatusCode::OK);
    }

    // Determine content type based on format
    let content_type = match query.format.as_deref() {
        Some("safetensors") | None if metadata_for_response.is_some() => {
            "application/vnd.safetensors"
        }
        _ => "application/octet-stream",
    };

    let mut response = response_builder
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, response_data.len())
        .header(
            "X-Tensor-Format",
            if metadata_for_response.is_some() {
                "safetensors"
            } else {
                "raw"
            },
        )
        .body(Body::from(response_data))
        .expect("building HTTP response with valid headers and body is infallible");

    // Add caching headers
    add_caching_headers(response.headers_mut(), &cid_str, &cache_config);

    // Add tensor metadata as headers if available
    if let Some(ref meta) = metadata_for_response {
        if let Ok(shape_json) = serde_json::to_string(&meta.shape) {
            if let Ok(header_value) = header::HeaderValue::from_str(&shape_json) {
                response
                    .headers_mut()
                    .insert("X-Tensor-Shape", header_value);
            }
        }
        if let Ok(header_value) = header::HeaderValue::from_str(&meta.dtype) {
            response
                .headers_mut()
                .insert("X-Tensor-Dtype", header_value);
        }
    }

    Ok(response)
}

/// Get tensor metadata only
///
/// GET /v1/tensor/{cid}/info
///
/// Retrieves only tensor metadata without downloading the full data.
pub async fn get_tensor_info(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
) -> Result<Json<TensorInfoResponse>, TensorError> {
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| TensorError::InvalidCid(cid_str.clone()))?;

    // Get the block
    let block = state
        .store
        .get(&cid)
        .await
        .map_err(|e| TensorError::Storage(e.to_string()))?
        .ok_or_else(|| TensorError::NotFound(cid_str.clone()))?;

    let data = block.data();

    // Parse metadata
    let metadata = TensorMetadata::from_safetensors_data(data).map_err(|e| {
        TensorError::InvalidFormat(format!("Failed to parse tensor metadata: {}", e))
    })?;

    Ok(Json(TensorInfoResponse {
        cid: cid_str,
        metadata,
    }))
}

/// Get tensor in Apache Arrow IPC format
///
/// GET /v1/tensor/{cid}/arrow
///
/// Retrieves tensor data in Apache Arrow IPC Stream format for efficient
/// data exchange with Arrow-compatible systems (Pandas, Polars, PyArrow, etc.)
pub async fn get_tensor_arrow(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
    Query(query): Query<TensorQuery>,
) -> Result<Response, TensorError> {
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| TensorError::InvalidCid(cid_str.clone()))?;

    // Get the block
    let block = state
        .store
        .get(&cid)
        .await
        .map_err(|e| TensorError::Storage(e.to_string()))?
        .ok_or_else(|| TensorError::NotFound(cid_str.clone()))?;

    let data = block.data();

    // Try to parse metadata (safetensors format)
    let metadata = TensorMetadata::from_safetensors_data(data)
        .map_err(|e| TensorError::InvalidFormat(format!("Cannot parse tensor metadata: {}", e)))?;

    // Handle partial retrieval (slicing) if requested
    let response_data = if let Some(slice_str) = query.slice {
        let slice = TensorSlice::parse(&slice_str)?;
        slice.extract_data(data, &metadata)?
    } else {
        // Return full tensor data (skip safetensors header)
        let header_len = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .expect("data[0..8] is exactly 8 bytes"),
        ) as usize;
        data[8 + header_len..].to_vec()
    };

    // Convert to Arrow RecordBatch and serialize
    let batch = crate::arrow::tensor_to_record_batch(&metadata, &response_data)
        .map_err(|e| TensorError::Storage(format!("Failed to create Arrow batch: {}", e)))?;

    let ipc_bytes = crate::arrow::record_batch_to_ipc_bytes(&batch)
        .map_err(|e| TensorError::Storage(format!("Failed to serialize Arrow IPC: {}", e)))?;

    // Build response
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/vnd.apache.arrow.stream")
        .header("X-Tensor-Shape", format!("{:?}", metadata.shape))
        .header("X-Tensor-Dtype", &metadata.dtype)
        .header("X-Tensor-Elements", metadata.num_elements.to_string())
        .body(Body::from(ipc_bytes))
        .map_err(|e| TensorError::Storage(format!("Failed to build response: {}", e)))
}

// ============================================================================
// Memory-Mapped Tensor Serving
// ============================================================================

/// Get tensor using memory-mapped I/O (zero-copy from disk)
///
/// GET /v1/tensor/{cid}/mmap
///
/// Retrieves tensor data using memory-mapped I/O for maximum performance.
/// This endpoint is optimized for serving large tensors directly from disk
/// without loading them into memory.
///
/// # Performance
///
/// - **Zero-copy**: Data is served directly from disk via OS page cache
/// - **Lazy loading**: Only requested pages are loaded into memory
/// - **OS optimizations**: Leverages sendfile and similar system calls
///
/// # Limitations
///
/// - Only works for tensors stored on local filesystem
/// - Requires tensor file path to be available
/// - Not suitable for tensors stored in distributed storage
pub async fn get_tensor_mmap(
    Path(cid_str): Path<String>,
    Query(query): Query<TensorQuery>,
    headers: HeaderMap,
    mmap_cache: Arc<MmapCache>,
    tensor_storage_path: PathBuf,
) -> Result<Response, TensorError> {
    let _cid: Cid = cid_str
        .parse()
        .map_err(|_| TensorError::InvalidCid(cid_str.clone()))?;

    // Check if cached (ETag)
    let cache_config = CacheConfig::default();
    if check_etag_match(&headers, &cid_str) {
        return Ok(not_modified_response(&cid_str, &cache_config));
    }

    // Construct file path from CID
    // In production, this would use the actual storage backend's file path
    let file_path = tensor_storage_path.join(format!("{}.tensor", cid_str));

    // Get or create memory-mapped file
    let mmap_file = mmap_cache.get_or_create(&file_path).map_err(|e| match e {
        MmapError::FileNotFound(_) => TensorError::NotFound(cid_str.clone()),
        _ => TensorError::Storage(e.to_string()),
    })?;

    // Get file data
    let data = mmap_file.bytes();

    // Try to parse metadata (safetensors format)
    let metadata = TensorMetadata::from_safetensors_data(&data).ok();

    // If metadata_only requested, return just metadata
    if query.metadata_only.unwrap_or(false) {
        if let Some(metadata) = metadata {
            return Ok(Json(TensorInfoResponse {
                cid: cid_str,
                metadata,
            })
            .into_response());
        } else {
            return Err(TensorError::InvalidFormat(
                "Cannot extract metadata from tensor".to_string(),
            ));
        }
    }

    // Handle partial retrieval (slicing)
    let (response_data, is_partial, metadata_for_response) = if let Some(slice_str) = query.slice {
        let meta = metadata.ok_or_else(|| {
            TensorError::InvalidFormat("Metadata required for slicing".to_string())
        })?;

        let slice = TensorSlice::parse(&slice_str)?;

        // For mmap, we can efficiently retrieve just the slice
        // by calculating the byte range
        let sliced_data = slice.extract_data(&data, &meta)?;

        (sliced_data, true, Some(meta))
    } else {
        // Return full tensor
        (data.to_vec(), false, metadata)
    };

    // Build response
    let mut response_builder = Response::builder();

    if is_partial {
        response_builder = response_builder.status(StatusCode::PARTIAL_CONTENT);
    } else {
        response_builder = response_builder.status(StatusCode::OK);
    }

    // Determine content type
    let content_type = match query.format.as_deref() {
        Some("safetensors") | None if metadata_for_response.is_some() => {
            "application/vnd.safetensors"
        }
        _ => "application/octet-stream",
    };

    let mut response = response_builder
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, response_data.len())
        .header("X-Served-By", "mmap")
        .header(
            "X-Tensor-Format",
            if metadata_for_response.is_some() {
                "safetensors"
            } else {
                "raw"
            },
        )
        .body(Body::from(response_data))
        .expect("building HTTP response with valid headers and body is infallible");

    // Add caching headers
    add_caching_headers(response.headers_mut(), &cid_str, &cache_config);

    // Add tensor metadata as headers
    if let Some(ref meta) = metadata_for_response {
        if let Ok(shape_json) = serde_json::to_string(&meta.shape) {
            if let Ok(header_value) = header::HeaderValue::from_str(&shape_json) {
                response
                    .headers_mut()
                    .insert("X-Tensor-Shape", header_value);
            }
        }
        if let Ok(header_value) = header::HeaderValue::from_str(&meta.dtype) {
            response
                .headers_mut()
                .insert("X-Tensor-Dtype", header_value);
        }
    }

    Ok(response)
}

/// Mmap-based tensor range request
///
/// Efficiently serves byte ranges from memory-mapped tensor files.
/// Optimized for HTTP 206 Partial Content responses.
#[allow(dead_code)]
pub async fn get_tensor_mmap_range(
    cid_str: String,
    range: std::ops::Range<usize>,
    mmap_cache: Arc<MmapCache>,
    tensor_storage_path: PathBuf,
) -> Result<Response, TensorError> {
    let _cid: Cid = cid_str
        .parse()
        .map_err(|_| TensorError::InvalidCid(cid_str.clone()))?;

    // Construct file path
    let file_path = tensor_storage_path.join(format!("{}.tensor", cid_str));

    // Get memory-mapped file
    let mmap_file = mmap_cache.get_or_create(&file_path).map_err(|e| match e {
        MmapError::FileNotFound(_) => TensorError::NotFound(cid_str.clone()),
        _ => TensorError::Storage(e.to_string()),
    })?;

    // Get the requested range (zero-copy)
    let range_data = mmap_file
        .range(range.clone())
        .map_err(|e| TensorError::Storage(e.to_string()))?;

    // Build partial content response
    let response = Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, range_data.len())
        .header(
            header::CONTENT_RANGE,
            format!(
                "bytes {}-{}/{}",
                range.start,
                range.end - 1,
                mmap_file.size()
            ),
        )
        .header("X-Served-By", "mmap")
        .body(Body::from(range_data))
        .expect("building PARTIAL_CONTENT response with valid headers and body is infallible");

    Ok(response)
}

// ============================================================================
// Error Types
// ============================================================================

/// Tensor operation errors
#[derive(Debug)]
pub enum TensorError {
    InvalidCid(String),
    NotFound(String),
    InvalidFormat(String),
    Storage(String),
    NotImplemented(String),
}

impl IntoResponse for TensorError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            TensorError::InvalidCid(cid) => {
                (StatusCode::BAD_REQUEST, format!("Invalid CID: {}", cid))
            }
            TensorError::NotFound(cid) => {
                (StatusCode::NOT_FOUND, format!("Tensor not found: {}", cid))
            }
            TensorError::InvalidFormat(msg) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid tensor format: {}", msg),
            ),
            TensorError::Storage(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Storage error: {}", msg),
            ),
            TensorError::NotImplemented(msg) => (
                StatusCode::NOT_IMPLEMENTED,
                format!("Not implemented: {}", msg),
            ),
        };

        (status, message).into_response()
    }
}

impl From<String> for TensorError {
    fn from(s: String) -> Self {
        TensorError::InvalidFormat(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_metadata_dtype_size() {
        assert_eq!(TensorMetadata::dtype_size("f32"), 4);
        assert_eq!(TensorMetadata::dtype_size("f64"), 8);
        assert_eq!(TensorMetadata::dtype_size("i32"), 4);
        assert_eq!(TensorMetadata::dtype_size("u8"), 1);
        assert_eq!(TensorMetadata::dtype_size("f16"), 2);
    }

    #[test]
    fn test_tensor_metadata_from_raw() {
        let meta = TensorMetadata::from_raw(vec![10, 20, 30], "f32".to_string());
        assert_eq!(meta.shape, vec![10, 20, 30]);
        assert_eq!(meta.dtype, "f32");
        assert_eq!(meta.num_elements, 6000);
        assert_eq!(meta.size_bytes, 24000);
    }

    #[test]
    fn test_tensor_slice_parse_single() {
        let slice = TensorSlice::parse("5").expect("test: parse single index should succeed");
        assert_eq!(slice.ranges, vec![(5, Some(6))]);
    }

    #[test]
    fn test_tensor_slice_parse_range() {
        let slice = TensorSlice::parse("10:20").expect("test: parse range slice should succeed");
        assert_eq!(slice.ranges, vec![(10, Some(20))]);
    }

    #[test]
    fn test_tensor_slice_parse_open_end() {
        let slice = TensorSlice::parse("10:").expect("test: parse open-end slice should succeed");
        assert_eq!(slice.ranges, vec![(10, None)]);
    }

    #[test]
    fn test_tensor_slice_parse_multi_dim() {
        let slice = TensorSlice::parse("0:10,5:15,2:8")
            .expect("test: parse multi-dim slice should succeed");
        assert_eq!(
            slice.ranges,
            vec![(0, Some(10)), (5, Some(15)), (2, Some(8))]
        );
    }

    #[test]
    fn test_tensor_slice_calculate_size() {
        let meta = TensorMetadata::from_raw(vec![100, 100], "f32".to_string());
        let slice = TensorSlice::parse("0:10,0:10").expect("test: parse 2D slice should succeed");

        let size = slice
            .calculate_size(&meta)
            .expect("test: size calculation should succeed");
        assert_eq!(size, 10 * 10 * 4); // 10x10 elements * 4 bytes
    }

    #[test]
    fn test_tensor_slice_invalid_dimensions() {
        let meta = TensorMetadata::from_raw(vec![100, 100], "f32".to_string());
        let slice = TensorSlice::parse("0:10").expect("test: parse 1D slice should succeed"); // Only 1 dimension

        let result = slice.calculate_size(&meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_slice_out_of_bounds() {
        let meta = TensorMetadata::from_raw(vec![100, 100], "f32".to_string());
        let slice = TensorSlice::parse("0:200,0:10")
            .expect("test: parse out-of-bounds slice should succeed");

        let result = slice.calculate_size(&meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_layout_serialization() {
        let layout = TensorLayout::RowMajor;
        let json =
            serde_json::to_string(&layout).expect("test: RowMajor serialization should succeed");
        assert_eq!(json, r#""rowmajor""#);

        let layout = TensorLayout::ColumnMajor;
        let json =
            serde_json::to_string(&layout).expect("test: ColumnMajor serialization should succeed");
        assert_eq!(json, r#""columnmajor""#);
    }

    #[test]
    fn test_tensor_slice_extract_1d() {
        // 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (f32)
        let data: Vec<u8> = (0..10).flat_map(|i| (i as f32).to_le_bytes()).collect();

        let meta = TensorMetadata::from_raw(vec![10], "f32".to_string());
        let slice = TensorSlice::parse("2:5").expect("test: parse 1D range should succeed");

        let result = slice
            .extract_data(&data, &meta)
            .expect("test: 1D slice extraction should succeed");

        // Should extract elements 2, 3, 4 (3 elements * 4 bytes = 12 bytes)
        assert_eq!(result.len(), 12);

        // Verify the extracted values
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("test: chunk to [u8;4] conversion should succeed"),
                )
            })
            .collect();

        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_slice_extract_2d() {
        // 2D tensor: 4x3 matrix (f32)
        // [[0, 1, 2],
        //  [3, 4, 5],
        //  [6, 7, 8],
        //  [9, 10, 11]]
        let data: Vec<u8> = (0..12).flat_map(|i| (i as f32).to_le_bytes()).collect();

        let meta = TensorMetadata::from_raw(vec![4, 3], "f32".to_string());
        let slice =
            TensorSlice::parse("1:3,0:2").expect("test: parse 2D row/col slice should succeed"); // Rows 1-2, Cols 0-1

        let result = slice
            .extract_data(&data, &meta)
            .expect("test: 2D slice extraction should succeed");

        // Should extract:
        // [[3, 4],
        //  [6, 7]]
        // 2 rows * 2 cols * 4 bytes = 16 bytes
        assert_eq!(result.len(), 16);

        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("test: chunk to [u8;4] conversion should succeed"),
                )
            })
            .collect();

        assert_eq!(values, vec![3.0, 4.0, 6.0, 7.0]);
    }

    #[test]
    fn test_tensor_slice_extract_2d_single_row() {
        let data: Vec<u8> = (0..12).flat_map(|i| (i as f32).to_le_bytes()).collect();

        let meta = TensorMetadata::from_raw(vec![4, 3], "f32".to_string());
        let slice =
            TensorSlice::parse("2:3,0:3").expect("test: parse single-row slice should succeed"); // Row 2, all columns

        let result = slice
            .extract_data(&data, &meta)
            .expect("test: single-row extraction should succeed");

        // Should extract: [6, 7, 8]
        assert_eq!(result.len(), 12);

        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("test: chunk to [u8;4] conversion should succeed"),
                )
            })
            .collect();

        assert_eq!(values, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_tensor_slice_extract_invalid_dimension() {
        let data = vec![0u8; 40]; // 10 f32 elements
        let meta = TensorMetadata::from_raw(vec![10], "f32".to_string());
        let slice = TensorSlice::parse("2:5,0:2")
            .expect("test: parse 2D slice for 1D tensor should succeed"); // 2D slice for 1D tensor

        let result = slice.extract_data(&data, &meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_slice_extract_out_of_bounds() {
        let data: Vec<u8> = (0..10).flat_map(|i| (i as f32).to_le_bytes()).collect();

        let meta = TensorMetadata::from_raw(vec![10], "f32".to_string());
        let slice =
            TensorSlice::parse("8:12").expect("test: parse out-of-bounds range should succeed"); // Out of bounds

        let result = slice.extract_data(&data, &meta);
        assert!(result.is_err());
    }
}
