//! Gateway route handlers
//!
//! HTTP route handlers for the IPFRS gateway REST API.

use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{
    body::Body,
    extract::{Multipart, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    Json,
};
use ipfrs_core::Cid;
use ipfrs_semantic::{DistanceMetric, QueryFilter};
use ipfrs_storage::BlockStoreTrait;
use ipfrs_tensorlogic::{Predicate, Proof, Rule, Substitution, Term};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::middleware::{
    add_caching_headers, check_etag_match, not_modified_response, CacheConfig,
};

use super::{AppError, GatewayState};

// ============================================================================
// Handler Functions
// ============================================================================

/// Health check endpoint
pub(super) async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "ipfrs-gateway",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Prometheus metrics endpoint
///
/// Returns metrics in Prometheus text exposition format for scraping
pub(super) async fn metrics_endpoint() -> impl IntoResponse {
    match crate::metrics::encode_metrics() {
        Ok(metrics) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            metrics,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode metrics: {}", e),
        )
            .into_response(),
    }
}

/// Get content via IPFS gateway with range request support
///
/// Supports:
/// - Single range requests (Range: bytes=0-100)
/// - Multi-range requests (Range: bytes=0-100,200-300)
/// - Conditional requests (If-None-Match)
/// - Caching headers (ETag, Cache-Control)
pub(super) async fn get_content(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| AppError::InvalidCid(cid_str.clone()))?;

    // Cache configuration for CID-based content
    let cache_config = CacheConfig::default();

    // Check for conditional request (If-None-Match)
    if check_etag_match(&headers, &cid_str) {
        return Ok(not_modified_response(&cid_str, &cache_config));
    }

    match state.store.get(&cid).await? {
        Some(block) => {
            let data = block.data();
            let total_size = data.len();

            // Check for Range header
            if let Some(range_header) = headers.get(header::RANGE) {
                if let Ok(range_str) = range_header.to_str() {
                    // Try parsing multi-range first
                    if let Some(ranges) = parse_multi_range(range_str, total_size) {
                        if ranges.len() == 1 {
                            // Single range - simple response
                            let (start, end) = ranges[0];
                            let slice = &data[start..end];
                            let content_range =
                                format!("bytes {}-{}/{}", start, end - 1, total_size);

                            let mut response = Response::builder()
                                .status(StatusCode::PARTIAL_CONTENT)
                                .header(header::CONTENT_RANGE, content_range)
                                .header(header::CONTENT_LENGTH, slice.len().to_string())
                                .header(header::CONTENT_TYPE, "application/octet-stream")
                                .header(header::ACCEPT_RANGES, "bytes")
                                .body(Body::from(slice.to_vec()))
                                .expect("building PARTIAL_CONTENT response with valid headers and body is infallible");

                            // Add caching headers
                            add_caching_headers(response.headers_mut(), &cid_str, &cache_config);

                            return Ok(response);
                        } else {
                            // Multi-range response with multipart/byteranges
                            return Ok(build_multipart_response(
                                data,
                                &ranges,
                                total_size,
                                &cid_str,
                                &cache_config,
                            ));
                        }
                    }
                }
            }

            // No range request or invalid range, return full content
            let mut response = Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(header::CONTENT_LENGTH, total_size.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .body(Body::from(data.to_vec()))
                .expect("building OK response with valid headers and body is infallible");

            // Add caching headers
            add_caching_headers(response.headers_mut(), &cid_str, &cache_config);

            Ok(response)
        }
        None => Err(AppError::BlockNotFound(cid_str)),
    }
}

/// Parse HTTP Range header for single range
/// Returns (start, end) where end is exclusive
#[allow(dead_code)]
pub(crate) fn parse_range(range_str: &str, total_size: usize) -> Option<(usize, usize)> {
    // Expected format: "bytes=start-end" or "bytes=start-"
    let range_str = range_str.strip_prefix("bytes=")?;

    if let Some((start_str, end_str)) = range_str.split_once('-') {
        let start: usize = start_str.parse().ok()?;

        let end = if end_str.is_empty() {
            total_size
        } else {
            end_str.parse::<usize>().ok()? + 1
        };

        if start < total_size && start < end && end <= total_size {
            Some((start, end))
        } else {
            None
        }
    } else {
        None
    }
}

/// Parse HTTP Range header for multiple ranges
/// Returns Vec of (start, end) tuples where end is exclusive
/// Supports formats: "bytes=0-100", "bytes=0-100,200-300", "bytes=0-100, 200-300"
pub(crate) fn parse_multi_range(range_str: &str, total_size: usize) -> Option<Vec<(usize, usize)>> {
    let range_str = range_str.strip_prefix("bytes=")?;

    let mut ranges = Vec::new();

    for part in range_str.split(',') {
        let part = part.trim();
        if let Some((start_str, end_str)) = part.split_once('-') {
            // Handle suffix range (e.g., "-500" means last 500 bytes)
            if start_str.is_empty() {
                let suffix_len: usize = end_str.parse().ok()?;
                let start = total_size.saturating_sub(suffix_len);
                ranges.push((start, total_size));
                continue;
            }

            let start: usize = start_str.parse().ok()?;

            let end = if end_str.is_empty() {
                total_size
            } else {
                end_str.parse::<usize>().ok()? + 1
            };

            if start < total_size && start < end && end <= total_size {
                ranges.push((start, end));
            } else {
                return None; // Invalid range
            }
        } else {
            return None; // Invalid format
        }
    }

    if ranges.is_empty() {
        None
    } else {
        // Merge overlapping and adjacent ranges for efficiency
        Some(merge_ranges(ranges))
    }
}

/// Merge overlapping and adjacent ranges
pub(crate) fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.len() <= 1 {
        return ranges;
    }

    // Sort by start position
    ranges.sort_by_key(|r| r.0);

    let mut merged = Vec::new();
    let mut current = ranges[0];

    for range in ranges.into_iter().skip(1) {
        // Check if ranges overlap or are adjacent
        if range.0 <= current.1 {
            // Extend current range
            current.1 = current.1.max(range.1);
        } else {
            merged.push(current);
            current = range;
        }
    }
    merged.push(current);

    merged
}

/// Build multipart/byteranges response for multi-range requests
pub(crate) fn build_multipart_response(
    data: &[u8],
    ranges: &[(usize, usize)],
    total_size: usize,
    cid: &str,
    cache_config: &CacheConfig,
) -> Response {
    // Generate a boundary string
    let boundary = format!("ipfrs_boundary_{:x}", rand::random::<u64>());

    // Build multipart body
    let mut body = Vec::new();

    for (start, end) in ranges {
        // Part header
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n");
        body.extend_from_slice(
            format!(
                "Content-Range: bytes {}-{}/{}\r\n\r\n",
                start,
                end - 1,
                total_size
            )
            .as_bytes(),
        );

        // Part data
        body.extend_from_slice(&data[*start..*end]);
        body.extend_from_slice(b"\r\n");
    }

    // Final boundary
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    let content_type = format!("multipart/byteranges; boundary={}", boundary);

    let mut response = Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, body.len().to_string())
        .header(header::ACCEPT_RANGES, "bytes")
        .body(Body::from(body))
        .expect("building multipart response with valid headers and body is infallible");

    // Add caching headers
    add_caching_headers(response.headers_mut(), cid, cache_config);

    response
}

/// Version endpoint
#[derive(Serialize)]
pub(super) struct VersionResponse {
    #[serde(rename = "Version")]
    version: String,
    #[serde(rename = "System")]
    system: String,
}

pub(super) async fn api_version() -> impl IntoResponse {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        system: "ipfrs/0.3.0".to_string(),
    })
}

/// Block get request
#[derive(Deserialize)]
pub(super) struct BlockRequest {
    arg: String,
}

pub(super) async fn api_block_get(
    State(state): State<GatewayState>,
    Json(req): Json<BlockRequest>,
) -> Result<Response, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => Ok((StatusCode::OK, block.data().to_vec()).into_response()),
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

/// Block stat response
#[derive(Serialize)]
pub(super) struct BlockStatResponse {
    #[serde(rename = "Key")]
    key: String,
    #[serde(rename = "Size")]
    size: u64,
}

pub(super) async fn api_block_stat(
    State(state): State<GatewayState>,
    Json(req): Json<BlockRequest>,
) -> Result<Json<BlockStatResponse>, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => Ok(Json(BlockStatResponse {
            key: req.arg,
            size: block.size(),
        })),
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

pub(super) async fn api_cat(
    State(state): State<GatewayState>,
    Json(req): Json<BlockRequest>,
) -> Result<Response, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => Ok((StatusCode::OK, block.data().to_vec()).into_response()),
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

/// Add response
#[derive(Serialize)]
pub(super) struct AddResponse {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Hash")]
    hash: String,
    #[serde(rename = "Size")]
    size: String,
}

pub(super) async fn api_add(
    State(state): State<GatewayState>,
    mut multipart: Multipart,
) -> Result<Json<AddResponse>, AppError> {
    use bytes::Bytes;
    use ipfrs_core::Block;

    // Process the first file field
    if let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::Upload(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.file_name().unwrap_or("upload").to_string();
        let data = field
            .bytes()
            .await
            .map_err(|e| AppError::Upload(format!("Failed to read file data: {}", e)))?;

        // Create block from uploaded data
        let bytes_data = Bytes::from(data.to_vec());
        let block = Block::new(bytes_data)
            .map_err(|e| AppError::Upload(format!("Failed to create block: {}", e)))?;
        let cid = *block.cid();
        let size = block.size();

        // Store block
        state.store.put(&block).await?;

        info!("Added file '{}' as {}", name, cid);

        Ok(Json(AddResponse {
            name,
            hash: cid.to_string(),
            size: size.to_string(),
        }))
    } else {
        Err(AppError::Upload("No file provided".to_string()))
    }
}

/// Block put request with raw data
pub(super) async fn api_block_put(
    State(state): State<GatewayState>,
    body: axum::body::Bytes,
) -> Result<Json<AddResponse>, AppError> {
    use ipfrs_core::Block;

    // Create block from raw bytes
    let block =
        Block::new(body).map_err(|e| AppError::Upload(format!("Failed to create block: {}", e)))?;
    let cid = *block.cid();
    let size = block.size();

    // Store block
    state.store.put(&block).await?;

    info!("Stored raw block {}", cid);

    Ok(Json(AddResponse {
        name: cid.to_string(),
        hash: cid.to_string(),
        size: size.to_string(),
    }))
}

/// DAG get request
#[derive(Deserialize)]
pub(super) struct DagRequest {
    arg: String,
}

/// DAG get response
#[derive(Serialize)]
pub(super) struct DagGetResponse {
    #[serde(rename = "Data")]
    data: String,
}

pub(super) async fn api_dag_get(
    State(state): State<GatewayState>,
    Json(req): Json<DagRequest>,
) -> Result<Json<DagGetResponse>, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => {
            // Convert block data to base64 for JSON transport
            let data_base64 =
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, block.data());

            Ok(Json(DagGetResponse { data: data_base64 }))
        }
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

/// DAG put response
#[derive(Serialize)]
pub(super) struct DagPutResponse {
    #[serde(rename = "Cid")]
    cid: CidInfo,
}

#[derive(Serialize)]
pub(super) struct CidInfo {
    #[serde(rename = "/")]
    cid: String,
}

pub(super) async fn api_dag_put(
    State(state): State<GatewayState>,
    body: axum::body::Bytes,
) -> Result<Json<DagPutResponse>, AppError> {
    use ipfrs_core::Block;

    // Create block from DAG data
    let block = Block::new(body)
        .map_err(|e| AppError::Upload(format!("Failed to create DAG block: {}", e)))?;
    let cid = *block.cid();

    // Store block
    state.store.put(&block).await?;

    info!("Stored DAG node {}", cid);

    Ok(Json(DagPutResponse {
        cid: CidInfo {
            cid: cid.to_string(),
        },
    }))
}

/// DAG resolve request
#[derive(Deserialize)]
pub(super) struct DagResolveRequest {
    arg: String,
}

/// DAG resolve response
#[derive(Serialize)]
pub(super) struct DagResolveResponse {
    #[serde(rename = "Cid")]
    cid: CidInfo,
    #[serde(rename = "RemPath")]
    rem_path: String,
}

pub(super) async fn api_dag_resolve(
    State(_state): State<GatewayState>,
    Json(req): Json<DagResolveRequest>,
) -> Result<Json<DagResolveResponse>, AppError> {
    // Parse the path (e.g., "/ipfs/Qm.../path/to/data")
    let path = req.arg.trim_start_matches("/ipfs/");
    let parts: Vec<&str> = path.splitn(2, '/').collect();

    let cid_str = parts[0];
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| AppError::InvalidCid(cid_str.to_string()))?;

    let sub_path = if parts.len() > 1 { parts[1] } else { "" };

    // For now, we just return the root CID and the remainder path
    // Full IPLD path resolution would require parsing DAG-CBOR/JSON
    Ok(Json(DagResolveResponse {
        cid: CidInfo {
            cid: cid.to_string(),
        },
        rem_path: sub_path.to_string(),
    }))
}

// ============================================================================
// Semantic Search Handlers
// ============================================================================

/// Semantic index request
#[derive(Deserialize)]
pub(super) struct SemanticIndexRequest {
    cid: String,
    embedding: Vec<f32>,
}

/// Semantic index response
#[derive(Serialize)]
pub(super) struct SemanticIndexResponse {
    indexed: bool,
}

pub(super) async fn api_semantic_index(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticIndexRequest>,
) -> Result<Json<SemanticIndexResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    let cid: Cid = req
        .cid
        .parse()
        .map_err(|_| AppError::InvalidCid(req.cid.clone()))?;

    semantic
        .add(&cid, &req.embedding)
        .map_err(|e| AppError::Semantic(format!("Failed to index: {}", e)))?;

    info!("Indexed content {} with embedding", cid);

    Ok(Json(SemanticIndexResponse { indexed: true }))
}

/// Semantic search request
#[derive(Deserialize)]
pub(super) struct SemanticSearchRequest {
    query: Vec<f32>,
    k: Option<usize>,
    filter: Option<QueryFilter>,
}

/// Semantic search response
#[derive(Serialize)]
pub(super) struct SemanticSearchResponse {
    results: Vec<SearchResultJson>,
}

#[derive(Serialize)]
pub(super) struct SearchResultJson {
    cid: String,
    score: f32,
}

pub(super) async fn api_semantic_search(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticSearchRequest>,
) -> Result<Json<SemanticSearchResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    let k = req.k.unwrap_or(10);

    let results = if let Some(filter) = req.filter {
        semantic
            .query_with_filter(&req.query, k, filter)
            .await
            .map_err(|e| AppError::Semantic(format!("Search failed: {}", e)))?
    } else {
        semantic
            .query(&req.query, k)
            .await
            .map_err(|e| AppError::Semantic(format!("Search failed: {}", e)))?
    };

    let results_json: Vec<SearchResultJson> = results
        .into_iter()
        .map(|r| SearchResultJson {
            cid: r.cid.to_string(),
            score: r.score,
        })
        .collect();

    Ok(Json(SemanticSearchResponse {
        results: results_json,
    }))
}

/// Semantic stats response
#[derive(Serialize)]
pub(super) struct SemanticStatsResponse {
    num_vectors: usize,
    dimension: usize,
    metric: String,
    cache_size: usize,
    cache_capacity: usize,
}

pub(super) async fn api_semantic_stats(
    State(state): State<GatewayState>,
) -> Result<Json<SemanticStatsResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    let router_stats = semantic.stats();
    let cache_stats = semantic.cache_stats();

    let metric_str = match router_stats.metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::L2 => "l2",
        DistanceMetric::DotProduct => "dotproduct",
    };

    Ok(Json(SemanticStatsResponse {
        num_vectors: router_stats.num_vectors,
        dimension: router_stats.dimension,
        metric: metric_str.to_string(),
        cache_size: cache_stats.size,
        cache_capacity: cache_stats.capacity,
    }))
}

/// Semantic save request
#[derive(Deserialize)]
pub(super) struct SemanticSaveRequest {
    path: String,
}

/// Semantic save response
#[derive(Serialize)]
pub(super) struct SemanticSaveResponse {
    success: bool,
    path: String,
}

pub(super) async fn api_semantic_save(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticSaveRequest>,
) -> Result<Json<SemanticSaveResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    semantic
        .save_index(&req.path)
        .await
        .map_err(|e| AppError::Semantic(format!("Failed to save index: {}", e)))?;

    info!("Saved semantic index to {}", req.path);

    Ok(Json(SemanticSaveResponse {
        success: true,
        path: req.path,
    }))
}

/// Semantic load request
#[derive(Deserialize)]
pub(super) struct SemanticLoadRequest {
    path: String,
}

/// Semantic load response
#[derive(Serialize)]
pub(super) struct SemanticLoadResponse {
    success: bool,
    path: String,
}

pub(super) async fn api_semantic_load(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticLoadRequest>,
) -> Result<Json<SemanticLoadResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    semantic
        .load_index(&req.path)
        .await
        .map_err(|e| AppError::Semantic(format!("Failed to load index: {}", e)))?;

    info!("Loaded semantic index from {}", req.path);

    Ok(Json(SemanticLoadResponse {
        success: true,
        path: req.path,
    }))
}

// ============================================================================
// TensorLogic Handlers
// ============================================================================

/// Logic store term request
#[derive(Deserialize)]
pub(super) struct LogicStoreTermRequest {
    term: Term,
}

/// Logic store response
#[derive(Serialize)]
pub(super) struct LogicStoreResponse {
    cid: String,
}

pub(super) async fn api_logic_store_term(
    State(state): State<GatewayState>,
    Json(req): Json<LogicStoreTermRequest>,
) -> Result<Json<LogicStoreResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid = tensorlogic
        .store_term(&req.term)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to store term: {}", e)))?;

    info!("Stored term as {}", cid);

    Ok(Json(LogicStoreResponse {
        cid: cid.to_string(),
    }))
}

/// Logic get term response
#[derive(Serialize)]
pub(super) struct LogicGetTermResponse {
    term: Term,
}

pub(super) async fn api_logic_get_term(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
) -> Result<Json<LogicGetTermResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid: Cid = cid_str
        .parse()
        .map_err(|_| AppError::InvalidCid(cid_str.clone()))?;

    let term = tensorlogic
        .get_term(&cid)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to get term: {}", e)))?
        .ok_or_else(|| AppError::NotFound(format!("Term not found: {}", cid_str)))?;

    Ok(Json(LogicGetTermResponse { term }))
}

/// Logic store predicate request
#[derive(Deserialize)]
pub(super) struct LogicStorePredicateRequest {
    predicate: Predicate,
}

pub(super) async fn api_logic_store_predicate(
    State(state): State<GatewayState>,
    Json(req): Json<LogicStorePredicateRequest>,
) -> Result<Json<LogicStoreResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid = tensorlogic
        .store_predicate(&req.predicate)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to store predicate: {}", e)))?;

    info!("Stored predicate as {}", cid);

    Ok(Json(LogicStoreResponse {
        cid: cid.to_string(),
    }))
}

/// Logic store rule request
#[derive(Deserialize)]
pub(super) struct LogicStoreRuleRequest {
    rule: Rule,
}

pub(super) async fn api_logic_store_rule(
    State(state): State<GatewayState>,
    Json(req): Json<LogicStoreRuleRequest>,
) -> Result<Json<LogicStoreResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid = tensorlogic
        .store_rule(&req.rule)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to store rule: {}", e)))?;

    info!("Stored rule as {}", cid);

    Ok(Json(LogicStoreResponse {
        cid: cid.to_string(),
    }))
}

/// Logic stats response
#[derive(Serialize)]
pub(super) struct LogicStatsResponse {
    enabled: bool,
}

pub(super) async fn api_logic_stats(
    State(state): State<GatewayState>,
) -> Result<Json<LogicStatsResponse>, AppError> {
    let enabled = state.tensorlogic.is_some();

    Ok(Json(LogicStatsResponse { enabled }))
}

/// Add fact request
#[derive(Deserialize)]
pub(super) struct AddFactRequest {
    fact: Predicate,
}

/// Add fact response
#[derive(Serialize)]
pub(super) struct AddFactResponse {
    success: bool,
}

pub(super) async fn api_logic_add_fact(
    State(state): State<GatewayState>,
    Json(req): Json<AddFactRequest>,
) -> Result<Json<AddFactResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .add_fact(req.fact)
        .map_err(|e| AppError::Logic(format!("Failed to add fact: {}", e)))?;

    Ok(Json(AddFactResponse { success: true }))
}

/// Add rule request
#[derive(Deserialize)]
pub(super) struct AddRuleRequest {
    rule: Rule,
}

/// Add rule response
#[derive(Serialize)]
pub(super) struct AddRuleResponse {
    success: bool,
}

pub(super) async fn api_logic_add_rule(
    State(state): State<GatewayState>,
    Json(req): Json<AddRuleRequest>,
) -> Result<Json<AddRuleResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .add_rule(req.rule)
        .map_err(|e| AppError::Logic(format!("Failed to add rule: {}", e)))?;

    Ok(Json(AddRuleResponse { success: true }))
}

/// Infer request
#[derive(Deserialize)]
pub(super) struct InferRequest {
    goal: Predicate,
}

/// Infer response
#[derive(Serialize)]
pub(super) struct InferResponse {
    solutions: Vec<Substitution>,
}

pub(super) async fn api_logic_infer(
    State(state): State<GatewayState>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let solutions = tensorlogic
        .infer(&req.goal)
        .map_err(|e| AppError::Logic(format!("Inference failed: {}", e)))?;

    Ok(Json(InferResponse { solutions }))
}

/// Prove request
#[derive(Deserialize)]
pub(super) struct ProveRequest {
    goal: Predicate,
}

/// Prove response
#[derive(Serialize)]
pub(super) struct ProveResponse {
    proof: Option<Proof>,
    cid: Option<String>,
}

/// Verify request
#[derive(Deserialize)]
pub(super) struct VerifyRequest {
    proof: Proof,
}

/// Verify response
#[derive(Serialize)]
pub(super) struct VerifyResponse {
    valid: bool,
}

pub(super) async fn api_logic_prove(
    State(state): State<GatewayState>,
    Json(req): Json<ProveRequest>,
) -> Result<Json<ProveResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let proof = tensorlogic
        .prove(&req.goal)
        .map_err(|e| AppError::Logic(format!("Proof generation failed: {}", e)))?;

    if let Some(ref p) = proof {
        let cid = tensorlogic
            .store_proof(p)
            .await
            .map_err(|e| AppError::Logic(format!("Failed to store proof: {}", e)))?;

        Ok(Json(ProveResponse {
            proof,
            cid: Some(cid.to_string()),
        }))
    } else {
        Ok(Json(ProveResponse {
            proof: None,
            cid: None,
        }))
    }
}

pub(super) async fn api_logic_verify(
    State(state): State<GatewayState>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let valid = tensorlogic
        .verify_proof(&req.proof)
        .map_err(|e| AppError::Logic(format!("Proof verification failed: {}", e)))?;

    Ok(Json(VerifyResponse { valid }))
}

/// Get proof response
#[derive(Serialize)]
pub(super) struct GetProofResponse {
    proof: Proof,
}

pub(super) async fn api_logic_get_proof(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
) -> Result<Json<GetProofResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid: Cid = cid_str
        .parse()
        .map_err(|e| AppError::InvalidCid(format!("Invalid CID: {}", e)))?;

    let proof = tensorlogic
        .get_proof(&cid)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to get proof: {}", e)))?
        .ok_or_else(|| AppError::NotFound(format!("Proof not found: {}", cid)))?;

    Ok(Json(GetProofResponse { proof }))
}

/// KB stats response
#[derive(Serialize)]
pub(super) struct KbStatsResponse {
    num_facts: usize,
    num_rules: usize,
}

pub(super) async fn api_logic_kb_stats(
    State(state): State<GatewayState>,
) -> Result<Json<KbStatsResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let kb_stats = tensorlogic.kb_stats();

    Ok(Json(KbStatsResponse {
        num_facts: kb_stats.num_facts,
        num_rules: kb_stats.num_rules,
    }))
}

/// KB save request
#[derive(Deserialize)]
pub(super) struct KbSaveRequest {
    path: String,
}

/// KB save response
#[derive(Serialize)]
pub(super) struct KbSaveResponse {
    success: bool,
    path: String,
}

pub(super) async fn api_logic_kb_save(
    State(state): State<GatewayState>,
    Json(req): Json<KbSaveRequest>,
) -> Result<Json<KbSaveResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .save_kb(&req.path)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to save knowledge base: {}", e)))?;

    info!("Saved knowledge base to {}", req.path);

    Ok(Json(KbSaveResponse {
        success: true,
        path: req.path,
    }))
}

/// KB load request
#[derive(Deserialize)]
pub(super) struct KbLoadRequest {
    path: String,
}

/// KB load response
#[derive(Serialize)]
pub(super) struct KbLoadResponse {
    success: bool,
    path: String,
}

pub(super) async fn api_logic_kb_load(
    State(state): State<GatewayState>,
    Json(req): Json<KbLoadRequest>,
) -> Result<Json<KbLoadResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .load_kb(&req.path)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to load knowledge base: {}", e)))?;

    info!("Loaded knowledge base from {}", req.path);

    Ok(Json(KbLoadResponse {
        success: true,
        path: req.path,
    }))
}

// ============================================================================
// Network Handlers
// ============================================================================

/// Network ID response
#[derive(Serialize)]
pub(super) struct NetworkIdResponse {
    #[serde(rename = "ID")]
    id: String,
    #[serde(rename = "Addresses")]
    addresses: Vec<String>,
}

pub(super) async fn api_network_id(
    State(state): State<GatewayState>,
) -> Result<Json<NetworkIdResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let (peer_id, listeners) = {
        let network = network.lock().await;
        (network.peer_id().to_string(), network.listeners())
    };

    let addresses = listeners
        .iter()
        .map(|addr| format!("{}/p2p/{}", addr, peer_id))
        .collect();

    Ok(Json(NetworkIdResponse {
        id: peer_id,
        addresses,
    }))
}

/// Swarm peers response
#[derive(Serialize)]
pub(super) struct SwarmPeersResponse {
    #[serde(rename = "Peers")]
    peers: Vec<PeerEntry>,
}

#[derive(Serialize)]
pub(super) struct PeerEntry {
    #[serde(rename = "Peer")]
    peer: String,
}

pub(super) async fn api_swarm_peers(
    State(state): State<GatewayState>,
) -> Result<Json<SwarmPeersResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let peers = {
        let network = network.lock().await;
        network.connected_peers()
    };

    let peer_entries: Vec<PeerEntry> = peers
        .into_iter()
        .map(|p| PeerEntry {
            peer: p.to_string(),
        })
        .collect();

    Ok(Json(SwarmPeersResponse {
        peers: peer_entries,
    }))
}

/// Swarm connect request
#[derive(Deserialize)]
pub(super) struct SwarmConnectRequest {
    arg: String,
}

/// Swarm connect response
#[derive(Serialize)]
pub(super) struct SwarmConnectResponse {
    #[serde(rename = "Strings")]
    strings: Vec<String>,
}

pub(super) async fn api_swarm_connect(
    State(state): State<GatewayState>,
    Json(req): Json<SwarmConnectRequest>,
) -> Result<Json<SwarmConnectResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let addr: ipfrs_network::libp2p::Multiaddr = req
        .arg
        .parse()
        .map_err(|e| AppError::Network(format!("Invalid multiaddr: {}", e)))?;

    {
        let mut network = network.lock().await;
        network
            .connect(addr.clone())
            .await
            .map_err(|e| AppError::Network(format!("Connect failed: {}", e)))?;
    }

    info!("Connected to peer: {}", req.arg);

    Ok(Json(SwarmConnectResponse {
        strings: vec![format!("connect {} success", req.arg)],
    }))
}

/// Swarm disconnect request
#[derive(Deserialize)]
pub(super) struct SwarmDisconnectRequest {
    arg: String,
}

/// Swarm disconnect response
#[derive(Serialize)]
pub(super) struct SwarmDisconnectResponse {
    #[serde(rename = "Strings")]
    strings: Vec<String>,
}

pub(super) async fn api_swarm_disconnect(
    State(state): State<GatewayState>,
    Json(req): Json<SwarmDisconnectRequest>,
) -> Result<Json<SwarmDisconnectResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let peer_id: ipfrs_network::libp2p::PeerId = req
        .arg
        .parse()
        .map_err(|e| AppError::Network(format!("Invalid peer ID: {}", e)))?;

    {
        let mut network = network.lock().await;
        network
            .disconnect(peer_id)
            .await
            .map_err(|e| AppError::Network(format!("Disconnect failed: {}", e)))?;
    }

    info!("Disconnected from peer: {}", req.arg);

    Ok(Json(SwarmDisconnectResponse {
        strings: vec![format!("disconnect {} success", req.arg)],
    }))
}

/// DHT findprovs request
#[derive(Deserialize)]
pub(super) struct DhtFindprovsRequest {
    arg: String,
}

/// DHT findprovs response
#[derive(Serialize)]
pub(super) struct DhtFindprovsResponse {
    #[serde(rename = "Responses")]
    responses: Vec<DhtProviderEntry>,
}

#[derive(Serialize)]
pub(super) struct DhtProviderEntry {
    #[serde(rename = "ID")]
    id: String,
}

pub(super) async fn api_dht_findprovs(
    State(state): State<GatewayState>,
    Json(req): Json<DhtFindprovsRequest>,
) -> Result<Json<DhtFindprovsResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    {
        let mut network = network.lock().await;
        network
            .find_providers(&cid)
            .await
            .map_err(|e| AppError::Network(format!("Find providers failed: {}", e)))?;
    }

    info!("Finding providers for: {}", req.arg);

    // Note: In a real implementation, we would wait for the query to complete
    // and return the actual providers. For now, return empty list as the query
    // is asynchronous and results come via events.
    Ok(Json(DhtFindprovsResponse { responses: vec![] }))
}

/// DHT provide request
#[derive(Deserialize)]
pub(super) struct DhtProvideRequest {
    arg: String,
}

/// DHT provide response
#[derive(Serialize)]
pub(super) struct DhtProvideResponse {
    #[serde(rename = "ID")]
    id: String,
}

pub(super) async fn api_dht_provide(
    State(state): State<GatewayState>,
    Json(req): Json<DhtProvideRequest>,
) -> Result<Json<DhtProvideResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    {
        let mut network = network.lock().await;
        network
            .provide(&cid)
            .await
            .map_err(|e| AppError::Network(format!("Provide failed: {}", e)))?;
    }

    info!("Announcing content to DHT: {}", req.arg);

    Ok(Json(DhtProvideResponse { id: req.arg }))
}

// ============================================================================
// GraphQL Handlers
// ============================================================================

/// GraphQL query handler
pub(super) async fn graphql_handler(
    State(state): State<GatewayState>,
    req: GraphQLRequest,
) -> Result<GraphQLResponse, AppError> {
    let schema = state
        .graphql_schema
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("GraphQL not enabled".to_string()))?;

    Ok(schema.execute(req.into_inner()).await.into())
}

/// GraphQL playground handler
pub(super) async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}
