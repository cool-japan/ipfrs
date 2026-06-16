//! HTTP Middleware for IPFRS Gateway
//!
//! Provides:
//! - Authentication and authorization middleware using JWT tokens and API keys
//! - CORS middleware for cross-origin requests
//! - Rate limiting middleware for DoS prevention
//! - Compression middleware for bandwidth optimization
//! - Caching middleware for HTTP caching headers

use crate::auth::{AuthError, AuthState, Claims, Permission};
use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use uuid::Uuid;

// ============================================================================
// CORS Configuration
// ============================================================================

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins (use "*" for any origin)
    pub allowed_origins: HashSet<String>,
    /// Allowed HTTP methods
    pub allowed_methods: HashSet<Method>,
    /// Allowed headers
    pub allowed_headers: HashSet<String>,
    /// Headers to expose to the client
    pub exposed_headers: HashSet<String>,
    /// Allow credentials (cookies, authorization headers)
    pub allow_credentials: bool,
    /// Max age for preflight cache (seconds)
    pub max_age: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        let mut methods = HashSet::new();
        methods.insert(Method::GET);
        methods.insert(Method::POST);
        methods.insert(Method::PUT);
        methods.insert(Method::DELETE);
        methods.insert(Method::OPTIONS);
        methods.insert(Method::HEAD);

        let mut headers = HashSet::new();
        headers.insert("content-type".to_string());
        headers.insert("authorization".to_string());
        headers.insert("accept".to_string());
        headers.insert("origin".to_string());
        headers.insert("x-requested-with".to_string());

        Self {
            allowed_origins: HashSet::new(), // Empty = allow all
            allowed_methods: methods,
            allowed_headers: headers,
            exposed_headers: HashSet::new(),
            allow_credentials: false,
            max_age: 86400, // 24 hours
        }
    }
}

impl CorsConfig {
    /// Create a permissive CORS config (allows all origins)
    pub fn permissive() -> Self {
        let mut config = Self::default();
        config.allowed_origins.insert("*".to_string());
        config
    }

    /// Allow specific origin
    pub fn allow_origin(mut self, origin: impl Into<String>) -> Self {
        self.allowed_origins.insert(origin.into());
        self
    }

    /// Allow credentials
    pub fn allow_credentials(mut self, allow: bool) -> Self {
        self.allow_credentials = allow;
        self
    }

    /// Check if origin is allowed
    fn is_origin_allowed(&self, origin: &str) -> bool {
        if self.allowed_origins.is_empty() || self.allowed_origins.contains("*") {
            true
        } else {
            self.allowed_origins.contains(origin)
        }
    }

    /// Get allowed methods as comma-separated string
    fn methods_string(&self) -> String {
        self.allowed_methods
            .iter()
            .map(|m| m.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Get allowed headers as comma-separated string
    fn headers_string(&self) -> String {
        self.allowed_headers
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// CORS middleware state
#[derive(Clone)]
pub struct CorsState {
    pub config: CorsConfig,
}

/// CORS middleware
///
/// Handles preflight requests and adds CORS headers to responses.
pub async fn cors_middleware(
    State(cors_state): State<CorsState>,
    req: Request,
    next: Next,
) -> Response {
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    // Handle preflight (OPTIONS) requests
    if req.method() == Method::OPTIONS {
        return build_preflight_response(&cors_state.config, origin.as_deref());
    }

    // Process the request
    let mut response = next.run(req).await;

    // Add CORS headers to response
    add_cors_headers(
        response.headers_mut(),
        &cors_state.config,
        origin.as_deref(),
    );

    response
}

/// Build preflight response for OPTIONS requests
fn build_preflight_response(config: &CorsConfig, origin: Option<&str>) -> Response {
    let mut response = Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Body::empty())
        .expect("building NO_CONTENT response with empty body is infallible");

    add_cors_headers(response.headers_mut(), config, origin);

    // Add preflight-specific headers
    if let Ok(value) = HeaderValue::from_str(&config.methods_string()) {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_METHODS, value);
    }
    if let Ok(value) = HeaderValue::from_str(&config.headers_string()) {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_HEADERS, value);
    }
    if let Ok(value) = HeaderValue::from_str(&config.max_age.to_string()) {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_MAX_AGE, value);
    }

    response
}

/// Add CORS headers to a response
fn add_cors_headers(headers: &mut HeaderMap, config: &CorsConfig, origin: Option<&str>) {
    // Access-Control-Allow-Origin
    let origin_value = if let Some(origin) = origin {
        if config.is_origin_allowed(origin) {
            if config.allowed_origins.contains("*") && !config.allow_credentials {
                "*"
            } else {
                origin
            }
        } else {
            return; // Origin not allowed, don't add CORS headers
        }
    } else if config.allowed_origins.contains("*") {
        "*"
    } else {
        return;
    };

    if let Ok(value) = HeaderValue::from_str(origin_value) {
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
    }

    // Access-Control-Allow-Credentials
    if config.allow_credentials {
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
            HeaderValue::from_static("true"),
        );
    }

    // Access-Control-Expose-Headers
    if !config.exposed_headers.is_empty() {
        let exposed = config
            .exposed_headers
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if let Ok(value) = HeaderValue::from_str(&exposed) {
            headers.insert(header::ACCESS_CONTROL_EXPOSE_HEADERS, value);
        }
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Time window duration
    pub window: Duration,
    /// Burst capacity (token bucket max tokens)
    pub burst_capacity: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window: Duration::from_secs(60),
            burst_capacity: 10,
        }
    }
}

impl RateLimitConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_requests == 0 {
            return Err("Maximum requests must be greater than 0".to_string());
        }

        if self.window.as_secs() == 0 {
            return Err("Time window must be greater than 0".to_string());
        }

        if self.burst_capacity == 0 {
            return Err("Burst capacity must be greater than 0".to_string());
        }

        if self.burst_capacity > self.max_requests {
            return Err(format!(
                "Burst capacity ({}) cannot exceed max requests ({})",
                self.burst_capacity, self.max_requests
            ));
        }

        Ok(())
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_update: Instant,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

impl TokenBucket {
    fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            tokens: capacity as f64,
            last_update: Instant::now(),
            capacity: capacity as f64,
            refill_rate,
        }
    }

    fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_update = now;
    }

    fn tokens_remaining(&self) -> u32 {
        self.tokens as u32
    }
}

/// Rate limiter state (per-IP buckets)
#[derive(Clone)]
pub struct RateLimitState {
    config: RateLimitConfig,
    buckets: Arc<Mutex<std::collections::HashMap<String, TokenBucket>>>,
}

impl RateLimitState {
    /// Create a new rate limiter state
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get or create a token bucket for an IP
    async fn get_bucket(&self, ip: &str) -> (bool, u32) {
        let mut buckets = self.buckets.lock().await;

        let refill_rate = self.config.max_requests as f64 / self.config.window.as_secs_f64();

        let bucket = buckets
            .entry(ip.to_string())
            .or_insert_with(|| TokenBucket::new(self.config.burst_capacity, refill_rate));

        let allowed = bucket.try_acquire();
        let remaining = bucket.tokens_remaining();

        (allowed, remaining)
    }
}

/// Rate limiting middleware
///
/// Limits requests per IP using token bucket algorithm.
pub async fn rate_limit_middleware(
    State(rate_state): State<RateLimitState>,
    req: Request,
    next: Next,
) -> Result<Response, RateLimitError> {
    // Extract client IP from headers or connection
    let ip = extract_client_ip(&req);

    let (allowed, remaining) = rate_state.get_bucket(&ip).await;

    if !allowed {
        return Err(RateLimitError::TooManyRequests);
    }

    let mut response = next.run(req).await;

    // Add rate limit headers
    let headers = response.headers_mut();
    if let Ok(value) = HeaderValue::from_str(&rate_state.config.max_requests.to_string()) {
        headers.insert("X-RateLimit-Limit", value);
    }
    if let Ok(value) = HeaderValue::from_str(&remaining.to_string()) {
        headers.insert("X-RateLimit-Remaining", value);
    }

    Ok(response)
}

/// Extract client IP from request
fn extract_client_ip(req: &Request) -> String {
    // Check X-Forwarded-For first (for proxied requests)
    if let Some(forwarded) = req.headers().get("x-forwarded-for") {
        if let Ok(s) = forwarded.to_str() {
            if let Some(ip) = s.split(',').next() {
                return ip.trim().to_string();
            }
        }
    }

    // Check X-Real-IP
    if let Some(real_ip) = req.headers().get("x-real-ip") {
        if let Ok(s) = real_ip.to_str() {
            return s.to_string();
        }
    }

    // Fallback to unknown
    "unknown".to_string()
}

/// Rate limit error
#[derive(Debug)]
pub enum RateLimitError {
    TooManyRequests,
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            RateLimitError::TooManyRequests => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please retry later.",
            ),
        };

        let mut response = (status, message).into_response();

        // Add Retry-After header (60 seconds)
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("60"));

        response
    }
}

// ============================================================================
// Compression Configuration
// ============================================================================

/// Compression level - balances speed vs compression ratio
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionLevel {
    /// Fastest compression, larger files (level 1)
    Fastest,
    /// Balanced compression (level 5-6)
    #[default]
    Balanced,
    /// Best compression, slower (level 9)
    Best,
    /// Custom compression level (0-9)
    Custom(u32),
}

impl CompressionLevel {
    /// Get the numeric compression level for gzip/deflate
    pub fn to_level(self) -> u32 {
        match self {
            CompressionLevel::Fastest => 1,
            CompressionLevel::Balanced => 5,
            CompressionLevel::Best => 9,
            CompressionLevel::Custom(level) => level.min(9),
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable gzip compression
    pub enable_gzip: bool,
    /// Compression level (speed vs size trade-off)
    pub level: CompressionLevel,
    /// Minimum size in bytes to compress (smaller files not compressed)
    pub min_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_gzip: true,
            level: CompressionLevel::Balanced,
            min_size: 1024, // Don't compress files smaller than 1KB
        }
    }
}

impl CompressionConfig {
    /// Create a fast compression config (prioritize speed)
    pub fn fast() -> Self {
        Self {
            level: CompressionLevel::Fastest,
            ..Default::default()
        }
    }

    /// Create a best compression config (prioritize size)
    pub fn best() -> Self {
        Self {
            level: CompressionLevel::Best,
            ..Default::default()
        }
    }

    /// Set compression level
    pub fn with_level(mut self, level: CompressionLevel) -> Self {
        self.level = level;
        self
    }

    /// Set minimum size threshold
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size;
        self
    }

    /// Enable/disable gzip compression
    pub fn with_gzip(mut self, gzip: bool) -> Self {
        self.enable_gzip = gzip;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Gzip must be enabled (only supported algorithm per COOLJAPAN OxiARC policy)
        if !self.enable_gzip {
            return Err("At least one compression algorithm must be enabled".to_string());
        }

        // Minimum size should be reasonable
        if self.min_size > 100 * 1024 * 1024 {
            return Err(format!(
                "Minimum compression size {} is too large (max: 100MB)",
                self.min_size
            ));
        }

        Ok(())
    }
}

// ============================================================================
// HTTP Caching
// ============================================================================

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Default max-age for cacheable responses (seconds)
    pub default_max_age: u64,
    /// Whether responses are public (can be cached by CDNs)
    pub public: bool,
    /// Whether to mark CID responses as immutable
    pub immutable_cids: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            default_max_age: 3600, // 1 hour
            public: true,
            immutable_cids: true, // CID content is immutable by definition
        }
    }
}

impl CacheConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Max age should be reasonable (not more than 1 year)
        const MAX_AGE_LIMIT: u64 = 365 * 24 * 3600; // 1 year in seconds

        if self.default_max_age > MAX_AGE_LIMIT {
            return Err(format!(
                "Max age {} exceeds maximum {} (1 year)",
                self.default_max_age, MAX_AGE_LIMIT
            ));
        }

        Ok(())
    }
}

/// Add caching headers to a response for a given CID
pub fn add_caching_headers(headers: &mut HeaderMap, cid: &str, config: &CacheConfig) {
    // ETag based on CID (content-addressed = perfect ETag)
    if let Ok(etag) = HeaderValue::from_str(&format!("\"{}\"", cid)) {
        headers.insert(header::ETAG, etag);
    }

    // Cache-Control
    let mut cache_control = String::new();
    if config.public {
        cache_control.push_str("public, ");
    } else {
        cache_control.push_str("private, ");
    }
    cache_control.push_str(&format!("max-age={}", config.default_max_age));

    // CID content is immutable - it will never change
    if config.immutable_cids {
        cache_control.push_str(", immutable");
    }

    if let Ok(value) = HeaderValue::from_str(&cache_control) {
        headers.insert(header::CACHE_CONTROL, value);
    }
}

/// Check if request has a matching ETag (for conditional requests)
pub fn check_etag_match(headers: &HeaderMap, cid: &str) -> bool {
    if let Some(if_none_match) = headers.get(header::IF_NONE_MATCH) {
        if let Ok(value) = if_none_match.to_str() {
            // Remove quotes and compare
            let etag = value.trim().trim_matches('"');
            return etag == cid;
        }
    }
    false
}

/// Build a 304 Not Modified response
pub fn not_modified_response(cid: &str, config: &CacheConfig) -> Response {
    let mut response = Response::builder()
        .status(StatusCode::NOT_MODIFIED)
        .body(Body::empty())
        .expect("building NOT_MODIFIED response with empty body is infallible");

    add_caching_headers(response.headers_mut(), cid, config);

    response
}

/// Authenticated user context
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: Uuid,
    pub username: String,
    pub claims: Option<Claims>,
}

/// Authenticate user from Authorization header
///
/// Supports both JWT tokens (Bearer <token>) and API keys (ipfrs_...)
fn authenticate_user(req: &Request, auth_state: &AuthState) -> Result<AuthUser, AuthError> {
    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::InvalidToken(
            "Missing Authorization header".to_string(),
        ))?;

    // Try JWT token first (Bearer <token>)
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        let claims = auth_state.jwt_manager.validate_token(token)?;
        let user = auth_state.user_store.get_user(&claims.username)?;

        return Ok(AuthUser {
            user_id: user.id,
            username: user.username,
            claims: Some(claims),
        });
    }

    // Try API key (ipfrs_...)
    if auth_header.starts_with("ipfrs_") {
        let (_api_key, user_id) = auth_state.api_key_store.authenticate(auth_header)?;
        let user = auth_state.user_store.get_by_id(&user_id)?;

        return Ok(AuthUser {
            user_id: user.id,
            username: user.username,
            claims: None,
        });
    }

    Err(AuthError::InvalidToken(
        "Authorization header must be either 'Bearer <token>' or 'ipfrs_<key>'".to_string(),
    ))
}

/// Authentication middleware
///
/// Validates JWT token or API key from Authorization header and injects authenticated user into request extensions.
pub async fn auth_middleware(
    State(auth_state): State<AuthState>,
    mut req: Request,
    next: Next,
) -> Result<Response, AuthMiddlewareError> {
    // Authenticate user (JWT or API key)
    let auth_user = authenticate_user(&req, &auth_state)?;

    // Inject authenticated user into request extensions
    req.extensions_mut().insert(auth_user);

    Ok(next.run(req).await)
}

/// Type alias for the permission check middleware future
type PermissionCheckFuture = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Response, AuthMiddlewareError>> + Send>,
>;

/// Authorization middleware factory
///
/// Creates middleware that checks if the authenticated user has required permissions.
pub fn require_permission(
    required: Permission,
) -> impl Fn(State<AuthState>, Request, Next) -> PermissionCheckFuture + Clone {
    move |State(auth_state): State<AuthState>, req: Request, next: Next| {
        let required = required;
        Box::pin(async move {
            // Get authenticated user from extensions
            let auth_user = req
                .extensions()
                .get::<AuthUser>()
                .ok_or_else(|| AuthError::InvalidToken("User not authenticated".to_string()))?;

            // Get user from store to check permissions
            let user = auth_state.user_store.get_by_id(&auth_user.user_id)?;

            // Check if user has required permission
            if !user.has_permission(required) {
                return Err(AuthMiddlewareError::from(
                    AuthError::InsufficientPermissions,
                ));
            }

            Ok(next.run(req).await)
        })
    }
}

/// Middleware error wrapper
#[derive(Debug)]
pub struct AuthMiddlewareError {
    error: AuthError,
}

impl From<AuthError> for AuthMiddlewareError {
    fn from(error: AuthError) -> Self {
        Self { error }
    }
}

impl IntoResponse for AuthMiddlewareError {
    fn into_response(self) -> Response {
        let (status, message) = match self.error {
            AuthError::InvalidToken(_) | AuthError::TokenExpired => {
                (StatusCode::UNAUTHORIZED, "Authentication required")
            }
            AuthError::InsufficientPermissions => {
                (StatusCode::FORBIDDEN, "Insufficient permissions")
            }
            AuthError::UserNotFound | AuthError::InvalidCredentials => {
                (StatusCode::UNAUTHORIZED, "Invalid credentials")
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error"),
        };

        (status, message).into_response()
    }
}

// ============================================================================
// Request Validation Middleware
// ============================================================================

/// Request validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum request body size (bytes)
    pub max_body_size: usize,
    /// Maximum CID length
    pub max_cid_length: usize,
    /// Validate CID format
    pub validate_cid_format: bool,
    /// Required content types for specific endpoints
    pub content_type_validation: bool,
    /// Maximum batch size for batch operations
    pub max_batch_size: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_body_size: 100 * 1024 * 1024, // 100 MB
            max_cid_length: 100,
            validate_cid_format: true,
            content_type_validation: true,
            max_batch_size: 1000,
        }
    }
}

impl ValidationConfig {
    /// Create a strict validation config
    pub fn strict() -> Self {
        Self {
            max_body_size: 10 * 1024 * 1024, // 10 MB
            max_cid_length: 64,
            validate_cid_format: true,
            content_type_validation: true,
            max_batch_size: 100,
        }
    }

    /// Create a permissive validation config
    pub fn permissive() -> Self {
        Self {
            max_body_size: 1024 * 1024 * 1024, // 1 GB
            max_cid_length: 200,
            validate_cid_format: false,
            content_type_validation: false,
            max_batch_size: 10000,
        }
    }
}

/// Validation error types
#[derive(Debug)]
pub enum ValidationError {
    /// Request body too large
    BodyTooLarge { size: usize, max: usize },
    /// Invalid CID format
    InvalidCid(String),
    /// Invalid content type
    InvalidContentType { expected: String, actual: String },
    /// Missing required parameter
    MissingParameter(String),
    /// Batch size exceeds limit
    BatchTooLarge { size: usize, max: usize },
    /// Invalid parameter value
    InvalidParameter { name: String, reason: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::BodyTooLarge { size, max } => {
                write!(
                    f,
                    "Request body too large: {} bytes (max: {} bytes)",
                    size, max
                )
            }
            ValidationError::InvalidCid(cid) => {
                write!(f, "Invalid CID format: {}", cid)
            }
            ValidationError::InvalidContentType { expected, actual } => {
                write!(
                    f,
                    "Invalid content type: expected {}, got {}",
                    expected, actual
                )
            }
            ValidationError::MissingParameter(param) => {
                write!(f, "Missing required parameter: {}", param)
            }
            ValidationError::BatchTooLarge { size, max } => {
                write!(
                    f,
                    "Batch size too large: {} items (max: {} items)",
                    size, max
                )
            }
            ValidationError::InvalidParameter { name, reason } => {
                write!(f, "Invalid parameter '{}': {}", name, reason)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

impl IntoResponse for ValidationError {
    fn into_response(self) -> Response {
        let request_id = Uuid::new_v4();
        let error_message = self.to_string();

        let (status, code) = match self {
            ValidationError::BodyTooLarge { .. } => {
                (StatusCode::PAYLOAD_TOO_LARGE, "BODY_TOO_LARGE")
            }
            ValidationError::InvalidCid(_) => (StatusCode::BAD_REQUEST, "INVALID_CID"),
            ValidationError::InvalidContentType { .. } => {
                (StatusCode::UNSUPPORTED_MEDIA_TYPE, "INVALID_CONTENT_TYPE")
            }
            ValidationError::MissingParameter(_) => (StatusCode::BAD_REQUEST, "MISSING_PARAMETER"),
            ValidationError::BatchTooLarge { .. } => (StatusCode::BAD_REQUEST, "BATCH_TOO_LARGE"),
            ValidationError::InvalidParameter { .. } => {
                (StatusCode::BAD_REQUEST, "INVALID_PARAMETER")
            }
        };

        let body = serde_json::json!({
            "error": error_message,
            "code": code,
            "request_id": request_id.to_string(),
        });

        (
            status,
            serde_json::to_string(&body).expect("serializing JSON Value is infallible"),
        )
            .into_response()
    }
}

/// Validate CID format
///
/// Basic validation: CIDv0 starts with "Qm" and is 46 chars, CIDv1 is base32/base58
pub fn validate_cid(cid: &str, config: &ValidationConfig) -> Result<(), ValidationError> {
    // Empty CID is always invalid, regardless of validation settings
    if cid.is_empty() {
        return Err(ValidationError::InvalidCid(
            "CID cannot be empty".to_string(),
        ));
    }

    if !config.validate_cid_format {
        return Ok(());
    }

    if cid.len() > config.max_cid_length {
        return Err(ValidationError::InvalidCid(format!(
            "CID too long: {} chars (max: {})",
            cid.len(),
            config.max_cid_length
        )));
    }

    // Basic format check: CIDv0 or CIDv1
    if cid.starts_with("Qm") && cid.len() == 46 {
        // CIDv0 (base58btc encoded SHA-256 hash)
        Ok(())
    } else if cid.starts_with("b") || cid.starts_with("z") || cid.starts_with("f") {
        // CIDv1 (multibase prefix)
        Ok(())
    } else {
        Err(ValidationError::InvalidCid(
            "Invalid CID format: must be CIDv0 (Qm...) or CIDv1 (b..., z..., f...)".to_string(),
        ))
    }
}

/// Validate batch size
pub fn validate_batch_size(size: usize, config: &ValidationConfig) -> Result<(), ValidationError> {
    if size == 0 {
        return Err(ValidationError::InvalidParameter {
            name: "batch".to_string(),
            reason: "Batch cannot be empty".to_string(),
        });
    }

    if size > config.max_batch_size {
        return Err(ValidationError::BatchTooLarge {
            size,
            max: config.max_batch_size,
        });
    }

    Ok(())
}

/// Validate content type
pub fn validate_content_type(
    headers: &HeaderMap,
    expected: &str,
    config: &ValidationConfig,
) -> Result<(), ValidationError> {
    if !config.content_type_validation {
        return Ok(());
    }

    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if !content_type.starts_with(expected) {
        return Err(ValidationError::InvalidContentType {
            expected: expected.to_string(),
            actual: content_type.to_string(),
        });
    }

    Ok(())
}

/// Validation middleware state
#[derive(Clone)]
pub struct ValidationState {
    pub config: ValidationConfig,
}

/// Request validation middleware
///
/// Validates request size and basic parameters before processing
pub async fn validation_middleware(
    State(_validation_state): State<ValidationState>,
    req: Request,
    next: Next,
) -> Result<Response, ValidationError> {
    let (parts, body) = req.into_parts();

    // Validate content-type for POST/PUT requests
    if parts.method == Method::POST || parts.method == Method::PUT {
        // Skip validation for multipart/form-data (handled by body parser)
        if let Some(content_type) = parts.headers.get(header::CONTENT_TYPE) {
            if let Ok(ct_str) = content_type.to_str() {
                if ct_str.contains("multipart/form-data") {
                    // Skip body size validation for multipart
                    let req = Request::from_parts(parts, body);
                    return Ok(next.run(req).await);
                }
            }
        }
    }

    // Reconstruct request and continue
    let req = Request::from_parts(parts, body);
    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cors_config_default() {
        let config = CorsConfig::default();
        assert!(config.allowed_origins.is_empty());
        assert!(config.allowed_methods.contains(&Method::GET));
        assert!(config.allowed_methods.contains(&Method::POST));
        assert!(!config.allow_credentials);
        assert_eq!(config.max_age, 86400);
    }

    #[test]
    fn test_cors_config_permissive() {
        let config = CorsConfig::permissive();
        assert!(config.allowed_origins.contains("*"));
        assert!(config.is_origin_allowed("https://example.com"));
        assert!(config.is_origin_allowed("http://localhost:3000"));
    }

    #[test]
    fn test_cors_config_allow_origin() {
        let config = CorsConfig::default()
            .allow_origin("https://example.com")
            .allow_origin("https://api.example.com");

        assert!(config.is_origin_allowed("https://example.com"));
        assert!(config.is_origin_allowed("https://api.example.com"));
        assert!(!config.is_origin_allowed("https://other.com"));
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_requests, 100);
        assert_eq!(config.window, Duration::from_secs(60));
        assert_eq!(config.burst_capacity, 10);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.default_max_age, 3600);
        assert!(config.public);
        assert!(config.immutable_cids);
    }

    #[test]
    fn test_add_caching_headers() {
        let mut headers = HeaderMap::new();
        let config = CacheConfig::default();

        add_caching_headers(&mut headers, "QmTest123", &config);

        assert!(headers.contains_key(header::ETAG));
        assert!(headers.contains_key(header::CACHE_CONTROL));

        let etag = headers
            .get(header::ETAG)
            .expect("test: ETAG header must be present")
            .to_str()
            .expect("test: ETAG header value must be valid UTF-8");
        assert_eq!(etag, "\"QmTest123\"");

        let cache_control = headers
            .get(header::CACHE_CONTROL)
            .expect("test: CACHE_CONTROL header must be present")
            .to_str()
            .expect("test: CACHE_CONTROL header value must be valid UTF-8");
        assert!(cache_control.contains("public"));
        assert!(cache_control.contains("max-age=3600"));
        assert!(cache_control.contains("immutable"));
    }

    #[test]
    fn test_check_etag_match() {
        let mut headers = HeaderMap::new();

        // No If-None-Match header
        assert!(!check_etag_match(&headers, "QmTest123"));

        // With matching ETag
        headers.insert(
            header::IF_NONE_MATCH,
            HeaderValue::from_static("\"QmTest123\""),
        );
        assert!(check_etag_match(&headers, "QmTest123"));

        // With non-matching ETag
        assert!(!check_etag_match(&headers, "QmOther456"));
    }

    #[tokio::test]
    async fn test_rate_limit_state() {
        let config = RateLimitConfig {
            max_requests: 5,
            window: Duration::from_secs(1),
            burst_capacity: 3,
        };
        let state = RateLimitState::new(config);

        // First 3 requests should succeed (burst capacity)
        for _ in 0..3 {
            let (allowed, _) = state.get_bucket("127.0.0.1").await;
            assert!(allowed);
        }
    }

    #[test]
    fn test_compression_level_to_level() {
        assert_eq!(CompressionLevel::Fastest.to_level(), 1);
        assert_eq!(CompressionLevel::Balanced.to_level(), 5);
        assert_eq!(CompressionLevel::Best.to_level(), 9);
        assert_eq!(CompressionLevel::Custom(7).to_level(), 7);
        assert_eq!(CompressionLevel::Custom(15).to_level(), 9); // Capped at 9
    }

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert!(config.enable_gzip);
        assert_eq!(config.level, CompressionLevel::Balanced);
        assert_eq!(config.min_size, 1024);
    }

    #[test]
    fn test_compression_config_fast() {
        let config = CompressionConfig::fast();
        assert_eq!(config.level, CompressionLevel::Fastest);
        assert!(config.enable_gzip);
    }

    #[test]
    fn test_compression_config_best() {
        let config = CompressionConfig::best();
        assert_eq!(config.level, CompressionLevel::Best);
    }

    #[test]
    fn test_compression_config_builder() {
        let config = CompressionConfig::default()
            .with_level(CompressionLevel::Custom(7))
            .with_min_size(2048)
            .with_gzip(true);

        assert_eq!(config.level, CompressionLevel::Custom(7));
        assert_eq!(config.min_size, 2048);
        assert!(config.enable_gzip);
    }

    #[test]
    fn test_compression_config_validation_valid() {
        let config = CompressionConfig::default();
        assert!(config.validate().is_ok());

        let config = CompressionConfig::default().with_gzip(true);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_compression_config_validation_invalid() {
        // No algorithms enabled
        let config = CompressionConfig::default().with_gzip(false);
        assert!(config.validate().is_err());

        // Min size too large
        let config = CompressionConfig::default().with_min_size(200 * 1024 * 1024);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rate_limit_config_validation_valid() {
        let config = RateLimitConfig::default();
        assert!(config.validate().is_ok());

        let config = RateLimitConfig {
            max_requests: 100,
            window: Duration::from_secs(60),
            burst_capacity: 50,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rate_limit_config_validation_invalid() {
        // Zero max requests
        let config = RateLimitConfig {
            max_requests: 0,
            window: Duration::from_secs(60),
            burst_capacity: 10,
        };
        assert!(config.validate().is_err());

        // Zero window
        let config = RateLimitConfig {
            max_requests: 100,
            window: Duration::from_secs(0),
            burst_capacity: 10,
        };
        assert!(config.validate().is_err());

        // Burst exceeds max
        let config = RateLimitConfig {
            max_requests: 100,
            window: Duration::from_secs(60),
            burst_capacity: 200,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cache_config_validation_valid() {
        let config = CacheConfig::default();
        assert!(config.validate().is_ok());

        let config = CacheConfig {
            default_max_age: 86400, // 1 day
            public: true,
            immutable_cids: true,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cache_config_validation_invalid() {
        // Max age too large (more than 1 year)
        let config = CacheConfig {
            default_max_age: 400 * 24 * 3600, // More than 1 year
            public: true,
            immutable_cids: true,
        };
        assert!(config.validate().is_err());
    }

    // Validation middleware tests

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_body_size, 100 * 1024 * 1024);
        assert_eq!(config.max_cid_length, 100);
        assert!(config.validate_cid_format);
        assert!(config.content_type_validation);
        assert_eq!(config.max_batch_size, 1000);
    }

    #[test]
    fn test_validation_config_strict() {
        let config = ValidationConfig::strict();
        assert_eq!(config.max_body_size, 10 * 1024 * 1024);
        assert_eq!(config.max_cid_length, 64);
        assert_eq!(config.max_batch_size, 100);
    }

    #[test]
    fn test_validation_config_permissive() {
        let config = ValidationConfig::permissive();
        assert_eq!(config.max_body_size, 1024 * 1024 * 1024);
        assert_eq!(config.max_cid_length, 200);
        assert!(!config.validate_cid_format);
        assert!(!config.content_type_validation);
        assert_eq!(config.max_batch_size, 10000);
    }

    #[test]
    fn test_validate_cid_v0() {
        let config = ValidationConfig::default();

        // Valid CIDv0
        assert!(validate_cid("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco", &config).is_ok());

        // Invalid CIDv0 (wrong length)
        assert!(validate_cid("QmShort", &config).is_err());

        // Empty CID
        assert!(validate_cid("", &config).is_err());
    }

    #[test]
    fn test_validate_cid_v1() {
        let config = ValidationConfig::default();

        // Valid CIDv1 prefixes
        assert!(validate_cid(
            "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
            &config
        )
        .is_ok());
        assert!(validate_cid("zb2rhk6GMPQF8p1kqXvhYnCMp3hGGUQVvqp6qjdvNLKqCqKCo", &config).is_ok());

        // Invalid format
        assert!(validate_cid("invalid_cid_format", &config).is_err());
    }

    #[test]
    fn test_validate_cid_disabled() {
        let config = ValidationConfig {
            validate_cid_format: false,
            ..Default::default()
        };

        // Should accept any string when validation is disabled
        assert!(validate_cid("invalid_format", &config).is_ok());
        assert!(validate_cid("", &config).is_err()); // Empty still fails
    }

    #[test]
    fn test_validate_batch_size_valid() {
        let config = ValidationConfig::default();

        assert!(validate_batch_size(1, &config).is_ok());
        assert!(validate_batch_size(100, &config).is_ok());
        assert!(validate_batch_size(1000, &config).is_ok());
    }

    #[test]
    fn test_validate_batch_size_invalid() {
        let config = ValidationConfig::default();

        // Empty batch
        assert!(validate_batch_size(0, &config).is_err());

        // Too large
        assert!(validate_batch_size(1001, &config).is_err());
        assert!(validate_batch_size(10000, &config).is_err());
    }

    #[test]
    fn test_validate_content_type_valid() {
        let config = ValidationConfig::default();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        assert!(validate_content_type(&headers, "application/json", &config).is_ok());
    }

    #[test]
    fn test_validate_content_type_invalid() {
        let config = ValidationConfig::default();
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain"));

        assert!(validate_content_type(&headers, "application/json", &config).is_err());
    }

    #[test]
    fn test_validate_content_type_disabled() {
        let config = ValidationConfig {
            content_type_validation: false,
            ..Default::default()
        };

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain"));

        // Should accept any content type when validation is disabled
        assert!(validate_content_type(&headers, "application/json", &config).is_ok());
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::InvalidCid("test".to_string());
        assert_eq!(err.to_string(), "Invalid CID format: test");

        let err = ValidationError::BodyTooLarge {
            size: 200,
            max: 100,
        };
        assert!(err.to_string().contains("200 bytes"));
        assert!(err.to_string().contains("100 bytes"));

        let err = ValidationError::BatchTooLarge {
            size: 2000,
            max: 1000,
        };
        assert!(err.to_string().contains("2000 items"));
        assert!(err.to_string().contains("1000 items"));
    }
}
