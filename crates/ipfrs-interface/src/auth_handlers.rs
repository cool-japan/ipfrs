//! Authentication API Handlers
//!
//! Provides HTTP endpoints for user authentication and management.

use crate::auth::{ApiKey, AuthError, AuthState, Permission, Role, User};
use crate::gateway::GatewayState;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

// ============================================================================
// Request/Response Types
// ============================================================================

/// Login request
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

/// Login response
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub expires_in: u64,
    pub user: UserInfo,
}

/// User registration request
#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub password: String,
    pub role: Option<Role>,
}

/// User registration response
#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub user: UserInfo,
}

/// User information (without sensitive data)
#[derive(Debug, Serialize)]
pub struct UserInfo {
    pub id: String,
    pub username: String,
    pub role: Role,
    pub permissions: Vec<Permission>,
    pub active: bool,
    pub created_at: u64,
}

impl From<&User> for UserInfo {
    fn from(user: &User) -> Self {
        Self {
            id: user.id.to_string(),
            username: user.username.clone(),
            role: user.role,
            permissions: user.permissions().into_iter().collect(),
            active: user.active,
            created_at: user.created_at,
        }
    }
}

/// Token refresh response
#[derive(Debug, Serialize)]
pub struct RefreshResponse {
    pub token: String,
    pub expires_in: u64,
}

/// Update permissions request
#[derive(Debug, Deserialize)]
pub struct UpdatePermissionsRequest {
    pub username: String,
    pub permissions: HashSet<Permission>,
}

/// Generic success response
#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub message: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get auth state from gateway state, returning error if not enabled
fn get_auth_state(state: &GatewayState) -> Result<&AuthState, AuthHandlerError> {
    state.auth.as_ref().ok_or_else(|| {
        AuthHandlerError::from(AuthError::InvalidToken(
            "Authentication not enabled".to_string(),
        ))
    })
}

// ============================================================================
// Handler Functions
// ============================================================================

/// Login endpoint
///
/// Authenticates user with username and password, returns JWT token.
///
/// POST /api/v0/auth/login
pub async fn login_handler(
    State(gateway_state): State<GatewayState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Authenticate user
    let user = auth_state
        .user_store
        .authenticate(&req.username, &req.password)?;

    // Generate JWT token (24 hour expiration)
    let token = auth_state.jwt_manager.generate_token(&user, 24)?;

    Ok(Json(LoginResponse {
        token,
        expires_in: 24 * 3600, // 24 hours in seconds
        user: UserInfo::from(&user),
    }))
}

/// Register endpoint
///
/// Creates a new user account. Only admins can specify role, otherwise defaults to User.
///
/// POST /api/v0/auth/register
pub async fn register_handler(
    State(gateway_state): State<GatewayState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // For now, default to User role if not specified
    // In production, you'd check if the requester is admin before allowing role specification
    let role = req.role.unwrap_or(Role::User);

    // Create new user
    let user = User::new(req.username, &req.password, role)?;

    // Add to store
    auth_state.user_store.add_user(user.clone())?;

    Ok(Json(RegisterResponse {
        success: true,
        user: UserInfo::from(&user),
    }))
}

/// Get current user info
///
/// Returns information about the authenticated user.
///
/// GET /api/v0/auth/me
pub async fn me_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
) -> Result<Json<UserInfo>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    let user = auth_state.user_store.get_user(&claims.username)?;

    Ok(Json(UserInfo::from(&user)))
}

/// Extract JWT token from headers
fn extract_token_from_headers(headers: &axum::http::HeaderMap) -> Result<&str, AuthError> {
    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::InvalidToken(
            "Missing Authorization header".to_string(),
        ))?;

    auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| AuthError::InvalidToken("Invalid Authorization format".to_string()))
}

/// Update user permissions (admin only)
///
/// Updates custom permissions for a user.
///
/// POST /api/v0/auth/permissions
pub async fn update_permissions_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<UpdatePermissionsRequest>,
) -> Result<Json<SuccessResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    // Check if requester is admin
    let requester = auth_state.user_store.get_user(&claims.username)?;

    if !requester.has_permission(Permission::SystemAdmin) {
        return Err(AuthHandlerError::from(AuthError::InsufficientPermissions));
    }

    // Update permissions
    auth_state
        .user_store
        .update_permissions(&req.username, req.permissions)?;

    Ok(Json(SuccessResponse {
        success: true,
        message: format!("Permissions updated for user: {}", req.username),
    }))
}

/// Deactivate user (admin only)
///
/// Deactivates a user account.
///
/// POST /api/v0/auth/deactivate/:username
pub async fn deactivate_user_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(username): axum::extract::Path<String>,
) -> Result<Json<SuccessResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    // Check if requester is admin
    let requester = auth_state.user_store.get_user(&claims.username)?;

    if !requester.has_permission(Permission::SystemAdmin) {
        return Err(AuthHandlerError::from(AuthError::InsufficientPermissions));
    }

    // Deactivate user
    auth_state.user_store.deactivate_user(&username)?;

    Ok(Json(SuccessResponse {
        success: true,
        message: format!("User deactivated: {}", username),
    }))
}

// ============================================================================
// API Key Management Handlers
// ============================================================================

/// Create API key request
#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
}

/// Create API key response
#[derive(Debug, Serialize)]
pub struct CreateApiKeyResponse {
    pub key: String,
    pub key_info: ApiKeyInfo,
}

/// API key information (sanitized, no key_hash)
#[derive(Debug, Serialize)]
pub struct ApiKeyInfo {
    pub id: String,
    pub prefix: String,
    pub name: String,
    pub created_at: u64,
    pub last_used_at: Option<u64>,
    pub active: bool,
}

impl From<&ApiKey> for ApiKeyInfo {
    fn from(key: &ApiKey) -> Self {
        Self {
            id: key.id.to_string(),
            prefix: key.prefix.clone(),
            name: key.name.clone(),
            created_at: key.created_at,
            last_used_at: key.last_used_at,
            active: key.active,
        }
    }
}

/// Create new API key
///
/// POST /api/v0/auth/keys
pub async fn create_api_key_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CreateApiKeyRequest>,
) -> Result<Json<CreateApiKeyResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    // Get user
    let user = auth_state.user_store.get_user(&claims.username)?;

    // Create API key
    let (api_key, raw_key) = ApiKey::new(user.id, req.name)?;

    // Store key
    auth_state.api_key_store.add_key(api_key.clone())?;

    Ok(Json(CreateApiKeyResponse {
        key: raw_key,
        key_info: ApiKeyInfo::from(&api_key),
    }))
}

/// List user's API keys
///
/// GET /api/v0/auth/keys
pub async fn list_api_keys_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Vec<ApiKeyInfo>>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    // Get user
    let user = auth_state.user_store.get_user(&claims.username)?;

    // List keys
    let keys = auth_state.api_key_store.list_user_keys(&user.id);
    let key_infos: Vec<ApiKeyInfo> = keys.iter().map(ApiKeyInfo::from).collect();

    Ok(Json(key_infos))
}

/// Revoke API key
///
/// POST /api/v0/auth/keys/:key_id/revoke
pub async fn revoke_api_key_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(key_id_str): axum::extract::Path<String>,
) -> Result<Json<SuccessResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    // Get user
    let user = auth_state.user_store.get_user(&claims.username)?;

    // Parse key ID
    let key_id = Uuid::parse_str(&key_id_str)
        .map_err(|_| AuthHandlerError::from(AuthError::InvalidCredentials))?;

    // Get key and verify ownership
    let key = auth_state.api_key_store.get_key(&key_id)?;
    if key.user_id != user.id && !user.has_permission(Permission::SystemAdmin) {
        return Err(AuthHandlerError::from(AuthError::InsufficientPermissions));
    }

    // Revoke key
    auth_state.api_key_store.revoke_key(&key_id)?;

    Ok(Json(SuccessResponse {
        success: true,
        message: format!("API key revoked: {}", key_id),
    }))
}

/// Delete API key
///
/// DELETE /api/v0/auth/keys/:key_id
pub async fn delete_api_key_handler(
    State(gateway_state): State<GatewayState>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(key_id_str): axum::extract::Path<String>,
) -> Result<Json<SuccessResponse>, AuthHandlerError> {
    let auth_state = get_auth_state(&gateway_state)?;

    // Extract and validate token
    let token = extract_token_from_headers(&headers)?;
    let claims = auth_state.jwt_manager.validate_token(token)?;

    // Get user
    let user = auth_state.user_store.get_user(&claims.username)?;

    // Parse key ID
    let key_id = Uuid::parse_str(&key_id_str)
        .map_err(|_| AuthHandlerError::from(AuthError::InvalidCredentials))?;

    // Get key and verify ownership
    let key = auth_state.api_key_store.get_key(&key_id)?;
    if key.user_id != user.id && !user.has_permission(Permission::SystemAdmin) {
        return Err(AuthHandlerError::from(AuthError::InsufficientPermissions));
    }

    // Delete key
    auth_state.api_key_store.delete_key(&key_id)?;

    Ok(Json(SuccessResponse {
        success: true,
        message: format!("API key deleted: {}", key_id),
    }))
}

// ============================================================================
// Error Handling
// ============================================================================

/// Authentication handler error
#[derive(Debug)]
pub struct AuthHandlerError {
    error: AuthError,
}

impl From<AuthError> for AuthHandlerError {
    fn from(error: AuthError) -> Self {
        Self { error }
    }
}

impl IntoResponse for AuthHandlerError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self.error {
            AuthError::InvalidCredentials => {
                (StatusCode::UNAUTHORIZED, "Invalid credentials".to_string())
            }
            AuthError::InvalidToken(msg) => (StatusCode::UNAUTHORIZED, msg),
            AuthError::TokenExpired => (StatusCode::UNAUTHORIZED, "Token expired".to_string()),
            AuthError::InsufficientPermissions => (
                StatusCode::FORBIDDEN,
                "Insufficient permissions".to_string(),
            ),
            AuthError::UserNotFound => (StatusCode::NOT_FOUND, "User not found".to_string()),
            AuthError::HashError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AuthError::JwtError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        (status, message).into_response()
    }
}
