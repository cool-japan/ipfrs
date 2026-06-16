//! Authentication and Authorization for IPFRS
//!
//! This module provides:
//! - API key authentication
//! - JWT token authentication
//! - OAuth2 integration
//! - Role-based access control (RBAC)
//! - Resource-level permissions

use anyhow::{anyhow, Context, Result};
use chrono::{Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// User roles for role-based access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Administrator with full access
    Admin,
    /// Regular user with read/write access
    User,
    /// Read-only access
    ReadOnly,
    /// Service account for automated operations
    Service,
}

impl Role {
    /// Check if this role has administrative privileges
    pub fn is_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }

    /// Get default permissions for this role
    pub fn default_permissions(&self) -> HashSet<Permission> {
        match self {
            Role::Admin => {
                // Admins get all permissions
                vec![
                    Permission::BlockRead,
                    Permission::BlockWrite,
                    Permission::BlockDelete,
                    Permission::DagRead,
                    Permission::DagWrite,
                    Permission::SemanticRead,
                    Permission::SemanticWrite,
                    Permission::LogicRead,
                    Permission::LogicWrite,
                    Permission::NetworkRead,
                    Permission::NetworkWrite,
                    Permission::AdminManage,
                ]
                .into_iter()
                .collect()
            }
            Role::User => vec![
                Permission::BlockRead,
                Permission::BlockWrite,
                Permission::DagRead,
                Permission::DagWrite,
                Permission::SemanticRead,
                Permission::SemanticWrite,
                Permission::LogicRead,
                Permission::LogicWrite,
                Permission::NetworkRead,
            ]
            .into_iter()
            .collect(),
            Role::ReadOnly => vec![
                Permission::BlockRead,
                Permission::DagRead,
                Permission::SemanticRead,
                Permission::LogicRead,
                Permission::NetworkRead,
            ]
            .into_iter()
            .collect(),
            Role::Service => {
                // Service accounts get automated operation permissions
                vec![
                    Permission::BlockRead,
                    Permission::BlockWrite,
                    Permission::DagRead,
                    Permission::DagWrite,
                    Permission::SemanticWrite,
                    Permission::LogicWrite,
                ]
                .into_iter()
                .collect()
            }
        }
    }
}

/// Fine-grained permissions for resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read blocks from storage
    BlockRead,
    /// Write blocks to storage
    BlockWrite,
    /// Delete blocks from storage
    BlockDelete,
    /// Read DAG structures
    DagRead,
    /// Write DAG structures
    DagWrite,
    /// Read semantic indexes
    SemanticRead,
    /// Write semantic indexes
    SemanticWrite,
    /// Read logic knowledge base
    LogicRead,
    /// Write logic knowledge base
    LogicWrite,
    /// Read network information
    NetworkRead,
    /// Modify network connections
    NetworkWrite,
    /// Manage users and permissions
    AdminManage,
}

/// Authentication token (API key or JWT)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    /// Token identifier
    pub id: String,
    /// User ID associated with this token
    pub user_id: String,
    /// Token type (api_key or jwt)
    pub token_type: TokenType,
    /// Token secret/value
    pub secret: String,
    /// Expiration time (None for API keys)
    pub expires_at: Option<chrono::DateTime<Utc>>,
    /// User roles
    pub roles: HashSet<Role>,
    /// Custom permissions (overrides role defaults if set)
    pub permissions: Option<HashSet<Permission>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl AuthToken {
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            Utc::now() > exp
        } else {
            false
        }
    }

    /// Get effective permissions for this token
    pub fn effective_permissions(&self) -> HashSet<Permission> {
        if let Some(ref perms) = self.permissions {
            perms.clone()
        } else {
            // Merge permissions from all roles
            self.roles
                .iter()
                .flat_map(|role| role.default_permissions())
                .collect()
        }
    }

    /// Check if token has a specific permission
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.effective_permissions().contains(&permission)
    }

    /// Check if token has admin role
    pub fn is_admin(&self) -> bool {
        self.roles.iter().any(|r| r.is_admin())
    }
}

/// Token type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    /// API key (long-lived, no expiration by default)
    ApiKey,
    /// JWT token (short-lived, expires)
    Jwt,
}

/// User account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// User ID
    pub id: String,
    /// Username
    pub username: String,
    /// Email address
    pub email: Option<String>,
    /// Password hash (bcrypt)
    pub password_hash: Option<String>,
    /// User roles
    pub roles: HashSet<Role>,
    /// Custom permissions (overrides role defaults if set)
    pub permissions: Option<HashSet<Permission>>,
    /// Account enabled
    pub enabled: bool,
    /// Account creation time
    pub created_at: chrono::DateTime<Utc>,
    /// Last login time
    pub last_login: Option<chrono::DateTime<Utc>>,
}

impl User {
    /// Create a new user
    pub fn new(id: String, username: String) -> Self {
        Self {
            id,
            username,
            email: None,
            password_hash: None,
            roles: HashSet::new(),
            permissions: None,
            enabled: true,
            created_at: Utc::now(),
            last_login: None,
        }
    }

    /// Add a role to this user
    pub fn add_role(&mut self, role: Role) {
        self.roles.insert(role);
    }

    /// Get effective permissions for this user
    pub fn effective_permissions(&self) -> HashSet<Permission> {
        if let Some(ref perms) = self.permissions {
            perms.clone()
        } else {
            // Merge permissions from all roles
            self.roles
                .iter()
                .flat_map(|role| role.default_permissions())
                .collect()
        }
    }

    /// Check if user has a specific permission
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.effective_permissions().contains(&permission)
    }
}

/// OAuth2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    /// OAuth2 provider name
    pub provider: String,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Authorization endpoint
    pub auth_url: String,
    /// Token endpoint
    pub token_url: String,
    /// Redirect URI
    pub redirect_uri: String,
    /// Scopes
    pub scopes: Vec<String>,
}

/// Authentication manager
pub struct AuthManager {
    /// Token storage (token_id -> token)
    tokens: Arc<RwLock<HashMap<String, AuthToken>>>,
    /// User storage (user_id -> user)
    users: Arc<RwLock<HashMap<String, User>>>,
    /// Username lookup (username -> user_id)
    username_lookup: Arc<RwLock<HashMap<String, String>>>,
    /// OAuth2 configurations
    oauth2_configs: Arc<RwLock<HashMap<String, OAuth2Config>>>,
    /// JWT secret for signing tokens
    jwt_secret: String,
    /// Default token expiration (for JWT)
    default_token_expiration: Duration,
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new(jwt_secret: String) -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
            users: Arc::new(RwLock::new(HashMap::new())),
            username_lookup: Arc::new(RwLock::new(HashMap::new())),
            oauth2_configs: Arc::new(RwLock::new(HashMap::new())),
            jwt_secret,
            default_token_expiration: Duration::hours(24),
        }
    }

    /// Set default token expiration
    pub fn with_token_expiration(mut self, duration: Duration) -> Self {
        self.default_token_expiration = duration;
        self
    }

    /// Create a new user
    pub fn create_user(
        &self,
        username: String,
        email: Option<String>,
        roles: HashSet<Role>,
    ) -> Result<User> {
        let user_id = uuid::Uuid::new_v4().to_string();

        // Check if username already exists
        {
            let lookup = self.username_lookup.read();
            if lookup.contains_key(&username) {
                return Err(anyhow!("Username already exists"));
            }
        }

        let mut user = User::new(user_id.clone(), username.clone());
        user.email = email;
        user.roles = roles;

        // Store user
        {
            let mut users = self.users.write();
            users.insert(user_id.clone(), user.clone());
        }

        // Update username lookup
        {
            let mut lookup = self.username_lookup.write();
            lookup.insert(username, user_id);
        }

        Ok(user)
    }

    /// Get user by ID
    pub fn get_user(&self, user_id: &str) -> Option<User> {
        self.users.read().get(user_id).cloned()
    }

    /// Get user by username
    pub fn get_user_by_username(&self, username: &str) -> Option<User> {
        let user_id = self.username_lookup.read().get(username).cloned()?;
        self.get_user(&user_id)
    }

    /// Update user
    pub fn update_user(&self, user: User) -> Result<()> {
        let mut users = self.users.write();
        users.insert(user.id.clone(), user);
        Ok(())
    }

    /// Delete user
    pub fn delete_user(&self, user_id: &str) -> Result<()> {
        let mut users = self.users.write();
        if let Some(user) = users.remove(user_id) {
            let mut lookup = self.username_lookup.write();
            lookup.remove(&user.username);
        }
        Ok(())
    }

    /// Create an API key for a user
    pub fn create_api_key(&self, user_id: &str, name: Option<String>) -> Result<AuthToken> {
        let user = self.get_user(user_id).context("User not found")?;

        let token_id = uuid::Uuid::new_v4().to_string();
        let secret = format!(
            "ipfrs_{}",
            uuid::Uuid::new_v4().to_string().replace('-', "")
        );

        let mut metadata = HashMap::new();
        if let Some(n) = name {
            metadata.insert("name".to_string(), n);
        }
        metadata.insert("created_at".to_string(), Utc::now().to_rfc3339());

        let token = AuthToken {
            id: token_id.clone(),
            user_id: user_id.to_string(),
            token_type: TokenType::ApiKey,
            secret: secret.clone(),
            expires_at: None,
            roles: user.roles.clone(),
            permissions: user.permissions.clone(),
            metadata,
        };

        // Store token
        {
            let mut tokens = self.tokens.write();
            tokens.insert(token_id, token.clone());
        }

        Ok(token)
    }

    /// Create a JWT token for a user
    pub fn create_jwt_token(&self, user_id: &str, duration: Option<Duration>) -> Result<AuthToken> {
        let user = self.get_user(user_id).context("User not found")?;

        let token_id = uuid::Uuid::new_v4().to_string();
        let expires_at = Utc::now() + duration.unwrap_or(self.default_token_expiration);

        // In a real implementation, this would use a proper JWT library
        // For now, we'll create a simple token structure
        let secret = self.encode_jwt(&token_id, user_id, &user.roles, expires_at)?;

        let token = AuthToken {
            id: token_id.clone(),
            user_id: user_id.to_string(),
            token_type: TokenType::Jwt,
            secret,
            expires_at: Some(expires_at),
            roles: user.roles.clone(),
            permissions: user.permissions.clone(),
            metadata: HashMap::new(),
        };

        // Store token
        {
            let mut tokens = self.tokens.write();
            tokens.insert(token_id, token.clone());
        }

        Ok(token)
    }

    /// Verify and decode a JWT token
    #[allow(dead_code)]
    fn encode_jwt(
        &self,
        token_id: &str,
        user_id: &str,
        _roles: &HashSet<Role>,
        expires_at: chrono::DateTime<Utc>,
    ) -> Result<String> {
        // Simplified JWT encoding - in production, use jsonwebtoken crate
        let payload = format!("{}:{}:{}", token_id, user_id, expires_at.timestamp());
        let signature = format!(
            "{:x}",
            md5::compute(format!("{}{}", payload, self.jwt_secret))
        );
        Ok(format!("{}.{}", payload, signature))
    }

    /// Verify a token (API key or JWT)
    pub fn verify_token(&self, secret: &str) -> Result<AuthToken> {
        // Check if it's an API key
        if secret.starts_with("ipfrs_") {
            let tokens = self.tokens.read();
            for token in tokens.values() {
                if token.secret == secret {
                    if token.is_expired() {
                        return Err(anyhow!("Token expired"));
                    }
                    return Ok(token.clone());
                }
            }
            return Err(anyhow!("Invalid API key"));
        }

        // Try to decode as JWT
        self.decode_jwt(secret)
    }

    /// Decode and verify JWT
    #[allow(dead_code)]
    fn decode_jwt(&self, jwt: &str) -> Result<AuthToken> {
        let parts: Vec<&str> = jwt.split('.').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid JWT format"));
        }

        let payload = parts[0];
        let signature = parts[1];

        // Verify signature
        let expected_sig = format!(
            "{:x}",
            md5::compute(format!("{}{}", payload, self.jwt_secret))
        );
        if signature != expected_sig {
            return Err(anyhow!("Invalid JWT signature"));
        }

        // Parse payload
        let payload_parts: Vec<&str> = payload.split(':').collect();
        if payload_parts.len() != 3 {
            return Err(anyhow!("Invalid JWT payload"));
        }

        let token_id = payload_parts[0];

        // Look up token
        let tokens = self.tokens.read();
        let token = tokens.get(token_id).context("Token not found")?;

        if token.is_expired() {
            return Err(anyhow!("Token expired"));
        }

        Ok(token.clone())
    }

    /// Revoke a token
    pub fn revoke_token(&self, token_id: &str) -> Result<()> {
        let mut tokens = self.tokens.write();
        tokens.remove(token_id);
        Ok(())
    }

    /// Check if a token has a specific permission
    pub fn check_permission(&self, token: &AuthToken, permission: Permission) -> Result<()> {
        if !token.has_permission(permission) {
            return Err(anyhow!(
                "Insufficient permissions: {:?} required",
                permission
            ));
        }
        Ok(())
    }

    /// Add OAuth2 provider configuration
    pub fn add_oauth2_provider(&self, config: OAuth2Config) -> Result<()> {
        let mut configs = self.oauth2_configs.write();
        configs.insert(config.provider.clone(), config);
        Ok(())
    }

    /// Get OAuth2 authorization URL
    pub fn get_oauth2_auth_url(&self, provider: &str, state: &str) -> Result<String> {
        let configs = self.oauth2_configs.read();
        let config = configs
            .get(provider)
            .context("OAuth2 provider not configured")?;

        let scopes = config.scopes.join(" ");
        Ok(format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}",
            config.auth_url,
            config.client_id,
            urlencoding::encode(&config.redirect_uri),
            urlencoding::encode(&scopes),
            state
        ))
    }

    /// List all users (admin only)
    pub fn list_users(&self) -> Vec<User> {
        self.users.read().values().cloned().collect()
    }

    /// List all tokens for a user
    pub fn list_user_tokens(&self, user_id: &str) -> Vec<AuthToken> {
        self.tokens
            .read()
            .values()
            .filter(|t| t.user_id == user_id)
            .cloned()
            .collect()
    }

    /// Clean up expired tokens
    pub fn cleanup_expired_tokens(&self) -> usize {
        let mut tokens = self.tokens.write();
        let before_count = tokens.len();
        tokens.retain(|_, token| !token.is_expired());
        before_count - tokens.len()
    }
}

impl Default for AuthManager {
    fn default() -> Self {
        Self::new("default_secret_change_in_production".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        let admin_perms = Role::Admin.default_permissions();
        assert!(admin_perms.contains(&Permission::AdminManage));
        assert!(admin_perms.contains(&Permission::BlockWrite));

        let readonly_perms = Role::ReadOnly.default_permissions();
        assert!(readonly_perms.contains(&Permission::BlockRead));
        assert!(!readonly_perms.contains(&Permission::BlockWrite));
    }

    #[test]
    fn test_user_creation() {
        let manager = AuthManager::new("test_secret".to_string());

        let result = manager.create_user(
            "alice".to_string(),
            Some("alice@example.com".to_string()),
            vec![Role::User].into_iter().collect(),
        );
        assert!(result.is_ok());

        let user = result.expect("test: user creation should succeed");
        assert_eq!(user.username, "alice");
        assert!(user.has_permission(Permission::BlockRead));
    }

    #[test]
    fn test_api_key_creation() {
        let manager = AuthManager::new("test_secret".to_string());

        let user = manager
            .create_user(
                "bob".to_string(),
                None,
                vec![Role::Admin].into_iter().collect(),
            )
            .expect("test: user creation should succeed");

        let token = manager
            .create_api_key(&user.id, Some("test_key".to_string()))
            .expect("test: API key creation should succeed");
        assert_eq!(token.token_type, TokenType::ApiKey);
        assert!(token.secret.starts_with("ipfrs_"));
        assert!(token.is_admin());
    }

    #[test]
    fn test_jwt_creation() {
        let manager = AuthManager::new("test_secret".to_string());

        let user = manager
            .create_user(
                "charlie".to_string(),
                None,
                vec![Role::User].into_iter().collect(),
            )
            .expect("test: user creation should succeed");

        let token = manager
            .create_jwt_token(&user.id, None)
            .expect("test: JWT creation should succeed");
        assert_eq!(token.token_type, TokenType::Jwt);
        assert!(!token.is_expired());
    }

    #[test]
    fn test_token_verification() {
        let manager = AuthManager::new("test_secret".to_string());

        let user = manager
            .create_user(
                "dave".to_string(),
                None,
                vec![Role::User].into_iter().collect(),
            )
            .expect("test: user creation should succeed");

        let token = manager
            .create_api_key(&user.id, None)
            .expect("test: API key creation should succeed");
        let verified = manager.verify_token(&token.secret);
        assert!(verified.is_ok());

        let invalid = manager.verify_token("ipfrs_invalid");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_permission_check() {
        let manager = AuthManager::new("test_secret".to_string());

        let user = manager
            .create_user(
                "eve".to_string(),
                None,
                vec![Role::ReadOnly].into_iter().collect(),
            )
            .expect("test: user creation should succeed");

        let token = manager
            .create_api_key(&user.id, None)
            .expect("test: API key creation should succeed");

        assert!(manager
            .check_permission(&token, Permission::BlockRead)
            .is_ok());
        assert!(manager
            .check_permission(&token, Permission::BlockWrite)
            .is_err());
    }

    #[test]
    fn test_token_revocation() {
        let manager = AuthManager::new("test_secret".to_string());

        let user = manager
            .create_user(
                "frank".to_string(),
                None,
                vec![Role::User].into_iter().collect(),
            )
            .expect("test: user creation should succeed");

        let token = manager
            .create_api_key(&user.id, None)
            .expect("test: API key creation should succeed");
        assert!(manager.verify_token(&token.secret).is_ok());

        manager
            .revoke_token(&token.id)
            .expect("test: token revocation should succeed");
        assert!(manager.verify_token(&token.secret).is_err());
    }
}
