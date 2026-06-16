//! Authentication and Authorization Module
//!
//! Provides JWT-based authentication and role-based access control (RBAC)
//! for the IPFRS HTTP API.

use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Token expired")]
    TokenExpired,

    #[error("Insufficient permissions")]
    InsufficientPermissions,

    #[error("User not found")]
    UserNotFound,

    #[error("Hashing error: {0}")]
    HashError(String),

    #[error("JWT error: {0}")]
    JwtError(#[from] jsonwebtoken::errors::Error),
}

pub type AuthResult<T> = Result<T, AuthError>;

/// User roles in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Administrator - full access to all operations
    Admin,
    /// User - can read and write data
    User,
    /// Read-only access
    ReadOnly,
}

impl Role {
    /// Check if this role has at least the required permission level
    pub fn has_permission(&self, required: Role) -> bool {
        matches!(
            (self, required),
            (Role::Admin, _)
                | (Role::User, Role::User)
                | (Role::User, Role::ReadOnly)
                | (Role::ReadOnly, Role::ReadOnly)
        )
    }
}

/// Permissions that can be granted to users
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    // Block operations
    BlockRead,
    BlockWrite,
    BlockDelete,

    // Semantic operations
    SemanticIndex,
    SemanticSearch,

    // Logic operations
    LogicRead,
    LogicWrite,

    // Network operations
    NetworkRead,
    NetworkWrite,

    // System operations
    SystemRead,
    SystemWrite,
    SystemAdmin,
}

impl Permission {
    /// Get all permissions for a role
    pub fn for_role(role: Role) -> HashSet<Permission> {
        match role {
            Role::Admin => {
                // Admin has all permissions
                vec![
                    Permission::BlockRead,
                    Permission::BlockWrite,
                    Permission::BlockDelete,
                    Permission::SemanticIndex,
                    Permission::SemanticSearch,
                    Permission::LogicRead,
                    Permission::LogicWrite,
                    Permission::NetworkRead,
                    Permission::NetworkWrite,
                    Permission::SystemRead,
                    Permission::SystemWrite,
                    Permission::SystemAdmin,
                ]
                .into_iter()
                .collect()
            }
            Role::User => {
                // User can read and write, but not admin operations
                vec![
                    Permission::BlockRead,
                    Permission::BlockWrite,
                    Permission::SemanticIndex,
                    Permission::SemanticSearch,
                    Permission::LogicRead,
                    Permission::LogicWrite,
                    Permission::NetworkRead,
                    Permission::SystemRead,
                ]
                .into_iter()
                .collect()
            }
            Role::ReadOnly => {
                // Read-only can only read
                vec![
                    Permission::BlockRead,
                    Permission::SemanticSearch,
                    Permission::LogicRead,
                    Permission::NetworkRead,
                    Permission::SystemRead,
                ]
                .into_iter()
                .collect()
            }
        }
    }
}

/// User information stored in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique user ID
    pub id: Uuid,
    /// Username
    pub username: String,
    /// Password hash (bcrypt)
    #[serde(skip_serializing)]
    pub password_hash: String,
    /// User role
    pub role: Role,
    /// Custom permissions (in addition to role permissions)
    pub custom_permissions: HashSet<Permission>,
    /// Whether the user is active
    pub active: bool,
    /// Creation timestamp
    pub created_at: u64,
}

impl User {
    /// Create a new user with hashed password
    pub fn new(username: String, password: &str, role: Role) -> AuthResult<Self> {
        let password_hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)
            .map_err(|e| AuthError::HashError(e.to_string()))?;

        Ok(Self {
            id: Uuid::new_v4(),
            username,
            password_hash,
            role,
            custom_permissions: HashSet::new(),
            active: true,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
        })
    }

    /// Verify password
    pub fn verify_password(&self, password: &str) -> AuthResult<bool> {
        bcrypt::verify(password, &self.password_hash)
            .map_err(|e| AuthError::HashError(e.to_string()))
    }

    /// Get all permissions for this user (role + custom)
    pub fn permissions(&self) -> HashSet<Permission> {
        let mut perms = Permission::for_role(self.role);
        perms.extend(self.custom_permissions.iter().copied());
        perms
    }

    /// Check if user has a specific permission
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.permissions().contains(&permission)
    }
}

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Username
    pub username: String,
    /// User role
    pub role: Role,
    /// OAuth2 scopes (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Issued at (UNIX timestamp)
    pub iat: u64,
    /// Expiration time (UNIX timestamp)
    pub exp: u64,
}

impl Claims {
    /// Create new claims for a user
    pub fn new(user: &User, expiration_hours: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs();

        Self {
            sub: user.id.to_string(),
            username: user.username.clone(),
            role: user.role,
            scope: None,
            iat: now,
            exp: now + (expiration_hours * 3600),
        }
    }

    /// Create new claims with OAuth2 scopes
    pub fn new_with_scopes(sub: &str, scope: &str, expiration_hours: usize) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs();

        Self {
            sub: sub.to_string(),
            username: sub.to_string(), // Use sub as username for OAuth2 tokens
            role: Role::User,          // Default role for OAuth2
            scope: Some(scope.to_string()),
            iat: now,
            exp: now + ((expiration_hours as u64) * 3600),
        }
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs();
        now > self.exp
    }
}

/// JWT token manager
#[derive(Clone)]
pub struct JwtManager {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
}

impl JwtManager {
    /// Create a new JWT manager with a secret key
    pub fn new(secret: &[u8]) -> Self {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;

        Self {
            encoding_key: EncodingKey::from_secret(secret),
            decoding_key: DecodingKey::from_secret(secret),
            validation,
        }
    }

    /// Generate a JWT token for a user
    pub fn generate_token(&self, user: &User, expiration_hours: u64) -> AuthResult<String> {
        let claims = Claims::new(user, expiration_hours);
        let token = encode(&Header::default(), &claims, &self.encoding_key)?;
        Ok(token)
    }

    /// Generate a JWT token with OAuth2 scopes
    pub fn generate_token_with_scopes(
        &self,
        sub: &str,
        scope: &str,
        expiration_hours: usize,
    ) -> AuthResult<String> {
        let claims = Claims::new_with_scopes(sub, scope, expiration_hours);
        let token = encode(&Header::default(), &claims, &self.encoding_key)?;
        Ok(token)
    }

    /// Validate and decode a JWT token
    pub fn validate_token(&self, token: &str) -> AuthResult<Claims> {
        let token_data = decode::<Claims>(token, &self.decoding_key, &self.validation)?;

        if token_data.claims.is_expired() {
            return Err(AuthError::TokenExpired);
        }

        Ok(token_data.claims)
    }
}

/// User store (in-memory for now, should be persisted in production)
#[derive(Clone)]
pub struct UserStore {
    users: dashmap::DashMap<String, User>,
}

impl UserStore {
    /// Create a new user store
    pub fn new() -> Self {
        Self {
            users: dashmap::DashMap::new(),
        }
    }

    /// Add a user to the store
    pub fn add_user(&self, user: User) -> AuthResult<()> {
        if self.users.contains_key(&user.username) {
            return Err(AuthError::InvalidCredentials);
        }
        self.users.insert(user.username.clone(), user);
        Ok(())
    }

    /// Get a user by username
    pub fn get_user(&self, username: &str) -> AuthResult<User> {
        self.users
            .get(username)
            .map(|entry| entry.clone())
            .ok_or(AuthError::UserNotFound)
    }

    /// Get a user by ID
    pub fn get_by_id(&self, user_id: &uuid::Uuid) -> AuthResult<User> {
        for entry in self.users.iter() {
            let user = entry.value();
            if user.id == *user_id {
                return Ok(user.clone());
            }
        }
        Err(AuthError::UserNotFound)
    }

    /// Authenticate a user with username and password
    pub fn authenticate(&self, username: &str, password: &str) -> AuthResult<User> {
        let user = self.get_user(username)?;

        if !user.active {
            return Err(AuthError::InvalidCredentials);
        }

        if !user.verify_password(password)? {
            return Err(AuthError::InvalidCredentials);
        }

        Ok(user)
    }

    /// Update user permissions
    pub fn update_permissions(
        &self,
        username: &str,
        permissions: HashSet<Permission>,
    ) -> AuthResult<()> {
        self.users
            .get_mut(username)
            .map(|mut entry| {
                entry.custom_permissions = permissions;
            })
            .ok_or(AuthError::UserNotFound)
    }

    /// Deactivate a user
    pub fn deactivate_user(&self, username: &str) -> AuthResult<()> {
        self.users
            .get_mut(username)
            .map(|mut entry| {
                entry.active = false;
            })
            .ok_or(AuthError::UserNotFound)
    }
}

impl Default for UserStore {
    fn default() -> Self {
        Self::new()
    }
}

/// API Key for long-lived authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// API key ID
    pub id: Uuid,
    /// The actual key (hashed in storage)
    #[serde(skip_serializing)]
    pub key_hash: String,
    /// Key prefix for identification (first 8 chars of key)
    pub prefix: String,
    /// Associated user ID
    pub user_id: Uuid,
    /// Optional key name/description
    pub name: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Last used timestamp
    pub last_used_at: Option<u64>,
    /// Whether the key is active
    pub active: bool,
}

impl ApiKey {
    /// Generate a new API key
    pub fn new(user_id: Uuid, name: String) -> AuthResult<(Self, String)> {
        let key_id = Uuid::new_v4();

        // Generate random API key (32 bytes = 64 hex chars)
        let key_bytes: [u8; 32] = rand::random();
        let raw_key = format!("ipfrs_{}", hex::encode(key_bytes));

        // Hash the key for storage
        let key_hash = bcrypt::hash(&raw_key, bcrypt::DEFAULT_COST)
            .map_err(|e| AuthError::HashError(e.to_string()))?;

        // Store prefix for identification
        let prefix = raw_key.chars().take(12).collect();

        let api_key = Self {
            id: key_id,
            key_hash,
            prefix,
            user_id,
            name,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
            last_used_at: None,
            active: true,
        };

        Ok((api_key, raw_key))
    }

    /// Verify an API key
    pub fn verify(&self, key: &str) -> AuthResult<bool> {
        bcrypt::verify(key, &self.key_hash).map_err(|e| AuthError::HashError(e.to_string()))
    }

    /// Update last used timestamp
    pub fn mark_used(&mut self) {
        self.last_used_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
        );
    }
}

/// API Key store
#[derive(Clone)]
pub struct ApiKeyStore {
    keys: dashmap::DashMap<Uuid, ApiKey>,
    user_keys: dashmap::DashMap<Uuid, Vec<Uuid>>, // user_id -> key_ids
}

impl ApiKeyStore {
    /// Create a new API key store
    pub fn new() -> Self {
        Self {
            keys: dashmap::DashMap::new(),
            user_keys: dashmap::DashMap::new(),
        }
    }

    /// Add an API key
    pub fn add_key(&self, key: ApiKey) -> AuthResult<()> {
        let user_id = key.user_id;
        let key_id = key.id;

        self.keys.insert(key_id, key);

        // Add to user's key list
        self.user_keys.entry(user_id).or_default().push(key_id);

        Ok(())
    }

    /// Get an API key by ID
    pub fn get_key(&self, key_id: &Uuid) -> AuthResult<ApiKey> {
        self.keys
            .get(key_id)
            .map(|entry| entry.clone())
            .ok_or(AuthError::InvalidCredentials)
    }

    /// Authenticate with API key
    pub fn authenticate(&self, key: &str) -> AuthResult<(ApiKey, Uuid)> {
        // Extract prefix for faster lookup
        let prefix: String = key.chars().take(12).collect();

        // Find key by prefix
        for entry in self.keys.iter() {
            let api_key = entry.value();
            if api_key.prefix == prefix && api_key.active && api_key.verify(key)? {
                // Update last used
                let mut key_mut = self
                    .keys
                    .get_mut(&api_key.id)
                    .expect("key exists: we just found it in the same map via iter()");
                key_mut.mark_used();

                return Ok((api_key.clone(), api_key.user_id));
            }
        }

        Err(AuthError::InvalidCredentials)
    }

    /// List API keys for a user
    pub fn list_user_keys(&self, user_id: &Uuid) -> Vec<ApiKey> {
        if let Some(key_ids) = self.user_keys.get(user_id) {
            key_ids
                .iter()
                .filter_map(|id| self.keys.get(id).map(|e| e.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Revoke (deactivate) an API key
    pub fn revoke_key(&self, key_id: &Uuid) -> AuthResult<()> {
        self.keys
            .get_mut(key_id)
            .map(|mut entry| {
                entry.active = false;
            })
            .ok_or(AuthError::InvalidCredentials)
    }

    /// Delete an API key
    pub fn delete_key(&self, key_id: &Uuid) -> AuthResult<()> {
        if let Some((_, key)) = self.keys.remove(key_id) {
            // Remove from user's key list
            if let Some(mut user_keys) = self.user_keys.get_mut(&key.user_id) {
                user_keys.retain(|id| id != key_id);
            }
            Ok(())
        } else {
            Err(AuthError::InvalidCredentials)
        }
    }
}

impl Default for ApiKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Authentication middleware state
#[derive(Clone)]
pub struct AuthState {
    pub jwt_manager: JwtManager,
    pub user_store: UserStore,
    pub api_key_store: ApiKeyStore,
}

impl AuthState {
    /// Create new auth state with a secret key
    pub fn new(secret: &[u8]) -> Self {
        Self {
            jwt_manager: JwtManager::new(secret),
            user_store: UserStore::new(),
            api_key_store: ApiKeyStore::new(),
        }
    }

    /// Initialize with a default admin user
    pub fn with_default_admin(secret: &[u8], admin_password: &str) -> AuthResult<Self> {
        let state = Self::new(secret);

        // Create default admin user
        let admin = User::new("admin".to_string(), admin_password, Role::Admin)?;
        state.user_store.add_user(admin)?;

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        assert!(Role::Admin.has_permission(Role::User));
        assert!(Role::Admin.has_permission(Role::ReadOnly));
        assert!(Role::User.has_permission(Role::ReadOnly));
        assert!(!Role::ReadOnly.has_permission(Role::User));
    }

    #[test]
    fn test_user_creation() {
        let user =
            User::new("test".to_string(), "password123", Role::User).expect("test: create user");
        assert_eq!(user.username, "test");
        assert_eq!(user.role, Role::User);
        assert!(user.active);
        assert!(user
            .verify_password("password123")
            .expect("test: verify correct password"));
        assert!(!user
            .verify_password("wrong")
            .expect("test: verify wrong password"));
    }

    #[test]
    fn test_user_permissions() {
        let user = User::new("test".to_string(), "password123", Role::User)
            .expect("test: user creation should succeed");
        assert!(user.has_permission(Permission::BlockRead));
        assert!(user.has_permission(Permission::BlockWrite));
        assert!(!user.has_permission(Permission::BlockDelete));
        assert!(!user.has_permission(Permission::SystemAdmin));
    }

    #[test]
    fn test_jwt_generation_and_validation() {
        let secret = b"test_secret_key_32_bytes_long!!!";
        let manager = JwtManager::new(secret);
        let user = User::new("test".to_string(), "password123", Role::User)
            .expect("test: user creation should succeed");

        let token = manager
            .generate_token(&user, 24)
            .expect("test: token generation should succeed");
        let claims = manager
            .validate_token(&token)
            .expect("test: token validation should succeed");

        assert_eq!(claims.username, "test");
        assert_eq!(claims.role, Role::User);
        assert!(!claims.is_expired());
    }

    #[test]
    fn test_user_store() {
        let store = UserStore::new();
        let user = User::new("test".to_string(), "password123", Role::User)
            .expect("test: user creation should succeed");

        store
            .add_user(user)
            .expect("test: user should be added to store");

        let authenticated = store
            .authenticate("test", "password123")
            .expect("test: authentication with correct credentials should succeed");
        assert_eq!(authenticated.username, "test");

        assert!(store.authenticate("test", "wrong").is_err());
        assert!(store.authenticate("nonexistent", "password123").is_err());
    }
}
