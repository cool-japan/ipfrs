//! OAuth2 Authentication Module
//!
//! Implements OAuth2 authentication flows including:
//! - Authorization Code Flow (with PKCE)
//! - Client Credentials Flow
//! - Refresh Token Flow
//!
//! Supports integration with external OAuth2 providers (Google, GitHub, etc.)
//! and can also act as an OAuth2 authorization server.

use crate::auth::{AuthError, AuthResult, JwtManager};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// OAuth2 grant types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantType {
    /// Authorization Code Flow
    AuthorizationCode,
    /// Client Credentials Flow
    ClientCredentials,
    /// Refresh Token Flow
    RefreshToken,
    /// Implicit Flow (deprecated, but included for compatibility)
    Implicit,
}

/// OAuth2 token type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    Bearer,
}

/// OAuth2 response type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseType {
    Code,
    Token,
}

/// OAuth2 scope
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope(String);

impl Scope {
    pub fn new(scope: impl Into<String>) -> Self {
        Self(scope.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Parse space-separated scopes
    pub fn parse_scopes(scopes: &str) -> Vec<Scope> {
        scopes.split_whitespace().map(Scope::new).collect()
    }

    /// Join scopes into space-separated string
    pub fn join_scopes(scopes: &[Scope]) -> String {
        scopes
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// OAuth2 client registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Client {
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uris: Vec<String>,
    pub grant_types: Vec<GrantType>,
    pub scopes: Vec<Scope>,
    pub name: String,
    pub created_at: u64,
}

impl OAuth2Client {
    pub fn new(
        name: String,
        redirect_uris: Vec<String>,
        grant_types: Vec<GrantType>,
        scopes: Vec<Scope>,
    ) -> Self {
        Self {
            client_id: Uuid::new_v4().to_string(),
            client_secret: Uuid::new_v4().to_string(),
            redirect_uris,
            grant_types,
            scopes,
            name,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
        }
    }

    /// Verify client secret
    pub fn verify_secret(&self, secret: &str) -> bool {
        self.client_secret == secret
    }

    /// Check if redirect URI is allowed
    pub fn is_redirect_uri_allowed(&self, uri: &str) -> bool {
        self.redirect_uris.iter().any(|u| u == uri)
    }

    /// Check if grant type is allowed
    pub fn is_grant_type_allowed(&self, grant_type: GrantType) -> bool {
        self.grant_types.contains(&grant_type)
    }

    /// Check if scope is allowed
    pub fn is_scope_allowed(&self, scope: &Scope) -> bool {
        self.scopes.contains(scope)
    }
}

/// Authorization code
#[derive(Debug, Clone)]
pub struct AuthorizationCode {
    pub code: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scopes: Vec<Scope>,
    pub user_id: String,
    pub expires_at: u64,
    /// PKCE code challenge
    pub code_challenge: Option<String>,
    /// PKCE code challenge method
    pub code_challenge_method: Option<CodeChallengeMethod>,
}

impl AuthorizationCode {
    pub fn new(
        client_id: String,
        redirect_uri: String,
        scopes: Vec<Scope>,
        user_id: String,
        ttl: Duration,
        code_challenge: Option<String>,
        code_challenge_method: Option<CodeChallengeMethod>,
    ) -> Self {
        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs()
            + ttl.as_secs();

        Self {
            code: Uuid::new_v4().to_string(),
            client_id,
            redirect_uri,
            scopes,
            user_id,
            expires_at,
            code_challenge,
            code_challenge_method,
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs();
        now > self.expires_at
    }

    /// Verify PKCE code verifier
    pub fn verify_code_verifier(&self, verifier: &str) -> bool {
        match (&self.code_challenge, &self.code_challenge_method) {
            (Some(challenge), Some(method)) => {
                let computed_challenge = method.compute_challenge(verifier);
                &computed_challenge == challenge
            }
            (None, None) => true, // No PKCE required
            _ => false,           // Inconsistent PKCE state
        }
    }
}

/// PKCE code challenge method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeChallengeMethod {
    #[serde(rename = "plain")]
    Plain,
    #[serde(rename = "S256")]
    S256,
}

impl CodeChallengeMethod {
    /// Compute challenge from verifier
    pub fn compute_challenge(&self, verifier: &str) -> String {
        match self {
            Self::Plain => verifier.to_string(),
            Self::S256 => {
                use sha2::{Digest, Sha256};
                let hash = Sha256::digest(verifier.as_bytes());
                base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, hash)
            }
        }
    }
}

/// OAuth2 access token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    pub token: String,
    pub token_type: TokenType,
    pub expires_in: u64,
    pub scopes: Vec<Scope>,
    pub user_id: String,
    pub created_at: u64,
}

impl AccessToken {
    pub fn new(token: String, scopes: Vec<Scope>, user_id: String, ttl: Duration) -> Self {
        Self {
            token,
            token_type: TokenType::Bearer,
            expires_in: ttl.as_secs(),
            scopes,
            user_id,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs();
        now > self.created_at + self.expires_in
    }
}

/// OAuth2 refresh token
#[derive(Debug, Clone)]
pub struct RefreshToken {
    pub token: String,
    pub client_id: String,
    pub user_id: String,
    pub scopes: Vec<Scope>,
    pub created_at: u64,
}

impl RefreshToken {
    pub fn new(client_id: String, user_id: String, scopes: Vec<Scope>) -> Self {
        Self {
            token: Uuid::new_v4().to_string(),
            client_id,
            user_id,
            scopes,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
        }
    }
}

/// OAuth2 token response
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// OAuth2 error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_description: Option<String>,
}

/// OAuth2 authorization server
pub struct OAuth2Server {
    clients: Arc<DashMap<String, OAuth2Client>>,
    authorization_codes: Arc<DashMap<String, AuthorizationCode>>,
    access_tokens: Arc<DashMap<String, AccessToken>>,
    refresh_tokens: Arc<DashMap<String, RefreshToken>>,
    jwt_manager: Arc<JwtManager>,
    /// Default access token TTL
    access_token_ttl: Duration,
    /// Default refresh token TTL
    #[allow(dead_code)]
    refresh_token_ttl: Duration,
    /// Default authorization code TTL
    code_ttl: Duration,
}

impl OAuth2Server {
    pub fn new(jwt_secret: &[u8]) -> Self {
        Self {
            clients: Arc::new(DashMap::new()),
            authorization_codes: Arc::new(DashMap::new()),
            access_tokens: Arc::new(DashMap::new()),
            refresh_tokens: Arc::new(DashMap::new()),
            jwt_manager: Arc::new(JwtManager::new(jwt_secret)),
            access_token_ttl: Duration::from_secs(3600), // 1 hour
            refresh_token_ttl: Duration::from_secs(86400 * 30), // 30 days
            code_ttl: Duration::from_secs(600),          // 10 minutes
        }
    }

    /// Register a new OAuth2 client
    pub fn register_client(
        &self,
        name: String,
        redirect_uris: Vec<String>,
        grant_types: Vec<GrantType>,
        scopes: Vec<Scope>,
    ) -> OAuth2Client {
        let client = OAuth2Client::new(name, redirect_uris, grant_types, scopes);
        self.clients
            .insert(client.client_id.clone(), client.clone());
        client
    }

    /// Get client by ID
    pub fn get_client(&self, client_id: &str) -> Option<OAuth2Client> {
        self.clients.get(client_id).map(|c| c.clone())
    }

    /// Authorize request (Authorization Code Flow)
    #[allow(clippy::too_many_arguments)]
    pub fn authorize(
        &self,
        client_id: &str,
        redirect_uri: &str,
        response_type: ResponseType,
        scopes: Vec<Scope>,
        user_id: String,
        code_challenge: Option<String>,
        code_challenge_method: Option<CodeChallengeMethod>,
    ) -> AuthResult<AuthorizationCode> {
        // Validate client
        let client = self
            .get_client(client_id)
            .ok_or(AuthError::InvalidCredentials)?;

        // Validate redirect URI
        if !client.is_redirect_uri_allowed(redirect_uri) {
            return Err(AuthError::InvalidCredentials);
        }

        // Validate grant type
        if !client.is_grant_type_allowed(GrantType::AuthorizationCode) {
            return Err(AuthError::InvalidCredentials);
        }

        // Validate scopes
        for scope in &scopes {
            if !client.is_scope_allowed(scope) {
                return Err(AuthError::InsufficientPermissions);
            }
        }

        // Validate response type
        if response_type != ResponseType::Code {
            return Err(AuthError::InvalidCredentials);
        }

        // Create authorization code
        let auth_code = AuthorizationCode::new(
            client_id.to_string(),
            redirect_uri.to_string(),
            scopes,
            user_id,
            self.code_ttl,
            code_challenge,
            code_challenge_method,
        );

        self.authorization_codes
            .insert(auth_code.code.clone(), auth_code.clone());

        Ok(auth_code)
    }

    /// Exchange authorization code for tokens
    pub fn exchange_code(
        &self,
        client_id: &str,
        client_secret: &str,
        code: &str,
        redirect_uri: &str,
        code_verifier: Option<&str>,
    ) -> AuthResult<(AccessToken, RefreshToken)> {
        // Validate client
        let client = self
            .get_client(client_id)
            .ok_or(AuthError::InvalidCredentials)?;

        if !client.verify_secret(client_secret) {
            return Err(AuthError::InvalidCredentials);
        }

        // Get and remove authorization code (one-time use)
        let auth_code = self
            .authorization_codes
            .remove(code)
            .ok_or(AuthError::InvalidToken("Invalid code".to_string()))?
            .1;

        // Validate code
        if auth_code.is_expired() {
            return Err(AuthError::TokenExpired);
        }

        if auth_code.client_id != client_id {
            return Err(AuthError::InvalidCredentials);
        }

        if auth_code.redirect_uri != redirect_uri {
            return Err(AuthError::InvalidCredentials);
        }

        // Verify PKCE code verifier if required
        if let Some(verifier) = code_verifier {
            if !auth_code.verify_code_verifier(verifier) {
                return Err(AuthError::InvalidCredentials);
            }
        } else if auth_code.code_challenge.is_some() {
            // Code challenge was provided but no verifier
            return Err(AuthError::InvalidCredentials);
        }

        // Generate access token using JWT
        let access_token_jwt = self
            .jwt_manager
            .generate_token_with_scopes(
                &auth_code.user_id,
                &Scope::join_scopes(&auth_code.scopes),
                (self.access_token_ttl.as_secs() / 3600) as usize,
            )
            .map_err(|_| AuthError::InvalidToken("Failed to generate token".to_string()))?;

        let access_token = AccessToken::new(
            access_token_jwt,
            auth_code.scopes.clone(),
            auth_code.user_id.clone(),
            self.access_token_ttl,
        );

        // Generate refresh token
        let refresh_token = RefreshToken::new(
            client_id.to_string(),
            auth_code.user_id.clone(),
            auth_code.scopes.clone(),
        );

        // Store tokens
        self.access_tokens
            .insert(access_token.token.clone(), access_token.clone());
        self.refresh_tokens
            .insert(refresh_token.token.clone(), refresh_token.clone());

        Ok((access_token, refresh_token))
    }

    /// Client Credentials Flow
    pub fn client_credentials(
        &self,
        client_id: &str,
        client_secret: &str,
        scopes: Vec<Scope>,
    ) -> AuthResult<AccessToken> {
        // Validate client
        let client = self
            .get_client(client_id)
            .ok_or(AuthError::InvalidCredentials)?;

        if !client.verify_secret(client_secret) {
            return Err(AuthError::InvalidCredentials);
        }

        // Validate grant type
        if !client.is_grant_type_allowed(GrantType::ClientCredentials) {
            return Err(AuthError::InvalidCredentials);
        }

        // Validate scopes
        for scope in &scopes {
            if !client.is_scope_allowed(scope) {
                return Err(AuthError::InsufficientPermissions);
            }
        }

        // Generate access token (use client_id as user_id for client credentials)
        let access_token_jwt = self
            .jwt_manager
            .generate_token_with_scopes(
                client_id,
                &Scope::join_scopes(&scopes),
                (self.access_token_ttl.as_secs() / 3600) as usize,
            )
            .map_err(|_| AuthError::InvalidToken("Failed to generate token".to_string()))?;

        let access_token = AccessToken::new(
            access_token_jwt,
            scopes,
            client_id.to_string(),
            self.access_token_ttl,
        );

        self.access_tokens
            .insert(access_token.token.clone(), access_token.clone());

        Ok(access_token)
    }

    /// Refresh access token
    pub fn refresh_token(
        &self,
        client_id: &str,
        client_secret: &str,
        refresh_token: &str,
    ) -> AuthResult<AccessToken> {
        // Validate client
        let client = self
            .get_client(client_id)
            .ok_or(AuthError::InvalidCredentials)?;

        if !client.verify_secret(client_secret) {
            return Err(AuthError::InvalidCredentials);
        }

        // Get refresh token
        let rt = self
            .refresh_tokens
            .get(refresh_token)
            .ok_or(AuthError::InvalidToken("Invalid refresh token".to_string()))?;

        if rt.client_id != client_id {
            return Err(AuthError::InvalidCredentials);
        }

        // Generate new access token
        let access_token_jwt = self
            .jwt_manager
            .generate_token_with_scopes(
                &rt.user_id,
                &Scope::join_scopes(&rt.scopes),
                (self.access_token_ttl.as_secs() / 3600) as usize,
            )
            .map_err(|_| AuthError::InvalidToken("Failed to generate token".to_string()))?;

        let access_token = AccessToken::new(
            access_token_jwt,
            rt.scopes.clone(),
            rt.user_id.clone(),
            self.access_token_ttl,
        );

        self.access_tokens
            .insert(access_token.token.clone(), access_token.clone());

        Ok(access_token)
    }

    /// Validate access token
    pub fn validate_token(&self, token: &str) -> AuthResult<AccessToken> {
        let access_token = self
            .access_tokens
            .get(token)
            .ok_or(AuthError::InvalidToken("Token not found".to_string()))?;

        if access_token.is_expired() {
            // Remove expired token
            drop(access_token);
            self.access_tokens.remove(token);
            return Err(AuthError::TokenExpired);
        }

        Ok(access_token.clone())
    }

    /// Revoke access token
    pub fn revoke_access_token(&self, token: &str) -> bool {
        self.access_tokens.remove(token).is_some()
    }

    /// Revoke refresh token
    pub fn revoke_refresh_token(&self, token: &str) -> bool {
        self.refresh_tokens.remove(token).is_some()
    }

    /// Clean up expired tokens and codes
    pub fn cleanup_expired(&self) {
        // Clean up expired authorization codes
        self.authorization_codes
            .retain(|_, code| !code.is_expired());

        // Clean up expired access tokens
        self.access_tokens.retain(|_, token| !token.is_expired());
    }
}

impl Default for OAuth2Server {
    fn default() -> Self {
        Self::new(b"default-secret-change-in-production")
    }
}

/// OAuth2 provider configuration for external providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2ProviderConfig {
    pub name: String,
    pub client_id: String,
    pub client_secret: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub redirect_uri: String,
    pub scopes: Vec<Scope>,
}

impl OAuth2ProviderConfig {
    /// Create configuration for Google OAuth2
    pub fn google(client_id: String, client_secret: String, redirect_uri: String) -> Self {
        Self {
            name: "google".to_string(),
            client_id,
            client_secret,
            authorization_endpoint: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_endpoint: "https://oauth2.googleapis.com/token".to_string(),
            redirect_uri,
            scopes: vec![
                Scope::new("openid"),
                Scope::new("email"),
                Scope::new("profile"),
            ],
        }
    }

    /// Create configuration for GitHub OAuth2
    pub fn github(client_id: String, client_secret: String, redirect_uri: String) -> Self {
        Self {
            name: "github".to_string(),
            client_id,
            client_secret,
            authorization_endpoint: "https://github.com/login/oauth/authorize".to_string(),
            token_endpoint: "https://github.com/login/oauth/access_token".to_string(),
            redirect_uri,
            scopes: vec![Scope::new("user:email"), Scope::new("read:user")],
        }
    }

    /// Build authorization URL
    pub fn build_auth_url(&self, state: &str) -> String {
        let scope = Scope::join_scopes(&self.scopes);
        format!(
            "{}?client_id={}&redirect_uri={}&scope={}&response_type=code&state={}",
            self.authorization_endpoint,
            urlencoding::encode(&self.client_id),
            urlencoding::encode(&self.redirect_uri),
            urlencoding::encode(&scope),
            state
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_scope_parsing() {
        let scopes = Scope::parse_scopes("read write admin");
        assert_eq!(scopes.len(), 3);
        assert_eq!(scopes[0].as_str(), "read");
        assert_eq!(scopes[1].as_str(), "write");
        assert_eq!(scopes[2].as_str(), "admin");
    }

    #[test]
    fn test_scope_joining() {
        let scopes = vec![Scope::new("read"), Scope::new("write")];
        let joined = Scope::join_scopes(&scopes);
        assert_eq!(joined, "read write");
    }

    #[test]
    fn test_client_creation() {
        let client = OAuth2Client::new(
            "test-client".to_string(),
            vec!["http://localhost:3000/callback".to_string()],
            vec![GrantType::AuthorizationCode],
            vec![Scope::new("read")],
        );

        assert!(!client.client_id.is_empty());
        assert!(!client.client_secret.is_empty());
        assert_eq!(client.name, "test-client");
    }

    #[test]
    fn test_client_verification() {
        let client = OAuth2Client::new(
            "test".to_string(),
            vec!["http://localhost/callback".to_string()],
            vec![GrantType::AuthorizationCode],
            vec![Scope::new("read")],
        );

        assert!(client.verify_secret(&client.client_secret));
        assert!(!client.verify_secret("wrong-secret"));
        assert!(client.is_redirect_uri_allowed("http://localhost/callback"));
        assert!(!client.is_redirect_uri_allowed("http://evil.com/callback"));
    }

    #[test]
    fn test_pkce_plain() {
        let method = CodeChallengeMethod::Plain;
        let verifier = "test-verifier";
        let challenge = method.compute_challenge(verifier);
        assert_eq!(challenge, verifier);
    }

    #[test]
    fn test_pkce_s256() {
        let method = CodeChallengeMethod::S256;
        let verifier = "test-verifier-with-sufficient-entropy";
        let challenge = method.compute_challenge(verifier);
        assert_ne!(challenge, verifier);
        assert!(!challenge.is_empty());

        // Verify consistency
        let challenge2 = method.compute_challenge(verifier);
        assert_eq!(challenge, challenge2);
    }

    #[test]
    fn test_authorization_code_expiry() {
        // Create a code that expires 1 second ago
        let code = AuthorizationCode {
            code: "test-code".to_string(),
            client_id: "client-id".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            scopes: vec![Scope::new("read")],
            user_id: "user-id".to_string(),
            expires_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs()
                - 1, // Expired 1 second ago
            code_challenge: None,
            code_challenge_method: None,
        };

        assert!(code.is_expired());
    }

    #[test]
    fn test_oauth2_server_client_registration() {
        let server = OAuth2Server::default();
        let client = server.register_client(
            "test-client".to_string(),
            vec!["http://localhost/callback".to_string()],
            vec![GrantType::AuthorizationCode],
            vec![Scope::new("read")],
        );

        let retrieved = server.get_client(&client.client_id);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved
                .expect("test: registered client should be retrievable by ID")
                .name,
            "test-client"
        );
    }

    #[test]
    fn test_oauth2_server_authorization() {
        let server = OAuth2Server::default();
        let client = server.register_client(
            "test".to_string(),
            vec!["http://localhost/callback".to_string()],
            vec![GrantType::AuthorizationCode],
            vec![Scope::new("read")],
        );

        let auth_code = server
            .authorize(
                &client.client_id,
                "http://localhost/callback",
                ResponseType::Code,
                vec![Scope::new("read")],
                "user-123".to_string(),
                None,
                None,
            )
            .expect("test: authorization code grant should succeed");

        assert!(!auth_code.code.is_empty());
        assert_eq!(auth_code.user_id, "user-123");
    }

    #[test]
    fn test_oauth2_server_client_credentials() {
        let server = OAuth2Server::default();
        let client = server.register_client(
            "test".to_string(),
            vec![],
            vec![GrantType::ClientCredentials],
            vec![Scope::new("read")],
        );

        let token = server
            .client_credentials(
                &client.client_id,
                &client.client_secret,
                vec![Scope::new("read")],
            )
            .expect("test: client credentials grant should succeed");

        assert!(!token.token.is_empty());
        assert_eq!(token.token_type, TokenType::Bearer);
    }

    #[test]
    fn test_provider_config_google() {
        let config = OAuth2ProviderConfig::google(
            "client-id".to_string(),
            "client-secret".to_string(),
            "http://localhost/callback".to_string(),
        );

        assert_eq!(config.name, "google");
        assert!(config.authorization_endpoint.contains("google"));

        let url = config.build_auth_url("random-state");
        assert!(url.contains("client_id=client-id"));
        assert!(url.contains("state=random-state"));
    }

    #[test]
    fn test_provider_config_github() {
        let config = OAuth2ProviderConfig::github(
            "client-id".to_string(),
            "client-secret".to_string(),
            "http://localhost/callback".to_string(),
        );

        assert_eq!(config.name, "github");
        assert!(config.authorization_endpoint.contains("github"));
    }

    #[test]
    fn test_token_validation() {
        let server = OAuth2Server::default();
        let client = server.register_client(
            "test".to_string(),
            vec![],
            vec![GrantType::ClientCredentials],
            vec![Scope::new("read")],
        );

        let token = server
            .client_credentials(
                &client.client_id,
                &client.client_secret,
                vec![Scope::new("read")],
            )
            .expect("test: client credentials grant should succeed for token validation");

        // Token should be valid
        let validated = server.validate_token(&token.token);
        assert!(validated.is_ok());

        // Revoke token
        server.revoke_access_token(&token.token);

        // Token should now be invalid
        let validated = server.validate_token(&token.token);
        assert!(validated.is_err());
    }
}
