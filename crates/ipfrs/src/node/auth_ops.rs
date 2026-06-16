//! Authentication and security operations for Node

use ipfrs_core::Result;

use crate::auth::{AuthToken, Permission, Role, User};

use super::Node;

impl Node {
    /// Create a new user (requires auth enabled)
    pub fn create_user(
        &self,
        username: String,
        email: Option<String>,
        roles: std::collections::HashSet<Role>,
    ) -> Result<User> {
        let auth = self.auth_manager()?;
        auth.create_user(username, email, roles)
            .map_err(|e| ipfrs_core::Error::Internal(format!("Failed to create user: {}", e)))
    }

    /// Verify an authentication token
    pub fn verify_token(&self, token: &str) -> Result<AuthToken> {
        let auth = self.auth_manager()?;
        auth.verify_token(token)
            .map_err(|e| ipfrs_core::Error::Internal(format!("Token verification failed: {}", e)))
    }

    /// Check if a token has a specific permission
    pub fn check_permission(&self, token: &AuthToken, permission: Permission) -> Result<()> {
        let auth = self.auth_manager()?;
        auth.check_permission(token, permission)
            .map_err(|e| ipfrs_core::Error::Internal(format!("Permission check failed: {}", e)))
    }
}
