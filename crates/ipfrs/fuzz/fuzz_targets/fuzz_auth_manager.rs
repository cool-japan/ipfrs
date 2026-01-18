//! Fuzz test for AuthManager user operations
//!
//! Tests robustness of user creation and management operations

#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs::{AuthManager, Role};
use std::collections::HashSet;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let manager = AuthManager::new("fuzz_secret".to_string());

    // Use fuzzed data to create usernames and operations
    let username = String::from_utf8_lossy(&data[0..data.len().min(32)]).to_string();
    let email = if data.len() > 32 {
        Some(String::from_utf8_lossy(&data[32..data.len().min(64)]).to_string())
    } else {
        None
    };

    // Choose role based on first byte
    let role = match data[0] % 4 {
        0 => Role::Admin,
        1 => Role::User,
        2 => Role::ReadOnly,
        _ => Role::Service,
    };

    let roles: HashSet<Role> = vec![role].into_iter().collect();

    // Try to create user with fuzzed data
    let user_result = manager.create_user(username.clone(), email, roles);

    match user_result {
        Ok(user) => {
            // Verify user properties are sane
            assert!(!user.id.is_empty());
            assert_eq!(user.username, username);
            assert!(user.enabled);

            // Test get_user
            let fetched = manager.get_user(&user.id);
            assert!(fetched.is_some());
            assert_eq!(fetched.unwrap().id, user.id);

            // Test get_user_by_username
            let fetched_by_name = manager.get_user_by_username(&username);
            assert!(fetched_by_name.is_some());
            assert_eq!(fetched_by_name.unwrap().id, user.id);

            // Test creating API key
            let api_key_result = manager.create_api_key(&user.id, None);
            if let Ok(token) = api_key_result {
                assert!(token.secret.starts_with("ipfrs_"));
                assert_eq!(token.user_id, user.id);

                // Test listing user tokens
                let tokens = manager.list_user_tokens(&user.id);
                assert!(!tokens.is_empty());
            }

            // Test creating JWT token
            let jwt_result = manager.create_jwt_token(&user.id, None);
            if let Ok(jwt) = jwt_result {
                assert!(!jwt.secret.is_empty());
                assert_eq!(jwt.user_id, user.id);
                assert!(jwt.expires_at.is_some());
            }

            // Test user update
            let mut updated_user = user.clone();
            updated_user.enabled = false;
            let update_result = manager.update_user(updated_user);
            assert!(update_result.is_ok());

            // Test user deletion
            let delete_result = manager.delete_user(&user.id);
            assert!(delete_result.is_ok());

            // Verify user is gone
            let fetched_after_delete = manager.get_user(&user.id);
            assert!(fetched_after_delete.is_none());
        }
        Err(_) => {
            // User creation failed, which is expected for some fuzzed inputs
            // (e.g., duplicate usernames, invalid data)
        }
    }

    // Test cleanup_expired_tokens (should not panic)
    let _cleaned = manager.cleanup_expired_tokens();

    // Test list_users (should not panic)
    let _users = manager.list_users();
});
