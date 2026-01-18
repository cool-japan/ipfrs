//! Fuzz test for AuthToken verification
//!
//! Tests robustness of token verification against malformed inputs

#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs::{AuthManager, Role};
use std::collections::HashSet;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to string (may contain invalid UTF-8)
    let maybe_token = String::from_utf8_lossy(data);

    // Create auth manager
    let manager = AuthManager::new("fuzz_secret_key".to_string());

    // Try to verify the fuzzed token
    let result = manager.verify_token(&maybe_token);

    // Should either succeed or fail gracefully (no panic)
    match result {
        Ok(token) => {
            // If verification succeeds, token should have valid properties
            assert!(!token.id.is_empty());
            assert!(!token.user_id.is_empty());
            assert!(!token.secret.is_empty());

            // Check expiration logic works
            let _ = token.is_expired();

            // Check permission logic works
            let perms = token.effective_permissions();
            for role in &token.roles {
                let role_perms = role.default_permissions();
                if token.permissions.is_none() {
                    for perm in &role_perms {
                        assert!(perms.contains(perm));
                    }
                }
            }
        }
        Err(_) => {
            // Verification failed, which is expected for most fuzzed inputs
        }
    }

    // Test token verification with known good tokens
    if data.len() > 10 {
        // Create a test user and token
        let user_result = manager.create_user(
            format!("user_{}", data[0]),
            None,
            vec![Role::User].into_iter().collect(),
        );

        if let Ok(user) = user_result {
            // Create API key
            if let Ok(api_key) = manager.create_api_key(&user.id, None) {
                // Verify it should work
                let verify_result = manager.verify_token(&api_key.secret);
                assert!(verify_result.is_ok());

                // Revoke and verify it fails
                let _ = manager.revoke_token(&api_key.id);
                let verify_after_revoke = manager.verify_token(&api_key.secret);
                assert!(verify_after_revoke.is_err());
            }
        }
    }
});
