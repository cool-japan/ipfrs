//! wasm32 compatibility layer for ipfrs-core.
//!
//! This module provides documentation and helpers for wasm32 targets.
//! Most of ipfrs-core is already wasm-compatible since it uses only
//! pure-Rust data structures.

/// Whether the current target is wasm32.
pub const IS_WASM32: bool = cfg!(target_arch = "wasm32");

/// A platform-independent timestamp in milliseconds.
///
/// On wasm32 this returns a stub value (0). For real browser timestamps,
/// use `js_sys::Date::now()` via the `wasm-bindgen` / `js-sys` crates.
/// On native targets this delegates to [`std::time::SystemTime`].
pub struct PlatformTime;

impl PlatformTime {
    /// Returns current time in milliseconds since Unix epoch.
    ///
    /// * On wasm32: returns `0` (stub — production builds should use `js_sys::Date::now()`).
    /// * On native: uses [`std::time::SystemTime`].
    pub fn now_ms() -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            // Stub: real wasm builds should use `js_sys::Date::now() as u64`
            // via the `js` feature of `getrandom` / `js-sys` crate.
            0u64
        }
    }
}

/// Capabilities available on the current compile target.
pub struct TargetCapabilities;

impl TargetCapabilities {
    /// Returns `true` when the filesystem APIs (`std::fs`) are available.
    pub fn has_filesystem() -> bool {
        !IS_WASM32
    }

    /// Returns `true` when OS threads are available.
    pub fn has_threads() -> bool {
        !IS_WASM32
    }

    /// Returns `true` when native network sockets (`std::net`) are available.
    pub fn has_network_sockets() -> bool {
        !IS_WASM32
    }

    /// Returns `true` when running inside a browser (wasm32 target).
    pub fn is_browser() -> bool {
        IS_WASM32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_wasm32_const_false_on_native() {
        // This test only runs on non-wasm32 targets.
        #[cfg(not(target_arch = "wasm32"))]
        const {
            assert!(!IS_WASM32)
        };
    }

    #[test]
    fn test_platform_time_now_ms_nonzero() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let ms = PlatformTime::now_ms();
            assert!(
                ms > 0,
                "expected non-zero timestamp on native target, got {ms}"
            );
        }
    }

    #[test]
    fn test_target_capabilities_native() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(TargetCapabilities::has_filesystem());
            assert!(TargetCapabilities::has_threads());
            assert!(TargetCapabilities::has_network_sockets());
        }
    }

    #[test]
    fn test_is_browser_false_on_native() {
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!TargetCapabilities::is_browser());
    }
}
