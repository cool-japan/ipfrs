//! FFI (Foreign Function Interface) bindings for C interoperability
//!
//! This module provides a C-compatible API for IPFRS, allowing the library
//! to be used from C, C++, and other languages that support C FFI.
//!
//! # Safety
//!
//! All functions are marked as `unsafe extern "C"` and handle panics to prevent
//! undefined behavior. Proper null checks are performed on all pointer arguments.
//!
//! # Memory Management
//!
//! - Opaque pointers are used to hide Rust types from C
//! - Callers must free resources using the provided `*_free` functions
//! - Strings passed from C must be valid UTF-8 null-terminated strings
//! - Strings returned to C must be freed using `ipfrs_string_free`

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

/// FFI error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpfrsErrorCode {
    /// Operation succeeded
    Success = 0,
    /// Null pointer was passed
    NullPointer = -1,
    /// Invalid UTF-8 string
    InvalidUtf8 = -2,
    /// Invalid CID format
    InvalidCid = -3,
    /// Block not found
    NotFound = -4,
    /// I/O error
    IoError = -5,
    /// Out of memory
    OutOfMemory = -6,
    /// Internal error (panic caught)
    InternalError = -7,
    /// Invalid argument
    InvalidArgument = -8,
    /// Operation timed out
    Timeout = -9,
    /// Unknown error
    Unknown = -99,
}

/// Opaque handle to IPFRS client
#[repr(C)]
pub struct IpfrsClient {
    _private: [u8; 0],
}

/// Opaque handle to a block
#[repr(C)]
pub struct IpfrsBlock {
    _private: [u8; 0],
}

/// Internal representation of IPFRS client
struct ClientInner {
    // In a real implementation, this would contain:
    // - Gateway configuration
    // - Blockstore handle
    // - Tokio runtime handle
    // For now, we'll keep it simple
    _placeholder: u8,
}

/// Internal representation of a block
#[allow(dead_code)]
struct BlockInner {
    cid: String,
    data: Vec<u8>,
}

// Thread-local for storing last error message
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Set the last error message
fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(msg);
    });
}

/// Clear the last error message
fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Initialize a new IPFRS client
///
/// # Arguments
///
/// * `config_path` - Path to configuration file (optional, can be NULL)
///
/// # Returns
///
/// Pointer to IpfrsClient on success, NULL on failure.
/// Use `ipfrs_get_last_error()` to retrieve error message.
///
/// # Safety
///
/// - `config_path` must be NULL or a valid null-terminated UTF-8 string
/// - Returned pointer must be freed with `ipfrs_client_free()`
#[no_mangle]
pub unsafe extern "C" fn ipfrs_client_new(config_path: *const c_char) -> *mut IpfrsClient {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Parse config path if provided
        let _config = if !config_path.is_null() {
            let c_str = unsafe { CStr::from_ptr(config_path) };
            match c_str.to_str() {
                Ok(s) => Some(s.to_string()),
                Err(_) => {
                    set_last_error("Invalid UTF-8 in config_path".to_string());
                    return ptr::null_mut();
                }
            }
        } else {
            None
        };

        // Create client inner
        let inner = Box::new(ClientInner { _placeholder: 0 });

        Box::into_raw(inner) as *mut IpfrsClient
    }));

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("Panic occurred in ipfrs_client_new".to_string());
            ptr::null_mut()
        }
    }
}

/// Free an IPFRS client
///
/// # Safety
///
/// - `client` must be a valid pointer returned from `ipfrs_client_new()`
/// - `client` must not be used after this call
/// - `client` must not be NULL
#[no_mangle]
pub unsafe extern "C" fn ipfrs_client_free(client: *mut IpfrsClient) {
    if client.is_null() {
        return;
    }

    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let _ = Box::from_raw(client as *mut ClientInner);
    }));
}

/// Add data to IPFRS and return its CID
///
/// # Arguments
///
/// * `client` - Pointer to IpfrsClient
/// * `data` - Pointer to data buffer
/// * `data_len` - Length of data in bytes
/// * `out_cid` - Output pointer to receive CID string (must be freed with ipfrs_string_free)
///
/// # Returns
///
/// Error code (0 for success)
///
/// # Safety
///
/// - `client` must be a valid pointer from `ipfrs_client_new()`
/// - `data` must point to at least `data_len` bytes
/// - `out_cid` must be a valid pointer to a char pointer
#[no_mangle]
pub unsafe extern "C" fn ipfrs_add(
    client: *mut IpfrsClient,
    data: *const u8,
    data_len: usize,
    out_cid: *mut *mut c_char,
) -> c_int {
    clear_last_error();

    // Null pointer checks
    if client.is_null() {
        set_last_error("client is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if data.is_null() {
        set_last_error("data is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if out_cid.is_null() {
        set_last_error("out_cid is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _inner = &*(client as *mut ClientInner);
        let data_slice = unsafe { slice::from_raw_parts(data, data_len) };

        // In a real implementation, this would:
        // 1. Chunk the data
        // 2. Create blocks
        // 3. Store them in the blockstore
        // 4. Return the root CID

        // For now, create a mock CID based on data length
        let mock_cid = format!("bafkreidummy{:016x}", data_slice.len());

        // Convert to C string
        match CString::new(mock_cid) {
            Ok(c_string) => {
                unsafe {
                    *out_cid = c_string.into_raw();
                }
                IpfrsErrorCode::Success as c_int
            }
            Err(_) => {
                set_last_error("Failed to create CID string".to_string());
                IpfrsErrorCode::InternalError as c_int
            }
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("Panic occurred in ipfrs_add".to_string());
            IpfrsErrorCode::InternalError as c_int
        }
    }
}

/// Get data from IPFRS by CID
///
/// # Arguments
///
/// * `client` - Pointer to IpfrsClient
/// * `cid` - Null-terminated CID string
/// * `out_data` - Output pointer to receive data buffer (must be freed with ipfrs_data_free)
/// * `out_len` - Output pointer to receive data length
///
/// # Returns
///
/// Error code (0 for success)
///
/// # Safety
///
/// - `client` must be a valid pointer from `ipfrs_client_new()`
/// - `cid` must be a valid null-terminated UTF-8 string
/// - `out_data` must be a valid pointer
/// - `out_len` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn ipfrs_get(
    client: *mut IpfrsClient,
    cid: *const c_char,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    clear_last_error();

    // Null pointer checks
    if client.is_null() {
        set_last_error("client is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if cid.is_null() {
        set_last_error("cid is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if out_data.is_null() {
        set_last_error("out_data is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if out_len.is_null() {
        set_last_error("out_len is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _inner = &*(client as *mut ClientInner);

        // Parse CID
        let c_str = unsafe { CStr::from_ptr(cid) };
        let cid_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid UTF-8 in CID".to_string());
                return IpfrsErrorCode::InvalidUtf8 as c_int;
            }
        };

        // In a real implementation, this would:
        // 1. Look up the CID in the blockstore
        // 2. Retrieve and reconstruct the data
        // 3. Return it to the caller

        // For now, return mock data
        let mock_data = format!("Data for CID: {}", cid_str).into_bytes();
        let len = mock_data.len();

        // Allocate buffer and copy data
        let mut boxed_data = mock_data.into_boxed_slice();
        let data_ptr = boxed_data.as_mut_ptr();
        std::mem::forget(boxed_data); // Prevent deallocation

        unsafe {
            *out_data = data_ptr;
            *out_len = len;
        }

        IpfrsErrorCode::Success as c_int
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("Panic occurred in ipfrs_get".to_string());
            IpfrsErrorCode::InternalError as c_int
        }
    }
}

/// Check if a block exists by CID
///
/// # Arguments
///
/// * `client` - Pointer to IpfrsClient
/// * `cid` - Null-terminated CID string
/// * `out_exists` - Output pointer to receive existence flag (1 = exists, 0 = not found)
///
/// # Returns
///
/// Error code (0 for success)
///
/// # Safety
///
/// - `client` must be a valid pointer from `ipfrs_client_new()`
/// - `cid` must be a valid null-terminated UTF-8 string
/// - `out_exists` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn ipfrs_has(
    client: *mut IpfrsClient,
    cid: *const c_char,
    out_exists: *mut c_int,
) -> c_int {
    clear_last_error();

    // Null pointer checks
    if client.is_null() {
        set_last_error("client is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if cid.is_null() {
        set_last_error("cid is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }
    if out_exists.is_null() {
        set_last_error("out_exists is NULL".to_string());
        return IpfrsErrorCode::NullPointer as c_int;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _inner = &*(client as *mut ClientInner);

        // Parse CID
        let c_str = unsafe { CStr::from_ptr(cid) };
        let _cid_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid UTF-8 in CID".to_string());
                return IpfrsErrorCode::InvalidUtf8 as c_int;
            }
        };

        // In a real implementation, check blockstore
        // For now, always return true
        unsafe {
            *out_exists = 1;
        }

        IpfrsErrorCode::Success as c_int
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("Panic occurred in ipfrs_has".to_string());
            IpfrsErrorCode::InternalError as c_int
        }
    }
}

/// Get the last error message
///
/// # Returns
///
/// Pointer to null-terminated error string, or NULL if no error.
/// The string is valid until the next FFI call on this thread.
/// DO NOT free this pointer.
#[no_mangle]
pub extern "C" fn ipfrs_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map_or(ptr::null(), |s| s.as_ptr() as *const c_char)
    })
}

/// Free a string returned by IPFRS functions
///
/// # Safety
///
/// - `s` must be a pointer returned by an IPFRS function (e.g., from ipfrs_add)
/// - `s` must not be used after this call
/// - `s` can be NULL (no-op)
#[no_mangle]
pub unsafe extern "C" fn ipfrs_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }

    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let _ = CString::from_raw(s);
    }));
}

/// Free data returned by ipfrs_get
///
/// # Safety
///
/// - `data` must be a pointer returned by `ipfrs_get()`
/// - `len` must be the length returned by `ipfrs_get()`
/// - `data` must not be used after this call
/// - `data` can be NULL (no-op)
#[no_mangle]
pub unsafe extern "C" fn ipfrs_data_free(data: *mut u8, len: usize) {
    if data.is_null() {
        return;
    }

    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let _ = Vec::from_raw_parts(data, len, len);
    }));
}

/// Get library version string
///
/// # Returns
///
/// Pointer to static version string. DO NOT free this pointer.
#[no_mangle]
pub extern "C" fn ipfrs_version() -> *const c_char {
    // Use a static string to avoid allocation
    static VERSION: &[u8] = b"ipfrs-interface 0.2.0\0";
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_lifecycle() {
        unsafe {
            let client = ipfrs_client_new(ptr::null());
            assert!(!client.is_null());
            ipfrs_client_free(client);
        }
    }

    #[test]
    fn test_add_and_get() {
        unsafe {
            let client = ipfrs_client_new(ptr::null());
            assert!(!client.is_null());

            // Add data
            let data = b"Hello, IPFRS!";
            let mut cid_ptr: *mut c_char = ptr::null_mut();
            let result = ipfrs_add(client, data.as_ptr(), data.len(), &mut cid_ptr);
            assert_eq!(result, IpfrsErrorCode::Success as c_int);
            assert!(!cid_ptr.is_null());

            // Get data back
            let mut out_data: *mut u8 = ptr::null_mut();
            let mut out_len: usize = 0;
            let result = ipfrs_get(client, cid_ptr, &mut out_data, &mut out_len);
            assert_eq!(result, IpfrsErrorCode::Success as c_int);
            assert!(!out_data.is_null());
            assert!(out_len > 0);

            // Clean up
            ipfrs_string_free(cid_ptr);
            ipfrs_data_free(out_data, out_len);
            ipfrs_client_free(client);
        }
    }

    #[test]
    fn test_has_block() {
        unsafe {
            let client = ipfrs_client_new(ptr::null());
            assert!(!client.is_null());

            let cid = CString::new("bafytest123")
                .expect("test: CString creation from valid string should succeed");
            let mut exists: c_int = 0;
            let result = ipfrs_has(client, cid.as_ptr(), &mut exists);
            assert_eq!(result, IpfrsErrorCode::Success as c_int);

            ipfrs_client_free(client);
        }
    }

    #[test]
    fn test_null_pointer_handling() {
        unsafe {
            // Test with null client
            let mut cid_ptr: *mut c_char = ptr::null_mut();
            let data = b"test";
            let result = ipfrs_add(ptr::null_mut(), data.as_ptr(), data.len(), &mut cid_ptr);
            assert_eq!(result, IpfrsErrorCode::NullPointer as c_int);
        }
    }

    #[test]
    fn test_version() {
        let version = ipfrs_version();
        assert!(!version.is_null());
        unsafe {
            let c_str = CStr::from_ptr(version);
            let version_str = c_str
                .to_str()
                .expect("test: version string should be valid UTF-8");
            assert!(version_str.contains("ipfrs-interface"));
        }
    }
}
