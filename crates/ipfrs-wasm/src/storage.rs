//! IndexedDB-compatible in-memory block store for wasm32.
//!
//! # Overview
//!
//! [`WasmBlockStore`] is a `#[wasm_bindgen]`-exposed wrapper suitable for use
//! in browser JavaScript.  In a production deployment the implementation would
//! delegate to `idb-keyval` (or a similar JS shim) to achieve real persistence;
//! for now it keeps all blocks in a Rust `HashMap` so that the async interface
//! can be tested in plain native Rust unit tests without a browser runtime.
//!
//! [`InMemoryBlockStore`] is the non-`wasm_bindgen` inner type.  It exposes the
//! full surface area of the store — including `export_entries` / `import_entries`
//! for snapshot/restore — and is the type under test.
//!
//! # Design notes
//!
//! - Keys are CID strings (arbitrary UTF-8; the store is content-agnostic).
//! - Values are raw byte payloads (`Vec<u8>`).
//! - Inserting the same CID twice overwrites the previous value (last-write-wins),
//!   which matches IndexedDB's `put` semantics.
//! - `total_bytes` tracks the sum of all currently stored payloads; it is kept
//!   consistent on every `put`, `delete`, and `clear`.

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// InMemoryBlockStore – pure-Rust inner type, testable without a browser
// ---------------------------------------------------------------------------

/// In-memory content-addressed block store.
///
/// Keys are CID strings; values are raw byte payloads.  This type is the
/// implementation nucleus shared by [`WasmBlockStore`] and is also used
/// directly in unit tests (no `wasm_bindgen` overhead).
pub struct InMemoryBlockStore {
    blocks: HashMap<String, Vec<u8>>,
    /// Cached sum of `blocks.values().map(|v| v.len())`.
    total_bytes_cached: usize,
}

impl InMemoryBlockStore {
    /// Create a new, empty block store.
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            total_bytes_cached: 0,
        }
    }

    /// Insert or overwrite the block identified by `cid`.
    ///
    /// If a block with the same CID already exists, it is replaced and the
    /// byte-count cache is adjusted accordingly (last-write-wins, matching
    /// IndexedDB `put` semantics).
    pub fn put(&mut self, cid: impl Into<String>, data: Vec<u8>) {
        let key = cid.into();
        let new_len = data.len();
        if let Some(old) = self.blocks.insert(key, data) {
            // Adjust total: remove old contribution, add new.
            self.total_bytes_cached = self
                .total_bytes_cached
                .saturating_sub(old.len())
                .saturating_add(new_len);
        } else {
            self.total_bytes_cached = self.total_bytes_cached.saturating_add(new_len);
        }
    }

    /// Return a reference to the raw bytes stored under `cid`, or `None`.
    pub fn get(&self, cid: &str) -> Option<&[u8]> {
        self.blocks.get(cid).map(|v| v.as_slice())
    }

    /// Return `true` if `cid` is present in the store.
    pub fn has(&self, cid: &str) -> bool {
        self.blocks.contains_key(cid)
    }

    /// Remove the block identified by `cid`.
    ///
    /// Returns `true` if the block existed and was removed, `false` otherwise.
    pub fn delete(&mut self, cid: &str) -> bool {
        if let Some(removed) = self.blocks.remove(cid) {
            self.total_bytes_cached = self.total_bytes_cached.saturating_sub(removed.len());
            true
        } else {
            false
        }
    }

    /// Return the number of blocks currently in the store.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Return `true` if the store contains no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Return all CID keys in sorted order.
    ///
    /// Sorting makes the output deterministic and simplifies tests.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.blocks.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Return the total size (in bytes) of all stored payloads.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes_cached
    }

    /// Remove all blocks, resetting the byte counter to zero.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.total_bytes_cached = 0;
    }

    /// Export all entries as a `Vec` of `(cid, data)` pairs.
    ///
    /// The pairs are returned in sorted CID order for deterministic output.
    /// This method is useful for snapshotting the store contents.
    pub fn export_entries(&self) -> Vec<(String, Vec<u8>)> {
        let mut entries: Vec<(String, Vec<u8>)> = self
            .blocks
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Import entries from a `Vec` of `(cid, data)` pairs.
    ///
    /// Any CID that already exists in the store is overwritten (last-write-wins).
    /// This method is the inverse of `export_entries`.
    pub fn import_entries(&mut self, entries: Vec<(String, Vec<u8>)>) {
        for (cid, data) in entries {
            self.put(cid, data);
        }
    }
}

impl Default for InMemoryBlockStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WasmBlockStore – wasm_bindgen-exposed wrapper
// ---------------------------------------------------------------------------

/// In-memory block store that mirrors the IndexedDB async interface.
///
/// In production browser builds this would delegate to `idb-keyval` JS glue
/// for real persistence.  The current implementation is fully synchronous and
/// keeps all blocks in Rust heap memory — data is lost when the JS object is
/// garbage-collected or the page is refreshed.
///
/// # JavaScript example
/// ```javascript
/// const store = new WasmBlockStore();
/// store.put("bafy…cid", new TextEncoder().encode("hello"));
/// const bytes = store.get("bafy…cid");   // Uint8Array | undefined
/// console.log(store.len(), store.total_bytes());
/// ```
#[wasm_bindgen]
pub struct WasmBlockStore {
    inner: InMemoryBlockStore,
}

#[wasm_bindgen]
impl WasmBlockStore {
    /// Create a new, empty `WasmBlockStore`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: InMemoryBlockStore::new(),
        }
    }

    /// Insert or overwrite the block identified by `cid` with `data`.
    ///
    /// If a block with the same CID already exists it is replaced
    /// (last-write-wins, matching IndexedDB `put` semantics).
    pub fn put(&mut self, cid: &str, data: &[u8]) {
        self.inner.put(cid, data.to_vec());
    }

    /// Retrieve the raw bytes stored under `cid`.
    ///
    /// Returns `undefined` in JavaScript when the CID is absent.
    pub fn get(&self, cid: &str) -> Option<Vec<u8>> {
        self.inner.get(cid).map(|s| s.to_vec())
    }

    /// Return `true` if `cid` is present in the store.
    pub fn has(&self, cid: &str) -> bool {
        self.inner.has(cid)
    }

    /// Remove the block identified by `cid`.
    ///
    /// Returns `true` if the block existed and was removed, `false` otherwise.
    pub fn delete(&mut self, cid: &str) -> bool {
        self.inner.delete(cid)
    }

    /// Return the number of blocks currently in the store.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if the store contains no blocks.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return all stored CID strings in sorted order.
    pub fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    /// Return the total size (in bytes) of all stored payloads.
    pub fn total_bytes(&self) -> usize {
        self.inner.total_bytes()
    }

    /// Remove all blocks from the store.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for WasmBlockStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::InMemoryBlockStore;

    fn store_with_entries() -> (InMemoryBlockStore, String, String) {
        let mut store = InMemoryBlockStore::new();
        let cid_a = "bafkreia".to_string();
        let cid_b = "bafkreib".to_string();
        store.put(cid_a.clone(), b"hello".to_vec());
        store.put(cid_b.clone(), b"world!".to_vec());
        (store, cid_a, cid_b)
    }

    // ------------------------------------------------------------------
    // put / get roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn test_put_and_get() {
        let mut store = InMemoryBlockStore::new();
        let cid = "bafkreitest1".to_string();
        let data = b"roundtrip data".to_vec();
        store.put(cid.clone(), data.clone());
        let retrieved = store.get(&cid).expect("block must be present");
        assert_eq!(retrieved, data.as_slice());
    }

    // ------------------------------------------------------------------
    // has
    // ------------------------------------------------------------------

    #[test]
    fn test_has() {
        let mut store = InMemoryBlockStore::new();
        let cid = "bafkreihas1".to_string();
        assert!(!store.has(&cid), "store must be empty initially");
        store.put(cid.clone(), b"data".to_vec());
        assert!(store.has(&cid), "block must be found after put");
        assert!(
            !store.has("bafkreinonexistent"),
            "absent CID must return false"
        );
    }

    // ------------------------------------------------------------------
    // delete
    // ------------------------------------------------------------------

    #[test]
    fn test_delete() {
        let mut store = InMemoryBlockStore::new();
        let cid = "bafkreidel1".to_string();
        store.put(cid.clone(), b"to delete".to_vec());
        assert!(
            store.delete(&cid),
            "delete must return true for existing key"
        );
        assert!(!store.has(&cid), "block must be gone after delete");
        assert!(
            !store.delete(&cid),
            "delete of absent key must return false"
        );
        assert!(
            !store.delete("bafkreinonexistent"),
            "delete of never-inserted key must return false"
        );
    }

    // ------------------------------------------------------------------
    // len / is_empty
    // ------------------------------------------------------------------

    #[test]
    fn test_len_and_is_empty() {
        let mut store = InMemoryBlockStore::new();
        assert!(store.is_empty(), "new store must be empty");
        assert_eq!(store.len(), 0);

        store.put("bafkreic1".to_string(), b"a".to_vec());
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);

        store.put("bafkreic2".to_string(), b"b".to_vec());
        assert_eq!(store.len(), 2);

        store.delete("bafkreic1");
        assert_eq!(store.len(), 1);

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    // ------------------------------------------------------------------
    // keys (sorted)
    // ------------------------------------------------------------------

    #[test]
    fn test_keys() {
        let (store, cid_a, cid_b) = store_with_entries();
        let keys = store.keys();
        // keys() returns sorted output
        assert_eq!(keys.len(), 2);
        // Sort expected to match
        let mut expected = vec![cid_a, cid_b];
        expected.sort();
        assert_eq!(keys, expected, "keys() must return sorted CID strings");
    }

    // ------------------------------------------------------------------
    // total_bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_total_bytes() {
        let mut store = InMemoryBlockStore::new();
        assert_eq!(store.total_bytes(), 0, "empty store has 0 bytes");

        store.put("cid1".to_string(), b"abc".to_vec()); // 3
        assert_eq!(store.total_bytes(), 3);

        store.put("cid2".to_string(), b"defgh".to_vec()); // 5
        assert_eq!(store.total_bytes(), 8);

        store.delete("cid1");
        assert_eq!(store.total_bytes(), 5);
    }

    // ------------------------------------------------------------------
    // clear
    // ------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let (mut store, _, _) = store_with_entries();
        assert!(!store.is_empty(), "store must have entries before clear");
        assert!(store.total_bytes() > 0);
        store.clear();
        assert!(store.is_empty(), "store must be empty after clear");
        assert_eq!(store.total_bytes(), 0, "total_bytes must be 0 after clear");
        assert_eq!(store.len(), 0);
    }

    // ------------------------------------------------------------------
    // export / import roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn test_export_import_roundtrip() {
        let (original, cid_a, cid_b) = store_with_entries();
        let exported = original.export_entries();

        // Exported entries are sorted by CID.
        let mut expected_cids: Vec<&str> = vec![cid_a.as_str(), cid_b.as_str()];
        expected_cids.sort();
        let exported_cids: Vec<&str> = exported.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(exported_cids, expected_cids);

        let mut restored = InMemoryBlockStore::new();
        restored.import_entries(exported);

        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.total_bytes(), original.total_bytes());
        assert_eq!(
            restored.get(&cid_a).map(|s| s.to_vec()),
            original.get(&cid_a).map(|s| s.to_vec())
        );
        assert_eq!(
            restored.get(&cid_b).map(|s| s.to_vec()),
            original.get(&cid_b).map(|s| s.to_vec())
        );
    }

    // ------------------------------------------------------------------
    // overwrite existing (last-write-wins)
    // ------------------------------------------------------------------

    #[test]
    fn test_overwrite_existing() {
        let mut store = InMemoryBlockStore::new();
        let cid = "bafkreioverwrite".to_string();

        store.put(cid.clone(), b"first value".to_vec());
        assert_eq!(store.total_bytes(), 11);

        store.put(cid.clone(), b"second".to_vec()); // 6 bytes
                                                    // last-write-wins: only 6 bytes now
        assert_eq!(store.len(), 1, "must not create a duplicate entry");
        assert_eq!(
            store.total_bytes(),
            6,
            "total_bytes must reflect the overwritten value"
        );
        assert_eq!(
            store.get(&cid).map(|s| s.to_vec()),
            Some(b"second".to_vec()),
            "get must return the most recently written value"
        );
    }
}
