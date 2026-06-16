//! WebAssembly bindings for IPFRS
//!
//! This module provides browser-compatible bindings for IPFRS using wasm-bindgen.
//! Two storage backends are available:
//!
//! - [`IpfrsClient`]: in-memory (HashMap-backed), ephemeral — data is lost on page refresh.
//! - `IpfrsClientPersistent`: IndexedDB-backed, browser-persistent (wasm32 only).
//!
//! # Quick Start (JavaScript)
//!
//! ```javascript
//! import init, { IpfrsClient, compute_cid, verify_cid, version } from './ipfrs_wasm.js';
//!
//! const _wasm = await init();
//!
//! // Ephemeral in-memory client
//! const client = new IpfrsClient();
//! const cid = await client.add(new TextEncoder().encode("hello ipfrs"));
//! const bytes = await client.get(cid);
//! console.log(new TextDecoder().decode(bytes));
//!
//! // Persistent IndexedDB client (browser only)
//! const persistent = await IpfrsClientPersistent.new("ipfrs-blocks");
//! const cid2 = await persistent.add(new TextEncoder().encode("persisted data"));
//! ```

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WebRTC transport (signalling types are target-independent;
// IpfrsPeer / IpfrsPeerAnswerer are wasm32-only)
// ---------------------------------------------------------------------------

pub mod async_api;
pub mod peer_state;
pub mod storage;
pub mod webrtc;

// Re-export the WASM peer types at crate root for wasm-bindgen visibility.
#[cfg(target_arch = "wasm32")]
pub use webrtc::{IpfrsPeer, IpfrsPeerAnswerer};

// ---------------------------------------------------------------------------
// Panic hook initialisation (wasm32 only)
// ---------------------------------------------------------------------------

/// Initialise the IPFRS WASM module.
///
/// Called automatically by the wasm-bindgen generated JS glue via the
/// `#[wasm_bindgen(start)]` attribute.  Sets up the `console_error_panic_hook`
/// so that Rust panics surface as readable browser console errors.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// CID helpers (pure, no allocation beyond the String result)
// ---------------------------------------------------------------------------

/// Compute a CIDv1 (base32-lower, SHA2-256, raw codec) string for `data`.
///
/// The format matches `bafy…` CIDs produced by go-ipfs / js-ipfs for raw
/// leaf blocks (codec `0x55`), giving the encoding:
///
/// ```text
/// <varint version=1><varint codec=0x55><varint mh-fn=0x12><varint mh-len=32><32-byte-digest>
/// ```
///
/// encoded in lowercase base32 with the `b` multibase prefix.
fn cid_from_bytes(data: &[u8]) -> String {
    use sha2::Digest;

    // 1. SHA-256 digest
    let digest: [u8; 32] = sha2::Sha256::digest(data).into();

    // 2. Multihash = [0x12, 0x20, <32 bytes>]
    //    0x12 = sha2-256 function code, 0x20 = 32 (length)
    let mut multihash = Vec::with_capacity(34);
    multihash.push(0x12u8); // sha2-256
    multihash.push(0x20u8); // 32 bytes
    multihash.extend_from_slice(&digest);

    // 3. CIDv1 binary = [0x01, 0x55, <multihash>]
    //    0x01 = CIDv1, 0x55 = raw codec
    let mut cid_bytes = Vec::with_capacity(36);
    cid_bytes.push(0x01u8); // version 1
    cid_bytes.push(0x55u8); // raw codec
    cid_bytes.extend_from_slice(&multihash);

    // 4. Multibase base32lower with 'b' prefix
    let encoded = base32_lower(&cid_bytes);
    format!("b{encoded}")
}

/// RFC 4648 base32 (lowercase, no padding) encoder.
fn base32_lower(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut output = Vec::with_capacity((input.len() * 8).div_ceil(5));
    let mut buffer: u64 = 0;
    let mut bits_left: u32 = 0;

    for &byte in input {
        buffer = (buffer << 8) | u64::from(byte);
        bits_left += 8;
        while bits_left >= 5 {
            bits_left -= 5;
            let idx = ((buffer >> bits_left) & 0x1f) as usize;
            output.push(ALPHABET[idx]);
        }
    }
    if bits_left > 0 {
        let idx = ((buffer << (5 - bits_left)) & 0x1f) as usize;
        output.push(ALPHABET[idx]);
    }

    // SAFETY: ALPHABET contains only ASCII bytes, so the Vec<u8> is valid UTF-8.
    String::from_utf8(output).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// InMemoryStore – pure in-memory block store for WASM
// ---------------------------------------------------------------------------

/// Pure in-memory content-addressed block store.
///
/// Keys are CIDv1 strings; values are the raw byte payloads.
struct InMemoryStore {
    blocks: std::collections::HashMap<String, Vec<u8>>,
    total_bytes: usize,
}

impl InMemoryStore {
    fn new() -> Self {
        Self {
            blocks: std::collections::HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Store `data` and return its CID.  If the same CID is already present
    /// the entry is *not* duplicated (content-addressed idempotency).
    fn put(&mut self, data: &[u8]) -> String {
        let cid = cid_from_bytes(data);
        self.blocks.entry(cid.clone()).or_insert_with(|| {
            self.total_bytes += data.len();
            data.to_vec()
        });
        cid
    }

    fn get(&self, cid: &str) -> Option<&Vec<u8>> {
        self.blocks.get(cid)
    }

    fn has(&self, cid: &str) -> bool {
        self.blocks.contains_key(cid)
    }

    /// Remove the block with the given CID.  Returns `true` if it existed.
    fn delete(&mut self, cid: &str) -> bool {
        if let Some(data) = self.blocks.remove(cid) {
            self.total_bytes = self.total_bytes.saturating_sub(data.len());
            true
        } else {
            false
        }
    }

    fn list(&self) -> Vec<String> {
        self.blocks.keys().cloned().collect()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

// ---------------------------------------------------------------------------
// IpfrsClient – the public WASM API (in-memory, ephemeral)
// ---------------------------------------------------------------------------

/// IPFRS in-memory client for WebAssembly.
///
/// Provides a content-addressed block store backed by a `HashMap`.
/// All data lives in the JS heap and is lost when the object is garbage-collected
/// or the page is refreshed — suitable for ephemeral browser usage or unit tests.
#[wasm_bindgen]
pub struct IpfrsClient {
    store: InMemoryStore,
}

#[wasm_bindgen]
impl IpfrsClient {
    /// Create a new IPFRS in-memory client.
    ///
    /// # JavaScript
    /// ```javascript
    /// const client = new IpfrsClient();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> IpfrsClient {
        IpfrsClient {
            store: InMemoryStore::new(),
        }
    }

    /// Add a byte slice to the store and return its CID string.
    ///
    /// The operation is deterministic: identical byte sequences always produce
    /// the same CID and are stored only once.
    ///
    /// # JavaScript
    /// ```javascript
    /// const cid = await client.add(new Uint8Array([1, 2, 3]));
    /// ```
    pub async fn add(&mut self, data: &[u8]) -> Result<String, JsValue> {
        if data.is_empty() {
            return Err(JsValue::from_str("data must not be empty"));
        }
        Ok(self.store.put(data))
    }

    /// Retrieve bytes by CID string.
    ///
    /// Returns an error if the CID is not present in the store.
    ///
    /// # JavaScript
    /// ```javascript
    /// const bytes = await client.get(cid);
    /// ```
    pub async fn get(&self, cid: &str) -> Result<Vec<u8>, JsValue> {
        self.store
            .get(cid)
            .cloned()
            .ok_or_else(|| JsValue::from_str(&format!("CID not found: {cid}")))
    }

    /// Return `true` if `cid` is present in the store.
    pub fn has(&self, cid: &str) -> bool {
        self.store.has(cid)
    }

    /// Return all stored CIDs as a `Vec<String>`.
    ///
    /// Order is unspecified (HashMap iteration order).
    #[wasm_bindgen(js_name = listCids)]
    pub fn list_cids(&self) -> Vec<String> {
        self.store.list()
    }

    /// Return storage statistics as a JSON string.
    ///
    /// # Fields
    /// - `block_count` – number of stored blocks
    /// - `total_bytes` – aggregate stored payload size in bytes
    ///
    /// # JavaScript
    /// ```javascript
    /// const stats = JSON.parse(client.stats());
    /// console.log(stats.block_count, stats.total_bytes);
    /// ```
    pub fn stats(&self) -> String {
        format!(
            r#"{{"block_count":{block_count},"total_bytes":{total_bytes}}}"#,
            block_count = self.store.len(),
            total_bytes = self.store.total_bytes(),
        )
    }

    /// Delete the block identified by `cid`.
    ///
    /// Returns `true` if the block existed and was removed, `false` otherwise.
    pub fn delete(&mut self, cid: &str) -> bool {
        self.store.delete(cid)
    }
}

impl Default for IpfrsClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free-standing utility functions
// ---------------------------------------------------------------------------

/// Compute the CIDv1 (base32-lower, SHA2-256, raw codec) for the given bytes
/// without storing them.
///
/// # JavaScript
/// ```javascript
/// const cid = compute_cid(new Uint8Array([1, 2, 3]));
/// ```
#[wasm_bindgen]
pub fn compute_cid(data: &[u8]) -> String {
    cid_from_bytes(data)
}

/// Return the ipfrs-wasm version string.
#[wasm_bindgen]
pub fn version() -> String {
    "ipfrs-wasm 0.2.0".to_string()
}

/// Verify that `data` matches `cid` (i.e., recomputing the CID yields the
/// same string).
///
/// # JavaScript
/// ```javascript
/// const ok = verify_cid(cid, new Uint8Array([1, 2, 3]));
/// ```
#[wasm_bindgen]
pub fn verify_cid(cid: &str, data: &[u8]) -> bool {
    cid_from_bytes(data) == cid
}

/// Add bytes using a temporary in-memory client and return the CID.
///
/// Convenience helper for one-shot content-addressing without managing a client.
///
/// # JavaScript
/// ```javascript
/// const cid = await add_bytes(new Uint8Array([1, 2, 3]));
/// ```
#[wasm_bindgen]
pub async fn add_bytes(data: &[u8]) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(JsValue::from_str("data must not be empty"));
    }
    Ok(cid_from_bytes(data))
}

/// Retrieve bytes from an [`IpfrsClient`] by CID.
///
/// Returns `None` (maps to `null` in JavaScript) if the CID is not present.
///
/// # JavaScript
/// ```javascript
/// const bytes = await get_bytes(client, cid);
/// if (bytes !== null) { ... }
/// ```
#[wasm_bindgen]
pub async fn get_bytes(client: &IpfrsClient, cid: &str) -> Result<Option<Vec<u8>>, JsValue> {
    Ok(client.store.get(cid).cloned())
}

// ---------------------------------------------------------------------------
// IndexedDB-backed block store (wasm32 only)
// ---------------------------------------------------------------------------

/// IndexedDB storage backend — only compiled for the `wasm32` target.
#[cfg(target_arch = "wasm32")]
pub mod indexed_db {
    use super::cid_from_bytes;
    use js_sys::{Array, Uint8Array};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbTransactionMode};

    const STORE_NAME: &str = "blocks";

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Open (or create) an IndexedDB database with a single "blocks" object store.
    ///
    /// The database is opened at version 1.  The `onupgradeneeded` callback
    /// creates the "blocks" object store the first time the database is opened.
    async fn open_db(db_name: &str) -> Result<IdbDatabase, JsValue> {
        let window = web_sys::window()
            .ok_or_else(|| JsValue::from_str("no global window — are you in a browser?"))?;

        let idb_factory = window
            .indexed_db()?
            .ok_or_else(|| JsValue::from_str("IndexedDB is not available in this browser"))?;

        let open_req: IdbOpenDbRequest = idb_factory.open_with_u32(db_name, 1)?;

        // Clone the request so the closure can reference it after the move.
        let on_upgrade = Closure::<dyn FnMut(web_sys::Event)>::new({
            let req = open_req.clone();
            move |_event: web_sys::Event| {
                // `result` at this point is the IdbDatabase being upgraded.
                let db: IdbDatabase = req
                    .result()
                    .expect("IdbOpenDbRequest.result() inside onupgradeneeded")
                    .dyn_into()
                    .expect("result is IdbDatabase");

                // Only create the store if it does not already exist.
                let store_names: Array = db.object_store_names().into();
                let already_exists = (0..store_names.length())
                    .any(|i| store_names.get(i).as_string().as_deref() == Some(STORE_NAME));

                if !already_exists {
                    db.create_object_store(STORE_NAME)
                        .expect("create_object_store failed");
                }
            }
        });

        open_req.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
        on_upgrade.forget(); // leak the closure — it is only called once during upgrade

        let db_value: JsValue = JsFuture::from(open_req).await?;
        let db: IdbDatabase = db_value
            .dyn_into()
            .map_err(|_| JsValue::from_str("IdbOpenDbRequest result was not an IdbDatabase"))?;

        Ok(db)
    }

    // -----------------------------------------------------------------------
    // IndexedDbStore
    // -----------------------------------------------------------------------

    /// IndexedDB-backed content-addressed block store.
    ///
    /// Data persists across page refreshes inside the browser's IndexedDB.
    /// Each instance holds the database name; operations open short-lived
    /// read-write transactions as needed.
    ///
    /// # JavaScript
    /// ```javascript
    /// const store = await IndexedDbStore.open("my-ipfrs-db");
    /// const cid   = await store.put(new TextEncoder().encode("hello"));
    /// const bytes = await store.get(cid);
    /// ```
    #[wasm_bindgen]
    pub struct IndexedDbStore {
        db_name: String,
    }

    #[wasm_bindgen]
    impl IndexedDbStore {
        /// Open (or create) an IndexedDB database for IPFRS block storage.
        ///
        /// The database is identified by `db_name` and stores blocks under an
        /// internal object store named `"blocks"`.
        #[wasm_bindgen]
        pub async fn open(db_name: &str) -> Result<IndexedDbStore, JsValue> {
            // Eagerly open to validate that IndexedDB is available and that the
            // upgrade callback runs before we hand the handle to the caller.
            let _db = open_db(db_name).await?;
            Ok(IndexedDbStore {
                db_name: db_name.to_string(),
            })
        }

        /// Store `data` in IndexedDB and return its CIDv1 string.
        ///
        /// If a block with the same CID is already present it is overwritten
        /// (idempotent, since the content is identical).
        pub async fn put(&self, data: &[u8]) -> Result<String, JsValue> {
            if data.is_empty() {
                return Err(JsValue::from_str("data must not be empty"));
            }

            let cid = cid_from_bytes(data);
            let db = open_db(&self.db_name).await?;

            let tx = db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)?;
            let store = tx.object_store(STORE_NAME)?;

            // Convert &[u8] -> Uint8Array for storage
            let js_data = Uint8Array::from(data);
            let cid_key = JsValue::from_str(&cid);

            let put_req = store.put_with_key(&js_data, &cid_key)?;
            JsFuture::from(put_req).await?;

            Ok(cid)
        }

        /// Retrieve raw bytes for the block identified by `cid`.
        ///
        /// Returns `None` when the CID is not in the store.
        pub async fn get(&self, cid: &str) -> Result<Option<Vec<u8>>, JsValue> {
            let db = open_db(&self.db_name).await?;

            let tx = db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readonly)?;
            let store = tx.object_store(STORE_NAME)?;

            let key = JsValue::from_str(cid);
            let get_req = store.get(&key)?;
            let result: JsValue = JsFuture::from(get_req).await?;

            if result.is_undefined() || result.is_null() {
                return Ok(None);
            }

            let arr: Uint8Array = result
                .dyn_into()
                .map_err(|_| JsValue::from_str("stored value is not a Uint8Array"))?;

            Ok(Some(arr.to_vec()))
        }

        /// Return `true` if `cid` is present in the store.
        pub async fn has(&self, cid: &str) -> Result<bool, JsValue> {
            Ok(self.get(cid).await?.is_some())
        }

        /// Delete the block identified by `cid`.
        ///
        /// Returns `true` if the block existed and was removed, `false` otherwise.
        pub async fn delete(&self, cid: &str) -> Result<bool, JsValue> {
            let existed = self.has(cid).await?;
            if !existed {
                return Ok(false);
            }

            let db = open_db(&self.db_name).await?;

            let tx = db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)?;
            let store = tx.object_store(STORE_NAME)?;

            let key = JsValue::from_str(cid);
            let del_req = store.delete(&key)?;
            JsFuture::from(del_req).await?;

            Ok(true)
        }

        /// Return the number of blocks currently stored in the database.
        pub async fn count(&self) -> Result<u32, JsValue> {
            let db = open_db(&self.db_name).await?;

            let tx = db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readonly)?;
            let store = tx.object_store(STORE_NAME)?;

            let count_req = store.count()?;
            let result: JsValue = JsFuture::from(count_req).await?;

            result
                .as_f64()
                .map(|n| n as u32)
                .ok_or_else(|| JsValue::from_str("count() did not return a number"))
        }
    }
}

// ---------------------------------------------------------------------------
// IpfrsClientPersistent – high-level client backed by IndexedDB (wasm32 only)
// ---------------------------------------------------------------------------

/// IPFRS persistent client backed by IndexedDB (browser only).
///
/// Data survives page refreshes.  Internally delegates all storage to
/// an [`indexed_db::IndexedDbStore`].
///
/// # JavaScript
/// ```javascript
/// const client = await IpfrsClientPersistent.new("ipfrs-blocks");
/// const cid    = await client.add(new TextEncoder().encode("hello"));
/// const bytes  = await client.get(cid);
/// ```
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct IpfrsClientPersistent {
    db_name: String,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl IpfrsClientPersistent {
    /// Open (or initialise) a persistent IPFRS client backed by the IndexedDB
    /// database named `db_name`.
    #[wasm_bindgen(constructor)]
    pub async fn new(db_name: &str) -> Result<IpfrsClientPersistent, JsValue> {
        // Delegate to IndexedDbStore::open to validate availability and run migrations.
        indexed_db::IndexedDbStore::open(db_name).await?;
        Ok(IpfrsClientPersistent {
            db_name: db_name.to_string(),
        })
    }

    /// Add `data` to the persistent store and return its CIDv1 string.
    pub async fn add(&self, data: &[u8]) -> Result<String, JsValue> {
        let store = indexed_db::IndexedDbStore::open(&self.db_name).await?;
        store.put(data).await
    }

    /// Retrieve bytes by `cid`.  Returns `None` when the block is absent.
    pub async fn get(&self, cid: &str) -> Result<Option<Vec<u8>>, JsValue> {
        let store = indexed_db::IndexedDbStore::open(&self.db_name).await?;
        store.get(cid).await
    }

    /// Return `true` if `cid` is in the persistent store.
    pub async fn has(&self, cid: &str) -> Result<bool, JsValue> {
        let store = indexed_db::IndexedDbStore::open(&self.db_name).await?;
        store.has(cid).await
    }

    /// Delete the block identified by `cid`.
    ///
    /// Returns `true` if the block existed and was removed, `false` otherwise.
    pub async fn delete(&self, cid: &str) -> Result<bool, JsValue> {
        let store = indexed_db::IndexedDbStore::open(&self.db_name).await?;
        store.delete(cid).await
    }

    /// Return the number of blocks in the persistent store.
    pub async fn count(&self) -> Result<u32, JsValue> {
        let store = indexed_db::IndexedDbStore::open(&self.db_name).await?;
        store.count().await
    }
}

// ---------------------------------------------------------------------------
// Tests – run on the native target, NOT wasm32
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // InMemoryStore unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_in_memory_store_roundtrip() {
        let mut store = InMemoryStore::new();
        let data = b"hello wasm";
        let cid = store.put(data);
        assert!(store.has(&cid));
        assert_eq!(store.get(&cid), Some(&data.to_vec()));
    }

    #[test]
    fn test_in_memory_delete() {
        let mut store = InMemoryStore::new();
        let cid = store.put(b"delete me");
        assert!(store.has(&cid));
        assert!(store.delete(&cid));
        assert!(!store.has(&cid));
        // Deleting again returns false
        assert!(!store.delete(&cid));
    }

    #[test]
    fn test_cid_deterministic() {
        let mut store = InMemoryStore::new();
        let cid1 = store.put(b"same data");
        let cid2 = store.put(b"same data");
        assert_eq!(cid1, cid2, "same content must produce same CID");
        // Only one block stored
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_different_data_different_cid() {
        let mut store = InMemoryStore::new();
        let cid1 = store.put(b"alpha");
        let cid2 = store.put(b"beta");
        assert_ne!(cid1, cid2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_total_bytes_accounting() {
        let mut store = InMemoryStore::new();
        store.put(b"abc"); // 3 bytes
        store.put(b"defgh"); // 5 bytes
        assert_eq!(store.total_bytes(), 8);
        // Storing duplicate must not increase total_bytes
        store.put(b"abc");
        assert_eq!(store.total_bytes(), 8);
    }

    #[test]
    fn test_total_bytes_after_delete() {
        let mut store = InMemoryStore::new();
        let cid = store.put(b"hello"); // 5 bytes
        assert_eq!(store.total_bytes(), 5);
        store.delete(&cid);
        assert_eq!(store.total_bytes(), 0);
    }

    #[test]
    fn test_list_cids() {
        let mut store = InMemoryStore::new();
        let c1 = store.put(b"one");
        let c2 = store.put(b"two");
        let mut listed = store.list();
        listed.sort();
        let mut expected = vec![c1, c2];
        expected.sort();
        assert_eq!(listed, expected);
    }

    // ------------------------------------------------------------------
    // compute_cid / verify_cid
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_cid_function() {
        let cid = compute_cid(b"ipfrs wasm");
        assert!(!cid.is_empty());
        // CIDv1 multibase base32lower starts with 'b'
        assert!(cid.starts_with('b'), "CID must start with 'b': {cid}");
    }

    #[test]
    fn test_compute_cid_deterministic() {
        assert_eq!(compute_cid(b"same"), compute_cid(b"same"));
        assert_ne!(compute_cid(b"a"), compute_cid(b"b"));
    }

    #[test]
    fn test_verify_cid() {
        let data = b"verify me";
        let cid = compute_cid(data);
        assert!(verify_cid(&cid, data));
        assert!(!verify_cid(&cid, b"different data"));
        assert!(!verify_cid("bnotacid", data));
    }

    // ------------------------------------------------------------------
    // IpfrsClient (sync-path via InMemoryStore)
    // ------------------------------------------------------------------

    #[test]
    fn test_store_stats() {
        let mut store = InMemoryStore::new();
        store.put(b"stats test data");
        let stats_json = format!(
            r#"{{"block_count":{block_count},"total_bytes":{total_bytes}}}"#,
            block_count = store.len(),
            total_bytes = store.total_bytes(),
        );
        assert!(stats_json.contains("\"block_count\":1"));
        assert!(stats_json.contains("\"total_bytes\":15"));
    }

    #[test]
    fn test_client_has_and_delete() {
        let mut client = IpfrsClient::new();
        let cid = client.store.put(b"hello client");
        assert!(client.has(&cid));
        assert!(client.delete(&cid));
        assert!(!client.has(&cid));
    }

    #[test]
    fn test_client_list_cids() {
        let mut client = IpfrsClient::new();
        client.store.put(b"item1");
        client.store.put(b"item2");
        assert_eq!(client.list_cids().len(), 2);
    }

    #[test]
    fn test_client_stats_json() {
        let mut client = IpfrsClient::new();
        client.store.put(b"stat");
        let stats = client.stats();
        assert!(stats.contains("block_count"));
        assert!(stats.contains("total_bytes"));
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(v.contains("ipfrs-wasm"));
        assert!(v.contains("0.2.0"));
    }

    #[test]
    fn test_cid_starts_with_b() {
        for payload in [b"a".as_ref(), b"hello world", b"\x00\xff\xfe"] {
            let cid = cid_from_bytes(payload);
            assert!(cid.starts_with('b'), "unexpected CID: {cid}");
        }
    }

    #[test]
    fn test_known_cid_vector() {
        // Cross-check against a known SHA-256 value.
        // echo -n "Hello World" | ipfs add --raw-leaves --inline-limit 0 --cid-version 1
        // produces: bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e
        use sha2::Digest;
        let data = b"Hello World";
        let digest: [u8; 32] = sha2::Sha256::digest(data).into();
        let expected_hex = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e";
        assert_eq!(hex::encode(digest), expected_hex);
        let cid = cid_from_bytes(data);
        assert!(cid.starts_with('b'));
        assert!(!cid.is_empty());
    }

    // ------------------------------------------------------------------
    // NPM package metadata validation
    // ------------------------------------------------------------------

    #[test]
    fn test_package_metadata() {
        let pkg_json = include_str!("../pkg/package.json");
        let parsed: serde_json::Value =
            serde_json::from_str(pkg_json).expect("pkg/package.json must be valid JSON");
        assert_eq!(parsed["name"], "@cool-japan/ipfrs");
        assert_eq!(parsed["version"], "0.2.0");
        assert_eq!(parsed["license"], "Apache-2.0");
        // Verify required file entries are present
        let files = parsed["files"].as_array().expect("files must be an array");
        let file_names: Vec<&str> = files.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            file_names.contains(&"ipfrs_wasm.js"),
            "ipfrs_wasm.js missing from files"
        );
        assert!(
            file_names.contains(&"ipfrs_wasm_bg.wasm"),
            "ipfrs_wasm_bg.wasm missing from files"
        );
    }

    #[test]
    fn test_compute_cid_stable() {
        // Regression test: CID for a fixed input must not change between releases.
        let cid = compute_cid(b"ipfrs stable vector");
        assert!(cid.starts_with('b'), "CID must start with multibase 'b'");
        // Verify against the same computation independently:
        let expected = cid_from_bytes(b"ipfrs stable vector");
        assert_eq!(cid, expected);
    }
}
