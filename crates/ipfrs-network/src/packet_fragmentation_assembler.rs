//! Packet Fragmentation and Reassembly Engine
//!
//! Provides production-quality splitting of large messages into MTU-sized fragments
//! with FNV-1a checksums, duplicate detection, stale-entry expiration, and rich
//! statistics – all without panics (`unwrap`-free).

use std::collections::{HashMap, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// Type aliases
// ──────────────────────────────────────────────────────────────────────────────

/// Opaque message identifier.
pub type PfaMessageId = u64;

/// Re-export the main struct under a longer alias for doc-clarity.
pub type PfaPacketFragmentationAssembler = PacketFragmentationAssembler;

// ──────────────────────────────────────────────────────────────────────────────
// Inline helpers (no external deps)
// ──────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Xorshift-64 PRNG.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a truncated to 32 bits (used as per-fragment checksum).
#[inline]
fn fnv1a_32(data: &[u8]) -> u32 {
    (fnv1a_64(data) & 0xFFFF_FFFF) as u32
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for [`PacketFragmentationAssembler`].
#[derive(Debug, Clone)]
pub struct PfaAssemblerConfig {
    /// Maximum transmission unit (payload bytes per fragment).
    pub mtu: usize,
    /// Upper limit on fragment count for a single message.
    pub max_fragments: u32,
    /// Seconds before a partially-received message is discarded.
    pub reassembly_timeout_secs: u64,
    /// Whether to verify FNV-1a checksums on received fragments.
    pub checksum_enabled: bool,
}

impl Default for PfaAssemblerConfig {
    fn default() -> Self {
        Self {
            mtu: 1_400,
            max_fragments: 4_096,
            reassembly_timeout_secs: 30,
            checksum_enabled: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Data types
// ──────────────────────────────────────────────────────────────────────────────

/// A single fragment of a fragmented message.
#[derive(Debug, Clone)]
pub struct PfaFragment {
    /// Message this fragment belongs to.
    pub msg_id: PfaMessageId,
    /// Zero-based index of this fragment within the message.
    pub fragment_index: u32,
    /// Total number of fragments that constitute the full message.
    pub total_fragments: u32,
    /// Byte offset of this fragment within the original message.
    pub offset: usize,
    /// Payload bytes.
    pub data: Vec<u8>,
    /// FNV-1a low-32 checksum of `data`.
    pub checksum: u32,
}

/// Internal buffer accumulating received fragments for one message.
#[derive(Debug)]
pub struct PfaReassemblyBuffer {
    /// Message identifier.
    pub msg_id: PfaMessageId,
    /// Slots for each expected fragment; `None` means not yet received.
    pub received: Vec<Option<PfaFragment>>,
    /// How many fragments are expected in total.
    pub total_fragments: u32,
    /// Unix-epoch seconds when the buffer was first created.
    pub created_at: u64,
    /// Unix-epoch seconds of the most recently received fragment.
    pub last_updated: u64,
    /// How many non-duplicate fragments have been received so far.
    received_count: u32,
}

impl PfaReassemblyBuffer {
    fn new(msg_id: PfaMessageId, total_fragments: u32, now_ts: u64) -> Self {
        Self {
            msg_id,
            received: vec![None; total_fragments as usize],
            total_fragments,
            created_at: now_ts,
            last_updated: now_ts,
            received_count: 0,
        }
    }

    /// Returns how many non-duplicate fragments have been stored.
    pub fn received_count(&self) -> u32 {
        self.received_count
    }

    /// Returns `true` when all fragments have been received.
    pub fn is_complete(&self) -> bool {
        self.received_count == self.total_fragments
    }
}

/// Audit record for the fragment log.
#[derive(Debug, Clone)]
pub struct PfaFragmentRecord {
    /// Timestamp of the event (Unix-epoch seconds).
    pub ts: u64,
    /// Message the fragment belongs to.
    pub msg_id: PfaMessageId,
    /// Fragment index.
    pub fragment_index: u32,
    /// Whether this was a duplicate fragment.
    pub is_dup: bool,
    /// Whether this fragment completed the message.
    pub assembled: bool,
}

/// Result returned by [`PacketFragmentationAssembler::receive_fragment`].
#[derive(Debug)]
pub enum PfaReceiveResult {
    /// Fragment stored; message is not yet complete.
    Buffered,
    /// Fragment was already received (duplicate).
    Duplicate,
    /// All fragments are now available; the reassembled payload is included.
    Assembled(Vec<u8>),
    /// An error occurred (checksum mismatch, out-of-range index, etc.).
    Error(String),
}

/// Snapshot of assembler-level counters.
#[derive(Debug, Clone, Default)]
pub struct PfaAssemblerStats {
    /// Number of messages that have been fragmented outbound.
    pub total_fragmented: u64,
    /// Number of messages that have been fully reassembled.
    pub total_assembled: u64,
    /// Fragments dropped due to checksum failure, range errors, or expiry.
    pub total_dropped: u64,
    /// Average fragment count across all fragmented messages.
    pub avg_fragments: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Core engine
// ──────────────────────────────────────────────────────────────────────────────

/// Packet fragmentation and reassembly engine.
///
/// # Thread safety
///
/// This struct is `Send` but not `Sync`; wrap in `Arc<Mutex<...>>` for shared use.
pub struct PacketFragmentationAssembler {
    config: PfaAssemblerConfig,
    /// Reassembly state keyed by message-id.
    buffers: HashMap<PfaMessageId, PfaReassemblyBuffer>,
    /// Rolling audit log (bounded to 1 000 entries).
    fragment_log: VecDeque<PfaFragmentRecord>,
    /// PRNG state used for unique message IDs.
    rng_state: u64,
    // Statistics counters
    total_fragmented: u64,
    total_assembled: u64,
    total_dropped: u64,
    /// Running sum of fragment-counts across all fragmented messages (for avg).
    fragment_count_sum: u64,
}

impl PacketFragmentationAssembler {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new assembler with the given configuration.
    pub fn new(config: PfaAssemblerConfig) -> Self {
        // Seed derived from config to give each instance a distinct PRNG stream.
        let seed = fnv1a_64(&config.mtu.to_le_bytes())
            ^ fnv1a_64(&config.reassembly_timeout_secs.to_le_bytes())
            ^ 0xDEAD_BEEF_CAFE_BABE;
        Self {
            config,
            buffers: HashMap::new(),
            fragment_log: VecDeque::with_capacity(1_000),
            rng_state: seed,
            total_fragmented: 0,
            total_assembled: 0,
            total_dropped: 0,
            fragment_count_sum: 0,
        }
    }

    /// Convenience constructor using default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PfaAssemblerConfig::default())
    }

    // ── PRNG helpers ──────────────────────────────────────────────────────────

    /// Generate the next pseudo-random u64.
    fn next_rand(&mut self) -> u64 {
        xorshift64(&mut self.rng_state)
    }

    /// Generate a new message ID that is not currently in use.
    pub fn generate_msg_id(&mut self) -> PfaMessageId {
        loop {
            let id = self.next_rand();
            if id != 0 && !self.buffers.contains_key(&id) {
                return id;
            }
        }
    }

    // ── Fragmentation ─────────────────────────────────────────────────────────

    /// Split `data` into MTU-sized fragments tagged with `msg_id`.
    ///
    /// Returns an empty `Vec` if `data` is empty.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the payload would require more than
    /// `config.max_fragments` fragments.
    pub fn fragment(
        &mut self,
        msg_id: PfaMessageId,
        data: &[u8],
    ) -> Result<Vec<PfaFragment>, String> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mtu = self.config.mtu.max(1);
        let total_fragments = data.len().div_ceil(mtu) as u32;

        if total_fragments > self.config.max_fragments {
            return Err(format!(
                "message requires {} fragments but max_fragments is {}",
                total_fragments, self.config.max_fragments
            ));
        }

        let mut fragments = Vec::with_capacity(total_fragments as usize);
        let mut offset = 0usize;

        for idx in 0..total_fragments {
            let end = (offset + mtu).min(data.len());
            let payload = data[offset..end].to_vec();
            let checksum = if self.config.checksum_enabled {
                fnv1a_32(&payload)
            } else {
                0
            };

            fragments.push(PfaFragment {
                msg_id,
                fragment_index: idx,
                total_fragments,
                offset,
                data: payload,
                checksum,
            });

            offset = end;
        }

        self.total_fragmented += 1;
        self.fragment_count_sum += total_fragments as u64;

        Ok(fragments)
    }

    // ── Reception ─────────────────────────────────────────────────────────────

    /// Process a received fragment.
    ///
    /// `now_ts` is the current Unix-epoch timestamp in seconds and is used for
    /// the fragment log and buffer creation timestamps.
    pub fn receive_fragment(&mut self, fragment: PfaFragment, now_ts: u64) -> PfaReceiveResult {
        // ── Validate checksum ─────────────────────────────────────────────────
        if self.config.checksum_enabled && !Self::verify_checksum_static(&fragment) {
            self.total_dropped += 1;
            self.push_log(PfaFragmentRecord {
                ts: now_ts,
                msg_id: fragment.msg_id,
                fragment_index: fragment.fragment_index,
                is_dup: false,
                assembled: false,
            });
            return PfaReceiveResult::Error(format!(
                "checksum mismatch for msg_id={} fragment_index={}",
                fragment.msg_id, fragment.fragment_index
            ));
        }

        // ── Basic sanity checks ───────────────────────────────────────────────
        if fragment.total_fragments == 0 {
            self.total_dropped += 1;
            return PfaReceiveResult::Error("total_fragments must be > 0".into());
        }
        if fragment.fragment_index >= fragment.total_fragments {
            self.total_dropped += 1;
            return PfaReceiveResult::Error(format!(
                "fragment_index {} out of range (total={})",
                fragment.fragment_index, fragment.total_fragments
            ));
        }
        if fragment.total_fragments > self.config.max_fragments {
            self.total_dropped += 1;
            return PfaReceiveResult::Error(format!(
                "total_fragments {} exceeds max_fragments {}",
                fragment.total_fragments, self.config.max_fragments
            ));
        }

        let msg_id = fragment.msg_id;
        let frag_idx = fragment.fragment_index;
        let total = fragment.total_fragments;

        // ── Ensure reassembly buffer exists ───────────────────────────────────
        let buf = self
            .buffers
            .entry(msg_id)
            .or_insert_with(|| PfaReassemblyBuffer::new(msg_id, total, now_ts));

        // Guard: total_fragments must match the buffer's expectation.
        if buf.total_fragments != total {
            self.total_dropped += 1;
            return PfaReceiveResult::Error(format!(
                "total_fragments mismatch: buffer expects {} but fragment says {}",
                buf.total_fragments, total
            ));
        }

        // Bounds check on the slot vec (should be redundant, but be safe).
        if frag_idx as usize >= buf.received.len() {
            self.total_dropped += 1;
            return PfaReceiveResult::Error(format!(
                "fragment_index {} out of buffer range {}",
                frag_idx,
                buf.received.len()
            ));
        }

        // ── Duplicate detection ───────────────────────────────────────────────
        if buf.received[frag_idx as usize].is_some() {
            self.push_log(PfaFragmentRecord {
                ts: now_ts,
                msg_id,
                fragment_index: frag_idx,
                is_dup: true,
                assembled: false,
            });
            return PfaReceiveResult::Duplicate;
        }

        // ── Store fragment ────────────────────────────────────────────────────
        buf.received[frag_idx as usize] = Some(fragment);
        buf.received_count += 1;
        buf.last_updated = now_ts;
        let complete = buf.is_complete();

        // ── Attempt reassembly if complete ────────────────────────────────────
        if complete {
            // Remove buffer and reassemble.
            if let Some(completed_buf) = self.buffers.remove(&msg_id) {
                match Self::assemble_buffer(completed_buf) {
                    Ok(data) => {
                        self.total_assembled += 1;
                        self.push_log(PfaFragmentRecord {
                            ts: now_ts,
                            msg_id,
                            fragment_index: frag_idx,
                            is_dup: false,
                            assembled: true,
                        });
                        return PfaReceiveResult::Assembled(data);
                    }
                    Err(e) => {
                        self.total_dropped += 1;
                        self.push_log(PfaFragmentRecord {
                            ts: now_ts,
                            msg_id,
                            fragment_index: frag_idx,
                            is_dup: false,
                            assembled: false,
                        });
                        return PfaReceiveResult::Error(e);
                    }
                }
            }
        }

        self.push_log(PfaFragmentRecord {
            ts: now_ts,
            msg_id,
            fragment_index: frag_idx,
            is_dup: false,
            assembled: false,
        });
        PfaReceiveResult::Buffered
    }

    // ── Reassembly ────────────────────────────────────────────────────────────

    /// Attempt to reassemble message `msg_id` if all fragments are present.
    ///
    /// Returns `None` if the buffer does not exist or is incomplete.
    pub fn reassemble(&mut self, msg_id: PfaMessageId) -> Option<Vec<u8>> {
        // Peek whether complete before removing.
        let complete = self
            .buffers
            .get(&msg_id)
            .map(|b| b.is_complete())
            .unwrap_or(false);

        if !complete {
            return None;
        }

        let buf = self.buffers.remove(&msg_id)?;
        match Self::assemble_buffer(buf) {
            Ok(data) => {
                self.total_assembled += 1;
                Some(data)
            }
            Err(_) => {
                self.total_dropped += 1;
                None
            }
        }
    }

    /// Internal: consume a complete buffer and produce the reassembled byte vector.
    fn assemble_buffer(buf: PfaReassemblyBuffer) -> Result<Vec<u8>, String> {
        // Sort slots by fragment index to reconstruct in-order.
        let mut indexed: Vec<(u32, Vec<u8>)> = Vec::with_capacity(buf.total_fragments as usize);
        for (idx, slot) in buf.received.into_iter().enumerate() {
            match slot {
                Some(frag) => indexed.push((frag.fragment_index, frag.data)),
                None => {
                    return Err(format!("missing fragment at slot {}", idx));
                }
            }
        }
        indexed.sort_unstable_by_key(|(i, _)| *i);

        let total_len: usize = indexed.iter().map(|(_, d)| d.len()).sum();
        let mut out = Vec::with_capacity(total_len);
        for (_, data) in indexed {
            out.extend_from_slice(&data);
        }
        Ok(out)
    }

    // ── Expiration ────────────────────────────────────────────────────────────

    /// Remove all buffers whose `created_at` is older than the configured timeout.
    pub fn expire_stale(&mut self, now_ts: u64) {
        let timeout = self.config.reassembly_timeout_secs;
        let before = self.buffers.len();
        self.buffers
            .retain(|_, buf| now_ts.saturating_sub(buf.created_at) < timeout);
        let dropped_count = before - self.buffers.len();
        self.total_dropped = self.total_dropped.saturating_add(dropped_count as u64);
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// List pending messages: `(msg_id, received_count, total_fragments)`.
    pub fn pending_messages(&self) -> Vec<(PfaMessageId, u32, u32)> {
        self.buffers
            .values()
            .map(|b| (b.msg_id, b.received_count(), b.total_fragments))
            .collect()
    }

    /// Verify the checksum of a fragment (instance method forwarding to static).
    pub fn verify_checksum(&self, fragment: &PfaFragment) -> bool {
        if !self.config.checksum_enabled {
            return true;
        }
        Self::verify_checksum_static(fragment)
    }

    /// Static checksum verifier (does not need a `&self` reference).
    fn verify_checksum_static(fragment: &PfaFragment) -> bool {
        fnv1a_32(&fragment.data) == fragment.checksum
    }

    /// Return a snapshot of assembler-level statistics.
    pub fn assembler_stats(&self) -> PfaAssemblerStats {
        let avg_fragments = if self.total_fragmented == 0 {
            0.0
        } else {
            self.fragment_count_sum as f64 / self.total_fragmented as f64
        };
        PfaAssemblerStats {
            total_fragmented: self.total_fragmented,
            total_assembled: self.total_assembled,
            total_dropped: self.total_dropped,
            avg_fragments,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Reference to the configuration.
    pub fn config(&self) -> &PfaAssemblerConfig {
        &self.config
    }

    /// Number of messages currently awaiting full reassembly.
    pub fn pending_count(&self) -> usize {
        self.buffers.len()
    }

    /// A slice of the most recent fragment log entries.
    pub fn fragment_log(&self) -> &VecDeque<PfaFragmentRecord> {
        &self.fragment_log
    }

    /// True if the engine holds any in-flight reassembly buffers.
    pub fn has_pending(&self) -> bool {
        !self.buffers.is_empty()
    }

    /// Return a reference to the reassembly buffer for `msg_id`, if any.
    pub fn get_buffer(&self, msg_id: PfaMessageId) -> Option<&PfaReassemblyBuffer> {
        self.buffers.get(&msg_id)
    }

    /// Drain and return all fragment log entries.
    pub fn drain_log(&mut self) -> Vec<PfaFragmentRecord> {
        self.fragment_log.drain(..).collect()
    }

    /// Reset all state while keeping configuration.
    pub fn reset(&mut self) {
        self.buffers.clear();
        self.fragment_log.clear();
        self.total_fragmented = 0;
        self.total_assembled = 0;
        self.total_dropped = 0;
        self.fragment_count_sum = 0;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Append an entry to the rolling fragment log, evicting the oldest if full.
    fn push_log(&mut self, record: PfaFragmentRecord) {
        if self.fragment_log.len() >= 1_000 {
            self.fragment_log.pop_front();
        }
        self.fragment_log.push_back(record);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_assembler() -> PacketFragmentationAssembler {
        PacketFragmentationAssembler::with_defaults()
    }

    fn make_assembler_cfg(
        mtu: usize,
        timeout: u64,
        checksum: bool,
    ) -> PacketFragmentationAssembler {
        PacketFragmentationAssembler::new(PfaAssemblerConfig {
            mtu,
            max_fragments: 4_096,
            reassembly_timeout_secs: timeout,
            checksum_enabled: checksum,
        })
    }

    fn receive_all(
        asm: &mut PacketFragmentationAssembler,
        frags: Vec<PfaFragment>,
        ts: u64,
    ) -> PfaReceiveResult {
        let len = frags.len();
        let mut last = PfaReceiveResult::Buffered;
        for (i, f) in frags.into_iter().enumerate() {
            last = asm.receive_fragment(f, ts);
            if i == len - 1 {
                return last;
            }
        }
        last
    }

    // ── fnv1a_64 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty() {
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a_known_value() {
        // "abc" → well-known FNV-1a-64 result
        let h = fnv1a_64(b"abc");
        assert_ne!(h, 0);
        assert_ne!(h, 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a_64(b"hello world");
        let b = fnv1a_64(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_nonzero() {
        let mut s = 12345u64;
        let v = xorshift64(&mut s);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift_changes_state() {
        let mut s = 99999u64;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b);
    }

    #[test]
    fn test_xorshift_reproducible() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = PfaAssemblerConfig::default();
        assert_eq!(cfg.mtu, 1_400);
        assert_eq!(cfg.max_fragments, 4_096);
        assert_eq!(cfg.reassembly_timeout_secs, 30);
        assert!(cfg.checksum_enabled);
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_empty_state() {
        let asm = make_assembler();
        assert_eq!(asm.pending_count(), 0);
        assert!(!asm.has_pending());
        let stats = asm.assembler_stats();
        assert_eq!(stats.total_fragmented, 0);
        assert_eq!(stats.total_assembled, 0);
        assert_eq!(stats.total_dropped, 0);
    }

    // ── generate_msg_id ───────────────────────────────────────────────────────

    #[test]
    fn test_generate_msg_id_nonzero() {
        let mut asm = make_assembler();
        let id = asm.generate_msg_id();
        assert_ne!(id, 0);
    }

    #[test]
    fn test_generate_msg_id_unique() {
        let mut asm = make_assembler();
        let ids: Vec<u64> = (0..100).map(|_| asm.generate_msg_id()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().cloned().collect();
        assert_eq!(unique.len(), 100);
    }

    // ── fragment – basic ──────────────────────────────────────────────────────

    #[test]
    fn test_fragment_empty_data() {
        let mut asm = make_assembler();
        let frags = asm
            .fragment(1, &[])
            .expect("test: fragment empty data should succeed");
        assert!(frags.is_empty());
    }

    #[test]
    fn test_fragment_single_chunk() {
        let mut asm = make_assembler_cfg(1_400, 30, true);
        let data: Vec<u8> = (0..100).collect();
        let frags = asm
            .fragment(1, &data)
            .expect("test: fragment single chunk should succeed");
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].fragment_index, 0);
        assert_eq!(frags[0].total_fragments, 1);
        assert_eq!(frags[0].offset, 0);
        assert_eq!(frags[0].data, data);
    }

    #[test]
    fn test_fragment_exact_mtu() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0xABu8; 10];
        let frags = asm
            .fragment(42, &data)
            .expect("test: fragment exact mtu should succeed");
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].data.len(), 10);
    }

    #[test]
    fn test_fragment_two_chunks() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 15];
        let frags = asm
            .fragment(1, &data)
            .expect("test: fragment two chunks should succeed");
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0].data.len(), 10);
        assert_eq!(frags[1].data.len(), 5);
    }

    #[test]
    fn test_fragment_many_chunks() {
        let mut asm = make_assembler_cfg(8, 30, true);
        let data = vec![0u8; 100];
        let frags = asm
            .fragment(1, &data)
            .expect("test: fragment many chunks should succeed");
        assert_eq!(frags.len(), 13); // ceil(100 / 8)
    }

    #[test]
    fn test_fragment_offsets_contiguous() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data: Vec<u8> = (0..35).collect();
        let frags = asm
            .fragment(1, &data)
            .expect("test: fragment offsets contiguous should succeed");
        let mut expected_offset = 0usize;
        for f in &frags {
            assert_eq!(f.offset, expected_offset);
            expected_offset += f.data.len();
        }
        assert_eq!(expected_offset, 35);
    }

    #[test]
    fn test_fragment_checksum_set() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![1u8, 2, 3];
        let frags = asm
            .fragment(1, &data)
            .expect("test: fragment checksum set should succeed");
        assert_ne!(frags[0].checksum, 0);
    }

    #[test]
    fn test_fragment_no_checksum() {
        let mut asm = make_assembler_cfg(10, 30, false);
        let data = vec![1u8, 2, 3];
        let frags = asm
            .fragment(1, &data)
            .expect("test: fragment no checksum should succeed");
        assert_eq!(frags[0].checksum, 0);
    }

    #[test]
    fn test_fragment_too_many() {
        let mut asm = PacketFragmentationAssembler::new(PfaAssemblerConfig {
            mtu: 1,
            max_fragments: 4,
            reassembly_timeout_secs: 30,
            checksum_enabled: true,
        });
        let data = vec![0u8; 10];
        assert!(asm.fragment(1, &data).is_err());
    }

    #[test]
    fn test_fragment_increments_stats() {
        let mut asm = make_assembler();
        let data = vec![0u8; 2_800];
        asm.fragment(1, &data)
            .expect("test: fragment increments stats should succeed");
        let stats = asm.assembler_stats();
        assert_eq!(stats.total_fragmented, 1);
        assert!(stats.avg_fragments >= 2.0);
    }

    // ── reassemble round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_small() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data: Vec<u8> = (0..25u8).collect();
        let frags = asm
            .fragment(1, &data)
            .expect("test: roundtrip small fragment should succeed");
        let result = receive_all(&mut asm, frags, 0);
        if let PfaReceiveResult::Assembled(reassembled) = result {
            assert_eq!(reassembled, data);
        } else {
            panic!("expected Assembled, got {:?}", result);
        }
    }

    #[test]
    fn test_roundtrip_exact_mtu() {
        let mut asm = make_assembler_cfg(8, 30, true);
        let data = vec![0xFFu8; 8];
        let frags = asm
            .fragment(1, &data)
            .expect("test: roundtrip exact mtu fragment should succeed");
        let result = receive_all(&mut asm, frags, 0);
        assert!(matches!(result, PfaReceiveResult::Assembled(d) if d == data));
    }

    #[test]
    fn test_roundtrip_single_byte() {
        let mut asm = make_assembler_cfg(1_400, 30, true);
        let data = vec![42u8];
        let frags = asm
            .fragment(99, &data)
            .expect("test: roundtrip single byte fragment should succeed");
        let result = receive_all(&mut asm, frags, 0);
        assert!(matches!(result, PfaReceiveResult::Assembled(d) if d == data));
    }

    #[test]
    fn test_roundtrip_large_payload() {
        let mut asm = make_assembler_cfg(500, 60, true);
        let data: Vec<u8> = (0..u8::MAX).cycle().take(10_000).collect();
        let frags = asm
            .fragment(7, &data)
            .expect("test: roundtrip large payload fragment should succeed");
        let result = receive_all(&mut asm, frags, 0);
        assert!(matches!(result, PfaReceiveResult::Assembled(d) if d == data));
    }

    #[test]
    fn test_roundtrip_no_checksum() {
        let mut asm = make_assembler_cfg(16, 30, false);
        let data: Vec<u8> = (0..48).collect();
        let frags = asm
            .fragment(5, &data)
            .expect("test: roundtrip no checksum fragment should succeed");
        let result = receive_all(&mut asm, frags, 0);
        assert!(matches!(result, PfaReceiveResult::Assembled(d) if d == data));
    }

    #[test]
    fn test_roundtrip_reverse_order() {
        let mut asm = make_assembler_cfg(10, 60, true);
        let data: Vec<u8> = (0..30u8).collect();
        let mut frags = asm
            .fragment(3, &data)
            .expect("test: roundtrip reverse order fragment should succeed");
        frags.reverse();
        let result = receive_all(&mut asm, frags, 0);
        assert!(matches!(result, PfaReceiveResult::Assembled(d) if d == data));
    }

    // ── duplicate detection ───────────────────────────────────────────────────

    #[test]
    fn test_duplicate_fragment() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: duplicate fragment setup should succeed");
        let dup = frags[0].clone();
        asm.receive_fragment(frags[0].clone(), 0);
        let result = asm.receive_fragment(dup, 0);
        assert!(matches!(result, PfaReceiveResult::Duplicate));
    }

    #[test]
    fn test_duplicate_last_fragment() {
        // With 3 fragments, duplicate the second (non-last) after it is received
        // but before the message is assembled.
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 30];
        let frags = asm
            .fragment(1, &data)
            .expect("test: duplicate last fragment setup should succeed");
        assert_eq!(frags.len(), 3);
        let dup = frags[1].clone();
        // Receive fragments 0 and 1 – not yet complete.
        asm.receive_fragment(frags[0].clone(), 0);
        asm.receive_fragment(frags[1].clone(), 0);
        // Send fragment 1 again before the message is assembled.
        let result = asm.receive_fragment(dup, 0);
        assert!(matches!(result, PfaReceiveResult::Duplicate));
    }

    // ── checksum errors ───────────────────────────────────────────────────────

    #[test]
    fn test_checksum_mismatch_is_error() {
        let mut asm = make_assembler_cfg(100, 30, true);
        let data = vec![1u8; 50];
        let mut frags = asm
            .fragment(1, &data)
            .expect("test: checksum mismatch setup should succeed");
        frags[0].checksum ^= 0xFFFF_FFFF; // corrupt
        let result = asm.receive_fragment(frags.remove(0), 0);
        assert!(matches!(result, PfaReceiveResult::Error(_)));
    }

    #[test]
    fn test_checksum_disabled_ignores_bad_checksum() {
        let mut asm = make_assembler_cfg(100, 30, false);
        let data = vec![1u8; 50];
        let mut frags = asm
            .fragment(1, &data)
            .expect("test: checksum disabled setup should succeed");
        frags[0].checksum = 0xDEAD_BEEF; // won't be checked
        let result = asm.receive_fragment(frags.remove(0), 0);
        // Only one fragment total → assembled
        assert!(matches!(result, PfaReceiveResult::Assembled(_)));
    }

    // ── out-of-range errors ───────────────────────────────────────────────────

    #[test]
    fn test_zero_total_fragments_error() {
        let mut asm = make_assembler();
        let bad = PfaFragment {
            msg_id: 1,
            fragment_index: 0,
            total_fragments: 0,
            offset: 0,
            data: vec![],
            checksum: 0,
        };
        let result = asm.receive_fragment(bad, 0);
        assert!(matches!(result, PfaReceiveResult::Error(_)));
    }

    #[test]
    fn test_index_gte_total_is_error() {
        let mut asm = make_assembler();
        let bad = PfaFragment {
            msg_id: 2,
            fragment_index: 5,
            total_fragments: 3,
            offset: 0,
            data: vec![],
            checksum: 0,
        };
        let result = asm.receive_fragment(bad, 0);
        assert!(matches!(result, PfaReceiveResult::Error(_)));
    }

    #[test]
    fn test_max_fragments_exceeded_error() {
        let mut asm = PacketFragmentationAssembler::new(PfaAssemblerConfig {
            mtu: 1,
            max_fragments: 2,
            reassembly_timeout_secs: 30,
            checksum_enabled: false,
        });
        let bad = PfaFragment {
            msg_id: 3,
            fragment_index: 0,
            total_fragments: 10,
            offset: 0,
            data: vec![0],
            checksum: 0,
        };
        let result = asm.receive_fragment(bad, 0);
        assert!(matches!(result, PfaReceiveResult::Error(_)));
    }

    // ── reassemble() explicit call ────────────────────────────────────────────

    #[test]
    fn test_reassemble_incomplete_returns_none() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: reassemble incomplete setup should succeed");
        asm.receive_fragment(frags[0].clone(), 0);
        // Only one of two received.
        assert!(asm.reassemble(1).is_none());
    }

    #[test]
    fn test_reassemble_unknown_msg_id_returns_none() {
        let mut asm = make_assembler();
        assert!(asm.reassemble(999_999).is_none());
    }

    #[test]
    fn test_reassemble_after_all_received() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data: Vec<u8> = (0..30).collect();
        let frags = asm
            .fragment(1, &data)
            .expect("test: reassemble after all received setup should succeed");
        for f in &frags {
            asm.receive_fragment(f.clone(), 0);
        }
        // receive_fragment already consumed; but let's check reassemble on its own
        // (it triggers Assembled internally, so buffer is removed).
        // Test a fresh scenario instead.
        let mut asm2 = make_assembler_cfg(10, 30, true);
        let frags2 = asm2
            .fragment(2, &data)
            .expect("test: reassemble asm2 fragment should succeed");
        let last_idx = frags2.len() - 1;
        for (i, f) in frags2.into_iter().enumerate() {
            if i < last_idx {
                asm2.receive_fragment(f, 0);
            } else {
                // Don't receive last; call reassemble first
                assert!(asm2.reassemble(2).is_none());
                asm2.receive_fragment(f, 0);
            }
        }
        // After the last receive_fragment, reassemble is triggered internally.
        assert!(asm2.reassemble(2).is_none()); // already consumed
    }

    // ── expire_stale ──────────────────────────────────────────────────────────

    #[test]
    fn test_expire_stale_removes_old_buffers() {
        let mut asm = make_assembler_cfg(10, 10, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: expire stale removes old buffers setup should succeed");
        asm.receive_fragment(frags[0].clone(), 0);
        assert_eq!(asm.pending_count(), 1);
        asm.expire_stale(100); // 100 >> 10 timeout
        assert_eq!(asm.pending_count(), 0);
    }

    #[test]
    fn test_expire_stale_keeps_fresh_buffers() {
        let mut asm = make_assembler_cfg(10, 60, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: expire stale keeps fresh buffers setup should succeed");
        asm.receive_fragment(frags[0].clone(), 50);
        asm.expire_stale(55); // only 5 secs old; timeout = 60
        assert_eq!(asm.pending_count(), 1);
    }

    #[test]
    fn test_expire_stale_increments_dropped() {
        let mut asm = make_assembler_cfg(10, 5, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: expire stale increments dropped setup should succeed");
        asm.receive_fragment(frags[0].clone(), 0);
        let before = asm.assembler_stats().total_dropped;
        asm.expire_stale(100);
        assert!(asm.assembler_stats().total_dropped > before);
    }

    #[test]
    fn test_expire_stale_no_op_on_empty() {
        let mut asm = make_assembler();
        asm.expire_stale(999_999); // should not panic
        assert_eq!(asm.pending_count(), 0);
    }

    // ── pending_messages ──────────────────────────────────────────────────────

    #[test]
    fn test_pending_messages_empty() {
        let asm = make_assembler();
        assert!(asm.pending_messages().is_empty());
    }

    #[test]
    fn test_pending_messages_one_entry() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 25];
        let frags = asm
            .fragment(7, &data)
            .expect("test: pending messages one entry setup should succeed");
        asm.receive_fragment(frags[0].clone(), 0);
        let pending = asm.pending_messages();
        assert_eq!(pending.len(), 1);
        let (mid, recv, total) = pending[0];
        assert_eq!(mid, 7);
        assert_eq!(recv, 1);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_pending_messages_multiple() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data1 = vec![0u8; 20];
        let data2 = vec![1u8; 30];
        let frags1 = asm
            .fragment(10, &data1)
            .expect("test: pending messages multiple frags1 should succeed");
        let frags2 = asm
            .fragment(20, &data2)
            .expect("test: pending messages multiple frags2 should succeed");
        asm.receive_fragment(frags1[0].clone(), 0);
        asm.receive_fragment(frags2[0].clone(), 0);
        let pending = asm.pending_messages();
        assert_eq!(pending.len(), 2);
    }

    // ── verify_checksum ───────────────────────────────────────────────────────

    #[test]
    fn test_verify_checksum_valid() {
        let asm = make_assembler();
        let data = vec![1u8, 2, 3, 4];
        let frag = PfaFragment {
            msg_id: 1,
            fragment_index: 0,
            total_fragments: 1,
            offset: 0,
            checksum: fnv1a_32(&data),
            data,
        };
        assert!(asm.verify_checksum(&frag));
    }

    #[test]
    fn test_verify_checksum_invalid() {
        let asm = make_assembler();
        let data = vec![1u8, 2, 3, 4];
        let frag = PfaFragment {
            msg_id: 1,
            fragment_index: 0,
            total_fragments: 1,
            offset: 0,
            checksum: 0xDEAD_BEEF,
            data,
        };
        assert!(!asm.verify_checksum(&frag));
    }

    #[test]
    fn test_verify_checksum_disabled() {
        let asm = make_assembler_cfg(100, 30, false);
        let frag = PfaFragment {
            msg_id: 1,
            fragment_index: 0,
            total_fragments: 1,
            offset: 0,
            checksum: 0xDEAD_BEEF, // wrong; should be ignored
            data: vec![42u8],
        };
        assert!(asm.verify_checksum(&frag));
    }

    // ── assembler_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_stats_after_full_roundtrip() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 30];
        let frags = asm
            .fragment(1, &data)
            .expect("test: stats after full roundtrip setup should succeed");
        receive_all(&mut asm, frags, 0);
        let stats = asm.assembler_stats();
        assert_eq!(stats.total_fragmented, 1);
        assert_eq!(stats.total_assembled, 1);
        assert_eq!(stats.total_dropped, 0);
        assert!((stats.avg_fragments - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_dropped_on_checksum_fail() {
        let mut asm = make_assembler_cfg(100, 30, true);
        let data = vec![1u8; 50];
        let mut frags = asm
            .fragment(1, &data)
            .expect("test: stats dropped on checksum fail setup should succeed");
        frags[0].checksum ^= 0xFF;
        asm.receive_fragment(frags.remove(0), 0);
        let stats = asm.assembler_stats();
        assert_eq!(stats.total_dropped, 1);
    }

    #[test]
    fn test_stats_avg_across_multiple() {
        let mut asm = make_assembler_cfg(10, 30, true);
        asm.fragment(1, &[0u8; 10])
            .expect("test: stats avg fragment 1 should succeed"); // 1 fragment
        asm.fragment(2, &[0u8; 20])
            .expect("test: stats avg fragment 2 should succeed"); // 2 fragments
        asm.fragment(3, &[0u8; 30])
            .expect("test: stats avg fragment 3 should succeed"); // 3 fragments
        let stats = asm.assembler_stats();
        assert_eq!(stats.total_fragmented, 3);
        // avg = (1+2+3)/3 = 2.0
        assert!((stats.avg_fragments - 2.0).abs() < f64::EPSILON);
    }

    // ── fragment log ─────────────────────────────────────────────────────────

    #[test]
    fn test_log_grows_on_receive() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 30];
        let frags = asm
            .fragment(1, &data)
            .expect("test: log grows on receive setup should succeed");
        for f in frags {
            asm.receive_fragment(f, 0);
        }
        assert!(!asm.fragment_log().is_empty());
    }

    #[test]
    fn test_log_capped_at_1000() {
        let mut asm = make_assembler_cfg(1, 30, false);
        // Send 1_100 single-byte single-fragment messages.
        for id in 1u64..=1_100 {
            let frags = asm
                .fragment(id, &[0u8])
                .expect("test: log capped at 1000 fragment should succeed");
            asm.receive_fragment(
                frags
                    .into_iter()
                    .next()
                    .expect("test: log capped at 1000 first fragment should exist"),
                0,
            );
        }
        assert_eq!(asm.fragment_log().len(), 1_000);
    }

    #[test]
    fn test_log_drain_empties() {
        let mut asm = make_assembler_cfg(1, 30, false);
        let frags = asm
            .fragment(1, &[42u8])
            .expect("test: log drain empties fragment should succeed");
        asm.receive_fragment(
            frags
                .into_iter()
                .next()
                .expect("test: log drain empties first fragment should exist"),
            0,
        );
        let drained = asm.drain_log();
        assert!(!drained.is_empty());
        assert!(asm.fragment_log().is_empty());
    }

    #[test]
    fn test_log_assembled_flag() {
        let mut asm = make_assembler_cfg(100, 30, true);
        let data = vec![0u8; 50];
        let frags = asm
            .fragment(1, &data)
            .expect("test: log assembled flag setup should succeed");
        receive_all(&mut asm, frags, 0);
        let log: Vec<_> = asm.fragment_log().iter().cloned().collect();
        let assembled_entry = log.iter().find(|r| r.assembled);
        assert!(assembled_entry.is_some());
    }

    #[test]
    fn test_log_dup_flag() {
        // Use a 3-fragment message so we can duplicate fragment 0 before assembly.
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 30];
        let frags = asm
            .fragment(1, &data)
            .expect("test: log dup flag setup should succeed");
        assert_eq!(frags.len(), 3);
        let dup = frags[0].clone();
        // Receive fragments 0 and 1 (not yet assembled).
        asm.receive_fragment(frags[0].clone(), 0);
        asm.receive_fragment(frags[1].clone(), 0);
        // Duplicate fragment 0.
        asm.receive_fragment(dup, 0);
        let log: Vec<_> = asm.fragment_log().iter().cloned().collect();
        let dup_entry = log.iter().find(|r| r.is_dup);
        assert!(dup_entry.is_some());
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_all() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: reset clears all setup should succeed");
        asm.receive_fragment(frags[0].clone(), 0);
        asm.reset();
        assert_eq!(asm.pending_count(), 0);
        assert!(asm.fragment_log().is_empty());
        let stats = asm.assembler_stats();
        assert_eq!(stats.total_fragmented, 0);
        assert_eq!(stats.total_assembled, 0);
    }

    // ── get_buffer ────────────────────────────────────────────────────────────

    #[test]
    fn test_get_buffer_present() {
        let mut asm = make_assembler_cfg(10, 30, true);
        let data = vec![0u8; 20];
        let frags = asm
            .fragment(1, &data)
            .expect("test: get buffer present setup should succeed");
        asm.receive_fragment(frags[0].clone(), 5);
        let buf = asm
            .get_buffer(1)
            .expect("test: get buffer present should return buffer");
        assert_eq!(buf.msg_id, 1);
        assert_eq!(buf.created_at, 5);
        assert_eq!(buf.received_count(), 1);
    }

    #[test]
    fn test_get_buffer_absent() {
        let asm = make_assembler();
        assert!(asm.get_buffer(42).is_none());
    }

    // ── PfaReassemblyBuffer helpers ───────────────────────────────────────────

    #[test]
    fn test_reassembly_buffer_not_complete_initially() {
        let buf = PfaReassemblyBuffer::new(1, 3, 0);
        assert!(!buf.is_complete());
        assert_eq!(buf.received_count(), 0);
    }

    #[test]
    fn test_total_fragments_mismatch_error() {
        let mut asm = make_assembler_cfg(10, 30, false);
        // First fragment claims total=2
        let f1 = PfaFragment {
            msg_id: 99,
            fragment_index: 0,
            total_fragments: 2,
            offset: 0,
            data: vec![0u8; 5],
            checksum: 0,
        };
        // Second fragment claims total=3 (mismatch)
        let f2 = PfaFragment {
            msg_id: 99,
            fragment_index: 1,
            total_fragments: 3,
            offset: 5,
            data: vec![0u8; 5],
            checksum: 0,
        };
        asm.receive_fragment(f1, 0);
        let result = asm.receive_fragment(f2, 0);
        assert!(matches!(result, PfaReceiveResult::Error(_)));
    }

    // ── Multi-message concurrency (sequential) ────────────────────────────────

    #[test]
    fn test_multiple_messages_in_flight() {
        let mut asm = make_assembler_cfg(8, 60, true);
        let payload_a: Vec<u8> = (0..24u8).collect();
        let payload_b: Vec<u8> = (0..16u8).map(|x| x * 2).collect();

        let frags_a = asm
            .fragment(100, &payload_a)
            .expect("test: multiple messages in flight frags_a should succeed"); // 3 frags
        let frags_b = asm
            .fragment(200, &payload_b)
            .expect("test: multiple messages in flight frags_b should succeed"); // 2 frags

        // Interleave receipt
        asm.receive_fragment(frags_a[0].clone(), 0);
        asm.receive_fragment(frags_b[0].clone(), 0);
        asm.receive_fragment(frags_a[1].clone(), 0);
        asm.receive_fragment(frags_b[1].clone(), 0);
        let result_a = asm.receive_fragment(frags_a[2].clone(), 0);

        assert_eq!(asm.pending_count(), 0); // both assembled
        assert!(matches!(result_a, PfaReceiveResult::Assembled(d) if d == payload_a));
    }

    #[test]
    fn test_stats_type_alias() {
        let stats: PfaAssemblerStats = PfaAssemblerStats::default();
        assert_eq!(stats.total_fragmented, 0);
    }

    #[test]
    fn test_type_alias_packet_fragmentation_assembler() {
        let _asm: PfaPacketFragmentationAssembler = PacketFragmentationAssembler::with_defaults();
    }
}
