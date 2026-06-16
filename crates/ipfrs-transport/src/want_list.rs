//! Enhanced want list management with priority queue
//!
//! Implements efficient priority-based block request scheduling with:
//! - Sub-microsecond priority updates
//! - CID deduplication
//! - Configurable timeouts with automatic cleanup
//! - Dynamic priority adjustment
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{WantList, WantListConfig, Priority};
//! use ipfrs_core::Cid;
//! use multihash::Multihash;
//!
//! // Create a want list with default configuration
//! let config = WantListConfig::default();
//! let mut want_list = WantList::new(config);
//!
//! // Create a test CID
//! let hash = Multihash::wrap(0x12, &[1, 2, 3, 4]).expect("test: wrapping valid bytes into multihash");
//! let cid = Cid::new_v1(0x55, hash);
//!
//! // Add a block request with normal priority
//! want_list.add_simple(cid.clone(), Priority::Normal as i32);
//!
//! // Update priority to high
//! want_list.update_priority(&cid, Priority::High as i32);
//!
//! // Get the highest priority block to request
//! if let Some(entry) = want_list.pop() {
//!     println!("Requesting block: {:?}", entry.cid);
//! }
//! ```

use ipfrs_core::Cid;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Want list configuration
#[derive(Debug, Clone)]
pub struct WantListConfig {
    /// Maximum number of concurrent wants
    pub max_wants: usize,
    /// Default timeout for want requests
    pub default_timeout: Duration,
    /// Maximum retry count
    pub max_retries: u32,
    /// Base delay for exponential backoff
    pub base_retry_delay: Duration,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
}

impl Default for WantListConfig {
    fn default() -> Self {
        Self {
            max_wants: 1024,
            default_timeout: Duration::from_secs(30),
            max_retries: 3,
            base_retry_delay: Duration::from_millis(100),
            max_retry_delay: Duration::from_secs(10),
        }
    }
}

/// Priority levels for block requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Background prefetch
    Low = 0,
    /// Normal request
    Normal = 50,
    /// Important for computation
    High = 100,
    /// Urgent - blocking computation
    Urgent = 200,
    /// Critical - past deadline
    Critical = 300,
}

impl From<i32> for Priority {
    fn from(value: i32) -> Self {
        match value {
            v if v >= 300 => Priority::Critical,
            v if v >= 200 => Priority::Urgent,
            v if v >= 100 => Priority::High,
            v if v >= 50 => Priority::Normal,
            _ => Priority::Low,
        }
    }
}

impl From<Priority> for i32 {
    fn from(value: Priority) -> Self {
        value as i32
    }
}

/// Want entry with metadata for scheduling
#[derive(Debug, Clone)]
pub struct WantEntry {
    /// CID of wanted block
    pub cid: Cid,
    /// Priority level
    pub priority: i32,
    /// When this want was created
    pub created_at: Instant,
    /// When this want expires
    pub expires_at: Instant,
    /// Number of retries attempted
    pub retry_count: u32,
    /// If this is a retry, when was the last attempt
    pub last_attempt: Option<Instant>,
    /// Send DONT_HAVE if unavailable
    pub send_dont_have: bool,
    /// User-defined deadline
    pub deadline: Option<Instant>,
    /// Request tag for grouping related requests
    pub tag: Option<String>,
}

impl WantEntry {
    /// Create a new want entry
    pub fn new(cid: Cid, priority: i32, timeout: Duration) -> Self {
        let now = Instant::now();
        Self {
            cid,
            priority,
            created_at: now,
            expires_at: now + timeout,
            retry_count: 0,
            last_attempt: None,
            send_dont_have: false,
            deadline: None,
            tag: None,
        }
    }

    /// Set deadline for priority elevation
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set tag for request grouping
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tag = Some(tag);
        self
    }

    /// Enable DONT_HAVE response
    pub fn with_dont_have(mut self) -> Self {
        self.send_dont_have = true;
        self
    }

    /// Check if this want has expired
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Check if we can retry this want
    pub fn can_retry(&self, max_retries: u32) -> bool {
        self.retry_count < max_retries
    }

    /// Calculate effective priority considering deadline
    pub fn effective_priority(&self) -> i32 {
        let mut priority = self.priority;

        // Boost priority if deadline is approaching
        if let Some(deadline) = self.deadline {
            let now = Instant::now();
            if now >= deadline {
                // Past deadline - critical priority
                priority = priority.max(Priority::Critical as i32);
            } else {
                let time_left = deadline.duration_since(now);
                if time_left < Duration::from_secs(1) {
                    priority = priority.max(Priority::Urgent as i32);
                } else if time_left < Duration::from_secs(5) {
                    priority = priority.max(Priority::High as i32);
                }
            }
        }

        priority
    }
}

/// Priority queue entry for the heap
#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry {
    cid: Cid,
    priority: i32,
    created_at: Instant,
    /// Version number for invalidation
    version: u64,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first
        self.priority
            .cmp(&other.priority)
            // Then earlier creation time
            .then_with(|| other.created_at.cmp(&self.created_at))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Enhanced want list with priority queue
pub struct WantList {
    /// Priority queue of wants
    heap: BinaryHeap<HeapEntry>,
    /// Map from CID to entry (for O(1) lookup and deduplication)
    entries: HashMap<Cid, (WantEntry, u64)>,
    /// Current version counter (for lazy deletion)
    version_counter: u64,
    /// Configuration
    config: WantListConfig,
}

impl WantList {
    /// Create a new want list with configuration
    pub fn new(config: WantListConfig) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(config.max_wants),
            entries: HashMap::with_capacity(config.max_wants),
            version_counter: 0,
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(WantListConfig::default())
    }

    /// Add a CID to the want list
    ///
    /// Returns true if added, false if already present or list full
    pub fn add(&mut self, entry: WantEntry) -> bool {
        // Check if already wanted
        if self.entries.contains_key(&entry.cid) {
            return false;
        }

        // Check capacity
        if self.entries.len() >= self.config.max_wants {
            return false;
        }

        let cid = entry.cid;
        let priority = entry.effective_priority();
        let created_at = entry.created_at;

        self.version_counter += 1;
        let version = self.version_counter;

        self.entries.insert(cid, (entry, version));
        self.heap.push(HeapEntry {
            cid,
            priority,
            created_at,
            version,
        });

        true
    }

    /// Add with simple parameters
    pub fn add_simple(&mut self, cid: Cid, priority: i32) -> bool {
        let entry = WantEntry::new(cid, priority, self.config.default_timeout);
        self.add(entry)
    }

    /// Remove a CID from the want list
    ///
    /// Uses lazy deletion - entry is invalidated but not removed from heap
    pub fn remove(&mut self, cid: &Cid) -> Option<WantEntry> {
        self.entries.remove(cid).map(|(entry, _)| entry)
    }

    /// Update priority for a CID
    ///
    /// Achieves sub-microsecond updates by using lazy re-insertion
    pub fn update_priority(&mut self, cid: &Cid, new_priority: i32) -> bool {
        if let Some((entry, _old_version)) = self.entries.get_mut(cid) {
            entry.priority = new_priority;

            // Create new heap entry with updated priority
            self.version_counter += 1;
            let version = self.version_counter;
            let created_at = entry.created_at;

            // Update version in entries map
            if let Some((_, v)) = self.entries.get_mut(cid) {
                *v = version;
            }

            self.heap.push(HeapEntry {
                cid: *cid,
                priority: new_priority,
                created_at,
                version,
            });

            true
        } else {
            false
        }
    }

    /// Boost priority for deadline-approaching entries
    pub fn boost_deadline_priorities(&mut self) {
        let now = Instant::now();
        let mut updates = Vec::new();

        for (cid, (entry, _)) in &self.entries {
            if let Some(deadline) = entry.deadline {
                let effective = entry.effective_priority();
                if effective > entry.priority {
                    updates.push((*cid, effective));
                }

                // Check if deadline passed
                if now >= deadline && entry.priority < Priority::Critical as i32 {
                    updates.push((*cid, Priority::Critical as i32));
                }
            }
        }

        for (cid, priority) in updates {
            self.update_priority(&cid, priority);
        }
    }

    /// Get the highest priority want
    pub fn pop(&mut self) -> Option<WantEntry> {
        loop {
            let heap_entry = self.heap.pop()?;

            // Check if entry is still valid
            if let Some((_entry, version)) = self.entries.get(&heap_entry.cid) {
                if *version == heap_entry.version {
                    // Entry is valid, remove and return
                    return self.entries.remove(&heap_entry.cid).map(|(e, _)| e);
                }
                // Entry has been updated, continue to next
            }
            // Entry was removed, continue to next
        }
    }

    /// Peek at the highest priority want without removing
    pub fn peek(&self) -> Option<&WantEntry> {
        // Note: This may return stale entry due to lazy deletion
        // For accurate peek, would need to rebuild heap
        self.entries
            .values()
            .max_by_key(|(e, _)| e.effective_priority())
            .map(|(e, _)| e)
    }

    /// Check if a CID is wanted
    pub fn contains(&self, cid: &Cid) -> bool {
        self.entries.contains_key(cid)
    }

    /// Get entry for a CID
    pub fn get(&self, cid: &Cid) -> Option<&WantEntry> {
        self.entries.get(cid).map(|(e, _)| e)
    }

    /// Get mutable entry for a CID
    pub fn get_mut(&mut self, cid: &Cid) -> Option<&mut WantEntry> {
        self.entries.get_mut(cid).map(|(e, _)| e)
    }

    /// Number of wants in the list
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all expired entries
    pub fn cleanup_expired(&mut self) -> Vec<WantEntry> {
        let mut expired = Vec::new();
        let cids_to_remove: Vec<Cid> = self
            .entries
            .iter()
            .filter(|(_, (entry, _))| entry.is_expired())
            .map(|(cid, _)| *cid)
            .collect();

        for cid in cids_to_remove {
            if let Some(entry) = self.remove(&cid) {
                expired.push(entry);
            }
        }

        expired
    }

    /// Get entries that should be retried
    pub fn get_retry_candidates(&self) -> Vec<Cid> {
        self.entries
            .iter()
            .filter(|(_, (entry, _))| {
                entry.can_retry(self.config.max_retries) && entry.last_attempt.is_some()
            })
            .map(|(cid, _)| *cid)
            .collect()
    }

    /// Mark an entry as attempted (for retry logic)
    pub fn mark_attempted(&mut self, cid: &Cid) {
        if let Some((entry, _)) = self.entries.get_mut(cid) {
            entry.retry_count += 1;
            entry.last_attempt = Some(Instant::now());
        }
    }

    /// Calculate retry delay with exponential backoff and jitter
    pub fn retry_delay(&self, retry_count: u32) -> Duration {
        let base = self.config.base_retry_delay.as_millis() as u64;
        let max = self.config.max_retry_delay.as_millis() as u64;

        // Exponential backoff: base * 2^retry_count
        let delay_ms = base.saturating_mul(1 << retry_count.min(10));
        let delay_ms = delay_ms.min(max);

        // Add jitter (10% random variation)
        let jitter = (delay_ms / 10) as i64;
        let jitter_offset = if jitter > 0 {
            // Simple pseudo-random using current time
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            (now % (jitter * 2)) - jitter
        } else {
            0
        };

        Duration::from_millis((delay_ms as i64 + jitter_offset).max(0) as u64)
    }

    /// Get all CIDs in the want list
    pub fn cids(&self) -> Vec<Cid> {
        self.entries.keys().copied().collect()
    }

    /// Get all entries sorted by priority
    pub fn entries_by_priority(&self) -> Vec<&WantEntry> {
        let mut entries: Vec<_> = self.entries.values().map(|(e, _)| e).collect();
        entries.sort_by_key(|b| std::cmp::Reverse(b.effective_priority()));
        entries
    }

    /// Get entries with specific tag
    pub fn entries_with_tag(&self, tag: &str) -> Vec<&WantEntry> {
        self.entries
            .values()
            .filter_map(|(e, _)| {
                if e.tag.as_deref() == Some(tag) {
                    Some(e)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.heap.clear();
        self.entries.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &WantListConfig {
        &self.config
    }

    /// Add multiple entries in batch for better performance
    ///
    /// Returns the number of successfully added entries
    pub fn add_batch(&mut self, entries: &[(Cid, i32)]) -> usize {
        let mut added = 0;
        for (cid, priority) in entries {
            if self.add_simple(*cid, *priority) {
                added += 1;
            }
        }
        added
    }

    /// Add multiple CIDs with the same priority in batch
    ///
    /// Returns the number of successfully added entries
    pub fn add_batch_same_priority(&mut self, cids: &[Cid], priority: i32) -> usize {
        let mut added = 0;
        for cid in cids {
            if self.add_simple(*cid, priority) {
                added += 1;
            }
        }
        added
    }

    /// Remove multiple entries in batch
    ///
    /// Returns the removed entries
    pub fn remove_batch(&mut self, cids: &[Cid]) -> Vec<WantEntry> {
        let mut removed = Vec::with_capacity(cids.len());
        for cid in cids {
            if let Some(entry) = self.remove(cid) {
                removed.push(entry);
            }
        }
        removed
    }

    /// Update priorities for multiple CIDs in batch
    ///
    /// Returns the number of successfully updated entries
    pub fn update_priorities_batch(&mut self, updates: &[(Cid, i32)]) -> usize {
        let mut updated = 0;
        for (cid, priority) in updates {
            if self.update_priority(cid, *priority) {
                updated += 1;
            }
        }
        updated
    }

    /// Check if any of the given CIDs are present
    pub fn contains_any(&self, cids: &[Cid]) -> bool {
        cids.iter().any(|cid| self.contains(cid))
    }

    /// Check if all of the given CIDs are present
    pub fn contains_all(&self, cids: &[Cid]) -> bool {
        cids.iter().all(|cid| self.contains(cid))
    }

    /// Get multiple entries by CID
    ///
    /// Returns only the entries that exist
    pub fn get_batch(&self, cids: &[Cid]) -> Vec<&WantEntry> {
        cids.iter().filter_map(|cid| self.get(cid)).collect()
    }
}

/// Thread-safe want list wrapper
pub struct ConcurrentWantList {
    inner: Arc<RwLock<WantList>>,
}

impl ConcurrentWantList {
    /// Create a new concurrent want list
    pub fn new(config: WantListConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(WantList::new(config))),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(WantListConfig::default())
    }

    /// Add an entry to the want list
    pub fn add(&self, entry: WantEntry) -> bool {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .add(entry)
    }

    /// Add with simple parameters
    pub fn add_simple(&self, cid: Cid, priority: i32) -> bool {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .add_simple(cid, priority)
    }

    /// Remove an entry
    pub fn remove(&self, cid: &Cid) -> Option<WantEntry> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(cid)
    }

    /// Update priority
    pub fn update_priority(&self, cid: &Cid, new_priority: i32) -> bool {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .update_priority(cid, new_priority)
    }

    /// Pop highest priority entry
    pub fn pop(&self) -> Option<WantEntry> {
        self.inner.write().unwrap_or_else(|e| e.into_inner()).pop()
    }

    /// Check if CID is wanted
    pub fn contains(&self, cid: &Cid) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(cid)
    }

    /// Get number of wants
    pub fn len(&self) -> usize {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Cleanup expired entries
    pub fn cleanup_expired(&self) -> Vec<WantEntry> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .cleanup_expired()
    }

    /// Boost deadline priorities
    pub fn boost_deadline_priorities(&self) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .boost_deadline_priorities()
    }

    /// Get all CIDs
    pub fn cids(&self) -> Vec<Cid> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).cids()
    }

    /// Mark entry as attempted
    pub fn mark_attempted(&self, cid: &Cid) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mark_attempted(cid)
    }

    /// Get retry delay
    pub fn retry_delay(&self, retry_count: u32) -> Duration {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .retry_delay(retry_count)
    }

    /// Clone the inner Arc
    pub fn clone_inner(&self) -> Arc<RwLock<WantList>> {
        Arc::clone(&self.inner)
    }

    /// Add multiple entries in batch for better performance
    ///
    /// Returns the number of successfully added entries
    pub fn add_batch(&self, entries: &[(Cid, i32)]) -> usize {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .add_batch(entries)
    }

    /// Add multiple CIDs with the same priority in batch
    ///
    /// Returns the number of successfully added entries
    pub fn add_batch_same_priority(&self, cids: &[Cid], priority: i32) -> usize {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .add_batch_same_priority(cids, priority)
    }

    /// Remove multiple entries in batch
    ///
    /// Returns the removed entries
    pub fn remove_batch(&self, cids: &[Cid]) -> Vec<WantEntry> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove_batch(cids)
    }

    /// Update priorities for multiple CIDs in batch
    ///
    /// Returns the number of successfully updated entries
    pub fn update_priorities_batch(&self, updates: &[(Cid, i32)]) -> usize {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .update_priorities_batch(updates)
    }

    /// Check if any of the given CIDs are present
    pub fn contains_any(&self, cids: &[Cid]) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_any(cids)
    }

    /// Check if all of the given CIDs are present
    pub fn contains_all(&self, cids: &[Cid]) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_all(cids)
    }

    /// Get multiple entries by CID
    ///
    /// Returns only the entries that exist (note: returns clones for thread safety)
    pub fn get_batch(&self, cids: &[Cid]) -> Vec<WantEntry> {
        let lock = self.inner.read().unwrap_or_else(|e| e.into_inner());
        cids.iter()
            .filter_map(|cid| lock.get(cid).cloned())
            .collect()
    }
}

impl Clone for ConcurrentWantList {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use multihash::Multihash;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: parse known-good CID string")
    }

    fn test_cid2() -> Cid {
        "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"
            .parse()
            .expect("test: parse known-good CID string")
    }

    #[test]
    fn test_want_list_add_remove() {
        let mut list = WantList::with_defaults();
        let cid = test_cid();

        assert!(list.add_simple(cid, 50));
        assert!(list.contains(&cid));
        assert_eq!(list.len(), 1);

        // Duplicate add should fail
        assert!(!list.add_simple(cid, 100));

        // Remove
        let entry = list.remove(&cid);
        assert!(entry.is_some());
        assert!(!list.contains(&cid));
    }

    #[test]
    fn test_priority_ordering() {
        let mut list = WantList::with_defaults();
        let cid1 = test_cid();
        let cid2 = test_cid2();

        list.add_simple(cid1, 10);
        list.add_simple(cid2, 100);

        // Higher priority should come first
        let first = list
            .pop()
            .expect("test: pop highest-priority entry from non-empty want list");
        assert_eq!(first.cid, cid2);
        assert_eq!(first.priority, 100);

        let second = list.pop().expect("test: pop second entry from want list");
        assert_eq!(second.cid, cid1);
    }

    #[test]
    fn test_priority_update() {
        let mut list = WantList::with_defaults();
        let cid1 = test_cid();
        let cid2 = test_cid2();

        list.add_simple(cid1, 10);
        list.add_simple(cid2, 20);

        // Update cid1 to higher priority
        assert!(list.update_priority(&cid1, 100));

        // Now cid1 should come first
        let first = list
            .pop()
            .expect("test: pop updated-priority entry from non-empty want list");
        assert_eq!(first.cid, cid1);
    }

    #[test]
    fn test_deadline_priority_boost() {
        let mut list = WantList::with_defaults();
        let cid = test_cid();

        // Create entry with imminent deadline
        let entry = WantEntry::new(cid, Priority::Low as i32, Duration::from_secs(60))
            .with_deadline(Instant::now() + Duration::from_millis(100));

        list.add(entry);

        // Wait for deadline to approach
        std::thread::sleep(Duration::from_millis(50));

        // Effective priority should be boosted
        let entry = list
            .get(&cid)
            .expect("test: get entry that was just added to want list");
        assert!(entry.effective_priority() > Priority::Low as i32);
    }

    #[test]
    fn test_retry_delay_exponential_backoff() {
        let list = WantList::with_defaults();

        let delay0 = list.retry_delay(0);
        let delay1 = list.retry_delay(1);
        let delay2 = list.retry_delay(2);

        // Each delay should be roughly double the previous
        assert!(delay1 >= delay0);
        assert!(delay2 >= delay1);
        assert!(delay2 <= list.config.max_retry_delay);
    }

    #[test]
    fn test_concurrent_want_list() {
        let list = ConcurrentWantList::with_defaults();
        let cid = test_cid();

        assert!(list.add_simple(cid, 50));
        assert!(list.contains(&cid));
        assert_eq!(list.len(), 1);

        let entry = list.pop();
        assert!(entry.is_some());
        assert!(list.is_empty());
    }

    #[test]
    fn test_add_batch() {
        let mut list = WantList::with_defaults();

        let cids: Vec<_> = (0u64..10)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        let entries: Vec<_> = cids.iter().map(|cid| (*cid, 100)).collect();
        let added = list.add_batch(&entries);

        assert_eq!(added, 10);
        assert_eq!(list.len(), 10);

        // Try adding duplicates
        let added = list.add_batch(&entries);
        assert_eq!(added, 0); // No new entries added
        assert_eq!(list.len(), 10);
    }

    #[test]
    fn test_add_batch_same_priority() {
        let mut list = WantList::with_defaults();

        let cids: Vec<_> = (0u64..5)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        let added = list.add_batch_same_priority(&cids, 200);
        assert_eq!(added, 5);
        assert_eq!(list.len(), 5);

        // All should have the same priority
        for cid in &cids {
            let entry = list
                .get(cid)
                .expect("test: get entry that was batch-added to want list");
            assert_eq!(entry.priority, 200);
        }
    }

    #[test]
    fn test_remove_batch() {
        let mut list = WantList::with_defaults();

        let cids: Vec<_> = (0u64..8)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        for cid in &cids {
            list.add_simple(*cid, 100);
        }
        assert_eq!(list.len(), 8);

        let removed = list.remove_batch(&cids[0..4]);
        assert_eq!(removed.len(), 4);
        assert_eq!(list.len(), 4);

        for cid in &cids[0..4] {
            assert!(!list.contains(cid));
        }
        for cid in &cids[4..] {
            assert!(list.contains(cid));
        }
    }

    #[test]
    fn test_update_priorities_batch() {
        let mut list = WantList::with_defaults();

        let cids: Vec<_> = (0u64..6)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        for cid in &cids {
            list.add_simple(*cid, 100);
        }

        let updates: Vec<_> = cids
            .iter()
            .enumerate()
            .map(|(i, cid)| (*cid, 200 + i as i32))
            .collect();
        let updated = list.update_priorities_batch(&updates);
        assert_eq!(updated, 6);

        for (i, cid) in cids.iter().enumerate() {
            let entry = list
                .get(cid)
                .expect("test: get entry after batch priority update");
            assert_eq!(entry.priority, 200 + i as i32);
        }
    }

    #[test]
    fn test_contains_any_all() {
        let mut list = WantList::with_defaults();

        let cids: Vec<_> = (0u64..5)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        // Add only first 3
        for cid in &cids[0..3] {
            list.add_simple(*cid, 100);
        }

        assert!(list.contains_any(&cids)); // At least one is present
        assert!(!list.contains_all(&cids)); // Not all are present
        assert!(list.contains_all(&cids[0..3])); // First 3 are all present
        assert!(!list.contains_any(&cids[4..5])); // Last one not present
    }

    #[test]
    fn test_get_batch() {
        let mut list = WantList::with_defaults();

        let cids: Vec<_> = (0u64..7)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        // Add only even indices
        for (i, cid) in cids.iter().enumerate() {
            if i % 2 == 0 {
                list.add_simple(*cid, 100);
            }
        }

        let entries = list.get_batch(&cids);
        assert_eq!(entries.len(), 4); // 0, 2, 4, 6

        for entry in entries {
            assert_eq!(entry.priority, 100);
        }
    }

    #[test]
    fn test_concurrent_batch_operations() {
        let list = ConcurrentWantList::with_defaults();

        let cids: Vec<_> = (0u64..10)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash =
                    Multihash::wrap(0x12, &data).expect("test: wrap raw bytes into Multihash");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        // Test batch add
        let added = list.add_batch_same_priority(&cids, 150);
        assert_eq!(added, 10);
        assert_eq!(list.len(), 10);

        // Test contains
        assert!(list.contains_all(&cids));

        // Test batch update
        let updates: Vec<_> = cids.iter().map(|cid| (*cid, 250)).collect();
        let updated = list.update_priorities_batch(&updates);
        assert_eq!(updated, 10);

        // Test batch get
        let entries = list.get_batch(&cids);
        assert_eq!(entries.len(), 10);
        for entry in entries {
            assert_eq!(entry.priority, 250);
        }

        // Test batch remove
        let removed = list.remove_batch(&cids[0..5]);
        assert_eq!(removed.len(), 5);
        assert_eq!(list.len(), 5);
    }
}
