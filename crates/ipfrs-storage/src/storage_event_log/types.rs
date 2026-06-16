//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::functions::event_checksum;

/// The storage operation that generated a [`SelStorageEvent`].
///
/// *Note*: `EventType` in this crate collides with the simpler `EventType`
/// re-exported from `event_log.rs`, so this type is named `SelEventType`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelEventType {
    /// Object was first written to storage.
    Created,
    /// Object was read.
    Read,
    /// Object content was updated in place.
    Updated,
    /// Object was permanently deleted.
    Deleted,
    /// Object was copied to a new location.
    Copied {
        /// Destination object identifier.
        dest: String,
    },
    /// Object was moved (rename / relocation).
    Moved {
        /// Destination object identifier.
        dest: String,
    },
    /// A new immutable version was captured.
    Versioned {
        /// Version number assigned.
        version: u32,
    },
    /// Object was compressed.
    Compressed,
    /// Object was encrypted.
    Encrypted,
    /// Object was migrated between storage tiers.
    Tiered {
        /// Source tier name.
        from: String,
        /// Destination tier name.
        to: String,
    },
    /// Marks the start of a batch of related operations.
    BatchStart(String),
    /// Marks the end of a batch of related operations.
    BatchEnd(String),
    /// A system-level error occurred during an operation.
    SystemError(String),
}
impl SelEventType {
    /// Return a canonical string name for the variant (used as an aggregation key).
    pub fn type_name(&self) -> &'static str {
        match self {
            SelEventType::Created => "Created",
            SelEventType::Read => "Read",
            SelEventType::Updated => "Updated",
            SelEventType::Deleted => "Deleted",
            SelEventType::Copied { .. } => "Copied",
            SelEventType::Moved { .. } => "Moved",
            SelEventType::Versioned { .. } => "Versioned",
            SelEventType::Compressed => "Compressed",
            SelEventType::Encrypted => "Encrypted",
            SelEventType::Tiered { .. } => "Tiered",
            SelEventType::BatchStart(_) => "BatchStart",
            SelEventType::BatchEnd(_) => "BatchEnd",
            SelEventType::SystemError(_) => "SystemError",
        }
    }
}
/// Monotonically increasing identifier for a [`SelStorageEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(pub u64);
/// Aggregate statistics over the entire event log.
///
/// *Note*: `EventLogStats` collides with the simpler `EventLogStats` exported from
/// `event_log.rs`, so this type is named `SelEventLogStats`.
#[derive(Debug, Clone)]
pub struct SelEventLogStats {
    /// Total number of events currently in the log.
    pub total_events: u64,
    /// Per-type event counts, sorted by type name.
    pub events_by_type: Vec<(String, u64)>,
    /// Sum of `size_bytes` across all events.
    pub total_bytes_logged: u64,
    /// Timestamp of the oldest retained event.
    pub oldest_event_ts: Option<u64>,
    /// Timestamp of the newest retained event.
    pub newest_event_ts: Option<u64>,
    /// Number of times retention was applied.
    pub retention_purges: u64,
}
/// A single, immutable audit record in the event log.
///
/// *Note*: `StorageEvent` collides with the simpler `StorageEvent` exported from
/// `event_log.rs`, so this type is named `SelStorageEvent`.
#[derive(Debug, Clone)]
pub struct SelStorageEvent {
    /// Unique, monotonically increasing identifier (starts at 1).
    pub id: EventId,
    /// The storage operation that was performed.
    pub event_type: SelEventType,
    /// Identifier of the object involved.
    pub object_id: String,
    /// Identifier of the user or service that triggered the operation.
    pub user_id: String,
    /// Wall-clock timestamp supplied by the caller (e.g. UNIX microseconds).
    pub timestamp: u64,
    /// Size of the object involved, in bytes.
    pub size_bytes: u64,
    /// Arbitrary key-value pairs for extra context.
    pub metadata: Vec<(String, String)>,
    /// FNV-1a 64-bit checksum over `id`, `object_id`, `user_id`, `timestamp`.
    pub checksum: u64,
    /// Optional caller-supplied identifier for grouping related events.
    pub correlation_id: Option<String>,
}
/// Aggregated statistics for one event type over a query window.
#[derive(Debug, Clone)]
pub struct EventAggregation {
    /// Canonical type name (e.g. `"Created"`, `"Tiered"`).
    pub event_type: String,
    /// Total number of matching events.
    pub count: u64,
    /// Sum of `size_bytes` across matching events.
    pub total_bytes: u64,
    /// Timestamp of the earliest matching event.
    pub first_seen: u64,
    /// Timestamp of the most recent matching event.
    pub last_seen: u64,
    /// Number of distinct `object_id` values.
    pub unique_objects: usize,
    /// Number of distinct `user_id` values.
    pub unique_users: usize,
}
/// Errors returned by [`SelStorageEventLog`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventLogError {
    /// No event with the requested [`EventId`] exists.
    EventNotFound(EventId),
    /// The query would scan more events than is practical.
    QueryTooExpensive {
        /// Estimated scan count.
        estimated: usize,
    },
    /// The retention policy could not be applied.
    RetentionError(String),
    /// The checksum of an event does not match its stored fields.
    CorruptedEvent(EventId),
}
/// Governs how events are pruned from the log.
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// Retain every event; never prune automatically.
    KeepAll,
    /// Keep only the most recent `n` events.
    KeepLast(usize),
    /// Keep events whose timestamp is within the last `duration` microseconds.
    KeepForDuration(u64),
    /// Keep at most `n` events per event-type name.
    KeepByType(HashMap<String, usize>),
    /// Prune oldest events once the total byte-size of all events exceeds this.
    SizeLimited(u64),
}
/// Bounded, append-only, queryable log of [`SelStorageEvent`]s.
pub struct SelStorageEventLog {
    pub(super) events: VecDeque<SelStorageEvent>,
    pub(super) next_id: u64,
    pub(super) config: EventLogConfig,
    /// Cumulative count of retention passes performed.
    pub(super) retention_purges: u64,
    /// Running count of events appended since last retention flush.
    pub(super) events_since_flush: usize,
}
impl SelStorageEventLog {
    /// Create a new event log with the supplied [`EventLogConfig`].
    pub fn new(config: EventLogConfig) -> Self {
        let capacity = if config.max_events > 0 {
            config.max_events.min(1_000_000)
        } else {
            4096
        };
        Self {
            events: VecDeque::with_capacity(capacity),
            next_id: 1,
            config,
            retention_purges: 0,
            events_since_flush: 0,
        }
    }
    /// Create a log with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EventLogConfig::default())
    }
    /// Append a new event and return its [`EventId`].
    ///
    /// Assigns a monotonically increasing id, computes the FNV-1a checksum
    /// (when `config.enable_checksums` is `true`), and appends the event.
    /// If `config.max_events > 0` and the log is at capacity, the oldest
    /// event is evicted first.  After every `batch_flush_size` appends,
    /// [`apply_retention`](Self::apply_retention) is called automatically.
    #[allow(clippy::too_many_arguments)]
    pub fn log(
        &mut self,
        event_type: SelEventType,
        object_id: String,
        user_id: String,
        size_bytes: u64,
        metadata: Vec<(String, String)>,
        correlation_id: Option<String>,
        current_ts: u64,
    ) -> EventId {
        let id = EventId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        let mut event = SelStorageEvent {
            id,
            event_type,
            object_id,
            user_id,
            timestamp: current_ts,
            size_bytes,
            metadata,
            checksum: 0,
            correlation_id,
        };
        if self.config.enable_checksums {
            event.checksum = event_checksum(&event);
        }
        if self.config.max_events > 0 && self.events.len() >= self.config.max_events {
            self.events.pop_front();
        }
        self.events.push_back(event);
        self.events_since_flush = self.events_since_flush.saturating_add(1);
        if self.config.batch_flush_size > 0
            && self.events_since_flush >= self.config.batch_flush_size
        {
            self.events_since_flush = 0;
            let _ = self.apply_retention(current_ts);
        }
        id
    }
    /// Return a reference to the event with the given [`EventId`].
    pub fn get(&self, id: EventId) -> Result<&SelStorageEvent, EventLogError> {
        self.events
            .iter()
            .find(|e| e.id == id)
            .ok_or(EventLogError::EventNotFound(id))
    }
    /// Return references to events matching the [`EventQuery`], respecting
    /// `offset` and `limit`.
    pub fn query<'a>(&'a self, q: &EventQuery) -> Result<Vec<&'a SelStorageEvent>, EventLogError> {
        let scan_bound = 1_000_000;
        if self.events.len() > scan_bound
            && q.event_types.is_empty()
            && q.object_id.is_none()
            && q.user_id.is_none()
            && q.time_range.is_none()
            && q.correlation_id.is_none()
            && q.limit == 0
        {
            return Err(EventLogError::QueryTooExpensive {
                estimated: self.events.len(),
            });
        }
        let matching = self.events.iter().filter(|e| q.matches(e)).skip(q.offset);
        let results: Vec<&SelStorageEvent> = if q.limit > 0 {
            matching.take(q.limit).collect()
        } else {
            matching.collect()
        };
        Ok(results)
    }
    /// Compute per-type aggregation statistics over a time range.
    ///
    /// `event_types` is a slice of variants to include; an empty slice means
    /// all types.  `time_range` is `[start, end]` in the same units as
    /// `timestamp`; `None` means no time constraint.
    pub fn aggregate(
        &self,
        event_types: &[SelEventType],
        time_range: Option<(u64, u64)>,
    ) -> Vec<EventAggregation> {
        struct Bucket {
            count: u64,
            total_bytes: u64,
            first_seen: u64,
            last_seen: u64,
            objects: std::collections::HashSet<String>,
            users: std::collections::HashSet<String>,
        }
        let mut buckets: HashMap<String, Bucket> = HashMap::new();
        for event in &self.events {
            if let Some((start, end)) = time_range {
                if event.timestamp < start || event.timestamp > end {
                    continue;
                }
            }
            if !event_types.is_empty()
                && !event_types
                    .iter()
                    .any(|t| std::mem::discriminant(t) == std::mem::discriminant(&event.event_type))
            {
                continue;
            }
            let key = event.event_type.type_name().to_string();
            let bucket = buckets.entry(key).or_insert_with(|| Bucket {
                count: 0,
                total_bytes: 0,
                first_seen: event.timestamp,
                last_seen: event.timestamp,
                objects: std::collections::HashSet::new(),
                users: std::collections::HashSet::new(),
            });
            bucket.count += 1;
            bucket.total_bytes = bucket.total_bytes.saturating_add(event.size_bytes);
            if event.timestamp < bucket.first_seen {
                bucket.first_seen = event.timestamp;
            }
            if event.timestamp > bucket.last_seen {
                bucket.last_seen = event.timestamp;
            }
            bucket.objects.insert(event.object_id.clone());
            bucket.users.insert(event.user_id.clone());
        }
        let mut result: Vec<EventAggregation> = buckets
            .into_iter()
            .map(|(type_name, b)| EventAggregation {
                event_type: type_name,
                count: b.count,
                total_bytes: b.total_bytes,
                first_seen: b.first_seen,
                last_seen: b.last_seen,
                unique_objects: b.objects.len(),
                unique_users: b.users.len(),
            })
            .collect();
        result.sort_by(|a, b| a.event_type.cmp(&b.event_type));
        result
    }
    /// Return all events with the given correlation identifier.
    pub fn correlate(&self, correlation_id: &str) -> Vec<&SelStorageEvent> {
        self.events
            .iter()
            .filter(|e| e.correlation_id.as_deref() == Some(correlation_id))
            .collect()
    }
    /// Recompute checksums for all events and return the [`EventId`]s of any
    /// whose stored checksum does not match.
    ///
    /// When `config.enable_checksums` is `false`, no events are considered
    /// corrupted (returns an empty `Vec`).
    pub fn verify_integrity(&self) -> Vec<EventId> {
        if !self.config.enable_checksums {
            return Vec::new();
        }
        self.events
            .iter()
            .filter(|e| event_checksum(e) != e.checksum)
            .map(|e| e.id)
            .collect()
    }
    /// Apply the configured retention policy and return the number of events
    /// pruned.
    pub fn apply_retention(&mut self, current_ts: u64) -> usize {
        let before = self.events.len();
        match &self.config.retention_policy.clone() {
            RetentionPolicy::KeepAll => {}
            RetentionPolicy::KeepLast(n) => {
                let keep = *n;
                while self.events.len() > keep {
                    self.events.pop_front();
                }
            }
            RetentionPolicy::KeepForDuration(duration_us) => {
                let cutoff = current_ts.saturating_sub(*duration_us);
                self.events.retain(|e| e.timestamp >= cutoff);
            }
            RetentionPolicy::KeepByType(limits) => {
                let mut type_counts: HashMap<&str, usize> = HashMap::new();
                let mut remove: std::collections::HashSet<EventId> =
                    std::collections::HashSet::new();
                for event in self.events.iter().rev() {
                    let type_name = event.event_type.type_name();
                    let count = type_counts.entry(type_name).or_insert(0);
                    *count += 1;
                    let limit = limits.get(type_name).copied().unwrap_or(usize::MAX);
                    if *count > limit {
                        remove.insert(event.id);
                    }
                }
                self.events.retain(|e| !remove.contains(&e.id));
            }
            RetentionPolicy::SizeLimited(max_bytes) => {
                let max = *max_bytes;
                const OVERHEAD: u64 = 256;
                let mut total: u64 = self
                    .events
                    .iter()
                    .map(|e| e.size_bytes.saturating_add(OVERHEAD))
                    .fold(0u64, |acc, b| acc.saturating_add(b));
                while total > max {
                    if let Some(front) = self.events.pop_front() {
                        total = total.saturating_sub(front.size_bytes.saturating_add(OVERHEAD));
                    } else {
                        break;
                    }
                }
            }
        }
        let purged = before.saturating_sub(self.events.len());
        if purged > 0 {
            self.retention_purges = self.retention_purges.saturating_add(1);
        }
        purged
    }
    /// Return cloned copies of all events matching `q` (for backup / export).
    ///
    /// Unlike [`query`](Self::query), this method always returns owned values.
    pub fn export(&self, q: &EventQuery) -> Vec<SelStorageEvent> {
        let matching = self.events.iter().filter(|e| q.matches(e)).skip(q.offset);
        if q.limit > 0 {
            matching.take(q.limit).cloned().collect()
        } else {
            matching.cloned().collect()
        }
    }
    /// Compute aggregate statistics over all currently retained events.
    pub fn stats(&self) -> SelEventLogStats {
        let mut type_counts: HashMap<&str, u64> = HashMap::new();
        let mut total_bytes: u64 = 0;
        let mut oldest: Option<u64> = None;
        let mut newest: Option<u64> = None;
        for event in &self.events {
            *type_counts.entry(event.event_type.type_name()).or_insert(0) += 1;
            total_bytes = total_bytes.saturating_add(event.size_bytes);
            oldest = Some(match oldest {
                None => event.timestamp,
                Some(o) => o.min(event.timestamp),
            });
            newest = Some(match newest {
                None => event.timestamp,
                Some(n) => n.max(event.timestamp),
            });
        }
        let mut events_by_type: Vec<(String, u64)> = type_counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        events_by_type.sort_by(|a, b| a.0.cmp(&b.0));
        SelEventLogStats {
            total_events: self.events.len() as u64,
            events_by_type,
            total_bytes_logged: total_bytes,
            oldest_event_ts: oldest,
            newest_event_ts: newest,
            retention_purges: self.retention_purges,
        }
    }
    /// Return the number of events currently in the log.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}
/// Predicate used to select a subset of events (previous API compatibility).
#[derive(Debug, Default, Clone)]
pub struct EventFilter {
    /// Match events whose kind name equals this string.
    pub kind_filter: Option<String>,
    /// Match events in this namespace.
    pub namespace_filter: Option<String>,
    /// Match events that reference this CID.
    pub cid_filter: Option<String>,
    /// Match events with `timestamp >= since_timestamp`.
    pub since_timestamp: Option<u64>,
    /// Match events with `timestamp <= until_timestamp`.
    pub until_timestamp: Option<u64>,
    /// Match events with this correlation ID.
    pub correlation_id_filter: Option<u64>,
}
/// Predicate for selecting a subset of events.
///
/// All non-`None` / non-empty fields must match; `None`/empty = unconstrained.
#[derive(Debug, Default, Clone)]
pub struct EventQuery {
    /// Restrict to events whose `event_type` is one of these variants.
    /// An empty `Vec` means *all types*.
    pub event_types: Vec<SelEventType>,
    /// Restrict to events for this object identifier.
    pub object_id: Option<String>,
    /// Restrict to events for this user identifier.
    pub user_id: Option<String>,
    /// Restrict to events whose timestamp falls in `[start, end]`.
    pub time_range: Option<(u64, u64)>,
    /// Restrict to events with this correlation identifier.
    pub correlation_id: Option<String>,
    /// Maximum number of matching events to return (`0` = no limit).
    pub limit: usize,
    /// Number of matching events to skip before collecting results.
    pub offset: usize,
}
impl EventQuery {
    /// Return `true` if `event` satisfies every non-`None`/non-empty constraint.
    pub(super) fn matches(&self, event: &SelStorageEvent) -> bool {
        if !self.event_types.is_empty()
            && !self
                .event_types
                .iter()
                .any(|t| std::mem::discriminant(t) == std::mem::discriminant(&event.event_type))
        {
            return false;
        }
        if let Some(ref oid) = self.object_id {
            if &event.object_id != oid {
                return false;
            }
        }
        if let Some(ref uid) = self.user_id {
            if &event.user_id != uid {
                return false;
            }
        }
        if let Some((start, end)) = self.time_range {
            if event.timestamp < start || event.timestamp > end {
                return false;
            }
        }
        if let Some(ref cid) = self.correlation_id {
            if event.correlation_id.as_deref() != Some(cid.as_str()) {
                return false;
            }
        }
        true
    }
}
/// The specific storage operation kind (previous `StorageEventKind` API).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageEventKind {
    /// A block was written to storage.
    Put {
        /// Content identifier of the block.
        cid: String,
        /// Size of the block in bytes.
        size_bytes: u64,
    },
    /// A block was read from storage.
    Get {
        /// Content identifier of the block.
        cid: String,
        /// Whether the block was found in cache.
        hit: bool,
    },
    /// A block was explicitly deleted.
    Delete {
        /// Content identifier of the deleted block.
        cid: String,
        /// Size of the deleted block in bytes.
        size_bytes: u64,
    },
    /// A block was evicted (e.g. by an LRU policy).
    Evict {
        /// Content identifier of the evicted block.
        cid: String,
        /// Human-readable reason for eviction.
        reason: String,
        /// Size of the evicted block in bytes.
        size_bytes: u64,
    },
    /// A block was replicated to another node.
    Replicate {
        /// Content identifier of the replicated block.
        cid: String,
        /// Address or name of the destination node.
        target_node: String,
    },
    /// Integrity of a block was verified.
    Verify {
        /// Content identifier of the verified block.
        cid: String,
        /// Whether the verification check passed.
        passed: bool,
    },
    /// A compaction cycle was completed.
    Compact {
        /// Number of bytes freed during compaction.
        freed_bytes: u64,
        /// Duration of the compaction cycle in milliseconds.
        duration_ms: u64,
    },
    /// A block was migrated between storage tiers.
    Migrate {
        /// Content identifier of the migrated block.
        cid: String,
        /// Source storage tier name.
        from_tier: String,
        /// Destination storage tier name.
        to_tier: String,
    },
}
impl StorageEventKind {
    /// Return the variant name as a static string slice.
    pub fn kind_name(&self) -> &'static str {
        match self {
            StorageEventKind::Put { .. } => "Put",
            StorageEventKind::Get { .. } => "Get",
            StorageEventKind::Delete { .. } => "Delete",
            StorageEventKind::Evict { .. } => "Evict",
            StorageEventKind::Replicate { .. } => "Replicate",
            StorageEventKind::Verify { .. } => "Verify",
            StorageEventKind::Compact { .. } => "Compact",
            StorageEventKind::Migrate { .. } => "Migrate",
        }
    }
}
/// Configuration for a [`SelStorageEventLog`].
#[derive(Debug, Clone)]
pub struct EventLogConfig {
    /// Hard upper bound on the number of in-memory events (0 = unlimited).
    pub max_events: usize,
    /// Policy applied during [`SelStorageEventLog::apply_retention`].
    pub retention_policy: RetentionPolicy,
    /// When `true`, checksums are computed and stored for every new event.
    pub enable_checksums: bool,
    /// Number of new events that trigger an automatic retention check.
    pub batch_flush_size: usize,
}
