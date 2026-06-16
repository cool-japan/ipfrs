//! Metadata storage and filtering for hybrid search
//!
//! This module provides metadata management for vectors, enabling
//! hybrid search that combines vector similarity with attribute filtering.

use ipfrs_core::{Cid, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// Type aliases for complex index structures
type StringIndexMap = HashMap<String, HashMap<String, HashSet<Cid>>>;
type NumericIndexMap = HashMap<String, BTreeMap<i64, HashSet<Cid>>>;

/// Metadata value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Timestamp (Unix epoch seconds)
    Timestamp(u64),
    /// Array of strings
    StringArray(Vec<String>),
    /// Null value
    Null,
}

impl MetadataValue {
    /// Create a timestamp for now
    pub fn now() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        MetadataValue::Timestamp(timestamp)
    }

    /// Get as string if possible
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            MetadataValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(f) => Some(*f),
            MetadataValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Get as timestamp if possible
    pub fn as_timestamp(&self) -> Option<u64> {
        match self {
            MetadataValue::Timestamp(t) => Some(*t),
            MetadataValue::Integer(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }

    /// Get as boolean if possible
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            MetadataValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Metadata record for a vector
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Key-value pairs
    pub fields: HashMap<String, MetadataValue>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last updated timestamp
    pub updated_at: u64,
}

impl Metadata {
    /// Create new metadata with current timestamp
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            fields: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Set a field value
    pub fn set(&mut self, key: impl Into<String>, value: MetadataValue) -> &mut Self {
        self.fields.insert(key.into(), value);
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self
    }

    /// Get a field value
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.fields.get(key)
    }

    /// Check if a field exists
    pub fn has(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    /// Remove a field
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.fields.remove(key)
    }

    /// Builder method for string field
    pub fn with_string(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, MetadataValue::String(value.into()));
        self
    }

    /// Builder method for integer field
    pub fn with_integer(mut self, key: impl Into<String>, value: i64) -> Self {
        self.set(key, MetadataValue::Integer(value));
        self
    }

    /// Builder method for timestamp field
    pub fn with_timestamp(mut self, key: impl Into<String>, value: u64) -> Self {
        self.set(key, MetadataValue::Timestamp(value));
        self
    }

    /// Builder method for boolean field
    pub fn with_boolean(mut self, key: impl Into<String>, value: bool) -> Self {
        self.set(key, MetadataValue::Boolean(value));
        self
    }
}

/// Metadata filter expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataFilter {
    /// Equality check: field == value
    Equals(String, MetadataValue),
    /// Not equal: field != value
    NotEquals(String, MetadataValue),
    /// Greater than: field > value
    GreaterThan(String, MetadataValue),
    /// Greater than or equal: field >= value
    GreaterThanOrEqual(String, MetadataValue),
    /// Less than: field < value
    LessThan(String, MetadataValue),
    /// Less than or equal: field <= value
    LessThanOrEqual(String, MetadataValue),
    /// String contains: field contains substring
    Contains(String, String),
    /// String starts with: field starts with prefix
    StartsWith(String, String),
    /// String ends with: field ends with suffix
    EndsWith(String, String),
    /// Value in set: field in \[values\]
    In(String, Vec<MetadataValue>),
    /// Value not in set: field not in \[values\]
    NotIn(String, Vec<MetadataValue>),
    /// Field exists
    Exists(String),
    /// Field does not exist
    NotExists(String),
    /// Timestamp range: created_at or updated_at within range
    TimeRange {
        field: String,
        start: Option<u64>,
        end: Option<u64>,
    },
    /// Logical AND of multiple filters
    And(Vec<MetadataFilter>),
    /// Logical OR of multiple filters
    Or(Vec<MetadataFilter>),
    /// Logical NOT
    Not(Box<MetadataFilter>),
}

impl MetadataFilter {
    /// Create an equals filter
    pub fn eq(field: impl Into<String>, value: MetadataValue) -> Self {
        MetadataFilter::Equals(field.into(), value)
    }

    /// Create a not equals filter
    pub fn ne(field: impl Into<String>, value: MetadataValue) -> Self {
        MetadataFilter::NotEquals(field.into(), value)
    }

    /// Create a greater than filter
    pub fn gt(field: impl Into<String>, value: MetadataValue) -> Self {
        MetadataFilter::GreaterThan(field.into(), value)
    }

    /// Create a greater than or equal filter
    pub fn gte(field: impl Into<String>, value: MetadataValue) -> Self {
        MetadataFilter::GreaterThanOrEqual(field.into(), value)
    }

    /// Create a less than filter
    pub fn lt(field: impl Into<String>, value: MetadataValue) -> Self {
        MetadataFilter::LessThan(field.into(), value)
    }

    /// Create a less than or equal filter
    pub fn lte(field: impl Into<String>, value: MetadataValue) -> Self {
        MetadataFilter::LessThanOrEqual(field.into(), value)
    }

    /// Create a time range filter
    pub fn time_range(field: impl Into<String>, start: Option<u64>, end: Option<u64>) -> Self {
        MetadataFilter::TimeRange {
            field: field.into(),
            start,
            end,
        }
    }

    /// Create an AND filter
    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        MetadataFilter::And(filters)
    }

    /// Create an OR filter
    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        MetadataFilter::Or(filters)
    }

    /// Create a NOT filter
    pub fn negate(filter: MetadataFilter) -> Self {
        MetadataFilter::Not(Box::new(filter))
    }

    /// Evaluate the filter against metadata
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            MetadataFilter::Equals(field, value) => metadata.get(field) == Some(value),
            MetadataFilter::NotEquals(field, value) => metadata.get(field) != Some(value),
            MetadataFilter::GreaterThan(field, value) => {
                Self::compare_gt(metadata.get(field), value)
            }
            MetadataFilter::GreaterThanOrEqual(field, value) => {
                Self::compare_gte(metadata.get(field), value)
            }
            MetadataFilter::LessThan(field, value) => Self::compare_lt(metadata.get(field), value),
            MetadataFilter::LessThanOrEqual(field, value) => {
                Self::compare_lte(metadata.get(field), value)
            }
            MetadataFilter::Contains(field, substring) => metadata
                .get(field)
                .and_then(|v| v.as_string())
                .is_some_and(|s| s.contains(substring)),
            MetadataFilter::StartsWith(field, prefix) => metadata
                .get(field)
                .and_then(|v| v.as_string())
                .is_some_and(|s| s.starts_with(prefix)),
            MetadataFilter::EndsWith(field, suffix) => metadata
                .get(field)
                .and_then(|v| v.as_string())
                .is_some_and(|s| s.ends_with(suffix)),
            MetadataFilter::In(field, values) => {
                metadata.get(field).is_some_and(|v| values.contains(v))
            }
            MetadataFilter::NotIn(field, values) => {
                metadata.get(field).is_none_or(|v| !values.contains(v))
            }
            MetadataFilter::Exists(field) => metadata.has(field),
            MetadataFilter::NotExists(field) => !metadata.has(field),
            MetadataFilter::TimeRange { field, start, end } => {
                let timestamp = if field == "created_at" {
                    Some(metadata.created_at)
                } else if field == "updated_at" {
                    Some(metadata.updated_at)
                } else {
                    metadata.get(field).and_then(|v| v.as_timestamp())
                };

                timestamp.is_some_and(|t| {
                    let after_start = start.is_none_or(|s| t >= s);
                    let before_end = end.is_none_or(|e| t <= e);
                    after_start && before_end
                })
            }
            MetadataFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            MetadataFilter::Not(filter) => !filter.matches(metadata),
        }
    }

    fn compare_gt(field_value: Option<&MetadataValue>, compare_value: &MetadataValue) -> bool {
        match (field_value, compare_value) {
            (Some(MetadataValue::Integer(a)), MetadataValue::Integer(b)) => a > b,
            (Some(MetadataValue::Float(a)), MetadataValue::Float(b)) => a > b,
            (Some(MetadataValue::Integer(a)), MetadataValue::Float(b)) => (*a as f64) > *b,
            (Some(MetadataValue::Float(a)), MetadataValue::Integer(b)) => *a > (*b as f64),
            (Some(MetadataValue::Timestamp(a)), MetadataValue::Timestamp(b)) => a > b,
            (Some(MetadataValue::String(a)), MetadataValue::String(b)) => a > b,
            _ => false,
        }
    }

    fn compare_gte(field_value: Option<&MetadataValue>, compare_value: &MetadataValue) -> bool {
        match (field_value, compare_value) {
            (Some(MetadataValue::Integer(a)), MetadataValue::Integer(b)) => a >= b,
            (Some(MetadataValue::Float(a)), MetadataValue::Float(b)) => a >= b,
            (Some(MetadataValue::Integer(a)), MetadataValue::Float(b)) => (*a as f64) >= *b,
            (Some(MetadataValue::Float(a)), MetadataValue::Integer(b)) => *a >= (*b as f64),
            (Some(MetadataValue::Timestamp(a)), MetadataValue::Timestamp(b)) => a >= b,
            (Some(MetadataValue::String(a)), MetadataValue::String(b)) => a >= b,
            _ => false,
        }
    }

    fn compare_lt(field_value: Option<&MetadataValue>, compare_value: &MetadataValue) -> bool {
        match (field_value, compare_value) {
            (Some(MetadataValue::Integer(a)), MetadataValue::Integer(b)) => a < b,
            (Some(MetadataValue::Float(a)), MetadataValue::Float(b)) => a < b,
            (Some(MetadataValue::Integer(a)), MetadataValue::Float(b)) => (*a as f64) < *b,
            (Some(MetadataValue::Float(a)), MetadataValue::Integer(b)) => *a < (*b as f64),
            (Some(MetadataValue::Timestamp(a)), MetadataValue::Timestamp(b)) => a < b,
            (Some(MetadataValue::String(a)), MetadataValue::String(b)) => a < b,
            _ => false,
        }
    }

    fn compare_lte(field_value: Option<&MetadataValue>, compare_value: &MetadataValue) -> bool {
        match (field_value, compare_value) {
            (Some(MetadataValue::Integer(a)), MetadataValue::Integer(b)) => a <= b,
            (Some(MetadataValue::Float(a)), MetadataValue::Float(b)) => a <= b,
            (Some(MetadataValue::Integer(a)), MetadataValue::Float(b)) => (*a as f64) <= *b,
            (Some(MetadataValue::Float(a)), MetadataValue::Integer(b)) => *a <= (*b as f64),
            (Some(MetadataValue::Timestamp(a)), MetadataValue::Timestamp(b)) => a <= b,
            (Some(MetadataValue::String(a)), MetadataValue::String(b)) => a <= b,
            _ => false,
        }
    }
}

/// Metadata store for CID-indexed metadata
pub struct MetadataStore {
    /// CID to metadata mapping
    data: Arc<RwLock<HashMap<Cid, Metadata>>>,
    /// Inverted index for string fields (field -> value -> CIDs)
    string_index: Arc<RwLock<StringIndexMap>>,
    /// Sorted index for numeric fields (field -> sorted (value, CID) pairs)
    numeric_index: Arc<RwLock<NumericIndexMap>>,
    /// Timestamp index for temporal queries
    timestamp_index: Arc<RwLock<BTreeMap<u64, HashSet<Cid>>>>,
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataStore {
    /// Create a new metadata store
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            string_index: Arc::new(RwLock::new(HashMap::new())),
            numeric_index: Arc::new(RwLock::new(HashMap::new())),
            timestamp_index: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Insert or update metadata for a CID
    pub fn insert(&self, cid: Cid, metadata: Metadata) -> Result<()> {
        // Remove old indexes if updating
        if self
            .data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&cid)
        {
            self.remove_from_indexes(&cid)?;
        }

        // Update indexes
        self.add_to_indexes(&cid, &metadata)?;

        // Store metadata
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(cid, metadata);

        Ok(())
    }

    /// Get metadata for a CID
    pub fn get(&self, cid: &Cid) -> Option<Metadata> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(cid)
            .cloned()
    }

    /// Remove metadata for a CID
    pub fn remove(&self, cid: &Cid) -> Result<Option<Metadata>> {
        self.remove_from_indexes(cid)?;
        Ok(self
            .data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(cid))
    }

    /// Check if metadata exists for a CID
    pub fn contains(&self, cid: &Cid) -> bool {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(cid)
    }

    /// Get all CIDs with metadata
    pub fn cids(&self) -> Vec<Cid> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .copied()
            .collect()
    }

    /// Get number of stored metadata records
    pub fn len(&self) -> usize {
        self.data.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Filter CIDs by metadata filter
    pub fn filter(&self, filter: &MetadataFilter) -> Vec<Cid> {
        // Try to use indexes for efficient filtering
        if let Some(cids) = self.filter_with_index(filter) {
            return cids;
        }

        // Fall back to linear scan
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|(_, m)| filter.matches(m))
            .map(|(cid, _)| *cid)
            .collect()
    }

    /// Filter using indexes if possible
    fn filter_with_index(&self, filter: &MetadataFilter) -> Option<Vec<Cid>> {
        match filter {
            MetadataFilter::Equals(field, MetadataValue::String(value)) => {
                let index = self.string_index.read().unwrap_or_else(|e| e.into_inner());
                index
                    .get(field)
                    .and_then(|field_index| field_index.get(value))
                    .map(|cids| cids.iter().copied().collect())
            }
            MetadataFilter::TimeRange { field, start, end } if field == "created_at" => {
                let index = self
                    .timestamp_index
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                let range_start = start.unwrap_or(0);
                let range_end = end.unwrap_or(u64::MAX);

                let cids: HashSet<Cid> = index
                    .range(range_start..=range_end)
                    .flat_map(|(_, cids)| cids.iter().copied())
                    .collect();

                Some(cids.into_iter().collect())
            }
            MetadataFilter::And(filters) => {
                // Intersect results from indexed filters
                let mut result: Option<HashSet<Cid>> = None;

                for f in filters {
                    if let Some(cids) = self.filter_with_index(f) {
                        let cid_set: HashSet<Cid> = cids.into_iter().collect();
                        result = Some(match result {
                            Some(existing) => existing.intersection(&cid_set).copied().collect(),
                            None => cid_set,
                        });
                    }
                }

                result.map(|s| s.into_iter().collect())
            }
            _ => None,
        }
    }

    /// Add metadata to indexes
    fn add_to_indexes(&self, cid: &Cid, metadata: &Metadata) -> Result<()> {
        // Index string fields
        for (key, value) in &metadata.fields {
            if let MetadataValue::String(s) = value {
                self.string_index
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .entry(key.clone())
                    .or_default()
                    .entry(s.clone())
                    .or_default()
                    .insert(*cid);
            }

            if let Some(i) = value.as_integer() {
                self.numeric_index
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .entry(key.clone())
                    .or_default()
                    .entry(i)
                    .or_default()
                    .insert(*cid);
            }
        }

        // Index creation timestamp
        self.timestamp_index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .entry(metadata.created_at)
            .or_default()
            .insert(*cid);

        Ok(())
    }

    /// Remove metadata from indexes
    fn remove_from_indexes(&self, cid: &Cid) -> Result<()> {
        let data = self.data.read().unwrap_or_else(|e| e.into_inner());
        if let Some(metadata) = data.get(cid) {
            // Remove from string index
            for (key, value) in &metadata.fields {
                if let MetadataValue::String(s) = value {
                    if let Some(field_index) = self
                        .string_index
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_mut(key)
                    {
                        if let Some(cids) = field_index.get_mut(s) {
                            cids.remove(cid);
                        }
                    }
                }

                if let Some(i) = value.as_integer() {
                    if let Some(field_index) = self
                        .numeric_index
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_mut(key)
                    {
                        if let Some(cids) = field_index.get_mut(&i) {
                            cids.remove(cid);
                        }
                    }
                }
            }

            // Remove from timestamp index
            if let Some(cids) = self
                .timestamp_index
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .get_mut(&metadata.created_at)
            {
                cids.remove(cid);
            }
        }

        Ok(())
    }

    /// Get CIDs created within a time range
    pub fn get_by_time_range(&self, start: Option<u64>, end: Option<u64>) -> Vec<Cid> {
        let index = self
            .timestamp_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let range_start = start.unwrap_or(0);
        let range_end = end.unwrap_or(u64::MAX);

        index
            .range(range_start..=range_end)
            .flat_map(|(_, cids)| cids.iter().copied())
            .collect()
    }

    /// Get unique values for a field
    pub fn get_field_values(&self, field: &str) -> Vec<MetadataValue> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|m| m.get(field).cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Get facet counts for a string field
    pub fn get_facet_counts(&self, field: &str) -> HashMap<String, usize> {
        let index = self.string_index.read().unwrap_or_else(|e| e.into_inner());
        index
            .get(field)
            .map(|field_index| {
                field_index
                    .iter()
                    .map(|(value, cids)| (value.clone(), cids.len()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Clear all metadata
    pub fn clear(&self) {
        self.data.write().unwrap_or_else(|e| e.into_inner()).clear();
        self.string_index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.numeric_index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.timestamp_index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// Temporal query options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOptions {
    /// Time range start (Unix timestamp)
    pub start: Option<u64>,
    /// Time range end (Unix timestamp)
    pub end: Option<u64>,
    /// Apply recency boost to scores
    pub recency_boost: bool,
    /// Recency decay factor (higher = faster decay)
    pub decay_factor: f32,
    /// Reference time for recency calculation (default: now)
    pub reference_time: Option<u64>,
}

impl Default for TemporalOptions {
    fn default() -> Self {
        Self {
            start: None,
            end: None,
            recency_boost: false,
            decay_factor: 1.0,
            reference_time: None,
        }
    }
}

impl TemporalOptions {
    /// Create options for a specific time range
    pub fn range(start: u64, end: u64) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            ..Default::default()
        }
    }

    /// Create options with recency boosting
    pub fn with_recency(decay_factor: f32) -> Self {
        Self {
            recency_boost: true,
            decay_factor,
            ..Default::default()
        }
    }

    /// Calculate recency boost multiplier for a timestamp
    pub fn recency_multiplier(&self, timestamp: u64) -> f32 {
        if !self.recency_boost {
            return 1.0;
        }

        let reference = self.reference_time.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });

        if timestamp >= reference {
            return 1.0;
        }

        let age_seconds = reference - timestamp;
        let age_days = age_seconds as f32 / 86400.0;

        // Exponential decay: e^(-decay_factor * age_days)
        (-self.decay_factor * age_days / 30.0).exp()
    }
}

impl std::hash::Hash for MetadataValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            MetadataValue::String(s) => {
                0u8.hash(state);
                s.hash(state);
            }
            MetadataValue::Integer(i) => {
                1u8.hash(state);
                i.hash(state);
            }
            MetadataValue::Float(f) => {
                2u8.hash(state);
                f.to_bits().hash(state);
            }
            MetadataValue::Boolean(b) => {
                3u8.hash(state);
                b.hash(state);
            }
            MetadataValue::Timestamp(t) => {
                4u8.hash(state);
                t.hash(state);
            }
            MetadataValue::StringArray(arr) => {
                5u8.hash(state);
                arr.hash(state);
            }
            MetadataValue::Null => {
                6u8.hash(state);
            }
        }
    }
}

impl Eq for MetadataValue {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: known-valid CID string should parse")
    }

    fn test_cid2() -> Cid {
        "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"
            .parse()
            .expect("test: known-valid CID string should parse")
    }

    #[test]
    fn test_metadata_creation() {
        let metadata = Metadata::new()
            .with_string("type", "document")
            .with_integer("size", 1024)
            .with_boolean("indexed", true);

        assert_eq!(
            metadata.get("type"),
            Some(&MetadataValue::String("document".to_string()))
        );
        assert_eq!(metadata.get("size"), Some(&MetadataValue::Integer(1024)));
        assert_eq!(metadata.get("indexed"), Some(&MetadataValue::Boolean(true)));
    }

    #[test]
    fn test_metadata_filter() {
        let metadata = Metadata::new()
            .with_string("category", "tech")
            .with_integer("views", 100)
            .with_timestamp("published", 1700000000);

        // Equals filter
        assert!(
            MetadataFilter::eq("category", MetadataValue::String("tech".to_string()))
                .matches(&metadata)
        );

        // Greater than filter
        assert!(MetadataFilter::gt("views", MetadataValue::Integer(50)).matches(&metadata));
        assert!(!MetadataFilter::gt("views", MetadataValue::Integer(200)).matches(&metadata));

        // Time range filter
        assert!(
            MetadataFilter::time_range("published", Some(1699999999), Some(1700000001))
                .matches(&metadata)
        );
    }

    #[test]
    fn test_metadata_store() {
        let store = MetadataStore::new();

        let cid1 = test_cid();
        let cid2 = test_cid2();

        let meta1 = Metadata::new()
            .with_string("type", "image")
            .with_integer("size", 1024);

        let meta2 = Metadata::new()
            .with_string("type", "document")
            .with_integer("size", 2048);

        store
            .insert(cid1, meta1)
            .expect("test: insert cid1 into store should succeed");
        store
            .insert(cid2, meta2)
            .expect("test: insert cid2 into store should succeed");

        assert_eq!(store.len(), 2);

        // Filter by type
        let filter = MetadataFilter::eq("type", MetadataValue::String("image".to_string()));
        let results = store.filter(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], cid1);
    }

    #[test]
    fn test_compound_filters() {
        let metadata = Metadata::new()
            .with_string("category", "tech")
            .with_integer("views", 100)
            .with_boolean("published", true);

        // AND filter
        let and_filter = MetadataFilter::and(vec![
            MetadataFilter::eq("category", MetadataValue::String("tech".to_string())),
            MetadataFilter::gt("views", MetadataValue::Integer(50)),
        ]);
        assert!(and_filter.matches(&metadata));

        // OR filter
        let or_filter = MetadataFilter::or(vec![
            MetadataFilter::eq("category", MetadataValue::String("science".to_string())),
            MetadataFilter::gt("views", MetadataValue::Integer(50)),
        ]);
        assert!(or_filter.matches(&metadata));

        // NOT filter
        let not_filter = MetadataFilter::negate(MetadataFilter::eq(
            "published",
            MetadataValue::Boolean(false),
        ));
        assert!(not_filter.matches(&metadata));
    }

    #[test]
    fn test_temporal_options() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test: system time should be after UNIX_EPOCH")
            .as_secs();

        let options = TemporalOptions {
            recency_boost: true,
            decay_factor: 1.0,
            reference_time: Some(now),
            ..Default::default()
        };

        // Recent timestamp should have high multiplier
        let recent_mult = options.recency_multiplier(now - 86400); // 1 day ago
        assert!(recent_mult > 0.9);

        // Old timestamp should have low multiplier
        let old_mult = options.recency_multiplier(now - 86400 * 90); // 90 days ago
        assert!(old_mult < 0.5);
    }

    #[test]
    fn test_facet_counts() {
        let store = MetadataStore::new();

        // Use known valid CIDs
        let valid_cids = [
            "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
            "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354",
            "bafybeibvfkifsqbapirjrj7zbfwddz5qz5awvbftjgktpcqcxjkzstszlm",
        ];

        for (i, cid_str) in valid_cids.iter().enumerate() {
            let cid: Cid = cid_str
                .parse()
                .expect("test: known-valid CID string should parse");
            let meta = Metadata::new().with_string("type", if i < 2 { "image" } else { "doc" });
            store
                .insert(cid, meta)
                .expect("test: insert CID metadata should succeed");
        }

        let counts = store.get_facet_counts("type");
        assert_eq!(counts.get("image").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("doc").copied().unwrap_or(0), 1);
    }
}
