//! High-level object store abstraction over raw block storage.
//!
//! Provides named object management, versioning, and metadata-rich retrieval.

use std::collections::HashMap;

/// A single version of a stored object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectVersion {
    /// Monotonically increasing version number, starts at 1.
    pub version: u32,
    /// Content identifier for this version.
    pub cid: String,
    /// Size of this version in bytes.
    pub size_bytes: u64,
    /// Unix timestamp (seconds) when this version was created.
    pub created_at_secs: u64,
    /// Who created this version.
    pub author: String,
    /// Version commit message.
    pub message: String,
}

/// A named, versioned object stored in the object store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredObject {
    /// Unique numeric identifier.
    pub object_id: u64,
    /// Human-readable name (unique within namespace).
    pub name: String,
    /// Namespace this object belongs to.
    pub namespace: String,
    /// All versions, ordered oldest (index 0) to newest (last index).
    pub versions: Vec<ObjectVersion>,
    /// Arbitrary tags attached to this object.
    pub tags: Vec<String>,
    /// Whether this object is pinned (prevents deletion).
    pub pinned: bool,
}

impl StoredObject {
    /// Returns the most recent version, or `None` if there are no versions.
    pub fn current_version(&self) -> Option<&ObjectVersion> {
        self.versions.last()
    }

    /// Returns the total number of versions.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Returns the sum of `size_bytes` across all versions.
    pub fn total_size_bytes(&self) -> u64 {
        self.versions.iter().map(|v| v.size_bytes).sum()
    }

    /// Finds a version by its version number, or `None` if not present.
    pub fn find_version(&self, v: u32) -> Option<&ObjectVersion> {
        self.versions.iter().find(|ov| ov.version == v)
    }
}

/// Aggregate statistics for a [`StorageObjectStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectStoreStats {
    /// Total number of objects.
    pub total_objects: usize,
    /// Total number of versions across all objects.
    pub total_versions: usize,
    /// Total bytes across all versions of all objects.
    pub total_size_bytes: u64,
    /// Number of pinned objects.
    pub pinned_objects: usize,
    /// Number of unique namespaces.
    pub namespaces: usize,
}

/// High-level object store providing named object management, versioning, and
/// metadata-rich retrieval on top of raw block storage.
#[derive(Debug, Default)]
pub struct StorageObjectStore {
    /// Objects keyed by their numeric id.
    pub objects: HashMap<u64, StoredObject>,
    /// Index from `(namespace, name)` to `object_id`.
    pub name_index: HashMap<(String, String), u64>,
    /// Counter used to assign the next object id.
    pub next_object_id: u64,
}

impl StorageObjectStore {
    /// Creates a new, empty object store.
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            name_index: HashMap::new(),
            next_object_id: 1,
        }
    }

    /// Creates a new object with version 1 and returns its `object_id`.
    ///
    /// If an object with the same `(namespace, name)` already exists the existing
    /// object id is returned without creating a duplicate entry.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &mut self,
        name: &str,
        namespace: &str,
        cid: &str,
        size_bytes: u64,
        author: &str,
        message: &str,
        created_at_secs: u64,
    ) -> u64 {
        let key = (namespace.to_owned(), name.to_owned());
        if let Some(&existing_id) = self.name_index.get(&key) {
            return existing_id;
        }

        let object_id = self.next_object_id;
        self.next_object_id += 1;

        let version = ObjectVersion {
            version: 1,
            cid: cid.to_owned(),
            size_bytes,
            created_at_secs,
            author: author.to_owned(),
            message: message.to_owned(),
        };

        let stored = StoredObject {
            object_id,
            name: name.to_owned(),
            namespace: namespace.to_owned(),
            versions: vec![version],
            tags: Vec::new(),
            pinned: false,
        };

        self.objects.insert(object_id, stored);
        self.name_index.insert(key, object_id);
        object_id
    }

    /// Appends a new version to an existing object.
    ///
    /// The new version number is `previous_max_version + 1`.
    /// Returns `false` if the object is not found.
    pub fn add_version(
        &mut self,
        object_id: u64,
        cid: &str,
        size_bytes: u64,
        author: &str,
        message: &str,
        created_at_secs: u64,
    ) -> bool {
        let Some(obj) = self.objects.get_mut(&object_id) else {
            return false;
        };

        let next_version = obj.versions.iter().map(|v| v.version).max().unwrap_or(0) + 1;

        obj.versions.push(ObjectVersion {
            version: next_version,
            cid: cid.to_owned(),
            size_bytes,
            created_at_secs,
            author: author.to_owned(),
            message: message.to_owned(),
        });

        true
    }

    /// Retrieves an object by its `(namespace, name)` pair.
    pub fn get_by_name(&self, namespace: &str, name: &str) -> Option<&StoredObject> {
        let key = (namespace.to_owned(), name.to_owned());
        let id = self.name_index.get(&key)?;
        self.objects.get(id)
    }

    /// Retrieves an object by its numeric id.
    pub fn get(&self, object_id: u64) -> Option<&StoredObject> {
        self.objects.get(&object_id)
    }

    /// Deletes an object by id.
    ///
    /// Returns `false` when the object is not found **or** when it is pinned.
    pub fn delete(&mut self, object_id: u64) -> bool {
        let Some(obj) = self.objects.get(&object_id) else {
            return false;
        };

        if obj.pinned {
            return false;
        }

        let key = (obj.namespace.clone(), obj.name.clone());
        self.objects.remove(&object_id);
        self.name_index.remove(&key);
        true
    }

    /// Pins an object, preventing it from being deleted.
    ///
    /// Returns `false` if the object is not found.
    pub fn pin(&mut self, object_id: u64) -> bool {
        let Some(obj) = self.objects.get_mut(&object_id) else {
            return false;
        };
        obj.pinned = true;
        true
    }

    /// Unpins an object, allowing it to be deleted.
    ///
    /// Returns `false` if the object is not found.
    pub fn unpin(&mut self, object_id: u64) -> bool {
        let Some(obj) = self.objects.get_mut(&object_id) else {
            return false;
        };
        obj.pinned = false;
        true
    }

    /// Adds a tag to an object (idempotent — duplicate tags are silently ignored).
    ///
    /// Returns `false` if the object is not found.
    pub fn add_tag(&mut self, object_id: u64, tag: &str) -> bool {
        let Some(obj) = self.objects.get_mut(&object_id) else {
            return false;
        };
        if !obj.tags.iter().any(|t| t == tag) {
            obj.tags.push(tag.to_owned());
        }
        true
    }

    /// Returns all objects in the given namespace, sorted ascending by name.
    pub fn objects_in_namespace(&self, namespace: &str) -> Vec<&StoredObject> {
        let mut result: Vec<&StoredObject> = self
            .objects
            .values()
            .filter(|o| o.namespace == namespace)
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Returns all objects that carry the given tag, sorted ascending by `object_id`.
    pub fn objects_with_tag(&self, tag: &str) -> Vec<&StoredObject> {
        let mut result: Vec<&StoredObject> = self
            .objects
            .values()
            .filter(|o| o.tags.iter().any(|t| t == tag))
            .collect();
        result.sort_by_key(|o| o.object_id);
        result
    }

    /// Computes aggregate statistics for the entire store.
    pub fn stats(&self) -> ObjectStoreStats {
        let total_objects = self.objects.len();
        let total_versions = self.objects.values().map(|o| o.versions.len()).sum();
        let total_size_bytes = self.objects.values().map(|o| o.total_size_bytes()).sum();
        let pinned_objects = self.objects.values().filter(|o| o.pinned).count();
        let namespaces = self
            .objects
            .values()
            .map(|o| o.namespace.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();

        ObjectStoreStats {
            total_objects,
            total_versions,
            total_size_bytes,
            pinned_objects,
            namespaces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> StorageObjectStore {
        StorageObjectStore::new()
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn create_simple(store: &mut StorageObjectStore, name: &str, namespace: &str) -> u64 {
        store.create(name, namespace, "cid-abc", 100, "alice", "initial", 1000)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let store = make_store();
        assert!(store.objects.is_empty());
        assert!(store.name_index.is_empty());
        assert_eq!(store.next_object_id, 1);
    }

    #[test]
    fn test_create_stores_object() {
        let mut store = make_store();
        let id = create_simple(&mut store, "obj1", "ns1");
        let obj = store.get(id).expect("object should exist");
        assert_eq!(obj.name, "obj1");
        assert_eq!(obj.namespace, "ns1");
        assert_eq!(obj.versions.len(), 1);
        assert_eq!(obj.versions[0].version, 1);
        assert_eq!(obj.versions[0].cid, "cid-abc");
    }

    #[test]
    fn test_create_returns_incrementing_ids() {
        let mut store = make_store();
        let id1 = create_simple(&mut store, "a", "ns");
        let id2 = create_simple(&mut store, "b", "ns");
        let id3 = create_simple(&mut store, "c", "ns");
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    #[test]
    fn test_create_registers_in_name_index() {
        let mut store = make_store();
        let id = create_simple(&mut store, "myobj", "mynamespace");
        let indexed_id = store
            .name_index
            .get(&("mynamespace".to_owned(), "myobj".to_owned()))
            .copied();
        assert_eq!(indexed_id, Some(id));
    }

    #[test]
    fn test_create_idempotent_same_namespace_name() {
        let mut store = make_store();
        let id1 = create_simple(&mut store, "dup", "ns");
        let id2 = create_simple(&mut store, "dup", "ns");
        assert_eq!(id1, id2);
        assert_eq!(store.objects.len(), 1);
    }

    #[test]
    fn test_add_version_appends_with_correct_version_number() {
        let mut store = make_store();
        let id = create_simple(&mut store, "v", "ns");
        let ok = store.add_version(id, "cid-v2", 200, "bob", "second", 2000);
        assert!(ok);
        let obj = store.get(id).expect("object should exist");
        assert_eq!(obj.versions.len(), 2);
        assert_eq!(obj.versions[1].version, 2);
        assert_eq!(obj.versions[1].cid, "cid-v2");

        let ok2 = store.add_version(id, "cid-v3", 300, "carol", "third", 3000);
        assert!(ok2);
        assert_eq!(store.get(id).unwrap().versions[2].version, 3);
    }

    #[test]
    fn test_add_version_false_for_unknown_object_id() {
        let mut store = make_store();
        let ok = store.add_version(9999, "cid", 100, "x", "msg", 0);
        assert!(!ok);
    }

    #[test]
    fn test_current_version_returns_last_version() {
        let mut store = make_store();
        let id = create_simple(&mut store, "cv", "ns");
        store.add_version(id, "cid-v2", 200, "bob", "second", 2000);
        let obj = store.get(id).unwrap();
        let cv = obj.current_version().expect("should have current version");
        assert_eq!(cv.version, 2);
        assert_eq!(cv.cid, "cid-v2");
    }

    #[test]
    fn test_current_version_none_when_no_versions() {
        // Construct a StoredObject with no versions manually.
        let obj = StoredObject {
            object_id: 1,
            name: "empty".to_owned(),
            namespace: "ns".to_owned(),
            versions: vec![],
            tags: vec![],
            pinned: false,
        };
        assert!(obj.current_version().is_none());
    }

    #[test]
    fn test_find_version_by_number() {
        let mut store = make_store();
        let id = create_simple(&mut store, "fv", "ns");
        store.add_version(id, "cid-v2", 200, "bob", "second", 2000);
        store.add_version(id, "cid-v3", 300, "carol", "third", 3000);
        let obj = store.get(id).unwrap();
        let v2 = obj.find_version(2).expect("version 2 should exist");
        assert_eq!(v2.cid, "cid-v2");
        assert!(obj.find_version(99).is_none());
    }

    #[test]
    fn test_version_count_correct() {
        let mut store = make_store();
        let id = create_simple(&mut store, "vc", "ns");
        assert_eq!(store.get(id).unwrap().version_count(), 1);
        store.add_version(id, "cid-v2", 200, "x", "m", 0);
        assert_eq!(store.get(id).unwrap().version_count(), 2);
    }

    #[test]
    fn test_get_by_name_some() {
        let mut store = make_store();
        let id = create_simple(&mut store, "named", "space");
        let obj = store.get_by_name("space", "named");
        assert!(obj.is_some());
        assert_eq!(obj.unwrap().object_id, id);
    }

    #[test]
    fn test_get_by_name_none() {
        let store = make_store();
        assert!(store.get_by_name("nonexistent", "nothing").is_none());
    }

    #[test]
    fn test_get_some() {
        let mut store = make_store();
        let id = create_simple(&mut store, "g", "ns");
        assert!(store.get(id).is_some());
    }

    #[test]
    fn test_get_none() {
        let store = make_store();
        assert!(store.get(12345).is_none());
    }

    #[test]
    fn test_delete_removes_object() {
        let mut store = make_store();
        let id = create_simple(&mut store, "del", "ns");
        let ok = store.delete(id);
        assert!(ok);
        assert!(store.get(id).is_none());
        assert!(!store
            .name_index
            .contains_key(&("ns".to_owned(), "del".to_owned())));
    }

    #[test]
    fn test_delete_false_when_pinned() {
        let mut store = make_store();
        let id = create_simple(&mut store, "pinned-obj", "ns");
        store.pin(id);
        let ok = store.delete(id);
        assert!(!ok);
        assert!(store.get(id).is_some());
    }

    #[test]
    fn test_delete_false_when_not_found() {
        let mut store = make_store();
        assert!(!store.delete(999));
    }

    #[test]
    fn test_pin_sets_pinned_true() {
        let mut store = make_store();
        let id = create_simple(&mut store, "p", "ns");
        assert!(!store.get(id).unwrap().pinned);
        let ok = store.pin(id);
        assert!(ok);
        assert!(store.get(id).unwrap().pinned);
    }

    #[test]
    fn test_pin_false_when_not_found() {
        let mut store = make_store();
        assert!(!store.pin(404));
    }

    #[test]
    fn test_unpin_sets_pinned_false() {
        let mut store = make_store();
        let id = create_simple(&mut store, "u", "ns");
        store.pin(id);
        let ok = store.unpin(id);
        assert!(ok);
        assert!(!store.get(id).unwrap().pinned);
    }

    #[test]
    fn test_unpin_false_when_not_found() {
        let mut store = make_store();
        assert!(!store.unpin(404));
    }

    #[test]
    fn test_add_tag_idempotent() {
        let mut store = make_store();
        let id = create_simple(&mut store, "t", "ns");
        assert!(store.add_tag(id, "important"));
        assert!(store.add_tag(id, "important")); // duplicate — should be ignored
        assert!(store.add_tag(id, "critical"));
        let obj = store.get(id).unwrap();
        assert_eq!(obj.tags.len(), 2);
        assert!(obj.tags.contains(&"important".to_owned()));
        assert!(obj.tags.contains(&"critical".to_owned()));
    }

    #[test]
    fn test_add_tag_false_when_not_found() {
        let mut store = make_store();
        assert!(!store.add_tag(999, "tag"));
    }

    #[test]
    fn test_objects_in_namespace_filtered_and_sorted() {
        let mut store = make_store();
        create_simple(&mut store, "zebra", "alpha");
        create_simple(&mut store, "apple", "alpha");
        create_simple(&mut store, "mango", "alpha");
        create_simple(&mut store, "only-here", "beta");

        let result = store.objects_in_namespace("alpha");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "apple");
        assert_eq!(result[1].name, "mango");
        assert_eq!(result[2].name, "zebra");
    }

    #[test]
    fn test_objects_in_namespace_empty_for_unknown() {
        let store = make_store();
        assert!(store.objects_in_namespace("no-such-ns").is_empty());
    }

    #[test]
    fn test_objects_with_tag_sorted_by_object_id() {
        let mut store = make_store();
        let id1 = create_simple(&mut store, "a", "ns");
        let id2 = create_simple(&mut store, "b", "ns");
        let id3 = create_simple(&mut store, "c", "ns");
        store.add_tag(id3, "hot");
        store.add_tag(id1, "hot");
        store.add_tag(id2, "hot");

        let result = store.objects_with_tag("hot");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].object_id, id1);
        assert_eq!(result[1].object_id, id2);
        assert_eq!(result[2].object_id, id3);
    }

    #[test]
    fn test_objects_with_tag_empty_for_unknown_tag() {
        let mut store = make_store();
        create_simple(&mut store, "a", "ns");
        assert!(store.objects_with_tag("nosuch").is_empty());
    }

    #[test]
    fn test_total_size_bytes_sums_all_versions() {
        let mut store = make_store();
        let id = store.create("s", "ns", "c1", 100, "a", "m", 0);
        store.add_version(id, "c2", 200, "a", "m", 0);
        store.add_version(id, "c3", 300, "a", "m", 0);
        let obj = store.get(id).unwrap();
        assert_eq!(obj.total_size_bytes(), 600);
    }

    #[test]
    fn test_stats_namespaces_count() {
        let mut store = make_store();
        create_simple(&mut store, "a", "alpha");
        create_simple(&mut store, "b", "alpha");
        create_simple(&mut store, "c", "beta");
        create_simple(&mut store, "d", "gamma");

        let stats = store.stats();
        assert_eq!(stats.namespaces, 3);
        assert_eq!(stats.total_objects, 4);
    }

    #[test]
    fn test_stats_pinned_objects_count() {
        let mut store = make_store();
        let id1 = create_simple(&mut store, "x", "ns");
        let id2 = create_simple(&mut store, "y", "ns");
        create_simple(&mut store, "z", "ns");
        store.pin(id1);
        store.pin(id2);

        let stats = store.stats();
        assert_eq!(stats.pinned_objects, 2);
    }

    #[test]
    fn test_stats_total_versions_and_size() {
        let mut store = make_store();
        let id1 = store.create("a", "ns", "c1", 50, "x", "m", 0);
        store.add_version(id1, "c2", 75, "x", "m", 0);
        let _id2 = store.create("b", "ns", "c3", 100, "x", "m", 0);

        let stats = store.stats();
        assert_eq!(stats.total_versions, 3);
        assert_eq!(stats.total_size_bytes, 225);
    }
}
