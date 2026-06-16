//! Version Control System for Differentiable Storage
//!
//! Provides Git-like version control for tensor models and gradients:
//! - Commit tracking with parent links (DAG structure)
//! - Branch and tag management
//! - Checkout to specific commits
//! - Merge support for collaborative training
//!
//! This enables reproducible model states and collaborative training workflows.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::{VersionControl, Author, SledBlockStore, BlockStoreConfig};
//! use ipfrs_core::Block;
//! use bytes::Bytes;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//! use std::path::PathBuf;
//!
//! # async fn example() -> ipfrs_core::Result<()> {
//! // Create a block store
//! let config = BlockStoreConfig {
//!     path: PathBuf::from(".ipfrs/vcs"),
//!     cache_size: 100 * 1024 * 1024,
//! };
//! let store = Arc::new(SledBlockStore::new(config)?);
//!
//! // Initialize version control
//! let vcs = VersionControl::new(store.clone());
//!
//! // Store model v1 and create initial commit
//! let model_v1 = Block::new(Bytes::from("model weights v1"))?;
//! store.put(&model_v1).await?;
//!
//! let author = Author {
//!     name: "AI Researcher".to_string(),
//!     email: "researcher@example.com".to_string(),
//! };
//!
//! let commit1 = vcs.commit(
//!     *model_v1.cid(),
//!     "Initial model".to_string(),
//!     author.clone(),
//!     HashMap::new(),
//! ).await?;
//!
//! // Train model, create v2, and commit
//! let model_v2 = Block::new(Bytes::from("model weights v2"))?;
//! store.put(&model_v2).await?;
//!
//! let commit2 = vcs.commit(
//!     *model_v2.cid(),
//!     "After 100 epochs".to_string(),
//!     author,
//!     HashMap::new(),
//! ).await?;
//!
//! // Checkout to previous version
//! let model_cid = vcs.checkout(&commit1).await?;
//! let previous_model = store.get(&model_cid).await?;
//!
//! // View commit history
//! let history = vcs.log(&commit2, 10).await?;
//! for commit in history {
//!     println!("{}: {}", commit.timestamp, commit.message);
//! }
//! # Ok(())
//! # }
//! ```

use crate::traits::BlockStore;
use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// Custom serialization for Cid
fn serialize_cid<S>(cid: &Cid, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_bytes(&cid.to_bytes())
}

fn deserialize_cid<'de, D>(deserializer: D) -> std::result::Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
    Cid::try_from(bytes).map_err(serde::de::Error::custom)
}

fn serialize_cid_vec<S>(cids: &[Cid], serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(cids.len()))?;
    for cid in cids {
        seq.serialize_element(&cid.to_bytes())?;
    }
    seq.end()
}

fn deserialize_cid_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<Cid>, D::Error>
where
    D: Deserializer<'de>,
{
    let bytes_vec: Vec<Vec<u8>> = Deserialize::deserialize(deserializer)?;
    bytes_vec
        .into_iter()
        .map(|bytes| Cid::try_from(bytes).map_err(serde::de::Error::custom))
        .collect()
}

/// IPLD schema for a commit in the version control system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Commit {
    /// CID of the commit (computed from serialized commit, not included in serialization)
    #[serde(skip)]
    pub cid: Option<Cid>,

    /// CIDs of parent commits (empty for initial commit)
    #[serde(
        serialize_with = "serialize_cid_vec",
        deserialize_with = "deserialize_cid_vec"
    )]
    pub parents: Vec<Cid>,

    /// CID of the root block this commit points to (e.g., model weights)
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub root: Cid,

    /// Commit message describing the changes
    pub message: String,

    /// Author of the commit
    pub author: Author,

    /// Unix timestamp when commit was created
    pub timestamp: u64,

    /// Optional metadata (e.g., training config, hyperparameters)
    pub metadata: HashMap<String, String>,
}

/// Author information for a commit
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Author {
    /// Author name
    pub name: String,
    /// Author email
    pub email: String,
}

impl Commit {
    /// Create a new commit
    pub fn new(
        parents: Vec<Cid>,
        root: Cid,
        message: String,
        author: Author,
        metadata: HashMap<String, String>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs();

        Self {
            cid: None,
            parents,
            root,
            message,
            author,
            timestamp,
            metadata,
        }
    }

    /// Serialize commit to bytes and compute CID
    pub fn finalize(&mut self) -> Result<Cid> {
        let bytes = oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize commit: {e}")))?;

        let block = Block::new(Bytes::from(bytes))?;
        let cid = *block.cid();
        self.cid = Some(cid);
        Ok(cid)
    }

    /// Create a commit from a block
    pub fn from_block(block: &Block) -> Result<Self> {
        let mut commit: Commit =
            oxicode::serde::decode_owned_from_slice(block.data(), oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| Error::Serialization(format!("Failed to deserialize commit: {e}")))?;
        commit.cid = Some(*block.cid());
        Ok(commit)
    }

    /// Convert commit to a block for storage
    pub fn to_block(&self) -> Result<Block> {
        let bytes = oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize commit: {e}")))?;
        Block::new(Bytes::from(bytes))
    }

    /// Check if this is an initial commit (no parents)
    pub fn is_initial(&self) -> bool {
        self.parents.is_empty()
    }

    /// Get a reference to the commit CID (panics if not finalized)
    pub fn cid(&self) -> &Cid {
        self.cid.as_ref().expect("Commit not finalized")
    }
}

/// Reference to a commit (branch or tag)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ref {
    /// Name of the reference (e.g., "main", "dev", "v1.0")
    pub name: String,
    /// CID of the commit this ref points to
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub commit: Cid,
    /// Type of reference
    pub ref_type: RefType,
}

/// Type of reference
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RefType {
    /// Mutable branch pointer
    Branch,
    /// Immutable tag pointer
    Tag,
}

impl Ref {
    /// Create a new branch reference
    pub fn branch(name: String, commit: Cid) -> Self {
        Self {
            name,
            commit,
            ref_type: RefType::Branch,
        }
    }

    /// Create a new tag reference
    pub fn tag(name: String, commit: Cid) -> Self {
        Self {
            name,
            commit,
            ref_type: RefType::Tag,
        }
    }

    /// Convert ref to a block for storage
    pub fn to_block(&self) -> Result<Block> {
        let bytes = oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize ref: {e}")))?;
        Block::new(Bytes::from(bytes))
    }

    /// Create a ref from a block
    pub fn from_block(block: &Block) -> Result<Self> {
        oxicode::serde::decode_owned_from_slice(block.data(), oxicode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| Error::Serialization(format!("Failed to deserialize ref: {e}")))
    }
}

/// Version Control System for managing commits, branches, and tags
pub struct VersionControl<S: BlockStore> {
    /// Underlying block store
    store: Arc<S>,
    /// Current branch name
    current_branch: parking_lot::RwLock<String>,
    /// HEAD pointer (current commit CID)
    head: parking_lot::RwLock<Option<Cid>>,
    /// In-memory refs cache (ref name -> commit CID)
    refs_cache: dashmap::DashMap<String, Cid>,
}

/// Merge strategy for combining branches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Fast-forward merge (only when target is ancestor of source)
    FastForward,
    /// Three-way merge (creates merge commit)
    ThreeWay,
    /// Ours (keep current branch's changes on conflict)
    Ours,
    /// Theirs (accept incoming branch's changes on conflict)
    Theirs,
}

/// Result of a merge operation
#[derive(Debug, Clone, PartialEq)]
pub enum MergeResult {
    /// Fast-forward merge succeeded
    FastForward { target: Cid },
    /// Merge commit created
    MergeCommit { commit: Cid },
    /// Conflicts detected (contains conflicting paths)
    Conflicts { conflicts: Vec<String> },
}

impl<S: BlockStore> VersionControl<S> {
    /// Create a new version control system
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            current_branch: parking_lot::RwLock::new("main".to_string()),
            head: parking_lot::RwLock::new(None),
            refs_cache: dashmap::DashMap::new(),
        }
    }

    /// List all refs
    pub fn list_refs(&self) -> Vec<(String, Cid)> {
        self.refs_cache
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Create a new commit
    ///
    /// # Arguments
    /// * `root` - CID of the root block (e.g., model weights)
    /// * `message` - Commit message
    /// * `author` - Author information
    /// * `metadata` - Optional metadata
    pub async fn commit(
        &self,
        root: Cid,
        message: String,
        author: Author,
        metadata: HashMap<String, String>,
    ) -> Result<Cid> {
        // Get parent commits (current HEAD)
        let parents = if let Some(head) = *self.head.read() {
            vec![head]
        } else {
            vec![] // Initial commit
        };

        // Create and finalize commit
        let mut commit = Commit::new(parents, root, message, author, metadata);
        let commit_cid = commit.finalize()?;

        // Store commit block
        let commit_block = commit.to_block()?;
        self.store.put(&commit_block).await?;

        // Update HEAD
        *self.head.write() = Some(commit_cid);

        // Update current branch ref
        let branch_name = self.current_branch.read().clone();
        self.update_ref(&branch_name, commit_cid, RefType::Branch)
            .await?;

        Ok(commit_cid)
    }

    /// Checkout to a specific commit
    ///
    /// Returns the root CID of the commit (e.g., model weights to load)
    pub async fn checkout(&self, commit_cid: &Cid) -> Result<Cid> {
        // Load commit
        let commit_block = self
            .store
            .get(commit_cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Commit not found: {commit_cid}")))?;

        let commit = Commit::from_block(&commit_block)?;

        // Update HEAD to this commit
        *self.head.write() = Some(*commit_cid);

        // Return root CID for application to load
        Ok(commit.root)
    }

    /// Checkout to a branch or tag
    pub async fn checkout_ref(&self, ref_name: &str) -> Result<Cid> {
        // Load ref
        let ref_obj = self.get_ref(ref_name).await?;

        // Update current branch if it's a branch
        if ref_obj.ref_type == RefType::Branch {
            *self.current_branch.write() = ref_name.to_string();
        }

        // Checkout to the commit
        self.checkout(&ref_obj.commit).await
    }

    /// Create a new branch at the current HEAD
    pub async fn create_branch(&self, branch_name: &str) -> Result<()> {
        let head = self
            .head
            .read()
            .ok_or_else(|| Error::Storage("No HEAD commit".to_string()))?;

        self.update_ref(branch_name, head, RefType::Branch).await
    }

    /// Create a new tag at the current HEAD
    pub async fn create_tag(&self, tag_name: &str) -> Result<()> {
        let head = self
            .head
            .read()
            .ok_or_else(|| Error::Storage("No HEAD commit".to_string()))?;

        self.update_ref(tag_name, head, RefType::Tag).await
    }

    /// Update a reference (branch or tag)
    async fn update_ref(&self, name: &str, commit: Cid, ref_type: RefType) -> Result<()> {
        // Store in cache
        self.refs_cache.insert(name.to_string(), commit);

        // Also persist the ref as a block for durability
        let ref_obj = Ref {
            name: name.to_string(),
            commit,
            ref_type,
        };

        let ref_block = ref_obj.to_block()?;
        self.store.put(&ref_block).await?;

        Ok(())
    }

    /// Get a reference by name
    #[allow(clippy::unused_async)]
    async fn get_ref(&self, name: &str) -> Result<Ref> {
        // Check cache first
        if let Some(commit_cid) = self.refs_cache.get(name) {
            // Determine ref type based on naming convention
            let ref_type = if name.starts_with("refs/tags/") || name.contains("/tags/") {
                RefType::Tag
            } else {
                RefType::Branch
            };

            return Ok(Ref {
                name: name.to_string(),
                commit: *commit_cid,
                ref_type,
            });
        }

        Err(Error::NotFound(format!("Ref not found: {name}")))
    }

    /// Get the current HEAD commit CID
    pub fn head(&self) -> Option<Cid> {
        *self.head.read()
    }

    /// Get the current branch name
    pub fn current_branch(&self) -> String {
        self.current_branch.read().clone()
    }

    /// Get commit history (walk the DAG backwards)
    pub async fn log(&self, commit_cid: &Cid, limit: usize) -> Result<Vec<Commit>> {
        let mut commits = Vec::new();
        let mut current = Some(*commit_cid);

        while let Some(cid) = current {
            if commits.len() >= limit {
                break;
            }

            // Load commit
            let commit_block = self
                .store
                .get(&cid)
                .await?
                .ok_or_else(|| Error::NotFound(format!("Commit not found: {cid}")))?;

            let commit = Commit::from_block(&commit_block)?;

            // Move to parent
            current = commit.parents.first().copied();

            commits.push(commit);
        }

        Ok(commits)
    }

    /// Get the underlying store
    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    /// Find common ancestor between two commits
    pub async fn find_common_ancestor(&self, commit1: &Cid, commit2: &Cid) -> Result<Option<Cid>> {
        // Get all ancestors of commit1
        let mut ancestors1 = std::collections::HashSet::new();
        let mut queue = vec![*commit1];

        while let Some(cid) = queue.pop() {
            if !ancestors1.insert(cid) {
                continue; // Already visited
            }

            let block = self
                .store
                .get(&cid)
                .await?
                .ok_or_else(|| Error::NotFound(format!("Commit not found: {cid}")))?;

            let commit = Commit::from_block(&block)?;
            queue.extend(commit.parents.iter().copied());
        }

        // Walk commit2's ancestors until we find one in ancestors1
        let mut queue = vec![*commit2];
        let mut visited = std::collections::HashSet::new();

        while let Some(cid) = queue.pop() {
            if !visited.insert(cid) {
                continue;
            }

            if ancestors1.contains(&cid) {
                return Ok(Some(cid));
            }

            let block = self
                .store
                .get(&cid)
                .await?
                .ok_or_else(|| Error::NotFound(format!("Commit not found: {cid}")))?;

            let commit = Commit::from_block(&block)?;
            queue.extend(commit.parents.iter().copied());
        }

        Ok(None)
    }

    /// Check if commit1 is an ancestor of commit2 (i.e., fast-forward is possible)
    pub async fn is_ancestor(&self, ancestor: &Cid, descendant: &Cid) -> Result<bool> {
        if ancestor == descendant {
            return Ok(true);
        }

        let mut queue = vec![*descendant];
        let mut visited = std::collections::HashSet::new();

        while let Some(cid) = queue.pop() {
            if !visited.insert(cid) {
                continue;
            }

            if &cid == ancestor {
                return Ok(true);
            }

            let block = self
                .store
                .get(&cid)
                .await?
                .ok_or_else(|| Error::NotFound(format!("Commit not found: {cid}")))?;

            let commit = Commit::from_block(&block)?;
            queue.extend(commit.parents.iter().copied());
        }

        Ok(false)
    }

    /// Merge a branch into the current HEAD
    ///
    /// # Arguments
    /// * `branch_cid` - The commit CID to merge into current HEAD
    /// * `message` - Merge commit message
    /// * `author` - Author of the merge commit
    /// * `strategy` - Merge strategy to use
    pub async fn merge(
        &self,
        branch_cid: &Cid,
        message: String,
        author: Author,
        strategy: MergeStrategy,
    ) -> Result<MergeResult> {
        let head_cid = self
            .head
            .read()
            .ok_or_else(|| Error::Storage("No HEAD commit".to_string()))?;

        // Check if already up to date
        if &head_cid == branch_cid {
            return Ok(MergeResult::FastForward { target: head_cid });
        }

        // Check if fast-forward is possible
        if self.is_ancestor(&head_cid, branch_cid).await? {
            // Fast-forward merge
            *self.head.write() = Some(*branch_cid);

            // Update current branch ref
            let branch_name = self.current_branch.read().clone();
            self.refs_cache.insert(branch_name.clone(), *branch_cid);

            return Ok(MergeResult::FastForward {
                target: *branch_cid,
            });
        }

        // Fast-forward only strategy fails if not possible
        if strategy == MergeStrategy::FastForward {
            return Err(Error::Storage(
                "Fast-forward not possible, branches have diverged".to_string(),
            ));
        }

        // Load both commits
        let head_block = self
            .store
            .get(&head_cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("HEAD commit not found: {head_cid}")))?;
        let head_commit = Commit::from_block(&head_block)?;

        let branch_block = self
            .store
            .get(branch_cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Branch commit not found: {branch_cid}")))?;
        let branch_commit = Commit::from_block(&branch_block)?;

        // Three-way merge: create a merge commit with both parents
        match strategy {
            MergeStrategy::ThreeWay | MergeStrategy::Ours | MergeStrategy::Theirs => {
                // For now, we'll use the branch's root for the merge
                // In a real implementation, we'd need to:
                // 1. Find common ancestor
                // 2. Compute diff from ancestor to head
                // 3. Compute diff from ancestor to branch
                // 4. Apply both diffs and resolve conflicts
                // For simplicity, we'll use the strategy to pick a root:
                let merge_root = match strategy {
                    MergeStrategy::Ours => head_commit.root,
                    MergeStrategy::Theirs => branch_commit.root,
                    MergeStrategy::ThreeWay => {
                        // Use branch's root (in real impl, would merge properly)
                        branch_commit.root
                    }
                    _ => unreachable!(),
                };

                // Create merge commit with both parents
                let mut merge_commit = Commit::new(
                    vec![head_cid, *branch_cid],
                    merge_root,
                    message,
                    author,
                    HashMap::new(),
                );

                let merge_cid = merge_commit.finalize()?;
                let merge_block = merge_commit.to_block()?;
                self.store.put(&merge_block).await?;

                // Update HEAD
                *self.head.write() = Some(merge_cid);

                // Update current branch ref
                let branch_name = self.current_branch.read().clone();
                self.refs_cache.insert(branch_name.clone(), merge_cid);

                Ok(MergeResult::MergeCommit { commit: merge_cid })
            }
            MergeStrategy::FastForward => unreachable!(), // Already handled above
        }
    }

    /// Merge a named branch into current HEAD
    pub async fn merge_branch(
        &self,
        branch_name: &str,
        message: String,
        author: Author,
        strategy: MergeStrategy,
    ) -> Result<MergeResult> {
        let branch_ref = self.get_ref(branch_name).await?;
        self.merge(&branch_ref.commit, message, author, strategy)
            .await
    }
}

/// Commit builder for ergonomic commit creation
pub struct CommitBuilder {
    parents: Vec<Cid>,
    root: Option<Cid>,
    message: Option<String>,
    author: Option<Author>,
    metadata: HashMap<String, String>,
}

impl CommitBuilder {
    /// Create a new commit builder
    pub fn new() -> Self {
        Self {
            parents: Vec::new(),
            root: None,
            message: None,
            author: None,
            metadata: HashMap::new(),
        }
    }

    /// Set parent commits
    #[must_use]
    pub fn parents(mut self, parents: Vec<Cid>) -> Self {
        self.parents = parents;
        self
    }

    /// Set root CID
    #[must_use]
    pub fn root(mut self, root: Cid) -> Self {
        self.root = Some(root);
        self
    }

    /// Set commit message
    #[must_use]
    pub fn message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }

    /// Set author
    #[must_use]
    pub fn author(mut self, author: Author) -> Self {
        self.author = Some(author);
        self
    }

    /// Add metadata entry
    #[must_use]
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Build the commit
    pub fn build(self) -> Result<Commit> {
        let root = self
            .root
            .ok_or_else(|| Error::Storage("Root CID is required".to_string()))?;
        let message = self
            .message
            .ok_or_else(|| Error::Storage("Commit message is required".to_string()))?;
        let author = self
            .author
            .ok_or_else(|| Error::Storage("Author is required".to_string()))?;

        Ok(Commit::new(
            self.parents,
            root,
            message,
            author,
            self.metadata,
        ))
    }
}

impl Default for CommitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};

    #[tokio::test]
    async fn test_commit_creation() {
        let author = Author {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };

        let root_block = Block::new(Bytes::from("model weights"))
            .expect("test: Block::new with valid bytes should succeed");
        let root_cid = *root_block.cid();

        let mut commit = Commit::new(
            vec![],
            root_cid,
            "Initial commit".to_string(),
            author,
            HashMap::new(),
        );

        let commit_cid = commit
            .finalize()
            .expect("test: commit finalize should succeed");
        assert!(commit.cid.is_some());
        assert_eq!(commit.cid(), &commit_cid);
        assert!(commit.is_initial());
    }

    #[tokio::test]
    async fn test_commit_serialization() {
        let author = Author {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };

        let root_block = Block::new(Bytes::from("model weights"))
            .expect("test: Block::new with valid bytes should succeed");
        let root_cid = *root_block.cid();

        let mut commit = Commit::new(
            vec![],
            root_cid,
            "Initial commit".to_string(),
            author.clone(),
            HashMap::new(),
        );

        commit
            .finalize()
            .expect("test: commit finalize should succeed");
        let commit_block = commit
            .to_block()
            .expect("test: commit to_block should succeed");
        let deserialized =
            Commit::from_block(&commit_block).expect("test: Commit from_block should succeed");

        assert_eq!(commit, deserialized);
    }

    #[tokio::test]
    async fn test_version_control_initial_commit() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-vcs-test-initial"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed"),
        );
        let vcs = VersionControl::new(store.clone());

        // Create root block (model)
        let model_block =
            Block::new(Bytes::from("model v1")).expect("test: Block::new should succeed");
        let model_cid = *model_block.cid();
        store
            .put(&model_block)
            .await
            .expect("test: store.put should succeed");

        // Create initial commit
        let author = Author {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };

        let commit_cid = vcs
            .commit(
                model_cid,
                "Initial commit".to_string(),
                author,
                HashMap::new(),
            )
            .await
            .expect("test: vcs.commit should succeed");

        // Verify HEAD is updated
        assert_eq!(vcs.head(), Some(commit_cid));

        // Verify we can load the commit
        let commit_block = store
            .get(&commit_cid)
            .await
            .expect("test: store.get should succeed")
            .expect("test: block should exist");
        let commit =
            Commit::from_block(&commit_block).expect("test: Commit::from_block should succeed");
        assert_eq!(commit.root, model_cid);
        assert_eq!(commit.message, "Initial commit");
        assert!(commit.is_initial());
    }

    #[tokio::test]
    async fn test_version_control_multiple_commits() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-vcs-test-multiple"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed"),
        );
        let vcs = VersionControl::new(store.clone());

        let author = Author {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };

        // First commit
        let model1 = Block::new(Bytes::from("model v1"))
            .expect("test: Block::new for model1 should succeed");
        store
            .put(&model1)
            .await
            .expect("test: store.put model1 should succeed");
        let commit1 = vcs
            .commit(
                *model1.cid(),
                "First commit".to_string(),
                author.clone(),
                HashMap::new(),
            )
            .await
            .expect("test: first commit should succeed");

        // Second commit
        let model2 = Block::new(Bytes::from("model v2"))
            .expect("test: Block::new for model2 should succeed");
        store
            .put(&model2)
            .await
            .expect("test: store.put model2 should succeed");
        let commit2 = vcs
            .commit(
                *model2.cid(),
                "Second commit".to_string(),
                author,
                HashMap::new(),
            )
            .await
            .expect("test: second commit should succeed");

        // Verify HEAD is at second commit
        assert_eq!(vcs.head(), Some(commit2));

        // Load second commit and verify it has first commit as parent
        let commit2_block = store
            .get(&commit2)
            .await
            .expect("test: store.get commit2 should succeed")
            .expect("test: commit2 block should exist");
        let commit2_obj =
            Commit::from_block(&commit2_block).expect("test: Commit::from_block should succeed");
        assert_eq!(commit2_obj.parents, vec![commit1]);
    }

    #[tokio::test]
    async fn test_checkout() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-vcs-test-checkout"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed"),
        );
        let vcs = VersionControl::new(store.clone());

        let author = Author {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };

        // Create two commits
        let model1 = Block::new(Bytes::from("model v1"))
            .expect("test: Block::new for model1 should succeed");
        store
            .put(&model1)
            .await
            .expect("test: store.put model1 should succeed");
        let commit1 = vcs
            .commit(
                *model1.cid(),
                "First".to_string(),
                author.clone(),
                HashMap::new(),
            )
            .await
            .expect("test: first commit should succeed");

        let model2 = Block::new(Bytes::from("model v2"))
            .expect("test: Block::new for model2 should succeed");
        store
            .put(&model2)
            .await
            .expect("test: store.put model2 should succeed");
        let _commit2 = vcs
            .commit(*model2.cid(), "Second".to_string(), author, HashMap::new())
            .await
            .expect("test: second commit should succeed");

        // Checkout to first commit
        let root = vcs
            .checkout(&commit1)
            .await
            .expect("test: checkout should succeed");
        assert_eq!(root, *model1.cid());
        assert_eq!(vcs.head(), Some(commit1));
    }

    #[tokio::test]
    async fn test_commit_log() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-vcs-test-log"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed"),
        );
        let vcs = VersionControl::new(store.clone());

        let author = Author {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };

        // Create three commits
        let mut commits = Vec::new();
        for i in 1..=3 {
            let model = Block::new(Bytes::from(format!("model v{}", i)))
                .expect("test: Block::new should succeed");
            store
                .put(&model)
                .await
                .expect("test: store.put should succeed");
            let commit = vcs
                .commit(
                    *model.cid(),
                    format!("Commit {}", i),
                    author.clone(),
                    HashMap::new(),
                )
                .await
                .expect("test: vcs.commit should succeed");
            commits.push(commit);
        }

        // Get log from HEAD
        let log = vcs
            .log(&commits[2], 10)
            .await
            .expect("test: vcs.log should succeed");
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].message, "Commit 3");
        assert_eq!(log[1].message, "Commit 2");
        assert_eq!(log[2].message, "Commit 1");
    }

    #[test]
    fn test_commit_builder() {
        let author = Author {
            name: "Builder".to_string(),
            email: "builder@example.com".to_string(),
        };

        let root_block = Block::new(Bytes::from("root")).expect("test: Block::new should succeed");

        let commit = CommitBuilder::new()
            .root(*root_block.cid())
            .message("Test commit".to_string())
            .author(author.clone())
            .metadata("key1".to_string(), "value1".to_string())
            .build()
            .expect("test: CommitBuilder::build should succeed");

        assert_eq!(commit.message, "Test commit");
        assert_eq!(commit.author, author);
        assert_eq!(
            commit
                .metadata
                .get("key1")
                .expect("test: metadata key1 should exist"),
            "value1"
        );
    }
}
