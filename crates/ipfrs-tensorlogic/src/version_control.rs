//! Model version control system for ML models
//!
//! This module provides Git-like version control for ML models:
//! - Commit/checkout operations
//! - Branching and merging
//! - Diff operations for models
//! - Model history tracking

use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during version control operations
#[derive(Debug, Error)]
pub enum VersionControlError {
    #[error("Commit not found: {0}")]
    CommitNotFound(String),

    #[error("Branch not found: {0}")]
    BranchNotFound(String),

    #[error("Branch already exists: {0}")]
    BranchAlreadyExists(String),

    #[error("Merge conflict in layer: {0}")]
    MergeConflict(String),

    #[error("Invalid commit ID: {0}")]
    InvalidCommitId(String),

    #[error("Cannot merge: {0}")]
    CannotMerge(String),

    #[error("Detached HEAD state")]
    DetachedHead,

    #[error("No parent commit")]
    NoParentCommit,
}

/// A commit in the model version history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCommit {
    /// Unique commit ID (CID of the commit itself)
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub id: Cid,

    /// Parent commit ID(s) (empty for initial commit, multiple for merges)
    #[serde(serialize_with = "serialize_cid_vec")]
    #[serde(deserialize_with = "deserialize_cid_vec")]
    pub parents: Vec<Cid>,

    /// Model CID (points to the actual model data)
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub model: Cid,

    /// Commit message
    pub message: String,

    /// Author
    pub author: String,

    /// Timestamp
    pub timestamp: i64,

    /// Metadata (hyperparameters, training info, etc.)
    pub metadata: HashMap<String, String>,
}

impl ModelCommit {
    /// Create a new commit
    pub fn new(id: Cid, parents: Vec<Cid>, model: Cid, message: String, author: String) -> Self {
        Self {
            id,
            parents,
            model,
            message,
            author,
            timestamp: chrono::Utc::now().timestamp(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the commit
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if this is a merge commit
    pub fn is_merge(&self) -> bool {
        self.parents.len() > 1
    }

    /// Check if this is an initial commit
    pub fn is_initial(&self) -> bool {
        self.parents.is_empty()
    }
}

/// A branch in the version control system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Branch name
    pub name: String,

    /// Current commit CID
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub head: Cid,

    /// Branch description
    pub description: Option<String>,
}

impl Branch {
    /// Create a new branch
    pub fn new(name: String, head: Cid) -> Self {
        Self {
            name,
            head,
            description: None,
        }
    }

    /// Add description to the branch
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// Model version control repository
#[derive(Debug, Clone)]
pub struct ModelRepository {
    /// All commits (commit ID -> commit)
    commits: HashMap<String, ModelCommit>,

    /// All branches (branch name -> branch)
    branches: HashMap<String, Branch>,

    /// Current branch name (None if detached HEAD)
    current_branch: Option<String>,

    /// Current HEAD commit
    head: Option<Cid>,
}

impl ModelRepository {
    /// Create a new empty repository
    pub fn new() -> Self {
        Self {
            commits: HashMap::new(),
            branches: HashMap::new(),
            current_branch: None,
            head: None,
        }
    }

    /// Initialize repository with an initial commit
    pub fn init(&mut self, initial_commit: ModelCommit) -> Result<(), VersionControlError> {
        let commit_id = initial_commit.id.to_string();
        let commit_cid = initial_commit.id;

        self.commits.insert(commit_id, initial_commit);

        // Create main branch
        let main_branch = Branch::new("main".to_string(), commit_cid);
        self.branches.insert("main".to_string(), main_branch);
        self.current_branch = Some("main".to_string());
        self.head = Some(commit_cid);

        Ok(())
    }

    /// Create a commit
    pub fn commit(
        &mut self,
        model: Cid,
        message: String,
        author: String,
    ) -> Result<ModelCommit, VersionControlError> {
        let parents = if let Some(head) = self.head {
            vec![head]
        } else {
            vec![]
        };

        // In a real implementation, we would compute the CID from the commit content
        // For now, use a placeholder
        let commit_id = Cid::default();

        let commit = ModelCommit::new(commit_id, parents, model, message, author);

        self.commits.insert(commit_id.to_string(), commit.clone());
        self.head = Some(commit_id);

        // Update current branch if we're on one
        if let Some(branch_name) = &self.current_branch {
            if let Some(branch) = self.branches.get_mut(branch_name) {
                branch.head = commit_id;
            }
        }

        Ok(commit)
    }

    /// Checkout to a specific commit or branch
    pub fn checkout(&mut self, target: &str) -> Result<(), VersionControlError> {
        // Try to interpret target as a branch name first
        if let Some(branch) = self.branches.get(target) {
            self.current_branch = Some(target.to_string());
            self.head = Some(branch.head);
            return Ok(());
        }

        // Try to interpret target as a commit ID
        if let Some(commit) = self.commits.get(target) {
            self.current_branch = None; // Detached HEAD
            self.head = Some(commit.id);
            return Ok(());
        }

        Err(VersionControlError::CommitNotFound(target.to_string()))
    }

    /// Create a new branch
    pub fn create_branch(
        &mut self,
        name: String,
        start_point: Option<Cid>,
    ) -> Result<(), VersionControlError> {
        if self.branches.contains_key(&name) {
            return Err(VersionControlError::BranchAlreadyExists(name));
        }

        let head = start_point
            .or(self.head)
            .ok_or(VersionControlError::NoParentCommit)?;

        let branch = Branch::new(name.clone(), head);
        self.branches.insert(name, branch);

        Ok(())
    }

    /// Delete a branch
    pub fn delete_branch(&mut self, name: &str) -> Result<(), VersionControlError> {
        if !self.branches.contains_key(name) {
            return Err(VersionControlError::BranchNotFound(name.to_string()));
        }

        if let Some(current) = &self.current_branch {
            if current == name {
                return Err(VersionControlError::CannotMerge(
                    "Cannot delete current branch".to_string(),
                ));
            }
        }

        self.branches.remove(name);
        Ok(())
    }

    /// List all branches
    pub fn list_branches(&self) -> Vec<&Branch> {
        self.branches.values().collect()
    }

    /// Get current branch name
    pub fn current_branch(&self) -> Option<&str> {
        self.current_branch.as_deref()
    }

    /// Get current HEAD commit
    pub fn head_commit(&self) -> Option<&ModelCommit> {
        self.head
            .as_ref()
            .and_then(|cid| self.commits.get(&cid.to_string()))
    }

    /// Get a commit by ID
    pub fn get_commit(&self, commit_id: &str) -> Option<&ModelCommit> {
        self.commits.get(commit_id)
    }

    /// Get commit history from a starting commit
    pub fn get_history(&self, start: &Cid, max_count: Option<usize>) -> Vec<&ModelCommit> {
        let mut history = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![start];

        while let Some(cid) = queue.pop() {
            if visited.contains(cid) {
                continue;
            }

            if let Some(max) = max_count {
                if history.len() >= max {
                    break;
                }
            }

            if let Some(commit) = self.commits.get(&cid.to_string()) {
                visited.insert(*cid);
                history.push(commit);

                // Add parents to queue
                for parent in &commit.parents {
                    if !visited.contains(parent) {
                        queue.push(parent);
                    }
                }
            }
        }

        history
    }

    /// Perform a fast-forward merge
    pub fn merge_fast_forward(&mut self, branch: &str) -> Result<(), VersionControlError> {
        let target_branch = self
            .branches
            .get(branch)
            .ok_or_else(|| VersionControlError::BranchNotFound(branch.to_string()))?;

        let target_head = target_branch.head;

        // Update current branch
        if let Some(current_name) = &self.current_branch {
            if let Some(current_branch) = self.branches.get_mut(current_name) {
                current_branch.head = target_head;
            }
        } else {
            return Err(VersionControlError::DetachedHead);
        }

        self.head = Some(target_head);

        Ok(())
    }

    /// Check if fast-forward merge is possible
    pub fn can_fast_forward(&self, branch: &str) -> Result<bool, VersionControlError> {
        let target_branch = self
            .branches
            .get(branch)
            .ok_or_else(|| VersionControlError::BranchNotFound(branch.to_string()))?;

        let current_head = self.head.ok_or(VersionControlError::NoParentCommit)?;

        // Check if current head is an ancestor of target head
        let history = self.get_history(&target_branch.head, None);

        Ok(history.iter().any(|c| c.id == current_head))
    }
}

impl Default for ModelRepository {
    fn default() -> Self {
        Self::new()
    }
}

/// Model difference representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiff {
    /// Layers that were added
    pub added_layers: Vec<String>,

    /// Layers that were removed
    pub removed_layers: Vec<String>,

    /// Layers that were modified
    pub modified_layers: Vec<LayerDiff>,
}

/// Difference in a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDiff {
    /// Layer name
    pub name: String,

    /// Shape changed
    pub shape_changed: bool,

    /// Previous shape
    pub old_shape: Vec<usize>,

    /// New shape
    pub new_shape: Vec<usize>,

    /// L2 norm of the difference
    pub l2_diff: f32,

    /// Maximum absolute difference
    pub max_diff: f32,
}

/// Model differ for computing differences between models
pub struct ModelDiffer;

impl ModelDiffer {
    /// Compute diff between two models
    pub fn diff(
        model_a: &HashMap<String, Vec<f32>>,
        model_b: &HashMap<String, Vec<f32>>,
    ) -> ModelDiff {
        let mut added_layers = Vec::new();
        let mut removed_layers = Vec::new();
        let mut modified_layers = Vec::new();

        let keys_a: HashSet<_> = model_a.keys().collect();
        let keys_b: HashSet<_> = model_b.keys().collect();

        // Find added layers
        for key in keys_b.difference(&keys_a) {
            added_layers.push((*key).clone());
        }

        // Find removed layers
        for key in keys_a.difference(&keys_b) {
            removed_layers.push((*key).clone());
        }

        // Find modified layers
        for key in keys_a.intersection(&keys_b) {
            let values_a = &model_a[*key];
            let values_b = &model_b[*key];

            let shape_changed = values_a.len() != values_b.len();

            if shape_changed || !values_equal(values_a, values_b) {
                let (l2_diff, max_diff) = compute_diffs(values_a, values_b);

                modified_layers.push(LayerDiff {
                    name: (*key).clone(),
                    shape_changed,
                    old_shape: vec![values_a.len()],
                    new_shape: vec![values_b.len()],
                    l2_diff,
                    max_diff,
                });
            }
        }

        ModelDiff {
            added_layers,
            removed_layers,
            modified_layers,
        }
    }

    /// Check if diff has any changes
    pub fn has_changes(diff: &ModelDiff) -> bool {
        !diff.added_layers.is_empty()
            || !diff.removed_layers.is_empty()
            || !diff.modified_layers.is_empty()
    }
}

/// Check if two float vectors are equal within tolerance
fn values_equal(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
}

/// Compute L2 and max difference between two vectors
fn compute_diffs(a: &[f32], b: &[f32]) -> (f32, f32) {
    let min_len = a.len().min(b.len());

    let mut l2_sum: f32 = 0.0;
    let mut max_diff: f32 = 0.0;

    for i in 0..min_len {
        let diff = (a[i] - b[i]).abs();
        l2_sum += diff * diff;
        max_diff = max_diff.max(diff);
    }

    let l2_diff = l2_sum.sqrt();

    (l2_diff, max_diff)
}

// Helper functions for serializing/deserializing Vec<Cid>
fn serialize_cid_vec<S>(cids: &[Cid], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    let strings: Vec<String> = cids.iter().map(|c| c.to_string()).collect();
    strings.serialize(serializer)
}

fn deserialize_cid_vec<'de, D>(deserializer: D) -> Result<Vec<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let strings = Vec::<String>::deserialize(deserializer)?;
    strings
        .into_iter()
        .map(|s| s.parse().map_err(serde::de::Error::custom))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_commit() {
        let commit = ModelCommit::new(
            Cid::default(),
            vec![],
            Cid::default(),
            "Initial commit".to_string(),
            "test@example.com".to_string(),
        );

        assert!(commit.is_initial());
        assert!(!commit.is_merge());
    }

    #[test]
    fn test_repository_init() {
        let mut repo = ModelRepository::new();

        let commit = ModelCommit::new(
            Cid::default(),
            vec![],
            Cid::default(),
            "Initial commit".to_string(),
            "test@example.com".to_string(),
        );

        repo.init(commit).expect("test: should succeed");

        assert_eq!(repo.current_branch(), Some("main"));
        assert!(repo.head_commit().is_some());
    }

    #[test]
    fn test_branch_creation() {
        let mut repo = ModelRepository::new();

        let commit = ModelCommit::new(
            Cid::default(),
            vec![],
            Cid::default(),
            "Initial commit".to_string(),
            "test@example.com".to_string(),
        );

        repo.init(commit).expect("test: should succeed");

        repo.create_branch("develop".to_string(), None)
            .expect("test: should succeed");

        assert_eq!(repo.list_branches().len(), 2);
    }

    #[test]
    fn test_checkout() {
        let mut repo = ModelRepository::new();

        let commit = ModelCommit::new(
            Cid::default(),
            vec![],
            Cid::default(),
            "Initial commit".to_string(),
            "test@example.com".to_string(),
        );

        repo.init(commit).expect("test: should succeed");
        repo.create_branch("develop".to_string(), None)
            .expect("test: should succeed");

        repo.checkout("develop").expect("test: should succeed");

        assert_eq!(repo.current_branch(), Some("develop"));
    }

    #[test]
    fn test_model_diff() {
        let mut model_a = HashMap::new();
        model_a.insert("layer1".to_string(), vec![1.0, 2.0, 3.0]);
        model_a.insert("layer2".to_string(), vec![4.0, 5.0]);

        let mut model_b = HashMap::new();
        model_b.insert("layer1".to_string(), vec![1.1, 2.1, 3.1]);
        model_b.insert("layer3".to_string(), vec![6.0, 7.0]);

        let diff = ModelDiffer::diff(&model_a, &model_b);

        assert_eq!(diff.added_layers.len(), 1);
        assert_eq!(diff.removed_layers.len(), 1);
        assert_eq!(diff.modified_layers.len(), 1);
        assert!(ModelDiffer::has_changes(&diff));
    }

    #[test]
    fn test_layer_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.5, 3.5];

        let (l2_diff, max_diff) = compute_diffs(&a, &b);

        assert!(l2_diff > 0.0);
        assert!((max_diff - 0.5).abs() < 1e-6);
    }
}
