//! Provenance tracking for ML models
//!
//! This module provides comprehensive provenance tracking including:
//! - Data lineage as Merkle DAG
//! - Backward tracing to source data
//! - Attribution metadata (contributors, datasets, licenses)
//! - Training history and reproducibility

use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during provenance operations
#[derive(Debug, Error)]
pub enum ProvenanceError {
    #[error("Provenance record not found: {0}")]
    RecordNotFound(String),

    #[error("Circular dependency detected")]
    CircularDependency,

    #[error("Invalid provenance chain")]
    InvalidChain,

    #[error("Missing required metadata: {0}")]
    MissingMetadata(String),
}

/// License types for datasets and models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum License {
    /// MIT License
    MIT,
    /// Apache 2.0
    Apache2,
    /// GPL v3
    GPLv3,
    /// BSD 3-Clause
    BSD3Clause,
    /// Creative Commons Attribution
    CCBY,
    /// Creative Commons Attribution-ShareAlike
    CCBYSA,
    /// Proprietary
    Proprietary,
    /// Custom license
    Custom(String),
    /// Unknown license
    Unknown,
}

impl std::fmt::Display for License {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            License::MIT => write!(f, "MIT"),
            License::Apache2 => write!(f, "Apache-2.0"),
            License::GPLv3 => write!(f, "GPL-3.0"),
            License::BSD3Clause => write!(f, "BSD-3-Clause"),
            License::CCBY => write!(f, "CC-BY"),
            License::CCBYSA => write!(f, "CC-BY-SA"),
            License::Proprietary => write!(f, "Proprietary"),
            License::Custom(s) => write!(f, "Custom: {}", s),
            License::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Attribution information for contributors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribution {
    /// Contributor name
    pub name: String,
    /// Email or contact
    pub contact: Option<String>,
    /// Organization
    pub organization: Option<String>,
    /// Role (e.g., "data provider", "model trainer", "code contributor")
    pub role: String,
    /// Contribution timestamp
    pub timestamp: i64,
}

impl Attribution {
    /// Create a new attribution
    pub fn new(name: String, role: String) -> Self {
        Self {
            name,
            contact: None,
            organization: None,
            role,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Add contact information
    pub fn with_contact(mut self, contact: String) -> Self {
        self.contact = Some(contact);
        self
    }

    /// Add organization
    pub fn with_organization(mut self, organization: String) -> Self {
        self.organization = Some(organization);
        self
    }
}

/// Dataset provenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetProvenance {
    /// Dataset CID
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub dataset_cid: Cid,

    /// Dataset name
    pub name: String,

    /// Dataset version
    pub version: String,

    /// License
    pub license: License,

    /// Attribution
    pub attributions: Vec<Attribution>,

    /// Source URLs (if applicable)
    pub sources: Vec<String>,

    /// Description
    pub description: Option<String>,

    /// Creation timestamp
    pub created_at: i64,
}

impl DatasetProvenance {
    /// Create a new dataset provenance record
    pub fn new(dataset_cid: Cid, name: String, version: String, license: License) -> Self {
        Self {
            dataset_cid,
            name,
            version,
            license,
            attributions: Vec::new(),
            sources: Vec::new(),
            description: None,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Add an attribution
    pub fn add_attribution(mut self, attribution: Attribution) -> Self {
        self.attributions.push(attribution);
        self
    }

    /// Add a source URL
    pub fn add_source(mut self, source: String) -> Self {
        self.sources.push(source);
        self
    }

    /// Add description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// Hyperparameters for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Learning rate
    pub learning_rate: Option<f32>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Number of epochs
    pub epochs: Option<usize>,
    /// Optimizer name
    pub optimizer: Option<String>,
    /// Additional parameters
    pub custom: HashMap<String, String>,
}

impl Hyperparameters {
    /// Create new hyperparameters
    pub fn new() -> Self {
        Self {
            learning_rate: None,
            batch_size: None,
            epochs: None,
            optimizer: None,
            custom: HashMap::new(),
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    /// Set optimizer
    pub fn with_optimizer(mut self, optimizer: String) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Add custom parameter
    pub fn add_param(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Training provenance for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProvenance {
    /// Model CID
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub model_cid: Cid,

    /// Parent model CID (if fine-tuning or transfer learning)
    #[serde(serialize_with = "serialize_optional_cid")]
    #[serde(deserialize_with = "deserialize_optional_cid")]
    pub parent_model: Option<Cid>,

    /// Training datasets
    #[serde(serialize_with = "serialize_cid_vec")]
    #[serde(deserialize_with = "deserialize_cid_vec")]
    pub training_datasets: Vec<Cid>,

    /// Validation datasets
    #[serde(serialize_with = "serialize_cid_vec")]
    #[serde(deserialize_with = "deserialize_cid_vec")]
    pub validation_datasets: Vec<Cid>,

    /// Hyperparameters
    pub hyperparameters: Hyperparameters,

    /// Training metrics (final)
    pub metrics: HashMap<String, f32>,

    /// Attribution
    pub attributions: Vec<Attribution>,

    /// License
    pub license: License,

    /// Training start time
    pub started_at: i64,

    /// Training end time
    pub completed_at: Option<i64>,

    /// Code repository (if applicable)
    pub code_repository: Option<String>,

    /// Code commit hash
    pub code_commit: Option<String>,

    /// Hardware used (e.g., "8x NVIDIA A100")
    pub hardware: Option<String>,

    /// Training framework (e.g., "PyTorch 2.0")
    pub framework: Option<String>,
}

impl TrainingProvenance {
    /// Create a new training provenance record
    pub fn new(model_cid: Cid, training_datasets: Vec<Cid>, license: License) -> Self {
        Self {
            model_cid,
            parent_model: None,
            training_datasets,
            validation_datasets: Vec::new(),
            hyperparameters: Hyperparameters::new(),
            metrics: HashMap::new(),
            attributions: Vec::new(),
            license,
            started_at: chrono::Utc::now().timestamp(),
            completed_at: None,
            code_repository: None,
            code_commit: None,
            hardware: None,
            framework: None,
        }
    }

    /// Set parent model
    pub fn with_parent(mut self, parent_cid: Cid) -> Self {
        self.parent_model = Some(parent_cid);
        self
    }

    /// Add validation dataset
    pub fn add_validation_dataset(mut self, dataset_cid: Cid) -> Self {
        self.validation_datasets.push(dataset_cid);
        self
    }

    /// Set hyperparameters
    pub fn with_hyperparameters(mut self, hyperparameters: Hyperparameters) -> Self {
        self.hyperparameters = hyperparameters;
        self
    }

    /// Add metric
    pub fn add_metric(mut self, name: String, value: f32) -> Self {
        self.metrics.insert(name, value);
        self
    }

    /// Add attribution
    pub fn add_attribution(mut self, attribution: Attribution) -> Self {
        self.attributions.push(attribution);
        self
    }

    /// Mark training as complete
    pub fn complete(mut self) -> Self {
        self.completed_at = Some(chrono::Utc::now().timestamp());
        self
    }

    /// Set code repository
    pub fn with_code_repository(mut self, repo: String, commit: String) -> Self {
        self.code_repository = Some(repo);
        self.code_commit = Some(commit);
        self
    }

    /// Set hardware info
    pub fn with_hardware(mut self, hardware: String) -> Self {
        self.hardware = Some(hardware);
        self
    }

    /// Set framework
    pub fn with_framework(mut self, framework: String) -> Self {
        self.framework = Some(framework);
        self
    }
}

/// Complete provenance graph for tracking lineage
#[derive(Debug, Clone)]
pub struct ProvenanceGraph {
    /// Dataset provenance records
    datasets: HashMap<String, DatasetProvenance>,

    /// Training provenance records
    training_records: HashMap<String, TrainingProvenance>,
}

impl ProvenanceGraph {
    /// Create a new provenance graph
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            training_records: HashMap::new(),
        }
    }

    /// Add a dataset provenance record
    pub fn add_dataset(&mut self, provenance: DatasetProvenance) {
        self.datasets
            .insert(provenance.dataset_cid.to_string(), provenance);
    }

    /// Add a training provenance record
    pub fn add_training(&mut self, provenance: TrainingProvenance) {
        self.training_records
            .insert(provenance.model_cid.to_string(), provenance);
    }

    /// Get dataset provenance
    pub fn get_dataset(&self, dataset_cid: &Cid) -> Option<&DatasetProvenance> {
        self.datasets.get(&dataset_cid.to_string())
    }

    /// Get training provenance
    pub fn get_training(&self, model_cid: &Cid) -> Option<&TrainingProvenance> {
        self.training_records.get(&model_cid.to_string())
    }

    /// Trace lineage backward from a model to all source datasets
    pub fn trace_lineage(&self, model_cid: &Cid) -> Result<LineageTrace, ProvenanceError> {
        let mut visited = HashSet::new();
        let mut datasets = Vec::new();
        let mut models = Vec::new();

        self.trace_recursive(model_cid, &mut visited, &mut datasets, &mut models)?;

        Ok(LineageTrace {
            target_model: *model_cid,
            datasets,
            models,
        })
    }

    /// Recursive helper for tracing lineage
    fn trace_recursive(
        &self,
        model_cid: &Cid,
        visited: &mut HashSet<Cid>,
        datasets: &mut Vec<Cid>,
        models: &mut Vec<Cid>,
    ) -> Result<(), ProvenanceError> {
        if visited.contains(model_cid) {
            return Err(ProvenanceError::CircularDependency);
        }

        visited.insert(*model_cid);

        let training = self
            .get_training(model_cid)
            .ok_or_else(|| ProvenanceError::RecordNotFound(model_cid.to_string()))?;

        models.push(*model_cid);

        // Add datasets
        for dataset_cid in &training.training_datasets {
            if !datasets.contains(dataset_cid) {
                datasets.push(*dataset_cid);
            }
        }

        for dataset_cid in &training.validation_datasets {
            if !datasets.contains(dataset_cid) {
                datasets.push(*dataset_cid);
            }
        }

        // Recursively trace parent model
        if let Some(parent_cid) = training.parent_model {
            self.trace_recursive(&parent_cid, visited, datasets, models)?;
        }

        Ok(())
    }

    /// Get all attributions for a model (including from datasets)
    pub fn get_all_attributions(
        &self,
        model_cid: &Cid,
    ) -> Result<Vec<Attribution>, ProvenanceError> {
        let lineage = self.trace_lineage(model_cid)?;
        let mut attributions = Vec::new();

        // Get model attributions
        if let Some(training) = self.get_training(model_cid) {
            attributions.extend(training.attributions.clone());
        }

        // Get dataset attributions
        for dataset_cid in &lineage.datasets {
            if let Some(dataset) = self.get_dataset(dataset_cid) {
                attributions.extend(dataset.attributions.clone());
            }
        }

        Ok(attributions)
    }

    /// Get all licenses in the lineage
    pub fn get_all_licenses(&self, model_cid: &Cid) -> Result<HashSet<License>, ProvenanceError> {
        let lineage = self.trace_lineage(model_cid)?;
        let mut licenses = HashSet::new();

        // Get model licenses
        for model in &lineage.models {
            if let Some(training) = self.get_training(model) {
                licenses.insert(training.license.clone());
            }
        }

        // Get dataset licenses
        for dataset_cid in &lineage.datasets {
            if let Some(dataset) = self.get_dataset(dataset_cid) {
                licenses.insert(dataset.license.clone());
            }
        }

        Ok(licenses)
    }

    /// Check if lineage is reproducible (has all necessary metadata)
    pub fn is_reproducible(&self, model_cid: &Cid) -> bool {
        if let Some(training) = self.get_training(model_cid) {
            // Check for required metadata
            training.code_repository.is_some()
                && training.code_commit.is_some()
                && training.hyperparameters.learning_rate.is_some()
                && !training.training_datasets.is_empty()
        } else {
            false
        }
    }
}

impl Default for ProvenanceGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of lineage tracing
#[derive(Debug, Clone)]
pub struct LineageTrace {
    /// Target model
    pub target_model: Cid,
    /// All datasets in the lineage
    pub datasets: Vec<Cid>,
    /// All models in the lineage (including target)
    pub models: Vec<Cid>,
}

impl LineageTrace {
    /// Get the depth of the lineage (number of model generations)
    pub fn depth(&self) -> usize {
        self.models.len()
    }

    /// Get the number of unique datasets
    pub fn dataset_count(&self) -> usize {
        self.datasets.len()
    }
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

fn serialize_optional_cid<S>(cid: &Option<Cid>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    match cid {
        Some(c) => Some(c.to_string()).serialize(serializer),
        None => None::<String>.serialize(serializer),
    }
}

fn deserialize_optional_cid<'de, D>(deserializer: D) -> Result<Option<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let opt = Option::<String>::deserialize(deserializer)?;
    opt.map(|s| s.parse().map_err(serde::de::Error::custom))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribution() {
        let attr = Attribution::new("John Doe".to_string(), "data provider".to_string())
            .with_contact("john@example.com".to_string())
            .with_organization("Example Corp".to_string());

        assert_eq!(attr.name, "John Doe");
        assert_eq!(attr.contact, Some("john@example.com".to_string()));
        assert_eq!(attr.organization, Some("Example Corp".to_string()));
    }

    #[test]
    fn test_dataset_provenance() {
        let dataset = DatasetProvenance::new(
            Cid::default(),
            "ImageNet".to_string(),
            "1.0".to_string(),
            License::CCBY,
        )
        .add_attribution(Attribution::new(
            "Stanford".to_string(),
            "creator".to_string(),
        ))
        .add_source("https://example.com/imagenet".to_string())
        .with_description("Large image dataset".to_string());

        assert_eq!(dataset.name, "ImageNet");
        assert_eq!(dataset.license, License::CCBY);
        assert_eq!(dataset.attributions.len(), 1);
    }

    #[test]
    fn test_hyperparameters() {
        let hparams = Hyperparameters::new()
            .with_learning_rate(0.001)
            .with_batch_size(32)
            .with_epochs(10)
            .with_optimizer("Adam".to_string())
            .add_param("weight_decay".to_string(), "0.0001".to_string());

        assert_eq!(hparams.learning_rate, Some(0.001));
        assert_eq!(hparams.batch_size, Some(32));
        assert_eq!(hparams.epochs, Some(10));
    }

    #[test]
    fn test_training_provenance() {
        let training = TrainingProvenance::new(Cid::default(), vec![Cid::default()], License::MIT)
            .with_hyperparameters(
                Hyperparameters::new()
                    .with_learning_rate(0.001)
                    .with_batch_size(32),
            )
            .add_metric("accuracy".to_string(), 0.95)
            .add_attribution(Attribution::new(
                "Jane Doe".to_string(),
                "trainer".to_string(),
            ))
            .complete();

        assert_eq!(training.training_datasets.len(), 1);
        assert_eq!(training.metrics.len(), 1);
        assert!(training.completed_at.is_some());
    }

    #[test]
    fn test_provenance_graph() {
        let mut graph = ProvenanceGraph::new();

        let dataset_cid = Cid::default();
        let dataset = DatasetProvenance::new(
            dataset_cid,
            "TestDataset".to_string(),
            "1.0".to_string(),
            License::MIT,
        );

        graph.add_dataset(dataset);

        let model_cid = Cid::default();
        let training = TrainingProvenance::new(model_cid, vec![dataset_cid], License::MIT);

        graph.add_training(training);

        assert!(graph.get_dataset(&dataset_cid).is_some());
        assert!(graph.get_training(&model_cid).is_some());
    }

    #[test]
    fn test_lineage_tracing() {
        let mut graph = ProvenanceGraph::new();

        let dataset_cid = Cid::default();
        let dataset = DatasetProvenance::new(
            dataset_cid,
            "TestDataset".to_string(),
            "1.0".to_string(),
            License::MIT,
        );
        graph.add_dataset(dataset);

        let model_cid = Cid::default();
        let training = TrainingProvenance::new(model_cid, vec![dataset_cid], License::MIT);
        graph.add_training(training);

        let lineage = graph
            .trace_lineage(&model_cid)
            .expect("test: should succeed");

        assert_eq!(lineage.depth(), 1);
        assert_eq!(lineage.dataset_count(), 1);
    }

    #[test]
    fn test_license_display() {
        assert_eq!(License::MIT.to_string(), "MIT");
        assert_eq!(License::Apache2.to_string(), "Apache-2.0");
        assert_eq!(
            License::Custom("Custom-1.0".to_string()).to_string(),
            "Custom: Custom-1.0"
        );
    }
}
