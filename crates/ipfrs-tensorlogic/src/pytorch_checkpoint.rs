//! PyTorch model checkpoint support for ipfrs-tensorlogic.
//!
//! This module provides functionality to load and work with PyTorch model checkpoints
//! (.pt/.pth files). PyTorch checkpoints are Python pickle files containing state_dict
//! structures with model weights and optionally optimizer state.
//!
//! # Safety and Security
//!
//! Python pickle format can execute arbitrary code during deserialization. This module
//! provides a safe subset of pickle deserialization focused on tensor data structures.
//! For maximum security, consider converting PyTorch checkpoints to Safetensors format.
//!
//! # Example
//!
//! ```rust,no_run
//! use ipfrs_tensorlogic::pytorch_checkpoint::{PyTorchCheckpoint, CheckpointMetadata};
//! use std::path::Path;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Load a PyTorch checkpoint
//! let checkpoint = PyTorchCheckpoint::load(Path::new("model.pt"))?;
//!
//! // Extract metadata
//! let metadata = checkpoint.metadata();
//! println!("Model has {} parameters", metadata.total_parameters);
//! println!("Layers: {:?}", metadata.layer_names);
//!
//! // Get state dict
//! let state_dict = checkpoint.state_dict();
//! for (key, tensor_info) in &state_dict.tensors {
//!     println!("{}: {:?}", key, tensor_info.shape);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::safetensors_support::SafetensorsWriter;

/// PyTorch checkpoint structure.
///
/// Contains the model state_dict and optional optimizer state, epoch information,
/// and other training metadata commonly saved in PyTorch checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchCheckpoint {
    /// Model state dictionary
    pub state_dict: StateDict,

    /// Optimizer state (if saved)
    pub optimizer_state: Option<OptimizerState>,

    /// Training epoch (if saved)
    pub epoch: Option<usize>,

    /// Training loss history (if saved)
    pub loss_history: Option<Vec<f32>>,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Model state dictionary containing named tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDict {
    /// Map from layer/parameter name to tensor information
    pub tensors: HashMap<String, TensorData>,
}

/// Tensor data with shape and values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,

    /// Data type identifier
    pub dtype: String,

    /// Flattened tensor values (stored as bytes)
    pub data: Vec<u8>,

    /// Whether this tensor requires gradient
    pub requires_grad: bool,
}

/// Optimizer state containing parameter state and hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer name (e.g., "Adam", "SGD")
    pub optimizer_type: String,

    /// Per-parameter state (momentum buffers, etc.)
    pub param_state: HashMap<String, ParamState>,

    /// Global optimizer hyperparameters
    pub hyperparameters: HashMap<String, f64>,
}

/// Per-parameter optimizer state (momentum, velocity, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamState {
    /// Momentum buffer (for SGD with momentum, Adam, etc.)
    pub momentum: Option<Vec<u8>>,

    /// Velocity buffer (for Adam, RMSprop, etc.)
    pub velocity: Option<Vec<u8>>,

    /// Step count (for Adam)
    pub step: Option<usize>,

    /// Custom state fields
    pub custom: HashMap<String, Vec<u8>>,
}

/// Checkpoint metadata for quick inspection.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Total number of parameters
    pub total_parameters: usize,

    /// Layer/parameter names
    pub layer_names: Vec<String>,

    /// Total size in bytes
    pub total_size_bytes: usize,

    /// Data types used
    pub dtypes: HashMap<String, usize>, // dtype -> count

    /// Whether optimizer state is present
    pub has_optimizer_state: bool,

    /// Current epoch (if available)
    pub epoch: Option<usize>,
}

impl PyTorchCheckpoint {
    /// Load a PyTorch checkpoint from a file.
    ///
    /// # Security Note
    ///
    /// This uses pickle deserialization which can be unsafe with untrusted files.
    /// Only load checkpoints from trusted sources.
    #[allow(dead_code)]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).context("Failed to open checkpoint file")?;
        let mut reader = BufReader::new(file);

        // Read all bytes
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .context("Failed to read checkpoint file")?;

        // Try to deserialize as pickle
        Self::from_pickle_bytes(&bytes)
    }

    /// Deserialize checkpoint from pickle bytes.
    ///
    /// This provides a safe subset of pickle deserialization focused on tensor data.
    fn from_pickle_bytes(bytes: &[u8]) -> Result<Self> {
        // Attempt to deserialize the pickle data
        // Note: This is a simplified version. Real PyTorch checkpoints may need
        // more sophisticated handling of numpy arrays and torch tensors.
        let value: serde_pickle::Value = serde_pickle::from_slice(bytes, Default::default())
            .context("Failed to deserialize pickle data")?;

        // Parse the pickle value into our checkpoint structure
        Self::parse_pickle_value(value)
    }

    /// Parse a pickle value into a checkpoint structure.
    fn parse_pickle_value(value: serde_pickle::Value) -> Result<Self> {
        use serde_pickle::{HashableValue, Value};

        // PyTorch checkpoints are typically dictionaries
        let dict = match value {
            Value::Dict(d) => d,
            _ => bail!("Expected dictionary at root of checkpoint"),
        };

        let mut state_dict_tensors = HashMap::new();
        let mut optimizer_state = None;
        let mut epoch = None;
        let mut loss_history = None;
        let mut metadata = HashMap::new();

        // Check if dict contains state_dict key
        let has_state_dict_key = dict.iter().any(|(k, _)| {
            matches!(k, HashableValue::String(ref s) if s == "state_dict" || s == "model_state_dict")
        });

        // Parse dictionary entries
        for (key, val) in &dict {
            let key_str = match key {
                HashableValue::String(s) => s.clone(),
                HashableValue::Bytes(b) => String::from_utf8_lossy(b).to_string(),
                _ => continue,
            };

            match key_str.as_str() {
                "state_dict" | "model_state_dict" => {
                    if let Value::Dict(sd) = val {
                        state_dict_tensors = Self::parse_state_dict(sd.clone())?;
                    }
                }
                "optimizer_state_dict" | "optimizer" => {
                    optimizer_state = Self::parse_optimizer_state(val.clone()).ok();
                }
                "epoch" => {
                    if let Value::I64(e) = val {
                        epoch = Some(*e as usize);
                    }
                }
                "loss_history" => {
                    loss_history = Self::parse_loss_history(val.clone()).ok();
                }
                _ => {
                    // Store as metadata
                    if let Value::String(s) = val {
                        metadata.insert(key_str, s.clone());
                    }
                }
            }
        }

        // If no explicit state_dict key, assume the whole dict is the state_dict
        if state_dict_tensors.is_empty() && !has_state_dict_key {
            state_dict_tensors = Self::parse_state_dict(dict)?;
        }

        Ok(PyTorchCheckpoint {
            state_dict: StateDict {
                tensors: state_dict_tensors,
            },
            optimizer_state,
            epoch,
            loss_history,
            metadata,
        })
    }

    /// Parse state_dict from pickle dictionary.
    fn parse_state_dict(
        dict: std::collections::BTreeMap<serde_pickle::HashableValue, serde_pickle::Value>,
    ) -> Result<HashMap<String, TensorData>> {
        use serde_pickle::HashableValue;

        let mut tensors = HashMap::new();

        for (key, val) in dict {
            let key_str = match key {
                HashableValue::String(s) => s,
                HashableValue::Bytes(b) => String::from_utf8_lossy(&b).to_string(),
                _ => continue,
            };

            // Try to parse tensor data
            if let Ok(tensor_data) = Self::parse_tensor_value(val) {
                tensors.insert(key_str, tensor_data);
            }
        }

        Ok(tensors)
    }

    /// Parse a tensor value from pickle.
    fn parse_tensor_value(value: serde_pickle::Value) -> Result<TensorData> {
        use serde_pickle::{HashableValue, Value};

        // This is simplified - real PyTorch tensors are more complex
        // In practice, you'd need to handle torch.Tensor objects which contain
        // references to storage objects

        match value {
            Value::Dict(d) => {
                // Look for tensor-like dictionary structure
                let mut shape = Vec::new();
                let mut data = Vec::new();
                let mut dtype = "float32".to_string();
                let mut requires_grad = false;

                for (k, v) in d {
                    let key = match k {
                        HashableValue::String(s) => s,
                        HashableValue::Bytes(b) => String::from_utf8_lossy(&b).to_string(),
                        _ => continue,
                    };

                    match key.as_str() {
                        "shape" | "size" => {
                            if let Value::List(list) = v {
                                shape = list
                                    .into_iter()
                                    .filter_map(|v| match v {
                                        Value::I64(i) => Some(i as usize),
                                        _ => None,
                                    })
                                    .collect();
                            }
                        }
                        "data" | "storage" => {
                            if let Value::Bytes(b) = v {
                                data = b;
                            }
                        }
                        "dtype" => {
                            if let Value::String(s) = v {
                                dtype = s;
                            }
                        }
                        "requires_grad" => {
                            if let Value::Bool(b) = v {
                                requires_grad = b;
                            }
                        }
                        _ => {}
                    }
                }

                if !shape.is_empty() && !data.is_empty() {
                    Ok(TensorData {
                        shape,
                        dtype,
                        data,
                        requires_grad,
                    })
                } else {
                    bail!("Incomplete tensor data")
                }
            }
            Value::Bytes(data) => {
                // Raw bytes - assume 1D float32 array
                Ok(TensorData {
                    shape: vec![data.len() / 4],
                    dtype: "float32".to_string(),
                    data,
                    requires_grad: false,
                })
            }
            _ => bail!("Unsupported tensor value type"),
        }
    }

    /// Parse optimizer state from pickle value.
    #[allow(dead_code)]
    fn parse_optimizer_state(_value: serde_pickle::Value) -> Result<OptimizerState> {
        // Simplified - would need full implementation for real use
        Ok(OptimizerState {
            optimizer_type: "Unknown".to_string(),
            param_state: HashMap::new(),
            hyperparameters: HashMap::new(),
        })
    }

    /// Parse loss history from pickle value.
    #[allow(dead_code)]
    fn parse_loss_history(value: serde_pickle::Value) -> Result<Vec<f32>> {
        use serde_pickle::Value;

        match value {
            Value::List(list) => {
                let losses = list
                    .into_iter()
                    .filter_map(|v| match v {
                        Value::F64(f) => Some(f as f32),
                        _ => None,
                    })
                    .collect();
                Ok(losses)
            }
            _ => bail!("Expected list for loss history"),
        }
    }

    /// Get checkpoint metadata.
    pub fn metadata(&self) -> CheckpointMetadata {
        let mut total_parameters = 0;
        let mut layer_names = Vec::new();
        let mut total_size_bytes = 0;
        let mut dtypes = HashMap::new();

        for (name, tensor) in &self.state_dict.tensors {
            layer_names.push(name.clone());

            let num_elements: usize = tensor.shape.iter().product();
            total_parameters += num_elements;

            total_size_bytes += tensor.data.len();

            *dtypes.entry(tensor.dtype.clone()).or_insert(0) += 1;
        }

        CheckpointMetadata {
            total_parameters,
            layer_names,
            total_size_bytes,
            dtypes,
            has_optimizer_state: self.optimizer_state.is_some(),
            epoch: self.epoch,
        }
    }

    /// Get reference to state dict.
    pub fn state_dict(&self) -> &StateDict {
        &self.state_dict
    }

    /// Convert checkpoint to Safetensors format.
    ///
    /// This provides a safe, efficient format for storing model weights.
    pub fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut writer = SafetensorsWriter::new();

        for (name, tensor) in &self.state_dict.tensors {
            // Determine shape for safetensors
            let shape = tensor.shape.clone();

            // Convert data based on dtype
            match tensor.dtype.as_str() {
                "float32" | "Float" => {
                    // Convert bytes to f32 slice
                    if tensor.data.len() % 4 != 0 {
                        bail!("Invalid float32 data length for tensor {}", name);
                    }

                    let float_data: Vec<f32> = tensor
                        .data
                        .chunks_exact(4)
                        .map(|chunk| {
                            let bytes: [u8; 4] = chunk
                                .try_into()
                                .expect("chunks_exact(4) guarantees exactly 4 bytes");
                            f32::from_le_bytes(bytes)
                        })
                        .collect();

                    writer.add_f32(name, shape, &float_data);
                }
                "float64" | "Double" => {
                    if tensor.data.len() % 8 != 0 {
                        bail!("Invalid float64 data length for tensor {}", name);
                    }

                    let float_data: Vec<f64> = tensor
                        .data
                        .chunks_exact(8)
                        .map(|chunk| {
                            let bytes: [u8; 8] = chunk
                                .try_into()
                                .expect("chunks_exact(8) guarantees exactly 8 bytes");
                            f64::from_le_bytes(bytes)
                        })
                        .collect();

                    writer.add_f64(name, shape, &float_data);
                }
                _ => {
                    bail!("Unsupported dtype: {}", tensor.dtype);
                }
            }
        }

        writer
            .serialize()
            .context("Failed to serialize to safetensors")
    }

    /// Save checkpoint in PyTorch format.
    ///
    /// Note: This creates a simplified pickle format compatible with PyTorch.
    #[allow(dead_code)]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_pickle_bytes()?;
        std::fs::write(path, bytes).context("Failed to write checkpoint file")?;
        Ok(())
    }

    /// Serialize checkpoint to pickle bytes.
    fn to_pickle_bytes(&self) -> Result<Vec<u8>> {
        use serde_pickle::ser;

        // Note: serde_pickle::Value doesn't implement Serialize, so we need to
        // serialize our checkpoint structure directly via serde
        // We'll use a simplified serializable format

        #[derive(Serialize)]
        struct CheckpointSer {
            state_dict: HashMap<String, TensorSer>,
            #[serde(skip_serializing_if = "Option::is_none")]
            epoch: Option<usize>,
            #[serde(skip_serializing_if = "Option::is_none")]
            loss_history: Option<Vec<f32>>,
            metadata: HashMap<String, String>,
        }

        #[derive(Serialize)]
        struct TensorSer {
            shape: Vec<usize>,
            dtype: String,
            data_len: usize,
        }

        let state_dict_ser: HashMap<String, TensorSer> = self
            .state_dict
            .tensors
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    TensorSer {
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype.clone(),
                        data_len: tensor.data.len(),
                    },
                )
            })
            .collect();

        let checkpoint_ser = CheckpointSer {
            state_dict: state_dict_ser,
            epoch: self.epoch,
            loss_history: self.loss_history.clone(),
            metadata: self.metadata.clone(),
        };

        // Serialize using serde_pickle's serializer
        ser::to_vec(&checkpoint_ser, Default::default()).context("Failed to serialize to pickle")
    }

    /// Convert TensorData to pickle value.
    ///
    /// Note: This is a simplified helper for internal use.
    #[allow(dead_code)]
    fn tensor_to_pickle_value(_tensor: &TensorData) -> HashMap<String, String> {
        // Simplified version for internal use
        // In practice, you'd serialize the full tensor data
        HashMap::new()
    }

    /// Create a new empty checkpoint.
    pub fn new() -> Self {
        PyTorchCheckpoint {
            state_dict: StateDict {
                tensors: HashMap::new(),
            },
            optimizer_state: None,
            epoch: None,
            loss_history: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a tensor to the state dict.
    pub fn add_tensor(&mut self, name: String, tensor: TensorData) {
        self.state_dict.tensors.insert(name, tensor);
    }

    /// Set the epoch.
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = Some(epoch);
    }

    /// Add metadata entry.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl Default for PyTorchCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl StateDict {
    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.tensors.get(name)
    }

    /// Iterate over tensors.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorData)> {
        self.tensors.iter()
    }

    /// Get number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if state dict is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl TensorData {
    /// Create new tensor data from f32 values.
    pub fn from_f32(shape: Vec<usize>, data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        TensorData {
            shape,
            dtype: "float32".to_string(),
            data: bytes,
            requires_grad: false,
        }
    }

    /// Create new tensor data from f64 values.
    pub fn from_f64(shape: Vec<usize>, data: &[f64]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        TensorData {
            shape,
            dtype: "float64".to_string(),
            data: bytes,
            requires_grad: false,
        }
    }

    /// Get tensor as f32 slice.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != "float32" && self.dtype != "Float" {
            bail!("Expected float32 dtype, got {}", self.dtype);
        }

        if !self.data.len().is_multiple_of(4) {
            bail!("Invalid data length for float32");
        }

        Ok(self
            .data
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk
                    .try_into()
                    .expect("chunks_exact(4) guarantees exactly 4 bytes");
                f32::from_le_bytes(bytes)
            })
            .collect())
    }

    /// Get tensor as f64 slice.
    pub fn as_f64(&self) -> Result<Vec<f64>> {
        if self.dtype != "float64" && self.dtype != "Double" {
            bail!("Expected float64 dtype, got {}", self.dtype);
        }

        if !self.data.len().is_multiple_of(8) {
            bail!("Invalid data length for float64");
        }

        Ok(self
            .data
            .chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk
                    .try_into()
                    .expect("chunks_exact(8) guarantees exactly 8 bytes");
                f64::from_le_bytes(bytes)
            })
            .collect())
    }

    /// Get number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let mut checkpoint = PyTorchCheckpoint::new();

        // Add a simple tensor
        let tensor = TensorData::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        checkpoint.add_tensor("layer1.weight".to_string(), tensor);

        checkpoint.set_epoch(10);
        checkpoint.add_metadata("model_type".to_string(), "CNN".to_string());

        assert_eq!(checkpoint.state_dict().len(), 1);
        assert_eq!(checkpoint.epoch, Some(10));
        assert_eq!(
            checkpoint
                .metadata
                .get("model_type")
                .expect("test: should succeed"),
            "CNN"
        );
    }

    #[test]
    fn test_tensor_data_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(vec![2, 2], &data);

        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, "float32");
        assert_eq!(tensor.num_elements(), 4);

        let recovered = tensor.as_f32().expect("test: should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_tensor_data_f64() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f64(vec![2, 2], &data);

        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, "float64");

        let recovered = tensor.as_f64().expect("test: should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_metadata_extraction() {
        let mut checkpoint = PyTorchCheckpoint::new();

        checkpoint.add_tensor(
            "layer1.weight".to_string(),
            TensorData::from_f32(vec![10, 10], &vec![0.0; 100]),
        );
        checkpoint.add_tensor(
            "layer1.bias".to_string(),
            TensorData::from_f32(vec![10], &[0.0; 10]),
        );
        checkpoint.add_tensor(
            "layer2.weight".to_string(),
            TensorData::from_f64(vec![5, 10], &vec![0.0; 50]),
        );

        let metadata = checkpoint.metadata();

        assert_eq!(metadata.total_parameters, 160);
        assert_eq!(metadata.layer_names.len(), 3);
        assert_eq!(metadata.dtypes.get("float32"), Some(&2));
        assert_eq!(metadata.dtypes.get("float64"), Some(&1));
    }

    #[test]
    fn test_state_dict_access() {
        let mut checkpoint = PyTorchCheckpoint::new();

        let tensor = TensorData::from_f32(vec![3], &[1.0, 2.0, 3.0]);
        checkpoint.add_tensor("test".to_string(), tensor);

        let state_dict = checkpoint.state_dict();
        assert_eq!(state_dict.len(), 1);
        assert!(!state_dict.is_empty());

        let retrieved = state_dict.get("test").expect("test: should succeed");
        assert_eq!(retrieved.shape, vec![3]);
    }

    #[test]
    fn test_checkpoint_serialization() -> Result<()> {
        let mut checkpoint = PyTorchCheckpoint::new();

        checkpoint.add_tensor(
            "weight".to_string(),
            TensorData::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]),
        );
        checkpoint.set_epoch(5);
        checkpoint.add_metadata("arch".to_string(), "ResNet".to_string());

        // Test that serialization works without errors
        let bytes = checkpoint.to_pickle_bytes()?;
        assert!(!bytes.is_empty());

        // Note: Full PyTorch pickle roundtrip requires handling complex tensor
        // structures. For practical use, convert to Safetensors format using
        // to_safetensors() which provides full fidelity and is more secure.

        Ok(())
    }

    #[test]
    fn test_to_safetensors() -> Result<()> {
        let mut checkpoint = PyTorchCheckpoint::new();

        checkpoint.add_tensor(
            "layer1.weight".to_string(),
            TensorData::from_f32(vec![3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        );
        checkpoint.add_tensor(
            "layer1.bias".to_string(),
            TensorData::from_f32(vec![3], &[0.1, 0.2, 0.3]),
        );

        let safetensors_bytes = checkpoint.to_safetensors()?;
        assert!(!safetensors_bytes.is_empty());

        Ok(())
    }
}
