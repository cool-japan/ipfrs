# ipfrs-tensorlogic

TensorLogic integration layer for IPFRS.

## Overview

`ipfrs-tensorlogic` bridges IPFRS with the TensorLogic AI language:

- **Zero-Copy Binding**: Direct memory sharing via Apache Arrow
- **Distributed Reasoning**: Network-wide backward chaining
- **Gradient Storage**: Version-controlled learning history
- **Proof Provenance**: Merkle DAG for inference traces

## Key Features

### Native TensorLogic Support
Seamless integration with TensorLogic runtime:

- **IR Serialization**: Convert `tensorlogic::ir::Term` to IPLD
- **Lazy Loading**: Stream large models on-demand
- **Computation Graph**: Store Einsum graphs as DAG
- **Rule Distribution**: Share inference rules across network

### Zero-Copy Interface
Direct memory access without serialization overhead:

- **Apache Arrow**: Columnar memory format
- **Safetensors**: PyTorch-compatible tensor storage
- **Shared Buffers**: mmap-based data sharing
- **FFI Boundary**: Optimized Rust-to-Rust calls

### Distributed Inference
Split computation across IPFRS network:

- **Remote Knowledge**: Fetch missing facts from peers
- **Proof Synthesis**: Assemble proofs from distributed fragments
- **Goal Decomposition**: Break queries into subgoals
- **Result Caching**: Store intermediate inference results

### Differentiable Storage
Git-like version control for neural models:

- **Gradient Tracking**: Store update deltas
- **Checkpoint Management**: Save training snapshots
- **Provenance Graph**: Track data lineage
- **Rollback Support**: Restore previous states

## Architecture

```
TensorLogic Runtime
         ↓
Zero-Copy FFI (Apache Arrow)
         ↓
ipfrs-tensorlogic
├── ir/            # TensorLogic IR codec
├── inference/     # Distributed reasoning
├── gradient/      # Gradient storage & tracking
└── ffi/           # Foreign function interface
         ↓
ipfrs-core (Blocks & CID)
```

## Design Principles

- **Performance Critical**: No unnecessary copies or allocations
- **Type Safe**: Leverage Rust's type system
- **Composable**: Work with standard IPFRS blocks
- **Explainable**: Full provenance for XAI

## Usage Example

```rust
use ipfrs_tensorlogic::{TensorLogicNode, InferenceEngine};
use tensorlogic::ir::Term;

// Initialize node with TensorLogic support
let node = TensorLogicNode::new(config).await?;

// Store logic term
let term = Term::from_str("knows(alice, bob)")?;
let cid = node.put_term(term).await?;

// Distributed inference
let query = Term::from_str("knows(alice, ?X)")?;
let solutions = node.infer(query).await?;

// Access tensor data (zero-copy)
let weights = node.get_tensor(cid).await?;
let array: ArrayView2<f32> = weights.as_arrow_array()?;
```

## Integration Points

### With TensorLogic
- Share memory space via FFI
- Use TensorLogic types directly
- Support inference callbacks
- Integrate with learning loop

### With IPFRS Core
- Store terms as IPLD blocks
- Content-address by hash
- Version control via CID links
- Network distribution

## Dependencies

- `tensorlogic` - Core AI language runtime
- `arrow` - Columnar data format
- `safetensors` - Tensor serialization
- `ipfrs-core` - IPFRS primitives

## References

- IPFRS v0.2.0 Whitepaper (TensorLogic Architecture)
- IPFRS v0.3.0 Whitepaper (Zero-Copy Tensor Transport) *(planned v0.3.0)*
- TensorLogic Paper: arXiv:2510.12269
