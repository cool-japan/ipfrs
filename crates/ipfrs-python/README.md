# IPFRS Python Bindings

Python bindings for IPFRS (Inter-Planetary File Rust System) - a content-addressed storage system with semantic search and logic programming capabilities.

## Features

- **Content-Addressed Storage**: Store and retrieve data using cryptographic content identifiers (CIDs)
- **Semantic Search**: Vector similarity search with HNSW indexing
- **Logic Programming**: TensorLogic inference engine with backward chaining
- **Persistence**: Save and load indexes and knowledge bases
- **Type Hints**: Full type stub support for IDE autocompletion

## Installation

### From Source

```bash
# Install maturin
pip install maturin

# Build and install
cd crates/ipfrs-python
maturin develop --release
```

### From PyPI (Coming Soon)

```bash
pip install ipfrs
```

## Quick Start

### Basic Block Operations

```python
from ipfrs import Node, NodeConfig

# Create and start a node
config = NodeConfig(storage_path="/tmp/ipfrs-python-demo")
node = Node(config)
node.start()

# Store data
data = b"Hello, IPFRS!"
cid = node.put_block(data)
print(f"Stored block with CID: {cid}")

# Retrieve data
block = node.get_block(cid)
if block:
    print(f"Retrieved: {block.data()}")

# Check existence
exists = node.has_block(cid)
print(f"Block exists: {exists}")

# Clean up
node.stop()
```

### Semantic Search

```python
from ipfrs import Node, NodeConfig, Block
import random

# Create node with semantic search enabled
config = NodeConfig(
    storage_path="/tmp/ipfrs-semantic",
    enable_semantic=True
)
node = Node(config)
node.start()

# Store content with embeddings
embedding_dim = 128
for i in range(10):
    data = f"Document {i}".encode()
    cid = node.put_block(data)

    # Generate random embedding (in real use, use a model)
    embedding = [random.random() for _ in range(embedding_dim)]
    node.index_content(cid, embedding)

# Search for similar content
query = [random.random() for _ in range(embedding_dim)]
results = node.search_similar(query, k=5)

print("Search results:")
for cid, score in results:
    print(f"  CID: {cid}, Score: {score:.4f}")

node.stop()
```

### Logic Programming

```python
from ipfrs import Node, NodeConfig, Predicate, Term, Rule

# Create node with TensorLogic enabled
config = NodeConfig(
    storage_path="/tmp/ipfrs-logic",
    enable_tensorlogic=True
)
node = Node(config)
node.start()

# Add facts
node.add_fact(Predicate("parent", [
    Term.string("Alice"),
    Term.string("Bob")
]))
node.add_fact(Predicate("parent", [
    Term.string("Bob"),
    Term.string("Charlie")
]))

# Add rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
rule = Rule.rule(
    Predicate("grandparent", [Term.var("X"), Term.var("Z")]),
    [
        Predicate("parent", [Term.var("X"), Term.var("Y")]),
        Predicate("parent", [Term.var("Y"), Term.var("Z")])
    ]
)
node.add_rule(rule)

# Query: Who is Charlie's grandparent?
goal = Predicate("grandparent", [
    Term.var("X"),
    Term.string("Charlie")
])
results = node.infer(goal)

print("Query results:")
for substitution in results:
    print(f"  Bindings: {substitution.bindings()}")

# Generate and verify proof
proof = node.prove(goal)
if proof:
    is_valid = node.verify_proof(proof)
    print(f"Proof valid: {is_valid}")

# Get statistics
stats = node.kb_stats()
print(f"KB stats: {stats}")

node.stop()
```

### Persistence

```python
from ipfrs import Node, NodeConfig

config = NodeConfig(storage_path="/tmp/ipfrs-persist")
node = Node(config)
node.start()

# ... index content and add facts/rules ...

# Save indexes
node.save_semantic_index("/tmp/semantic_index.bin")
node.save_kb("/tmp/knowledge_base.bin")

# Later, load them back
node.load_semantic_index("/tmp/semantic_index.bin")
node.load_kb("/tmp/knowledge_base.bin")

node.stop()
```

## API Reference

### Node

The main interface for all IPFRS operations.

**Methods:**
- `start()` - Start the node
- `stop()` - Stop the node
- `put_block(data: bytes) -> Cid` - Store a block
- `get_block(cid: Cid) -> Optional[Block]` - Retrieve a block
- `has_block(cid: Cid) -> bool` - Check block existence
- `delete_block(cid: Cid)` - Delete a block
- `index_content(cid: Cid, embedding: List[float]) -> int` - Index for semantic search
- `search_similar(query: List[float], k: int) -> List[Tuple[Cid, float]]` - Search similar content
- `search_filtered(query: List[float], k: int, filter: Optional[Filter]) -> List[Tuple[Cid, float]]` - Search with filters
- `add_fact(fact: Predicate)` - Add a fact
- `add_rule(rule: Rule)` - Add a rule
- `infer(goal: Predicate) -> List[Substitution]` - Run inference
- `prove(goal: Predicate) -> Optional[Proof]` - Generate proof
- `verify_proof(proof: Proof) -> bool` - Verify proof
- `kb_stats() -> Dict[str, int]` - Get KB statistics
- `save_semantic_index(path: str)` - Save semantic index
- `load_semantic_index(path: str)` - Load semantic index
- `save_kb(path: str)` - Save knowledge base
- `load_kb(path: str)` - Load knowledge base

### Term

Logical term constructors:
- `Term.int(value: int)` - Integer constant
- `Term.float(value: float)` - Float constant
- `Term.string(value: str)` - String constant
- `Term.var(name: str)` - Variable

### Predicate

Logical predicate:
- `Predicate(name: str, args: List[Term])` - Create predicate

### Rule

Logical rule:
- `Rule.fact(head: Predicate)` - Create fact
- `Rule.rule(head: Predicate, body: List[Predicate])` - Create rule

### Filter

Search filters:
- `Filter.min_score(min_score: float)` - Minimum score threshold
- `Filter.max_score(max_score: float)` - Maximum score threshold

## Examples

See the `examples/` directory for more complete examples:
- `basic_blocks.py` - Basic block storage operations
- `semantic_search.py` - Semantic search with embeddings
- `logic_programming.py` - TensorLogic inference
- `persistence.py` - Saving and loading indexes

## Requirements

- Python 3.8+
- Rust 1.70+ (for building from source)

## License

Apache-2.0
