# Basic Concepts

Understanding these core concepts will help you get the most out of IPFRS.

## Content Addressing

### What is Content Addressing?

Content addressing means data is identified by its cryptographic hash rather than its location. This provides:

- **Immutability**: Content cannot be changed without changing its identifier
- **Deduplication**: Identical content has the same identifier
- **Verification**: Content integrity can be verified using the identifier
- **Location Independence**: Content can be retrieved from any source

### CIDs (Content Identifiers)

A CID is a self-describing content-addressed identifier. Example:

```
QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

CIDs encode:
- **Version**: CID format version (v0 or v1)
- **Codec**: How the content is encoded (e.g., raw, dag-pb, dag-cbor)
- **Hash**: The cryptographic hash of the content

## Blocks

A **block** is the fundamental unit of storage in IPFRS:

- Contains raw bytes of data
- Identified by a CID
- Immutable once created
- Can be any size (recommended: < 1MB for optimal performance)

```rust
// Adding a block
let data = b"Hello, IPFRS!";
let cid = node.add(data).await?;

// Retrieving a block
let data = node.cat(&cid).await?;
```

## DAGs (Directed Acyclic Graphs)

IPFRS uses DAGs to represent complex data structures:

- **Directed**: Links have a direction (parent → child)
- **Acyclic**: No circular references
- **Merkle DAGs**: Each node is identified by a hash of its content and children

### IPLD (InterPlanetary Linked Data)

IPLD is a data model for content-addressed data structures:

```json
{
  "name": "Alice",
  "age": 30,
  "friends": [
    { "/": "QmFriend1..." },
    { "/": "QmFriend2..." }
  ]
}
```

Links are represented as `{ "/": "CID" }`.

## Semantic Search

### Vector Embeddings

Semantic search uses vector embeddings to represent content meaning:

```python
# Text to vector
embedding = model.encode("machine learning")
# → [0.12, -0.45, 0.89, ...]

# Index content
await node.index_content(cid, embedding)

# Search for similar content
results = await node.search_similar(query_embedding, top_k=5)
```

### HNSW Index

HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for fast approximate nearest neighbor search:

- **Logarithmic search time**: O(log n) instead of O(n)
- **High recall**: Finds 95-99% of true nearest neighbors
- **Tunable parameters**: Trade-off between speed and accuracy

## Logic Programming

### Facts

Facts are basic statements about the world:

```prolog
parent(tom, bob).
parent(bob, ann).
age(tom, 50).
```

### Rules

Rules define logical relationships:

```prolog
% X is an ancestor of Y if X is a parent of Y
ancestor(X, Y) :- parent(X, Y).

% Or if X is a parent of Z and Z is an ancestor of Y
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

### Queries

Queries ask questions about the knowledge base:

```prolog
?- ancestor(tom, X).

Solutions:
  X = bob
  X = ann
```

### Inference

Inference is the process of deriving new facts from existing facts and rules:

1. **Forward Chaining**: Start with facts, apply rules to derive new facts
2. **Backward Chaining**: Start with a goal, work backward to find supporting facts

IPFRS uses **backward chaining** for efficient goal-oriented reasoning.

### Proofs

IPFRS can generate and verify proofs of inference:

```rust
// Generate a proof
let proof = node.prove("ancestor(tom, ann)").await?;

// Verify a proof
let valid = node.verify_proof(&proof).await?;
```

Proofs are content-addressed and can be shared or stored.

## Networking

### libp2p

IPFRS uses libp2p for peer-to-peer networking:

- **Multiaddress**: Universal address format
  - Example: `/ip4/127.0.0.1/tcp/4001`
- **Peer ID**: Unique identifier for each node
  - Example: `12D3KooWAbCdEfGh...`
- **Protocols**: Pluggable protocol system

### DHT (Distributed Hash Table)

The DHT enables content discovery:

```bash
# Announce content
ipfrs dht provide QmXXX...

# Find providers
ipfrs dht findprovs QmXXX...
```

### Bitswap

Bitswap is IPFRS's block exchange protocol:

1. **Want List**: Blocks you want to retrieve
2. **Have List**: Blocks you can provide
3. **Exchange**: Peers exchange blocks based on wants/haves

## Storage

### Block Store

The block store is a key-value database:

- **Key**: CID (content identifier)
- **Value**: Raw block data
- **Backend**: Sled (embedded database)

### Caching

IPFRS uses multi-level caching:

- **L1 Cache**: Hot cache (fast, small)
- **L2 Cache**: Warm cache (larger, slower)
- **Promotion**: Frequently accessed blocks move to L1

## Authentication & Authorization

### Authentication Methods

IPFRS supports multiple authentication methods:

1. **API Keys**: Long-lived tokens for services
2. **JWT**: Short-lived tokens for users
3. **OAuth**: Third-party authentication (future)

### RBAC (Role-Based Access Control)

Users are assigned roles with specific permissions:

- **Admin**: Full access to all operations
- **User**: Read/write access to data
- **ReadOnly**: Read-only access
- **Service**: Automated operations

### Permissions

Fine-grained permissions control access:

- `BlockRead`, `BlockWrite`, `BlockDelete`
- `SemanticRead`, `SemanticWrite`
- `LogicRead`, `LogicWrite`
- `NetworkRead`, `NetworkWrite`
- `AdminManage`

## Observability

### Metrics

IPFRS exposes Prometheus metrics:

```bash
curl http://localhost:9000/metrics
```

Metrics include:
- Operation counts and latencies
- Cache hit rates
- Storage size
- Network traffic

### Logging

Structured logging with configurable levels:

```bash
# Set log level
RUST_LOG=ipfrs=debug ipfrs daemon
```

### Tracing

Distributed tracing with OpenTelemetry:

- Trace entire request flows
- Identify bottlenecks
- Debug distributed systems

## Next Steps

Now that you understand the basics:

- [Storage Deep Dive](../core/storage.md)
- [Semantic Search Guide](../core/semantic.md)
- [Logic Programming Tutorial](../tutorials/knowledge-graph.md)
- [Network Configuration](../core/networking.md)
