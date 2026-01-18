# IPFRS Interface Examples

This directory contains example implementations and integration tests for the IPFRS HTTP Gateway API.

## Examples

### 1. Python Client (`python_client.py`)

A complete Python client implementation demonstrating all IPFRS HTTP API features.

**Requirements:**
```bash
pip install requests pyarrow
```

**Usage:**
```bash
# Start IPFRS gateway first
cd ../ipfrs-cli
cargo run --release

# Run Python client example
python3 examples/python_client.py
```

**Features:**
- File upload/download (Kubo v0 API)
- Batch operations
- Streaming uploads/downloads
- Tensor operations with Apache Arrow support
- Node information queries

**Example code:**
```python
from python_client import IPFRSClient

client = IPFRSClient("http://localhost:8080")

# Upload a file
result = client.add_file("myfile.txt")
cid = result['Hash']

# Download it back
content = client.cat(cid)

# Get tensor as Arrow table
table, metadata = client.get_tensor_arrow(tensor_cid)
df = table.to_pandas()  # Convert to Pandas
```

### 2. JavaScript Client (`javascript_client.js`)

A complete Node.js/JavaScript client implementation.

**Requirements:**
```bash
npm install axios form-data apache-arrow ws
```

**Usage:**
```bash
# Start IPFRS gateway first
cd ../ipfrs-cli
cargo run --release

# Run JavaScript client example
node examples/javascript_client.js
```

**Features:**
- File upload/download (Kubo v0 API)
- Batch operations
- Streaming uploads/downloads
- Tensor operations with Apache Arrow support
- WebSocket real-time events
- Node information queries

**Example code:**
```javascript
const IPFRSClient = require('./javascript_client');

const client = new IPFRSClient('http://localhost:8080');

// Upload a file
const result = await client.addFile('myfile.txt');
const cid = result.Hash;

// Download it back
const content = await client.cat(cid);

// Get tensor as Arrow table
const { table, metadata } = await client.getTensorArrow(tensorCid);
console.log(`Rows: ${table.numRows}`);
```

### 3. gRPC Server (`grpc_server.rs`)

Complete gRPC server implementation with all IPFRS services.

**Usage:**
```bash
# Start the gRPC server
cargo run --example grpc_server
```

**Features:**
- BlockService - Raw block operations (Get, Put, Has, Delete, Batch)
- DagService - DAG operations (Get, Put, Resolve, Traverse)
- FileService - File operations (Add, Get, List, Pin)
- TensorService - Tensor operations (Get, Slice, Stream)
- Interceptors - Logging and metrics

**Server runs on:** `localhost:50051`

### 4. gRPC Client (Rust) (`grpc_client.rs`)

Rust client example demonstrating gRPC API usage.

**Usage:**
```bash
# Start server first (in another terminal)
cargo run --example grpc_server

# Run client
cargo run --example grpc_client
```

**Example operations:**
- Put a block
- Check if block exists
- Retrieve a block
- Verify data integrity

### 5. gRPC Client (Python) (`grpc_python_client.py`)

Python client example for gRPC API.

**Prerequisites:**
```bash
pip install grpcio grpcio-tools

# Generate Python proto files
python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. \
    ../proto/block.proto ../proto/dag.proto ../proto/file.proto ../proto/tensor.proto
```

**Usage:**
```bash
# Start server first
cargo run --example grpc_server

# Run Python client
python3 examples/grpc_python_client.py
```

**Features:**
- Block storage and retrieval
- Batch operations with streaming
- Full gRPC API coverage

### 6. Integration Tests (`integration_test.rs`)

Comprehensive integration tests for HTTP endpoints.

**Usage:**
```bash
# Run all integration tests (requires running server)
cargo test --example integration_test --release -- --ignored

# Run specific test
cargo test --example integration_test --release test_add_and_cat_file -- --ignored
```

**Tests:**
- Health check endpoint
- File upload and download (Kubo v0)
- Gateway GET with content-type detection
- Range requests (HTTP 206)
- Batch operations
- ETag caching and conditional requests
- Version endpoint

### 7. Memory-Mapped Tensor Server (`mmap_tensor_server.rs`)

High-performance zero-copy tensor serving using memory-mapped I/O.

**Usage:**
```bash
# Run the mmap tensor server
cargo run --example mmap_tensor_server

# In another terminal, test the endpoints
curl http://localhost:8080/tensor/example1
curl http://localhost:8080/tensor/example2?start=0&end=1024
curl http://localhost:8080/cache/stats
```

**Features:**
- Zero-copy file serving via memory-mapped I/O
- Efficient byte range requests (HTTP 206)
- Memory-mapped file cache with LRU eviction
- Automatic test tensor file generation
- Cache statistics and management endpoints

**Example code:**
```rust
// Create mmap cache
let cache = Arc::new(MmapCache::new(100));

// Get or create memory-mapped file
let mmap_file = cache.get_or_create(&file_path)?;

// Serve full file (zero-copy)
let data = mmap_file.bytes();

// Serve range (zero-copy)
let range_data = mmap_file.range(0..1024)?;
```

**API Endpoints:**
- `GET /health` - Health check
- `GET /tensor/:name` - Get full tensor (zero-copy)
- `GET /tensor/:name?start=N&end=M` - Get byte range
- `GET /cache/stats` - Cache statistics
- `GET /cache/clear` - Clear cache

**Performance Benefits:**
- 2-10x faster than traditional file I/O for large files
- <1μs latency for cached files
- Minimal memory overhead (OS page cache)
- Efficient sparse file access

**Documentation:**
See [MMAP_GUIDE.md](./MMAP_GUIDE.md) for detailed usage patterns and best practices.

## API Endpoints

### gRPC API (Port 50051)

**BlockService:**
```
GetBlock(cid) -> BlockResponse
PutBlock(data) -> PutBlockResponse
HasBlock(cid) -> HasBlockResponse
DeleteBlock(cid) -> DeleteBlockResponse
BatchGetBlocks(cids) -> stream BlockResponse
BatchPutBlocks(stream data) -> BatchPutBlocksResponse
StreamBlocks(stream request) -> stream response
```

**DagService:**
```
GetDag(cid) -> DagResponse
PutDag(node) -> PutDagResponse
ResolvePath(path) -> ResolvePathResponse
TraverseDag(cid) -> stream DagNode
GetDagStats(cid) -> DagStatsResponse
```

**FileService:**
```
AddFile(stream chunks) -> AddFileResponse
GetFile(cid) -> stream FileChunk
ListDirectory(cid) -> ListDirectoryResponse
GetFileInfo(cid) -> FileInfoResponse
PinFile(cid) -> PinFileResponse
UnpinFile(cid) -> UnpinFileResponse
```

**TensorService:**
```
GetTensor(cid) -> stream TensorChunk
PutTensor(stream chunks) -> PutTensorResponse
GetTensorInfo(cid) -> TensorInfoResponse
SliceTensor(cid, ranges) -> stream TensorChunk
GetTensorStats(cid) -> TensorStatsResponse
StreamTensors(stream request) -> stream response
```

### Kubo v0 API (IPFS Compatible)

```
POST /api/v0/add              # Upload file
POST /api/v0/cat?arg=<cid>    # Download file
POST /api/v0/block/get        # Get raw block
POST /api/v0/block/put        # Store raw block
POST /api/v0/dag/get          # Get DAG node
POST /api/v0/dag/put          # Store DAG node
POST /api/v0/id               # Node identity
POST /api/v0/version          # Version info
POST /api/v0/swarm/peers      # List peers
POST /api/v0/stats/bw         # Bandwidth stats
POST /api/v0/pin/add          # Pin content
```

### HTTP Gateway

```
GET  /ipfs/<cid>              # Retrieve content
GET  /health                  # Health check
```

### High-Speed v1 API

```
POST /v1/block/batch/get      # Batch retrieve blocks
POST /v1/block/batch/put      # Batch store blocks
POST /v1/block/batch/has      # Batch check existence
POST /v1/stream/upload        # Streaming upload
GET  /v1/stream/download/:cid # Streaming download
GET  /v1/progress/:op_id      # Progress updates (SSE)
```

### Tensor API (Zero-Copy)

```
GET  /v1/tensor/:cid          # Get tensor (raw/safetensors)
GET  /v1/tensor/:cid/info     # Get tensor metadata
GET  /v1/tensor/:cid/arrow    # Get tensor (Apache Arrow IPC)
```

Query parameters:
- `slice`: Tensor slice specification (e.g., `0:10,5:15`)

### WebSocket

```
GET  /ws                      # WebSocket connection
```

Messages:
- `Subscribe`: Subscribe to topic (blocks, peers, dht)
- `Unsubscribe`: Unsubscribe from topic
- `Ping`: Keepalive ping
- `Event`: Real-time event notification

## Common Use Cases

### Upload and Download Files

```python
# Python
client = IPFRSClient()
result = client.add_file("document.pdf")
cid = result['Hash']
content = client.cat(cid)
```

```javascript
// JavaScript
const client = new IPFRSClient();
const result = await client.addFile('document.pdf');
const cid = result.Hash;
const content = await client.cat(cid);
```

### Batch Operations

```python
# Check multiple blocks exist
cids = ["Qm...", "Qm...", "Qm..."]
results = client.batch_has_blocks(cids)
for result in results:
    print(f"{result['cid']}: {result['exists']}")
```

```javascript
// Check multiple blocks exist
const cids = ["Qm...", "Qm...", "Qm..."];
const results = await client.batchHasBlocks(cids);
results.forEach(r => console.log(`${r.cid}: ${r.exists}`));
```

### Tensor Operations with Arrow

```python
# Get tensor as Arrow table, convert to Pandas
table, metadata = client.get_tensor_arrow(tensor_cid)
df = table.to_pandas()
print(f"Shape: {metadata['tensor_shape']}")
print(f"DType: {metadata['tensor_dtype']}")

# Get a slice
table, metadata = client.get_tensor_arrow(tensor_cid, slice_spec="0:100,5:25")
```

```javascript
// Get tensor as Arrow table
const { table, metadata } = await client.getTensorArrow(tensorCid);
console.log(`Shape: ${metadata.tensor_shape}`);
console.log(`Rows: ${table.numRows}`);

// Get a slice
const result = await client.getTensorArrow(tensorCid, '0:100,5:25');
```

### WebSocket Real-Time Events

```python
# Python (using websocket-client)
import websocket
import json

ws = websocket.create_connection("ws://localhost:8080/ws")
ws.send(json.dumps({"type": "Subscribe", "topic": "blocks"}))

while True:
    message = json.loads(ws.recv())
    if message['type'] == 'Event':
        event = json.loads(message['payload'])
        print(f"New block: {event}")
```

```javascript
// JavaScript
const ws = client.connectWebSocket(
    (message) => {
        if (message.type === 'Event') {
            const event = JSON.parse(message.payload);
            console.log('Block added:', event.BlockAdded);
        }
    },
    (error) => console.error('WebSocket error:', error)
);

client.subscribe(ws, 'blocks');
```

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

```bash
# Start gateway in one terminal
cd ../ipfrs-cli
cargo run --release

# Run integration tests in another terminal
cargo test --example integration_test --release -- --ignored
```

### Manual Testing with curl

```bash
# Upload file
curl -X POST -F "file=@test.txt" http://localhost:8080/api/v0/add

# Download file
curl -X POST "http://localhost:8080/api/v0/cat?arg=QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"

# Gateway GET
curl "http://localhost:8080/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"

# Range request
curl -H "Range: bytes=0-99" "http://localhost:8080/ipfs/Qm..."

# Batch operations
curl -X POST http://localhost:8080/v1/block/batch/has \
  -H "Content-Type: application/json" \
  -d '{"cids": ["Qm..."]}'

# Tensor info
curl "http://localhost:8080/v1/tensor/QmTensor.../info"

# Tensor as Arrow
curl "http://localhost:8080/v1/tensor/QmTensor.../arrow" > tensor.arrow
```

## Performance Tips

1. **Use batch operations** for multiple blocks instead of individual requests
2. **Use streaming API** (`/v1/stream/*`) for large files (>1MB)
3. **Enable compression** by sending `Accept-Encoding: gzip, br` header
4. **Use range requests** to download specific byte ranges
5. **Cache CIDs** - content is immutable, safe to cache indefinitely
6. **Use Arrow format** for tensors when integrating with Pandas/Polars

## Error Handling

All endpoints return JSON error responses:

```json
{
  "error": "Content not found",
  "code": "NOT_FOUND",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Common status codes:
- `200 OK` - Success
- `206 Partial Content` - Range request success
- `304 Not Modified` - Cached content valid (ETag match)
- `400 Bad Request` - Invalid input
- `404 Not Found` - Content not found
- `416 Range Not Satisfiable` - Invalid range
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Additional Resources

- [OpenAPI Specification](../openapi.yaml) - Complete API documentation
- [Configuration Guide](../CONFIGURATION.md) - Server configuration
- [Client Examples](../examples/CLIENT_EXAMPLES.md) - More usage examples
