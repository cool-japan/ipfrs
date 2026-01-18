# Quick Start

Get up and running with IPFRS in just a few minutes!

## Starting the Daemon

Start the IPFRS daemon:

```bash
ipfrs daemon
```

The daemon will start on `http://localhost:8080` by default.

You should see output like:

```
🚀 IPFRS daemon starting...
📦 Storage initialized at ~/.ipfrs/blocks
🔍 Semantic search enabled
🧠 TensorLogic enabled
🌐 Network initialized: 12D3KooWAbCdEfGh...
✅ HTTP API listening on http://127.0.0.1:8080
✅ Metrics server listening on http://127.0.0.1:9000
```

## Basic Operations

### Storing Content

Add a file to IPFRS:

```bash
echo "Hello, IPFRS!" > hello.txt
ipfrs add hello.txt
```

Output:

```
added QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX hello.txt
```

### Retrieving Content

Get content by CID:

```bash
ipfrs cat QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Output:

```
Hello, IPFRS!
```

### Semantic Search

Index content with embeddings:

```bash
ipfrs semantic index "machine learning" QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Search for similar content:

```bash
ipfrs semantic search "artificial intelligence" --top-k 5
```

### Logic Programming

Add facts to the knowledge base:

```bash
ipfrs logic add-fact "parent(tom, bob)"
ipfrs logic add-fact "parent(bob, ann)"
```

Add inference rules:

```bash
ipfrs logic add-rule "ancestor(X, Y) :- parent(X, Y)"
ipfrs logic add-rule "ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)"
```

Query the knowledge base:

```bash
ipfrs logic infer "ancestor(tom, X)"
```

Output:

```
Solutions:
  - ancestor(tom, bob)
  - ancestor(tom, ann)
```

## Using the HTTP API

### Store a Block

```bash
curl -X POST http://localhost:8080/api/v0/block/put \
  -H "Content-Type: application/json" \
  -d '{"data": "SGVsbG8sIElQRlJTIQ=="}'
```

Response:

```json
{
  "cid": "QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "size": 13
}
```

### Retrieve a Block

```bash
curl http://localhost:8080/api/v0/block/get/QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Semantic Search

```bash
curl -X POST http://localhost:8080/api/v0/semantic/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [0.1, 0.2, 0.3, ...],
    "top_k": 5
  }'
```

## Using GraphQL

Access the GraphQL playground at `http://localhost:8080/graphql`.

Example query:

```graphql
query {
  version
  blockStats {
    totalBlocks
    totalSize
  }
}
```

Example mutation:

```graphql
mutation {
  addBlock(data: "SGVsbG8sIElQRlJTIQ==") {
    cid
    size
  }
}
```

## Using Python Bindings

```python
import ipfrs

# Create a node
node = await ipfrs.Node.new()

# Add content
cid = await node.add(b"Hello, IPFRS!")
print(f"Added: {cid}")

# Get content
data = await node.cat(cid)
print(f"Retrieved: {data.decode()}")

# Semantic indexing
await node.index_content(cid, [0.1, 0.2, 0.3, ...])

# Search
results = await node.search_similar([0.15, 0.25, 0.35, ...], top_k=5)
for result in results:
    print(f"CID: {result.cid}, Score: {result.score}")

# Logic programming
await node.add_fact("parent(tom, bob)")
await node.add_rule("ancestor(X, Y) :- parent(X, Y)")
solutions = await node.infer("ancestor(tom, X)")
print(f"Solutions: {solutions}")
```

## Using JavaScript Bindings

```javascript
const ipfrs = require('@ipfrs/core');

(async () => {
  // Create a node
  const node = await ipfrs.createNode();

  // Add content
  const cid = await node.add(Buffer.from("Hello, IPFRS!"));
  console.log(`Added: ${cid}`);

  // Get content
  const data = await node.cat(cid);
  console.log(`Retrieved: ${data.toString()}`);

  // Semantic indexing
  await node.indexContent(cid, [0.1, 0.2, 0.3, ...]);

  // Search
  const results = await node.searchSimilar([0.15, 0.25, 0.35, ...], 5);
  results.forEach(result => {
    console.log(`CID: ${result.cid}, Score: ${result.score}`);
  });

  // Logic programming
  await node.addFact("parent(tom, bob)");
  await node.addRule("ancestor(X, Y) :- parent(X, Y)");
  const solutions = await node.infer("ancestor(tom, X)");
  console.log(`Solutions: ${JSON.stringify(solutions)}`);
})();
```

## Configuration

Create a custom configuration file at `~/.ipfrs/config.toml`:

```toml
[storage]
path = "~/.ipfrs/blocks"
cache_size_mb = 512

[network]
listen_addresses = ["/ip4/0.0.0.0/tcp/4001", "/ip4/0.0.0.0/udp/4001/quic"]
enable_dht = true
enable_mdns = true

[http]
address = "127.0.0.1:8080"
enable_cors = true

[semantic]
enable = true
dimension = 384

[tensorlogic]
enable = true
max_depth = 100
```

## Next Steps

Now that you have IPFRS running, explore:

- [Core Features](../core/storage.md) - Deep dive into storage, semantic search, and logic
- [API Documentation](../api/node.md) - Complete API reference
- [Tutorials](../tutorials/knowledge-graph.md) - Build real-world applications
- [Advanced Topics](../advanced/performance.md) - Optimization and production deployment

## Getting Help

- Check the [FAQ](../reference/faq.md)
- Read the [full documentation](../introduction.md)
- Open an [issue on GitHub](https://github.com/ipfrs/ipfrs/issues)
