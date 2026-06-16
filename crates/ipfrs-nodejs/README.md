# IPFRS Node.js Bindings

Node.js/TypeScript bindings for IPFRS (Inter-Planetary File Rust System) - a content-addressed storage system with semantic search and logic programming capabilities.

## Features

- **Content-Addressed Storage**: Store and retrieve data using cryptographic content identifiers (CIDs)
- **Semantic Search**: Vector similarity search with HNSW indexing
- **Logic Programming**: TensorLogic inference engine with backward chaining
- **Persistence**: Save and load indexes and knowledge bases
- **Promise-based API**: Full async/await support
- **TypeScript**: Complete type definitions included

## Installation

```bash
npm install @ipfrs/core
# or
yarn add @ipfrs/core
```

## Quick Start

### Basic Block Operations

```javascript
const { Node } = require('@ipfrs/core');

async function main() {
  // Create and start a node
  const node = new Node({
    storagePath: '/tmp/ipfrs-nodejs-demo',
    enableSemantic: true,
    enableTensorlogic: true
  });

  await node.start();

  // Store data
  const data = Buffer.from('Hello, IPFRS!');
  const cid = await node.putBlock(data);
  console.log(`Stored block with CID: ${cid}`);

  // Retrieve data
  const retrieved = await node.getBlock(cid);
  if (retrieved) {
    console.log(`Retrieved: ${retrieved.toString()}`);
  }

  // Check existence
  const exists = await node.hasBlock(cid);
  console.log(`Block exists: ${exists}`);

  // Clean up
  await node.stop();
}

main().catch(console.error);
```

### TypeScript Example

```typescript
import { Node, NodeConfig, SearchResult } from '@ipfrs/core';

async function semanticSearch() {
  const config: NodeConfig = {
    storagePath: '/tmp/ipfrs-semantic',
    enableSemantic: true,
    enableTensorlogic: false
  };

  const node = new Node(config);
  await node.start();

  // Store content with embeddings
  const embeddingDim = 128;
  for (let i = 0; i < 10; i++) {
    const data = Buffer.from(`Document ${i}`);
    const cid = await node.putBlock(data);

    // Generate random embedding (in real use, use a model)
    const embedding = Array.from({ length: embeddingDim }, () => Math.random());
    await node.indexContent(cid, embedding);
  }

  // Search for similar content
  const query = Array.from({ length: embeddingDim }, () => Math.random());
  const results: SearchResult[] = await node.searchSimilar(query, 5);

  console.log('Search results:');
  for (const result of results) {
    console.log(`  CID: ${result.cid}, Score: ${result.score.toFixed(4)}`);
  }

  await node.stop();
}

semanticSearch().catch(console.error);
```

### Logic Programming

```javascript
const { Node } = require('@ipfrs/core');

async function logicProgramming() {
  const node = new Node({
    storagePath: '/tmp/ipfrs-logic',
    enableTensorlogic: true
  });

  await node.start();

  // Add facts
  node.addFact({
    name: 'parent',
    args: [
      { kind: 'string', value: 'Alice' },
      { kind: 'string', value: 'Bob' }
    ]
  });

  node.addFact({
    name: 'parent',
    args: [
      { kind: 'string', value: 'Bob' },
      { kind: 'string', value: 'Charlie' }
    ]
  });

  // Add rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
  node.addRule({
    head: {
      name: 'grandparent',
      args: [
        { kind: 'var', value: 'X' },
        { kind: 'var', value: 'Z' }
      ]
    },
    body: [
      {
        name: 'parent',
        args: [
          { kind: 'var', value: 'X' },
          { kind: 'var', value: 'Y' }
        ]
      },
      {
        name: 'parent',
        args: [
          { kind: 'var', value: 'Y' },
          { kind: 'var', value: 'Z' }
        ]
      }
    ]
  });

  // Query: Who is Charlie's grandparent?
  const goal = {
    name: 'grandparent',
    args: [
      { kind: 'var', value: 'X' },
      { kind: 'string', value: 'Charlie' }
    ]
  };

  const results = node.infer(goal);
  console.log('Query results:', results);

  // Generate proof
  const proof = node.prove(goal);
  if (proof) {
    console.log('Proof:', proof);
  }

  // Get statistics
  const stats = node.kbStats();
  console.log(`KB stats: ${stats.numFacts} facts, ${stats.numRules} rules`);

  await node.stop();
}

logicProgramming().catch(console.error);
```

### Persistence

```javascript
const { Node } = require('@ipfrs/core');

async function persistence() {
  const node = new Node({ storagePath: '/tmp/ipfrs-persist' });
  await node.start();

  // ... index content and add facts/rules ...

  // Save indexes
  await node.saveSemanticIndex('/tmp/semantic_index.bin');
  await node.saveKb('/tmp/knowledge_base.bin');

  // Later, load them back
  await node.loadSemanticIndex('/tmp/semantic_index.bin');
  await node.loadKb('/tmp/knowledge_base.bin');

  await node.stop();
}

persistence().catch(console.error);
```

## API Reference

### Node

The main interface for all IPFRS operations.

**Constructor:**
- `new Node(config?: NodeConfig)` - Create a new IPFRS node

**Methods:**

**Lifecycle:**
- `start(): Promise<void>` - Start the node
- `stop(): Promise<void>` - Stop the node

**Block Operations:**
- `putBlock(data: Buffer): Promise<string>` - Store a block, returns CID
- `getBlock(cid: string): Promise<Buffer | null>` - Retrieve a block
- `hasBlock(cid: string): Promise<boolean>` - Check block existence
- `deleteBlock(cid: string): Promise<void>` - Delete a block

**Semantic Search:**
- `indexContent(cid: string, embedding: number[]): Promise<void>` - Index for semantic search
- `searchSimilar(query: number[], k: number): Promise<SearchResult[]>` - Search similar content
- `searchFiltered(query: number[], k: number, filter?: QueryFilter): Promise<SearchResult[]>` - Search with filters

**Logic Programming:**
- `addFact(fact: Predicate): void` - Add a fact
- `addRule(rule: Rule): void` - Add a rule
- `infer(goal: Predicate): string[]` - Run inference
- `prove(goal: Predicate): string | null` - Generate proof
- `kbStats(): KbStats` - Get KB statistics

**Persistence:**
- `saveSemanticIndex(path: string): Promise<void>` - Save semantic index
- `loadSemanticIndex(path: string): Promise<void>` - Load semantic index
- `saveKb(path: string): Promise<void>` - Save knowledge base
- `loadKb(path: string): Promise<void>` - Load knowledge base

### Types

**NodeConfig:**
```typescript
interface NodeConfig {
  storagePath?: string;
  enableSemantic?: boolean;
  enableTensorlogic?: boolean;
}
```

**SearchResult:**
```typescript
interface SearchResult {
  cid: string;
  score: number;
}
```

**QueryFilter:**
```typescript
interface QueryFilter {
  minScore?: number;
  maxScore?: number;
  maxResults?: number;
  cidPrefix?: string;
}
```

**Term:**
```typescript
interface Term {
  kind: 'int' | 'float' | 'string' | 'bool' | 'var';
  value: string;
}
```

**Predicate:**
```typescript
interface Predicate {
  name: string;
  args: Term[];
}
```

**Rule:**
```typescript
interface Rule {
  head: Predicate;
  body: Predicate[];
}
```

**KbStats:**
```typescript
interface KbStats {
  numFacts: number;
  numRules: number;
}
```

## Building from Source

```bash
# Install dependencies
npm install

# Build debug version
npm run build:debug

# Build release version
npm run build

# Run tests
npm test
```

## Requirements

- Node.js 14+
- Rust 1.70+ (for building from source)

## License

Apache-2.0
