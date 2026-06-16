# IPFRS WebAssembly Bindings

WebAssembly bindings for IPFRS (Inter-Planetary File Rust System) - enabling content-addressed storage with semantic search and logic programming in the browser.

## Features

- **Browser-Native**: Run IPFRS directly in web browsers
- **Content-Addressed Storage**: Store and retrieve data using cryptographic identifiers
- **Logic Programming**: TensorLogic inference engine in the browser
- **Semantic Search**: Vector similarity search (limited in browser)
- **TypeScript Support**: Full type definitions included
- **Zero Dependencies**: Compiled to WebAssembly for maximum compatibility

## Installation

```bash
npm install @ipfrs/wasm
# or
yarn add @ipfrs/wasm
```

## Quick Start

### Basic Usage (ES Modules)

```javascript
import init, { Node, NodeConfig } from '@ipfrs/wasm';

async function main() {
  // Initialize the WASM module
  await init();

  // Create a new node
  const config = new NodeConfig()
    .setEnableSemantic(false)  // Simplified for browser
    .setEnableTensorlogic(true);

  const node = new Node(config);

  // Logic programming example
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

  // Add a rule
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

  // Query
  const results = node.infer({
    name: 'grandparent',
    args: [
      { kind: 'var', value: 'X' },
      { kind: 'string', value: 'Charlie' }
    ]
  });

  console.log('Results:', results);

  // Get statistics
  const stats = node.kbStats();
  console.log('KB Stats:', stats);
}

main().catch(console.error);
```

### TypeScript Example

```typescript
import init, { Node, NodeConfig } from '@ipfrs/wasm';

interface Fact {
  name: string;
  args: Array<{ kind: string; value: string }>;
}

async function logicExample() {
  await init();

  const config = new NodeConfig()
    .setEnableTensorlogic(true);

  const node = new Node(config);

  // Type-safe fact
  const fact: Fact = {
    name: 'likes',
    args: [
      { kind: 'string', value: 'Alice' },
      { kind: 'string', value: 'programming' }
    ]
  };

  node.addFact(fact);

  const stats = node.kbStats();
  console.log(`Facts: ${stats.num_facts}, Rules: ${stats.num_rules}`);
}

logicExample();
```

### HTML Usage

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>IPFRS WASM Demo</title>
</head>
<body>
  <h1>IPFRS in the Browser</h1>
  <div id="output"></div>

  <script type="module">
    import init, { Node, NodeConfig } from './node_modules/@ipfrs/wasm/index.js';

    async function run() {
      await init();

      const config = new NodeConfig();
      const node = new Node(config);

      // Add some facts
      node.addFact({
        name: 'color',
        args: [
          { kind: 'string', value: 'sky' },
          { kind: 'string', value: 'blue' }
        ]
      });

      const stats = node.kbStats();
      document.getElementById('output').innerText =
        `Knowledge base has ${stats.num_facts} facts`;
    }

    run().catch(console.error);
  </script>
</body>
</html>
```

## API Reference

### NodeConfig

Configuration for IPFRS node.

**Methods:**
- `new NodeConfig()` - Create new configuration
- `setStoragePath(path: string): NodeConfig` - Set storage path (not used in browser)
- `setEnableSemantic(enable: boolean): NodeConfig` - Enable semantic search
- `setEnableTensorlogic(enable: boolean): NodeConfig` - Enable logic programming

### Node

Main IPFRS interface.

**Constructor:**
- `new Node(config?: NodeConfig)` - Create new node

**Methods:**

**Lifecycle:**
- `start(): void` - Start the node (no-op in browser)
- `stop(): void` - Stop the node

**Block Operations (Limited in Browser):**
- `putBlock(data: Uint8Array): string` - Store block, returns CID
- `hasBlock(cid: string): boolean` - Check if block exists

**Logic Programming:**
- `addFact(fact: Predicate): void` - Add a fact
- `addRule(rule: Rule): void` - Add a rule
- `infer(goal: Predicate): string[]` - Run inference
- `kbStats(): KbStats` - Get KB statistics

### Types

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
  num_facts: number;
  num_rules: number;
}
```

## Building from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
npm run build

# Build for Node.js
npm run build:nodejs

# Build for bundlers (webpack, etc.)
npm run build:bundler

# Run tests
npm test
```

## Browser Compatibility

- Modern browsers with WebAssembly support
- Chrome/Edge 57+
- Firefox 52+
- Safari 11+
- Opera 44+

## Limitations

Due to browser environment constraints:

1. **No File System**: Storage path is not used in browser
2. **Simplified Async**: Some async operations are synchronous in WASM
3. **Limited Semantic Search**: Full HNSW index may be too large for browser memory
4. **No Network**: Distributed features not available in browser

## Use Cases

- **Education**: Interactive logic programming tutorials
- **Demos**: Showcase IPFRS capabilities without backend
- **Client-Side**: Local data processing and reasoning
- **Prototyping**: Quick experiments with content-addressed storage

## Size Optimization

The WASM binary is optimized for size:

- Release build: ~800KB (compressed)
- Gzip: ~250KB
- Brotli: ~180KB

## License

Apache-2.0
