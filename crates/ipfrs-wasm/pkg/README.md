# @cool-japan/ipfrs

WebAssembly bindings for **IPFRS** (Inter-Planet File RUST System) — a content-addressed,
distributed block store written in pure Rust and compiled to WASM.

## Installation

```sh
npm install @cool-japan/ipfrs
```

## Usage

### Ephemeral in-memory client

```javascript
import init, { IpfrsClient, compute_cid, verify_cid, version } from '@cool-japan/ipfrs';

// Initialise the WASM module (must be awaited once)
await init();

// Create an in-memory client (data lives only for the lifetime of this object)
const client = new IpfrsClient();

// Add data — returns a CIDv1 (base32-lower, sha2-256, raw codec) string
const enc = new TextEncoder();
const cid = await client.add(enc.encode('hello ipfrs'));
console.log('CID:', cid);

// Retrieve bytes by CID
const bytes = await client.get(cid);
console.log('Data:', new TextDecoder().decode(bytes));

// Verify a CID matches its data
const ok = verify_cid(cid, enc.encode('hello ipfrs'));
console.log('Valid:', ok); // true

// Storage statistics
const stats = JSON.parse(client.stats());
console.log(stats.block_count, stats.total_bytes);
```

### Persistent IndexedDB client (browser only)

```javascript
import init, { IpfrsClientPersistent } from '@cool-japan/ipfrs';

await init();

// Opens (or creates) an IndexedDB database named "ipfrs-blocks"
const client = await IpfrsClientPersistent.new('ipfrs-blocks');

const enc = new TextEncoder();
const cid = await client.add(enc.encode('persistent hello'));
console.log('Persisted CID:', cid);

// Data survives page refreshes
const bytes = await client.get(cid);
console.log('Retrieved:', new TextDecoder().decode(bytes));

const count = await client.count();
console.log('Blocks stored:', count);
```

### Utility helpers

```javascript
import init, { compute_cid, verify_cid, add_bytes } from '@cool-japan/ipfrs';

await init();

const data = new TextEncoder().encode('any data');

// Compute a CID without storing
const cid = compute_cid(data);

// Verify integrity
const valid = verify_cid(cid, data);

// Standalone add (creates a temporary in-memory client)
const cid2 = await add_bytes(data);
```

## API

| Export | Type | Description |
|---|---|---|
| `IpfrsClient` | class | Ephemeral in-memory block store |
| `IpfrsClientPersistent` | class | Browser-persistent IndexedDB-backed block store |
| `compute_cid(data)` | function | Compute CIDv1 without storing |
| `verify_cid(cid, data)` | function | Verify data matches a CID |
| `add_bytes(data)` | async function | Add bytes using a temporary in-memory client |
| `get_bytes(client, cid)` | async function | Retrieve bytes from an `IpfrsClient` |
| `version()` | function | Return the library version string |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
