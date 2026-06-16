# IPFRS Storage Guide

Comprehensive guide for storage backend selection, performance tuning, and migration from IPFS.

## Table of Contents

1. [Backend Selection](#backend-selection)
2. [Performance Tuning](#performance-tuning)
3. [Migration from IPFS](#migration-from-ipfs)
4. [Multi-Datacenter Deployment](#multi-datacenter-deployment)
5. [ARM Optimization and Low-Power Operation](#arm-optimization-and-low-power-operation)

---

## Backend Selection

### Overview

IPFRS Storage supports two high-performance embedded database backends:

- **Sled** - Pure Rust, ACID-compliant, optimized for SSDs
- **ParityDB** - Column-based, optimized for write-heavy workloads

### When to Use Sled

**Best For:**
- General-purpose workloads with balanced read/write ratios
- Development and prototyping
- Systems requiring strong ACID guarantees
- Pure Rust stack requirements (no C/C++ dependencies)
- Medium-sized datasets (<100GB)

**Advantages:**
- Pure Rust implementation (no FFI overhead)
- ACID compliance with full crash recovery
- Mature and battle-tested in production
- Excellent read performance
- Automatic background compaction
- Lower memory overhead for small datasets

**Configuration:**
```rust
use ipfrs_storage::{SledBlockStore, BlockStoreConfig};
use std::path::PathBuf;

let config = BlockStoreConfig {
    path: PathBuf::from(".ipfrs/blocks"),
    cache_size: 100 * 1024 * 1024, // 100MB cache
};

let store = SledBlockStore::new(config)?;
```

**Performance Characteristics:**
- Single block write: ~500μs - 1ms
- Single block read (cache miss): ~200-500μs
- Batch write (100 blocks): 30-50ms
- Suitable for: Most general-purpose applications

---

### When to Use ParityDB

**Best For:**
- Write-heavy ingestion workloads
- Large datasets (>100GB)
- Blockchain and append-heavy workloads
- Systems with aggressive write amplification requirements
- SSD-optimized deployments

**Advantages:**
- 2-3x better write performance than Sled
- Lower write amplification (better SSD longevity)
- Column-based storage layout
- Built-in compression (LZ4)
- Three configuration presets for different use cases
- Excellent for sequential writes

**Configuration Presets:**

#### 1. Fast Write (High Throughput Ingestion)
```rust
use ipfrs_storage::{ParityDbBlockStore, ParityDbConfig};
use std::path::PathBuf;

let config = ParityDbConfig::fast_write(
    PathBuf::from(".ipfrs/blocks-paritydb")
);

let store = ParityDbBlockStore::new(config)?;
```
- **Use Case:** Bulk data ingestion, initial sync, backup restoration
- **Trade-offs:** Async WAL (less durable), higher memory usage
- **Write Performance:** 2-3x faster than Sled
- **Recommended For:** Ingestion pipelines, batch processing

#### 2. Balanced (General Purpose)
```rust
let config = ParityDbConfig::balanced(
    PathBuf::from(".ipfrs/blocks-paritydb")
);
let store = ParityDbBlockStore::new(config)?;
```
- **Use Case:** General-purpose storage with good durability
- **Trade-offs:** Balanced between performance and safety
- **Sync WAL:** Enabled (better crash recovery)
- **Recommended For:** Production deployments, long-running nodes

#### 3. Low Memory (Constrained Devices)
```rust
let config = ParityDbConfig::low_memory(
    PathBuf::from(".ipfrs/blocks-paritydb")
);
let store = ParityDbBlockStore::new(config)?;
```
- **Use Case:** Edge devices, embedded systems, Raspberry Pi
- **Trade-offs:** No B-tree index (saves memory), slightly slower lookups
- **Memory Footprint:** ~30% less than Balanced preset
- **Recommended For:** ARM devices, low-memory environments (<4GB RAM)

**Performance Characteristics:**
- Single block write: ~200-400μs
- Single block read (cache miss): ~300-600μs
- Batch write (100 blocks): 15-30ms
- Suitable for: High-throughput, write-heavy applications

---

### Decision Matrix

| Criterion | Sled | ParityDB |
|-----------|------|----------|
| **Write Performance** | Good | Excellent (2-3x) |
| **Read Performance** | Excellent | Good |
| **Memory Efficiency** | Good | Excellent (with low_memory preset) |
| **SSD Longevity** | Good | Excellent (lower write amp) |
| **ACID Compliance** | Full | Good |
| **Pure Rust** | ✅ | ✅ |
| **Compression** | No | Yes (LZ4) |
| **Best Dataset Size** | <100GB | >100GB |
| **ARM Optimization** | Good | Excellent |

### Benchmark Comparison

See `benches/blockstore_bench.rs` for comprehensive benchmarks. Run with:

```bash
cargo bench --bench blockstore_bench
```

**Typical Results (x86_64, NVMe SSD):**

| Operation | Block Size | Sled | ParityDB (fast_write) | Winner |
|-----------|------------|------|-----------------------|--------|
| Single PUT | 1KB | 0.8ms | 0.3ms | ParityDB |
| Single PUT | 100KB | 1.2ms | 0.5ms | ParityDB |
| Single GET | 1KB | 0.3ms | 0.4ms | Sled |
| Single GET | 100KB | 0.5ms | 0.6ms | Sled |
| Batch PUT (100) | 1KB each | 45ms | 20ms | ParityDB |

---

## Performance Tuning

### Hardware-Specific Optimizations

#### SSD Deployment (Recommended)

Both Sled and ParityDB are optimized for SSD storage.

**Configuration Tips:**
- Increase cache size for better read performance
- Enable compression for ParityDB to reduce I/O
- Use `fast_write` preset for ParityDB in write-heavy scenarios

**Sled on SSD:**
```rust
let config = BlockStoreConfig {
    path: PathBuf::from("/fast-ssd/ipfrs/blocks"),
    cache_size: 500 * 1024 * 1024, // 500MB cache for faster reads
};
```

**ParityDB on SSD:**
```rust
let config = ParityDbConfig::fast_write(
    PathBuf::from("/fast-ssd/ipfrs/blocks-paritydb")
);
// Already optimized for SSD with async writes
```

---

#### HDD Deployment (Budget Systems)

While not recommended, IPFRS Storage can work on HDDs with tuning.

**Key Considerations:**
- Use ParityDB `balanced` preset (better sequential writes)
- Enable hot/cold tiering to keep frequently accessed blocks on fast storage
- Reduce cache size to conserve memory
- Consider using memory-mapped I/O for large blocks

**Configuration:**
```rust
use ipfrs_storage::{ParityDbConfig, ParityDbPreset};

let config = ParityDbConfig::new(
    PathBuf::from("/slow-hdd/ipfrs/blocks"),
    ParityDbPreset::Balanced
);

let store = ParityDbBlockStore::new(config)?;
```

**Performance Expectations:**
- 5-10x slower than SSD
- Use with tiered storage for best results

---

#### ARM Devices (Raspberry Pi, Jetson, Mobile)

ARM devices benefit from ParityDB's `low_memory` preset.

**Raspberry Pi 4 (4GB RAM):**
```rust
let config = ParityDbConfig::low_memory(
    PathBuf::from("/home/pi/.ipfrs/blocks")
);

let store = ParityDbBlockStore::new(config)?;
```

**Optimizations:**
- No B-tree index (saves ~30% memory)
- LZ4 compression reduces disk I/O
- Lower cache requirements
- Better power efficiency

**Expected Performance:**
- Single block write: 1-2ms
- Single block read: 0.5-1ms
- Suitable for edge computing and IoT

---

#### x86_64 High-Performance Servers

For maximum throughput on server hardware:

**ParityDB Fast Write:**
```rust
let config = ParityDbConfig::fast_write(
    PathBuf::from("/nvme/ipfrs/blocks")
);
let store = ParityDbBlockStore::new(config)?;
```

**Additional Optimization Layers:**

1. **Bloom Filter (Fast Negative Lookups):**
```rust
use ipfrs_storage::{BloomBlockStore, BloomConfig};

let bloom_config = BloomConfig::default(); // 1% false positive rate
let store = BloomBlockStore::new(store, bloom_config);
// 10x faster has() checks for missing blocks
```

2. **LRU Cache (Fast Reads):**
```rust
use ipfrs_storage::CachedBlockStore;

let store = CachedBlockStore::new(store, 1024 * 1024 * 1024); // 1GB cache
// 100x faster for cached blocks
```

3. **Tiered Storage (Hot/Cold):**
```rust
use ipfrs_storage::{TieredStore, TierConfig};

let hot_store = ParityDbBlockStore::new(
    ParityDbConfig::fast_write(PathBuf::from("/nvme/hot"))
)?;
let cold_store = ParityDbBlockStore::new(
    ParityDbConfig::low_memory(PathBuf::from("/hdd/cold"))
)?;

let tier_config = TierConfig::default();
let store = TieredStore::new(hot_store, cold_store, tier_config)?;
// Automatically migrates cold blocks to slower storage
```

**Full Stack Example:**
```rust
// Layer 1: ParityDB backend (fast writes)
let paritydb = ParityDbBlockStore::new(
    ParityDbConfig::fast_write(PathBuf::from("/nvme/blocks"))
)?;

// Layer 2: Bloom filter (fast negative lookups)
let bloom = BloomBlockStore::new(paritydb, BloomConfig::default());

// Layer 3: LRU cache (fast reads)
let cached = CachedBlockStore::new(bloom, 2 * 1024 * 1024 * 1024); // 2GB

// Result: Optimized for all access patterns
```

---

### Memory Constraints

#### Low Memory (<2GB RAM)

Use ParityDB `low_memory` preset with minimal caching:

```rust
let config = ParityDbConfig::low_memory(
    PathBuf::from(".ipfrs/blocks")
);
let store = ParityDbBlockStore::new(config)?;

// Skip LRU cache or use very small cache (50MB)
let cached = CachedBlockStore::new(store, 50 * 1024 * 1024);
```

#### Medium Memory (4-8GB RAM)

Use balanced preset with moderate caching:

```rust
let config = ParityDbConfig::balanced(
    PathBuf::from(".ipfrs/blocks")
);
let store = ParityDbBlockStore::new(config)?;

let cached = CachedBlockStore::new(store, 512 * 1024 * 1024); // 512MB cache
```

#### High Memory (>16GB RAM)

Maximize caching for best performance:

```rust
let config = ParityDbConfig::fast_write(
    PathBuf::from(".ipfrs/blocks")
);
let store = ParityDbBlockStore::new(config)?;

let cached = CachedBlockStore::new(store, 4 * 1024 * 1024 * 1024); // 4GB cache
```

---

## Migration from IPFS

### Overview

IPFRS Storage provides CAR (Content Addressable aRchive) format support for importing/exporting blocks, making migration from IPFS straightforward.

### Migration Strategies

#### Strategy 1: CAR Export/Import (Recommended)

**Step 1: Export from IPFS (Kubo)**

```bash
# Export entire IPFS datastore to CAR file
ipfs dag export <root-cid> > ipfs-export.car

# Or export specific paths
ipfs dag export /ipfs/QmXxx... > my-data.car
```

**Step 2: Import into IPFRS**

```rust
use ipfrs_storage::{SledBlockStore, BlockStoreConfig, import_from_car};
use std::path::PathBuf;
use std::fs::File;

// Create IPFRS blockstore
let config = BlockStoreConfig::default();
let store = SledBlockStore::new(config)?;

// Import CAR file
let car_file = File::open("ipfs-export.car")?;
let stats = import_from_car(&store, car_file).await?;

println!("Imported {} blocks ({} bytes)",
         stats.blocks_read, stats.bytes_read);
```

**Benefits:**
- Standard format (IPFS compatible)
- Verifies CIDs during import
- Atomic import (all or nothing)
- Progress tracking

---

#### Strategy 2: Direct Datastore Migration

For advanced users with direct access to IPFS datastore files.

**Supported IPFS Datastores:**
- Badger (default in Kubo)
- LevelDB (legacy)
- Flatfs (file-based)

**Migration Steps:**

1. **Stop IPFS daemon:**
```bash
ipfs shutdown
```

2. **Export all blocks to CAR:**
```bash
# Find all blocks in datastore
ipfs refs local > all-refs.txt

# Export to CAR (may take time for large datastores)
ipfs dag export $(cat all-refs.txt) > full-export.car
```

3. **Import into IPFRS:**
```rust
use ipfrs_storage::{ParityDbBlockStore, ParityDbConfig, import_from_car};

let config = ParityDbConfig::fast_write(
    PathBuf::from(".ipfrs/blocks-paritydb")
);
let store = ParityDbBlockStore::new(config)?;

let car_file = File::open("full-export.car")?;
let stats = import_from_car(&store, car_file).await?;
```

---

#### Strategy 3: Live Migration (Experimental)

Migrate while IPFS continues running (requires HTTP gateway access).

```rust
use ipfrs_storage::{HybridBlockStore, GatewayConfig};

// Create hybrid store with IPFS gateway fallback
let local = SledBlockStore::new(BlockStoreConfig::default())?;

let gateway_config = GatewayConfig {
    gateways: vec![
        "http://localhost:8080".to_string(), // Local IPFS gateway
    ],
    cache_locally: true, // Cache retrieved blocks
    ..Default::default()
};

let store = HybridBlockStore::new(local, gateway_config)?;

// Blocks not in local store will be fetched from IPFS gateway
// and cached locally
```

---

### Migration Checklist

- [ ] Backup IPFS datastore before migration
- [ ] Verify disk space (IPFRS needs ~1.2x IPFS datastore size during migration)
- [ ] Choose backend (Sled for <100GB, ParityDB for >100GB)
- [ ] Export IPFS data to CAR format
- [ ] Import CAR into IPFRS
- [ ] Verify block counts match
- [ ] Test data integrity (random sampling)
- [ ] Update application to use IPFRS APIs
- [ ] Monitor performance after migration

---

### Verification

After migration, verify data integrity:

```rust
use ipfrs_storage::traits::BlockStore;

// Check block count
let ipfrs_count = store.len();
println!("IPFRS block count: {}", ipfrs_count);

// Verify specific CIDs
let cid = Cid::try_from("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")?;
let block = store.get(&cid).await?;
assert!(block.is_some(), "Block not found after migration");
```

---

### Performance Comparison: IPFS vs IPFRS

| Operation | IPFS (Badger) | IPFRS (Sled) | IPFRS (ParityDB) |
|-----------|---------------|--------------|------------------|
| Single block write | 1-2ms | 0.5-1ms | 0.2-0.4ms |
| Single block read | 0.5-1ms | 0.3-0.5ms | 0.4-0.6ms |
| Batch write (100 blocks) | 100-150ms | 45ms | 20ms |
| Memory overhead (100K blocks) | ~300MB | ~150MB | ~100MB |
| Disk write amplification | High | Medium | Low |

**Expected Improvements:**
- 2-5x faster writes
- 1.5-2x faster reads (with caching)
- 30-50% lower memory usage
- Better SSD longevity (lower write amplification)

---

## Advanced Features

### Garbage Collection

IPFRS includes built-in garbage collection to reclaim space:

```rust
use ipfrs_storage::{GarbageCollector, GcConfig, PinManager};

// Pin important blocks to prevent deletion
let pin_manager = PinManager::new();
pin_manager.pin(&important_cid, PinType::Direct)?;

// Configure GC
let gc_config = GcConfig {
    incremental: true,
    batch_size: Some(1000),
    ..Default::default()
};

let gc = GarbageCollector::new(store, pin_manager, gc_config);

// Run GC (removes unpinned blocks)
let result = gc.collect().await?;
println!("Collected {} blocks, freed {} bytes",
         result.blocks_collected, result.bytes_freed);
```

### Streaming Large Blocks

For blocks >1MB, use streaming interface:

```rust
use ipfrs_storage::{StreamingBlockStore, BlockReader};
use tokio::io::AsyncReadExt;

// Read large block in chunks
let mut reader = store.read(&large_cid, StreamConfig::default()).await?;
let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer

while let Ok(n) = reader.read(&mut buffer).await {
    if n == 0 { break; }
    // Process chunk
}
```

### S3 Backend (Cloud Storage)

For hybrid cloud deployments:

```rust
#[cfg(feature = "s3")]
use ipfrs_storage::{S3BlockStore, S3Config};

let s3_config = S3Config {
    bucket: "my-ipfrs-blocks".to_string(),
    region: "us-west-2".to_string(),
    ..Default::default()
};

let s3_store = S3BlockStore::new(s3_config).await?;
```

### Block Compression

For storage-constrained deployments, enable transparent block compression:

```rust
#[cfg(feature = "compression")]
use ipfrs_storage::{
    CompressionBlockStore, CompressionConfig, CompressionAlgorithm
};

// Wrap any block store with compression
let store = SledBlockStore::new(config)?;

// Configure compression
let compression_config = CompressionConfig::new(CompressionAlgorithm::Zstd)
    .with_level(3)                    // Compression level (1-22 for Zstd)
    .with_threshold(512)              // Only compress blocks > 512 bytes
    .with_max_ratio(0.9);             // Reject if compression < 10% savings

let compressed_store = CompressionBlockStore::new(store, compression_config);

// Use normally - compression is transparent
compressed_store.put(&block).await?;
let retrieved = compressed_store.get(&cid).await?;

// Check compression stats
let stats = compressed_store.stats();
println!("Compression ratio: {:.2}%", stats.compression_ratio() * 100.0);
println!("Space saved: {} MB", stats.bytes_saved() / 1_000_000);
```

**Available Algorithms:**
- **Zstd** - Best compression ratio (3-5x), fast decompression, recommended for most use cases
- **Lz4** - Very fast (10-20x faster than Zstd), moderate ratio (2-3x), good for real-time systems
- **Snappy** - Fastest, good for streaming, moderate ratio (1.5-2.5x)

**Performance Impact:**
- Zstd level 3: ~5-10% CPU overhead, 60-80% storage reduction (text-heavy data)
- Lz4: ~2-3% CPU overhead, 40-60% storage reduction
- Snappy: ~1-2% CPU overhead, 30-50% storage reduction

**When to Enable:**
- Disk space is limited
- Network bandwidth is expensive (cloud storage)
- Data is highly compressible (text, logs, JSON, structured data)
- CPU capacity is available

**When to Avoid:**
- Data is already compressed (images, video, compressed archives)
- CPU is bottleneck
- Ultra-low latency requirements (<1ms per block)

**Combining with Deduplication:**
```rust
use ipfrs_storage::{DedupBlockStore, ChunkingConfig};

// Chain compression + deduplication for maximum savings
let base_store = SledBlockStore::new(config)?;
let compressed = CompressionBlockStore::new(base_store, compression_config);
let dedup_store = DedupBlockStore::new(compressed, ChunkingConfig::default());

// Now benefits from both compression AND deduplication
```

---

## Troubleshooting

### High Memory Usage

**Symptoms:** Process consuming excessive RAM

**Solutions:**
1. Reduce cache size in BlockStoreConfig
2. Switch to ParityDB `low_memory` preset
3. Enable tiered storage to offload cold blocks
4. Disable bloom filter (saves ~10MB per 1M blocks)

### Slow Write Performance

**Symptoms:** Block writes taking >5ms

**Solutions:**
1. Switch to ParityDB `fast_write` preset
2. Use batch operations (`put_many`) instead of single puts
3. Verify SSD health (write amplification)
4. Disable sync writes in ParityDB (less durable)

### Slow Read Performance

**Symptoms:** Cache miss reads taking >2ms

**Solutions:**
1. Increase LRU cache size
2. Enable bloom filter for faster negative lookups
3. Use memory-mapped I/O for large blocks (feature: `mmap`)
4. Verify disk I/O (run `iostat` or `iotop`)

### Disk Space Issues

**Symptoms:** Running out of disk space

**Solutions:**
1. Run garbage collection regularly
2. Enable transparent block compression (feature: `compression`) - can reduce storage by 50-80%
3. Use content-defined deduplication (feature: `dedup`)
4. Switch to ParityDB with built-in LZ4 compression
5. Use tiered storage with cheaper cold storage
6. Export old data to CAR archives and delete locally

---

## Multi-Datacenter Deployment

### Overview

IPFRS Storage supports geo-distributed deployments with datacenter-aware routing, latency-based node selection, and flexible replication policies.

### Setting Up Multi-Datacenter RAFT

```rust
use ipfrs_storage::{
    MultiDatacenterCoordinator, Datacenter, DatacenterId, Region,
    ReplicationPolicy, LatencyAwareSelector,
    RaftNode, RaftConfig, NodeId,
    InMemoryTransport, MemoryBlockStore,
};
use std::sync::Arc;
use std::time::Duration;

// 1. Create datacenter coordinator
let mut dc_coord = MultiDatacenterCoordinator::new();

// 2. Define datacenters in different regions
let us_east = Datacenter::new(
    DatacenterId::new("us-east-1"),
    Region::new("us-east"),
);
let us_west = Datacenter::new(
    DatacenterId::new("us-west-2"),
    Region::new("us-west"),
);
let eu_west = Datacenter::new(
    DatacenterId::new("eu-west-1"),
    Region::new("eu-west"),
);

dc_coord.add_datacenter(us_east);
dc_coord.add_datacenter(us_west);
dc_coord.add_datacenter(eu_west);

// 3. Register RAFT nodes in datacenters
let node1 = NodeId(1);
let node2 = NodeId(2);
let node3 = NodeId(3);

dc_coord.register_node(node1, DatacenterId::new("us-east-1"))?;
dc_coord.register_node(node2, DatacenterId::new("us-west-2"))?;
dc_coord.register_node(node3, DatacenterId::new("eu-west-1"))?;

// 4. Record cross-datacenter latencies
dc_coord.record_latency(
    DatacenterId::new("us-east-1"),
    DatacenterId::new("us-west-2"),
    70, // 70ms
);
dc_coord.record_latency(
    DatacenterId::new("us-east-1"),
    DatacenterId::new("eu-west-1"),
    120, // 120ms
);

// 5. Create latency-aware selector for reads
let dc_coord = Arc::new(dc_coord);
let selector = LatencyAwareSelector::new(dc_coord.clone())
    .with_local_preference(true)
    .with_max_latency(200); // 200ms max

// 6. Select optimal nodes for reads
let available_nodes = vec![node1, node2, node3];
let selected_nodes = selector.select_read_nodes(&available_nodes, &node1);
// Returns nodes sorted by: local first, then by latency
```

### Replication Policies

IPFRS Storage supports multiple replication policies:

#### All Datacenters
```rust
let policy = ReplicationPolicy::AllDatacenters;
let targets = policy.select_datacenters(&dc_coord, &source_dc);
// Replicates to all datacenters
```

#### Regional Replication
```rust
let policy = ReplicationPolicy::Regions(vec![
    Region::new("us-east"),
    Region::new("us-west"),
]);
let targets = policy.select_datacenters(&dc_coord, &source_dc);
// Replicates only within US regions
```

#### N-Closest Datacenters
```rust
let policy = ReplicationPolicy::NClosest(2);
let targets = policy.select_datacenters(&dc_coord, &source_dc);
// Replicates to 2 nearest datacenters by latency
```

#### Custom Policy
```rust
let policy = ReplicationPolicy::Custom(vec![
    DatacenterId::new("us-east-1"),
    DatacenterId::new("eu-west-1"),
]);
// Explicitly specify target datacenters
```

### Monitoring Cross-Datacenter Traffic

```rust
use ipfrs_storage::CrossDcStats;

let mut stats = CrossDcStats::new();

// Track operations
stats.record_local();          // Local DC read
stats.record_cross_dc(75);     // Cross-DC read (75ms)

// Get statistics
println!("Cross-DC percentage: {:.1}%", stats.cross_dc_percentage());
println!("Avg cross-DC latency: {:.1}ms", stats.avg_cross_dc_latency_ms);
```

### Best Practices

1. **Latency Measurement:** Measure actual latencies between datacenters periodically
2. **Local Preference:** Enable local datacenter preference for read-heavy workloads
3. **Replication Strategy:** Balance between consistency (more replicas) and cost
4. **Region Affinity:** Use regional replication policies to comply with data residency laws
5. **Monitoring:** Track cross-datacenter traffic to optimize placement

---

## ARM Optimization and Low-Power Operation

### Overview

IPFRS Storage includes optimizations for ARM devices (Raspberry Pi, NVIDIA Jetson, mobile devices) with NEON SIMD acceleration and power-efficient operation modes.

### ARM Feature Detection

```rust
use ipfrs_storage::ArmFeatures;

let features = ArmFeatures::detect();

println!("Running on ARM: {}", features.is_arm());
println!("NEON support: {}", features.has_neon);

if features.is_aarch64 {
    println!("Platform: AArch64");
} else if features.is_armv7 {
    println!("Platform: ARMv7");
}
```

### NEON-Optimized Operations

IPFRS automatically uses NEON SIMD instructions on AArch64 for hash computations:

```rust
use ipfrs_storage::hash_block;

// Automatically uses NEON on AArch64, fallback on other platforms
let data = vec![0u8; 4096];
let hash = hash_block(&data);
```

**Performance:** Up to 2x faster hash computation on AArch64 devices with NEON.

### Power Profiles for Battery-Powered Devices

```rust
use ipfrs_storage::{PowerProfile, LowPowerBatcher};

// Choose power profile based on deployment
let profile = if on_battery {
    PowerProfile::LowPower        // Batch size: 50, delay: 100ms
} else if plugged_in {
    PowerProfile::Balanced        // Batch size: 10, delay: 10ms
} else {
    PowerProfile::Performance     // Batch size: 1, delay: 0ms
};

// Create a batcher for operations
let batcher: LowPowerBatcher<BlockOp> = LowPowerBatcher::new(profile);

// Operations are automatically batched
for op in operations {
    if let Some(batch) = batcher.push(op) {
        // Process batch when full
        process_batch(batch);
    }
}

// Flush remaining
let remaining = batcher.flush();
if !remaining.is_empty() {
    process_batch(remaining);
}
```

### Power Profiles Comparison

| Profile | Batch Size | Delay | Use Case |
|---------|------------|-------|----------|
| **Performance** | 1 | 0ms | Server deployments, no power constraints |
| **Balanced** | 10 | 10ms | Desktop, general-purpose |
| **LowPower** | 50 | 100ms | Battery-powered, edge devices |
| **Custom** | User-defined | User-defined | Fine-tuned for specific hardware |

### Performance Monitoring

```rust
use ipfrs_storage::{ArmPerfCounter, ArmPerfReport};

// Create performance counters
let put_counter = ArmPerfCounter::new("block_put");
let get_counter = ArmPerfCounter::new("block_get");

// Time operations
{
    let _timer = put_counter.start();
    store.put(&cid, data).await?;
}

{
    let _timer = get_counter.start();
    let data = store.get(&cid).await?;
}

// Generate report
let report = ArmPerfReport::from_counters(&[put_counter, get_counter]);
report.print();
```

### Power Statistics

```rust
use ipfrs_storage::PowerStats;

let mut stats = PowerStats::new();

// Record batch operations
stats.record_batch(10, Duration::from_millis(5));
stats.record_batch(15, Duration::from_millis(8));

// Analyze power efficiency
println!("Average ops/wakeup: {:.1}", stats.avg_ops_per_wakeup());
println!("Power saving ratio: {:.3}", stats.power_saving_ratio());
// Lower ratio = better (fewer wakeups per operation)
```

### ARM Deployment Recommendations

#### Raspberry Pi (ARM Cortex-A72)
- **Backend:** ParityDB balanced preset
- **Cache:** 50-100MB (limited RAM)
- **Power Profile:** LowPower or Balanced
- **Features:** Enable compression, deduplication
- **Expected Performance:** 100-200 block writes/sec

#### NVIDIA Jetson Nano/Xavier (ARM Cortex-A57/Carmel)
- **Backend:** ParityDB fast_write preset
- **Cache:** 200-500MB
- **Power Profile:** Balanced or Performance
- **Features:** Enable mmap for zero-copy reads
- **Expected Performance:** 500-1000 block writes/sec

#### Mobile/Edge Devices (ARMv8)
- **Backend:** Sled (lower memory footprint)
- **Cache:** 20-50MB
- **Power Profile:** LowPower
- **Features:** Enable tiered storage, S3 cold storage
- **Expected Performance:** 50-100 block writes/sec

### Optimizing for ARM

1. **Use NEON-Optimized Operations:** Hash computation is automatically accelerated
2. **Enable Power Profiles:** Reduce CPU wake-ups on battery-powered devices
3. **Batch Operations:** Use `put_many()` and `get_many()` for better efficiency
4. **Monitor Performance:** Use `ArmPerfCounter` to identify bottlenecks
5. **Tune Cache Size:** Balance between performance and available RAM
6. **Consider Tiering:** Use local storage for hot data, cloud for cold data

---

## Resources

- **API Documentation:** Run `cargo doc --open` in the `ipfrs-storage` crate
- **Benchmarks:** `cargo bench --bench blockstore_bench`
- **ARM Benchmarks:** `cargo bench --bench arm_optimization_bench`
- **Examples:** See `ipfrs/examples/` directory
- **Integration Tests:** See `/tmp/multi_dc_raft_integration_test.rs`
- **Issues:** https://github.com/your-repo/ipfrs/issues

---

## License

IPFRS Storage is licensed under Apache-2.0.
