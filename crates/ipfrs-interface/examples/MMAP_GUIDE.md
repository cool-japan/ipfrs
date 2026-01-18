# Memory-Mapped I/O Guide for IPFRS

This guide explains how to use memory-mapped I/O (mmap) for zero-copy tensor serving in IPFRS.

## What is Memory-Mapped I/O?

Memory-mapped I/O allows files to be accessed as if they were in memory, without explicitly reading them into buffers. The operating system handles loading file pages on-demand, providing several benefits:

- **Zero-copy**: Data is served directly from disk via OS page cache
- **Lazy loading**: Only requested pages are loaded into memory
- **OS optimizations**: Leverages system calls like `sendfile` for efficient transfers
- **Reduced memory usage**: No need to allocate buffers for entire files

## When to Use Mmap

### Good Use Cases

1. **Large tensor files** (>1MB) stored on local filesystem
2. **Random access patterns** (tensor slicing, partial requests)
3. **High-throughput serving** of the same files repeatedly
4. **Range requests** (HTTP 206 Partial Content)

### When NOT to Use Mmap

1. **Small files** (<64KB) - overhead may exceed benefits
2. **Distributed storage** - only works with local filesystem
3. **Sequential-only reads** - buffered I/O may be simpler
4. **Write-heavy workloads** - mmap is optimized for reading

## Basic Usage

### 1. Creating a Memory-Mapped File

```rust
use ipfrs_interface::mmap::MmapFile;

// Open a file and create memory map
let mmap_file = MmapFile::new("/path/to/tensor.bin")?;

// Get file size
let size = mmap_file.size();

// Check if empty
if mmap_file.is_empty() {
    println!("File is empty");
}
```

### 2. Reading Data

```rust
// Read full file (zero-copy)
let data = mmap_file.bytes();

// Read a specific range
let range_data = mmap_file.range(0..1024)?;

// Read multiple ranges
let ranges = vec![0..1024, 10240..11264, 102400..103424];
let multi_range_data = mmap_file.multi_range(&ranges)?;
```

### 3. Using the Cache

```rust
use ipfrs_interface::mmap::MmapCache;
use std::sync::Arc;

// Create cache (max 100 files)
let cache = Arc::new(MmapCache::new(100));

// Get or create mapping (reuses existing if cached)
let mmap_file = cache.get_or_create("/path/to/tensor.bin")?;

// Cache statistics
println!("Cache size: {}", cache.len());
println!("Cache empty: {}", cache.is_empty());

// Clear cache
cache.clear();
```

## Platform-Specific Optimizations

Configure mmap behavior based on access patterns:

```rust
use ipfrs_interface::mmap::MmapConfig;

// Sequential access (streaming downloads)
let config = MmapConfig::sequential();

// Random access (tensor slicing)
let config = MmapConfig::random();

// Large files with hugepages
let config = MmapConfig::hugepages();

// Custom configuration
let config = MmapConfig {
    use_hugepages: true,
    sequential_access: false,
    random_access: true,
    populate: true,
};
```

## Integration with HTTP Endpoints

### Example: Tensor Serving Endpoint

```rust
use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::Response,
};
use ipfrs_interface::mmap::{MmapCache, MmapError};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    mmap_cache: Arc<MmapCache>,
    storage_path: PathBuf,
}

async fn get_tensor(
    State(state): State<AppState>,
    Path(cid): Path<String>,
) -> Result<Response, MmapError> {
    // Construct file path
    let file_path = state.storage_path.join(format!("{}.tensor", cid));

    // Get or create memory-mapped file
    let mmap_file = state.mmap_cache.get_or_create(&file_path)?;

    // Get data (zero-copy)
    let data = mmap_file.bytes();

    // Build response
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, data.len())
        .header("X-Served-By", "mmap")
        .body(data.into())
        .unwrap())
}
```

### Example: Range Request Support

```rust
async fn get_tensor_range(
    State(state): State<AppState>,
    Path(cid): Path<String>,
    Query(range): Query<RangeQuery>,
) -> Result<Response, MmapError> {
    let file_path = state.storage_path.join(format!("{}.tensor", cid));
    let mmap_file = state.mmap_cache.get_or_create(&file_path)?;

    // Get requested range
    let data = mmap_file.range(range.start..range.end)?;

    // HTTP 206 Partial Content
    Ok(Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, data.len())
        .header(
            header::CONTENT_RANGE,
            format!("bytes {}-{}/{}", range.start, range.end - 1, mmap_file.size()),
        )
        .body(data.into())
        .unwrap())
}
```

## Performance Tips

### 1. Cache Sizing

```rust
// Size cache based on working set
let num_hot_files = 100;
let cache = MmapCache::new(num_hot_files);
```

### 2. Sequential vs Random Access

- **Sequential**: Set `sequential_access: true` for streaming
- **Random**: Set `random_access: true` for tensor slicing

### 3. Hugepages for Large Files

On Linux, enable hugepages for files >2MB:

```rust
let config = MmapConfig::hugepages();
```

### 4. Reuse Mappings

Always use `MmapCache` to reuse mappings across requests:

```rust
// Good: Reuses mapping
let mmap = cache.get_or_create(path)?;

// Bad: Creates new mapping every time
let mmap = MmapFile::new(path)?;
```

## Error Handling

```rust
use ipfrs_interface::mmap::MmapError;

match mmap_file.range(0..1024) {
    Ok(data) => println!("Got {} bytes", data.len()),
    Err(MmapError::InvalidRange(msg)) => {
        eprintln!("Invalid range: {}", msg);
    }
    Err(MmapError::FileNotFound(path)) => {
        eprintln!("File not found: {}", path);
    }
    Err(e) => {
        eprintln!("Mmap error: {}", e);
    }
}
```

## Benchmarking

Run benchmarks to measure mmap performance:

```bash
cargo bench --bench http_benchmarks -- mmap
```

Expected results:
- **Cache hit**: <1μs
- **Cache miss**: ~10-50μs
- **Range access**: ~1-10μs depending on size
- **Mmap vs file read**: 2-10x faster for large files

## Complete Example

See `examples/mmap_tensor_server.rs` for a complete working example:

```bash
cargo run --example mmap_tensor_server
```

Test with curl:

```bash
# Full file
curl http://localhost:8080/tensor/example1

# Byte range
curl http://localhost:8080/tensor/example1?start=0&end=100

# Cache stats
curl http://localhost:8080/cache/stats
```

## Limitations

1. **Local filesystem only** - doesn't work with network storage
2. **Read-only** - current implementation is read-only
3. **Linux/Unix focus** - Windows support is basic
4. **File size limits** - very large files (>1TB) may need special handling

## Best Practices

1. ✅ Use `MmapCache` for frequently accessed files
2. ✅ Configure access pattern hints (sequential/random)
3. ✅ Handle errors gracefully (file not found, invalid ranges)
4. ✅ Monitor cache size and eviction
5. ✅ Use for files >1MB on local storage
6. ❌ Don't use for small files (<64KB)
7. ❌ Don't use for remote/distributed storage
8. ❌ Don't hold mappings open indefinitely

## Further Reading

- [Linux mmap(2) man page](https://man7.org/linux/man-pages/man2/mmap.2.html)
- [memmap2 crate documentation](https://docs.rs/memmap2/)
- [Zero-copy networking](https://en.wikipedia.org/wiki/Zero-copy)
- [Page cache](https://en.wikipedia.org/wiki/Page_cache)
