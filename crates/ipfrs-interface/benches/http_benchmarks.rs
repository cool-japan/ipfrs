//! HTTP Endpoint Performance Benchmarks
//!
//! This benchmark suite measures the performance of various HTTP endpoints
//! to ensure they meet the target performance characteristics:
//! - Request latency: < 10ms (simple GET)
//! - Throughput: > 1GB/s (range requests)
//! - Concurrent connections: 10,000+
//! - Memory per connection: < 100KB

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

// Mock types for benchmarking (in real benchmarks, would use actual gateway)
struct MockGateway;

impl MockGateway {
    fn new() -> Self {
        Self
    }

    async fn handle_get(&self, _cid: &str) -> Result<Bytes, String> {
        // Simulate block retrieval
        Ok(Bytes::from(vec![0u8; 1024 * 64])) // 64KB
    }

    async fn handle_range_get(&self, _cid: &str, _range: (usize, usize)) -> Result<Bytes, String> {
        // Simulate range request
        let (start, end) = _range;
        Ok(Bytes::from(vec![0u8; end - start]))
    }

    async fn handle_batch_get(&self, cids: &[String]) -> Result<Vec<Bytes>, String> {
        // Simulate batch retrieval
        Ok(cids.iter().map(|_| Bytes::from(vec![0u8; 1024])).collect())
    }

    async fn handle_upload(&self, data: Bytes) -> Result<String, String> {
        // Simulate upload and CID generation
        let cid = format!("Qm{:x}", data.len());
        Ok(cid)
    }
}

/// Benchmark simple GET requests (target: <10ms)
fn bench_simple_get(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let gateway = Arc::new(MockGateway::new());

    c.bench_function("simple_get", |b| {
        b.to_async(&rt).iter(|| async {
            let cid = black_box("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco");
            gateway.handle_get(cid).await.unwrap()
        });
    });
}

/// Benchmark range requests with various sizes
fn bench_range_requests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let gateway = Arc::new(MockGateway::new());

    let mut group = c.benchmark_group("range_requests");

    for size in [1024, 64 * 1024, 1024 * 1024, 10 * 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let cid = black_box("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco");
                gateway.handle_range_get(cid, (0, size)).await.unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark batch operations with varying batch sizes
fn bench_batch_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let gateway = Arc::new(MockGateway::new());

    let mut group = c.benchmark_group("batch_operations");

    for batch_size in [1, 10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let cids: Vec<String> = (0..batch_size)
                    .map(|i| format!("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP{:04}", i))
                    .collect();

                b.to_async(&rt)
                    .iter(|| async { gateway.handle_batch_get(black_box(&cids)).await.unwrap() });
            },
        );
    }
    group.finish();
}

/// Benchmark upload operations with various file sizes
fn bench_uploads(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let gateway = Arc::new(MockGateway::new());

    let mut group = c.benchmark_group("uploads");

    for size in [1024, 64 * 1024, 1024 * 1024, 10 * 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let data = Bytes::from(vec![0u8; size]);

            b.to_async(&rt).iter(|| async {
                gateway
                    .handle_upload(black_box(data.clone()))
                    .await
                    .unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark concurrent requests
fn bench_concurrent_requests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let gateway = Arc::new(MockGateway::new());

    let mut group = c.benchmark_group("concurrent_requests");

    for concurrency in [1, 10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut tasks = Vec::new();
                    for _ in 0..concurrency {
                        let gw = gateway.clone();
                        tasks.push(tokio::spawn(async move {
                            gw.handle_get("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco")
                                .await
                        }));
                    }

                    for task in tasks {
                        task.await.unwrap().unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark ETag validation (304 responses)
fn bench_etag_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("etag_validation", |b| {
        b.to_async(&rt).iter(|| async {
            // Simulate ETag comparison (should be very fast)
            let cid = black_box("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco");
            let etag = format!("\"{}\"", cid);
            let if_none_match = black_box(&etag);

            // ETag match check
            &etag == if_none_match
        });
    });
}

/// Benchmark compression overhead
fn bench_compression(c: &mut Criterion) {
    use oxiarc_deflate::gzip_compress;

    let mut group = c.benchmark_group("compression");
    let data = vec![0u8; 1024 * 1024]; // 1MB

    for level in [1u8, 3, 6, 9].iter() {
        group.bench_with_input(BenchmarkId::new("gzip", level), level, |b, &level| {
            b.iter(|| gzip_compress(black_box(&data), level).expect("gzip compression failed"));
        });
    }

    group.finish();
}

/// Benchmark rate limiting overhead
fn bench_rate_limiting(c: &mut Criterion) {
    use std::time::Instant;

    c.bench_function("rate_limit_check", |b| {
        let mut last_refill = Instant::now();
        let mut tokens = 100.0f64;
        let rate = 100.0; // tokens per second
        let capacity = 100.0;

        b.iter(|| {
            // Simulate token bucket rate limiting
            let now = Instant::now();
            let elapsed = now.duration_since(last_refill).as_secs_f64();
            tokens = (tokens + elapsed * rate).min(capacity);
            last_refill = now;

            if tokens >= 1.0 {
                tokens -= 1.0;
                true
            } else {
                false
            }
        });
    });
}

/// Benchmark CID validation
fn bench_cid_validation(c: &mut Criterion) {
    c.bench_function("cid_validation", |b| {
        let cid = black_box("QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco");

        b.iter(|| {
            // Simulate CID validation
            cid.starts_with("Qm") && cid.len() == 46
        });
    });
}

/// Benchmark memory-mapped file access
fn bench_mmap_access(c: &mut Criterion) {
    use ipfrs_interface::mmap::MmapFile;
    use std::io::Write;

    // Create a temporary test file
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let test_data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect(); // 1MB
    temp_file.write_all(&test_data).unwrap();
    temp_file.flush().unwrap();

    let mut group = c.benchmark_group("mmap_access");
    group.throughput(Throughput::Bytes(test_data.len() as u64));

    // Benchmark full file access via mmap
    group.bench_function("full_file", |b| {
        let mmap = MmapFile::new(temp_file.path()).unwrap();
        b.iter(|| {
            let bytes = black_box(mmap.bytes());
            bytes.len()
        });
    });

    // Benchmark range access via mmap
    group.bench_function("range_access", |b| {
        let mmap = MmapFile::new(temp_file.path()).unwrap();
        b.iter(|| {
            let range = black_box(0..1024);
            let bytes = mmap.range(range).unwrap();
            bytes.len()
        });
    });

    // Benchmark multi-range access
    group.bench_function("multi_range", |b| {
        let mmap = MmapFile::new(temp_file.path()).unwrap();
        let ranges = vec![0..1024, 10240..11264, 102400..103424];
        b.iter(|| {
            let results = mmap.multi_range(black_box(&ranges)).unwrap();
            results.iter().map(|b| b.len()).sum::<usize>()
        });
    });

    group.finish();
}

/// Benchmark mmap cache performance
fn bench_mmap_cache(c: &mut Criterion) {
    use ipfrs_interface::mmap::MmapCache;
    use std::io::Write;

    // Create test files
    let mut temp_files = Vec::new();
    for i in 0..10 {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..1024).map(|j| ((i + j) % 256) as u8).collect();
        file.write_all(&data).unwrap();
        file.flush().unwrap();
        temp_files.push(file);
    }

    let mut group = c.benchmark_group("mmap_cache");

    // Benchmark cache hit (file already mapped)
    group.bench_function("cache_hit", |b| {
        let cache = MmapCache::new(100);
        // Pre-populate cache
        for file in &temp_files {
            cache.get_or_create(file.path()).unwrap();
        }

        b.iter(|| {
            let file = black_box(&temp_files[0]);
            cache.get_or_create(file.path()).unwrap();
        });
    });

    // Benchmark cache miss (new file mapping)
    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            let cache = MmapCache::new(100);
            let file = black_box(&temp_files[0]);
            cache.get_or_create(file.path()).unwrap();
        });
    });

    group.finish();
}

/// Benchmark mmap vs regular file I/O
fn bench_mmap_vs_read(c: &mut Criterion) {
    use ipfrs_interface::mmap::MmapFile;
    use std::io::{Read, Write};

    // Create a test file (1MB)
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    let test_data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
    temp_file.write_all(&test_data).unwrap();
    temp_file.flush().unwrap();

    let mut group = c.benchmark_group("mmap_vs_read");
    group.throughput(Throughput::Bytes(test_data.len() as u64));

    // Benchmark mmap access
    group.bench_function("mmap", |b| {
        let mmap = MmapFile::new(temp_file.path()).unwrap();
        b.iter(|| {
            let bytes = black_box(mmap.bytes());
            bytes.len()
        });
    });

    // Benchmark traditional file read
    group.bench_function("file_read", |b| {
        b.iter(|| {
            let mut file = std::fs::File::open(temp_file.path()).unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();
            black_box(buffer.len())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_get,
    bench_range_requests,
    bench_batch_operations,
    bench_uploads,
    bench_concurrent_requests,
    bench_etag_validation,
    bench_compression,
    bench_rate_limiting,
    bench_cid_validation,
    bench_mmap_access,
    bench_mmap_cache,
    bench_mmap_vs_read,
);

criterion_main!(benches);
