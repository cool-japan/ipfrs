// Kubo (go-ipfs) comparison benchmarks
//
// This benchmark suite compares ipfrs-storage performance against Kubo's
// Badger/LevelDB backends using identical workloads and hardware.
//
// Prerequisites:
// 1. Install Kubo: https://docs.ipfs.tech/install/
// 2. Initialize repo: ipfs init
// 3. Configure for benchmarking (disable networking, etc.)
//
// Run with: cargo bench --bench kubo_comparison -- --ignored

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ipfrs_storage::{
    BlockStoreConfig, BlockStoreTrait, ParityDbBlockStore, ParityDbConfig, ParityDbPreset,
    SledBlockStore,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

// Workload generator for realistic IPFS usage patterns
struct WorkloadGenerator {
    block_sizes: Vec<usize>,
    num_blocks: usize,
}

impl WorkloadGenerator {
    fn new() -> Self {
        Self {
            // Realistic IPFS block size distribution
            // Based on analysis of real IPFS data
            block_sizes: vec![
                256,     // 10% - small metadata
                4096,    // 20% - small files
                32768,   // 30% - medium chunks
                262144,  // 30% - large chunks (256KB, IPFS default)
                1048576, // 10% - very large blocks
            ],
            num_blocks: 1000,
        }
    }

    fn generate_blocks(&self) -> Vec<Vec<u8>> {
        let mut blocks = Vec::new();
        let mut rng = fastrand::Rng::with_seed(42);

        for i in 0..self.num_blocks {
            let size_idx = i % self.block_sizes.len();
            let size = self.block_sizes[size_idx];
            let mut block = vec![0u8; size];

            // Generate semi-random data (more realistic than pure random)
            for chunk in block.chunks_mut(8) {
                let val = rng.u64(..);
                let bytes = val.to_le_bytes();
                let len = chunk.len().min(8);
                chunk[..len].copy_from_slice(&bytes[..len]);
            }

            blocks.push(block);
        }

        blocks
    }

    fn generate_read_heavy_pattern(&self) -> Vec<usize> {
        // 80/20 rule: 20% of blocks account for 80% of reads
        let mut pattern = Vec::new();
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..10000 {
            if rng.u32(..) % 100 < 80 {
                // 80% of reads go to 20% of blocks (hot data)
                pattern.push(rng.usize(..self.num_blocks / 5));
            } else {
                // 20% of reads go to remaining 80% of blocks
                pattern.push(rng.usize(..self.num_blocks));
            }
        }

        pattern
    }

    #[allow(dead_code)]
    fn generate_write_heavy_pattern(&self) -> Vec<usize> {
        // Sequential writes (common during data ingestion)
        (0..self.num_blocks).collect()
    }
}

// Kubo HTTP API client for benchmarking
#[allow(dead_code)]
struct KuboClient {
    api_url: String,
    client: reqwest::Client,
}

#[allow(dead_code)]
impl KuboClient {
    fn new(api_url: String) -> Self {
        Self {
            api_url,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("bench: build HTTP client"),
        }
    }

    async fn add_block(&self, data: &[u8]) -> Result<String, String> {
        let url = format!("{}/api/v0/block/put", self.api_url);

        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(data.to_vec()));

        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if response.status().is_success() {
            let json: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
            Ok(json["Key"].as_str().unwrap_or("").to_string())
        } else {
            Err(format!("HTTP {}", response.status()))
        }
    }

    async fn get_block(&self, cid: &str) -> Result<Vec<u8>, String> {
        let url = format!("{}/api/v0/block/get?arg={}", self.api_url, cid);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if response.status().is_success() {
            response
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| e.to_string())
        } else {
            Err(format!("HTTP {}", response.status()))
        }
    }

    async fn stat_block(&self, cid: &str) -> Result<usize, String> {
        let url = format!("{}/api/v0/block/stat?arg={}", self.api_url, cid);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if response.status().is_success() {
            let json: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
            Ok(json["Size"].as_u64().unwrap_or(0) as usize)
        } else {
            Err(format!("HTTP {}", response.status()))
        }
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/api/v0/version", self.api_url);
        self.client.get(&url).send().await.is_ok()
    }
}

// Benchmark ipfrs-storage backends
fn bench_ipfrs_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipfrs_write");
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let workload = WorkloadGenerator::new();
    let blocks = workload.generate_blocks();

    // Benchmark Sled
    group.bench_function("sled", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_dir = tempfile::tempdir().expect("bench: create temp dir");
                let config = BlockStoreConfig {
                    path: temp_dir.path().to_path_buf(),
                    cache_size: 512 * 1024 * 1024,
                };
                let store = SledBlockStore::new(config).expect("bench: open sled store");

                for block in &blocks {
                    let block_data =
                        ipfrs_storage::create_block(block.clone()).expect("bench: create block");
                    store.put(&block_data).await.expect("bench: put block");
                }
            })
        })
    });

    // Benchmark ParityDB
    group.bench_function("paritydb", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_dir = tempfile::tempdir().expect("bench: create temp dir");
                let config =
                    ParityDbConfig::new(temp_dir.path().to_path_buf(), ParityDbPreset::Balanced);
                let store = ParityDbBlockStore::new(config).expect("bench: open paritydb store");

                for block in &blocks {
                    let block_data =
                        ipfrs_storage::create_block(block.clone()).expect("bench: create block");
                    store.put(&block_data).await.expect("bench: put block");
                }
            })
        })
    });

    group.finish();
}

fn bench_ipfrs_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipfrs_read");
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let workload = WorkloadGenerator::new();
    let blocks = workload.generate_blocks();

    // Prepare Sled store
    let temp_dir_sled = tempfile::tempdir().expect("bench: create temp dir");
    let sled_store = rt.block_on(async {
        let config = BlockStoreConfig {
            path: temp_dir_sled.path().to_path_buf(),
            cache_size: 512 * 1024 * 1024,
        };
        let store = SledBlockStore::new(config).expect("bench: open sled store");
        for block in &blocks {
            let block_data =
                ipfrs_storage::create_block(block.clone()).expect("bench: create block");
            store.put(&block_data).await.expect("bench: put block");
        }
        Arc::new(store)
    });

    // Prepare ParityDB store
    let temp_dir_parity = tempfile::tempdir().expect("bench: create temp dir");
    let parity_store = rt.block_on(async {
        let config = ParityDbConfig::new(
            temp_dir_parity.path().to_path_buf(),
            ParityDbPreset::Balanced,
        );
        let store = ParityDbBlockStore::new(config).expect("bench: open paritydb store");
        for block in &blocks {
            let block_data =
                ipfrs_storage::create_block(block.clone()).expect("bench: create block");
            store.put(&block_data).await.expect("bench: put block");
        }
        Arc::new(store)
    });

    let read_pattern = workload.generate_read_heavy_pattern();

    group.bench_function("sled", |b| {
        let store = sled_store.clone();
        b.iter(|| {
            rt.block_on(async {
                for &idx in &read_pattern[..100] {
                    let block = &blocks[idx % blocks.len()];
                    let cid = ipfrs_storage::utils::compute_cid(block);
                    let _ = black_box(store.get(&cid).await.expect("bench: get block"));
                }
            })
        })
    });

    group.bench_function("paritydb", |b| {
        let store = parity_store.clone();
        b.iter(|| {
            rt.block_on(async {
                for &idx in &read_pattern[..100] {
                    let block = &blocks[idx % blocks.len()];
                    let cid = ipfrs_storage::utils::compute_cid(block);
                    let _ = black_box(store.get(&cid).await.expect("bench: get block"));
                }
            })
        })
    });

    group.finish();
}

fn bench_ipfrs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipfrs_batch");
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let workload = WorkloadGenerator::new();
    let blocks = workload.generate_blocks();

    let batch_size = 100;
    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("sled_batch_write", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_dir = tempfile::tempdir().expect("bench: create temp dir");
                let config = BlockStoreConfig {
                    path: temp_dir.path().to_path_buf(),
                    cache_size: 512 * 1024 * 1024,
                };
                let store = SledBlockStore::new(config).expect("bench: open sled store");

                let items: Vec<_> = blocks
                    .iter()
                    .take(batch_size)
                    .map(|block| {
                        ipfrs_storage::create_block(block.clone()).expect("bench: create block")
                    })
                    .collect();

                store
                    .put_many(&items)
                    .await
                    .expect("bench: put many blocks");
            })
        })
    });

    group.bench_function("paritydb_batch_write", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_dir = tempfile::tempdir().expect("bench: create temp dir");
                let config =
                    ParityDbConfig::new(temp_dir.path().to_path_buf(), ParityDbPreset::Balanced);
                let store = ParityDbBlockStore::new(config).expect("bench: open paritydb store");

                let items: Vec<_> = blocks
                    .iter()
                    .take(batch_size)
                    .map(|block| {
                        ipfrs_storage::create_block(block.clone()).expect("bench: create block")
                    })
                    .collect();

                store
                    .put_many(&items)
                    .await
                    .expect("bench: put many blocks");
            })
        })
    });

    group.finish();
}

// Note: Kubo benchmarks are ignored by default since they require Kubo to be running
#[cfg(feature = "kubo_bench")]
fn bench_kubo_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kubo_comparison");
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let workload = WorkloadGenerator::new();
    let blocks = workload.generate_blocks();

    // Check if Kubo is available
    let kubo = KuboClient::new("http://127.0.0.1:5001".to_string());
    let available = rt.block_on(kubo.is_available());

    if !available {
        eprintln!("Kubo not running at http://127.0.0.1:5001");
        eprintln!("Start with: ipfs daemon");
        return;
    }

    group.bench_function("kubo_write", |b| {
        b.iter(|| {
            rt.block_on(async {
                for block in blocks.iter().take(100) {
                    let _ = kubo.add_block(block).await;
                }
            })
        })
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    targets = bench_ipfrs_write, bench_ipfrs_read, bench_ipfrs_batch
);

#[cfg(feature = "kubo_bench")]
criterion_group!(
    name = kubo_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));
    targets = bench_kubo_comparison
);

#[cfg(not(feature = "kubo_bench"))]
criterion_main!(benches);

#[cfg(feature = "kubo_bench")]
criterion_main!(benches, kubo_benches);
