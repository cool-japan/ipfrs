//! Benchmarks for ipfrs-core
//!
//! Run with: cargo bench -p ipfrs-core
//! Results are saved to target/criterion/

use criterion::criterion_main;

#[path = "core_benchmarks_parts/block_cid_bench.rs"]
mod block_cid_bench;

#[path = "core_benchmarks_parts/memory_hash_bench.rs"]
mod memory_hash_bench;

#[path = "core_benchmarks_parts/batch_codec_bench.rs"]
mod batch_codec_bench;

#[path = "core_benchmarks_parts/car_compression_dag_bench.rs"]
mod car_compression_dag_bench;

use batch_codec_bench::{batch_benches, batch_compression_benches, codec_benches};
use block_cid_bench::{
    block_benches, chunking_benches, cid_benches, ipld_benches, streaming_benches,
};
use car_compression_dag_bench::{
    car_benches, car_compression_benches, compression_benches, dag_benches,
};
use memory_hash_bench::{cdc_benches, hash_benches, memory_benches, pool_benches};

criterion_main!(
    cid_benches,
    block_benches,
    ipld_benches,
    chunking_benches,
    streaming_benches,
    memory_benches,
    cdc_benches,
    pool_benches,
    hash_benches,
    batch_benches,
    batch_compression_benches,
    codec_benches,
    compression_benches,
    car_benches,
    car_compression_benches,
    dag_benches,
);
