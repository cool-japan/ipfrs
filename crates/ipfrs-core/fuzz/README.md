# Fuzz Testing for IPFRS Core

This directory contains fuzz targets for testing the robustness of IPFRS core components.

## Installation

First, install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

To run a specific fuzz target:

```bash
# Fuzz block creation
cargo fuzz run fuzz_block_creation

# Fuzz CID parsing
cargo fuzz run fuzz_cid_parsing

# Fuzz IPLD parsing
cargo fuzz run fuzz_ipld_parsing

# Fuzz chunking
cargo fuzz run fuzz_chunking
```

## Fuzz Targets

### fuzz_block_creation
Tests the robustness of `Block::new()` against arbitrary data inputs. Verifies:
- Block creation doesn't panic
- CID determinism (same data → same CID)
- Data integrity (block data matches input)
- Size correctness

### fuzz_cid_parsing
Tests the robustness of CID parsing from strings. Verifies:
- Parser doesn't panic on invalid input
- Round-trip serialization (CID → string → CID)
- String format consistency

### fuzz_ipld_parsing
Tests the robustness of IPLD deserialization. Verifies:
- JSON/CBOR parsing doesn't panic
- Round-trip serialization maintains equivalence
- Handles malformed data gracefully

### fuzz_chunking
Tests data chunking operations. Verifies:
- Chunking and reassembly preserves data
- Different chunk sizes work correctly
- Edge cases (empty data, single chunk, etc.)

## Analyzing Results

Fuzz test artifacts (crashes, hangs) are stored in:
- `fuzz/artifacts/<target_name>/`

To reproduce a failure:
```bash
cargo fuzz run <target_name> <artifact_file>
```

## Coverage

To generate coverage reports:
```bash
cargo fuzz coverage <target_name>
```

## Continuous Fuzzing

For continuous fuzzing in CI/CD:
```bash
# Run for 60 seconds
cargo fuzz run <target_name> -- -max_total_time=60
```

## Notes

- Fuzzing requires a nightly Rust compiler
- Use `rustup default nightly` or `cargo +nightly fuzz run ...`
- Fuzzing can be resource-intensive; monitor CPU/memory usage
