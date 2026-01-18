# IPFRS Fuzzing Tests

This directory contains fuzzing tests for IPFRS core components using cargo-fuzz.

## Prerequisites

Install cargo-fuzz:
```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

### Fuzz Auth Token Verification

Test token verification with arbitrary inputs:
```bash
cargo fuzz run fuzz_auth_token
```

### Fuzz Auth Manager Operations

Test user creation and management with arbitrary inputs:
```bash
cargo fuzz run fuzz_auth_manager
```

## Options

### Run with timeout
```bash
cargo fuzz run fuzz_auth_token -- -max_total_time=60
```

### Run with specific number of runs
```bash
cargo fuzz run fuzz_auth_token -- -runs=1000000
```

### Run with corpus
```bash
cargo fuzz run fuzz_auth_token corpus/fuzz_auth_token
```

## Coverage

Generate coverage report:
```bash
cargo fuzz coverage fuzz_auth_token
```

## Continuous Fuzzing

For continuous fuzzing, run in the background:
```bash
cargo fuzz run fuzz_auth_token -- -max_total_time=3600 > fuzz.log 2>&1 &
```

## Found Issues

Any crashes or hangs discovered by fuzzing will be saved in:
- `fuzz/artifacts/fuzz_auth_token/`
- `fuzz/artifacts/fuzz_auth_manager/`

To reproduce an issue:
```bash
cargo fuzz run fuzz_auth_token fuzz/artifacts/fuzz_auth_token/crash-<hash>
```

## Targets

- **fuzz_auth_token**: Tests token verification, creation, and validation
- **fuzz_auth_manager**: Tests user management operations (create, update, delete, list)
- **fuzz_block_operations**: Tests block creation, CID generation, and data retrieval
- **fuzz_cid_parsing**: Tests CID parsing from arbitrary strings
- **fuzz_dag_cbor**: Tests IPLD DAG-CBOR serialization and deserialization

## Notes

- Fuzzing tests use libfuzzer-sys which requires a nightly Rust toolchain
- Tests should never panic - all inputs should be handled gracefully
- Run fuzzing for extended periods (hours/days) to find edge cases
