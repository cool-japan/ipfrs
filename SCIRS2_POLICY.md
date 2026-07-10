# SciRS2 Policy — IPFRS

**IPFRS does not depend on SciRS2-Core.** It uses the `rand` crate directly for its
randomness needs and does not use `ndarray`. This document records that decision, per
the COOLJAPAN convention that a project either adopts the SciRS2-Core policy or writes
down, explicitly, why it does not.

## Policy

IPFRS depends on `rand` (workspace-pinned `rand = "0.10"`, resolving to `0.10.2`)
directly in six of its twelve crates: `ipfrs-semantic`, `ipfrs-interface`,
`ipfrs-network`, `ipfrs-transport`, `ipfrs-storage`, `ipfrs-tensorlogic`. IPFRS does
**not** depend on `scirs2-core` or any other `scirs2-*` crate anywhere in the
workspace, and does not use `ndarray`. This is a deliberate exemption, not an oversight.

## Rationale

IPFRS is a distributed content-addressed storage / IPFS implementation, not a
numerical-computing library. Its `rand` usage is overwhelmingly for systems concerns —
connection/backoff jitter, Raft election-timeout randomization, non-cryptographic
ID/tag suffixes, metrics sampling, synthetic benchmark and load-test data, and test
fixtures — not linear algebra or scientific sampling. SciRS2-Core is a large numerics
stack (array types, BLAS/LAPACK-backed linear algebra, statistical distributions);
adopting it to replace this narrow, non-numerical usage would add substantial
dependency weight and compile time for no functional benefit.

The small amount of vector/matrix math IPFRS does need already uses narrowly-scoped,
purpose-built crates instead of a general numerics stack: `nalgebra` (`0.35`) for OPQ
rotation-matrix learning in `ipfrs-semantic`, and `hnsw_rs` for approximate
nearest-neighbor vector indexing. Neither warrants pulling in SciRS2-Core either.

## Security note on randomness (read carefully)

Most call sites above are non-cryptographic systems jitter, as described — but that is
**not universally true**, and the exceptions matter:

- **Differential-privacy noise** (`ipfrs-tensorlogic/src/differential_privacy.rs`,
  `gradient_noise.rs`) is the rigorous case: noise comes from an explicit, struct-held
  `rand::rngs::StdRng` (a ChaCha-backed CSPRNG), constructed via
  `rand::make_rng::<StdRng>()` in production so it is seeded from OS entropy; only
  explicitly-named test constructors install a fixed seed. Production noise is
  therefore never reproducible.
- **Other DP-flavored noise does not use that hardened pattern.**
  `ipfrs-semantic/src/privacy.rs` (`add_noise`),
  `ipfrs-semantic/src/federated.rs` (`apply_privacy_noise`), and
  `ipfrs-tensorlogic/src/gradient/federated.rs` (`add_gaussian_noise`,
  `add_laplacian_noise`) inject privacy noise using plain `rand::rng()` instead. This
  is an inconsistency worth closing, not a considered exemption.
- **Symmetric-encryption key/nonce generation** (`ipfrs-storage/src/encryption.rs`) and
  **API-key generation** (`ipfrs-interface/src/auth.rs`) likewise use plain
  `rand::rng()` / `rand::random()`, not a dedicated CSPRNG construction.

None of this is a known weak-PRNG bug: `rand`'s default source (`rand::rng()` /
`rand::random()`, i.e. `ThreadRng`) is itself documented upstream as a ChaCha12,
OS-seeded, auto-reseeding cryptographic-quality generator — the same generator family
as `StdRng`, just reached through a thread-local convenience API rather than an
explicit, storable, reseedable struct. So the claim "DP is the only place randomness
is security-sensitive" is **not accurate**; it is simply the only place that has made
the CSPRNG choice explicitly and defensively. Peer identity keys
(`ipfrs-network/src/identity.rs`, via `libp2p::identity::Keypair::generate_ed25519`),
session IDs (`ipfrs-tensorlogic/src/session_manager.rs`, via `getrandom::fill`), and
TLS (`rustls`) bypass the `rand` crate entirely.

## Scope / what would change this decision

If IPFRS grows genuine numerical-computing needs beyond narrow OPQ/DP hooks — e.g.
on-device model-training math or statistics-heavy features rather than utility-level
randomness — revisit adopting SciRS2-Core for those specific modules. Until then, the
dependency-weight cost is not justified by the actual usage pattern.

## Current `rand` footprint

| Crate | Representative files | What it's used for |
|---|---|---|
| `ipfrs-transport` | `src/metrics.rs` | Probabilistic + reservoir sampling of metric samples |
| `ipfrs-storage` | `src/raft.rs`, `src/encryption.rs`, `src/workload.rs`, `src/retry.rs` | Election-timeout jitter; encryption key/nonce generation; synthetic benchmark datasets; retry backoff jitter |
| `ipfrs-network` | `src/utils.rs`, `src/network_simulator.rs`, `benchmarking.rs`, `load_tester.rs` | Connection backoff jitter; simulated packet loss/latency/success rate for test harnesses |
| `ipfrs-semantic` | `src/hnsw.rs`, `router.rs`, `prod_tests.rs`, `privacy.rs`, `federated.rs`, `multimodal.rs` | Test-only random embeddings; DP noise on embeddings; Johnson–Lindenstrauss projection init |
| `ipfrs-interface` | `src/auth.rs`, `src/gateway/routes.rs` | API-key generation; multipart boundary strings |
| `ipfrs-tensorlogic` | `src/differential_privacy.rs`, `gradient_noise.rs`, `gradient/federated.rs`, `gradient/tensor.rs` | Hardened DP noise (`StdRng`); DP noise on gradients (plain `rand::rng()`); stochastic gradient sparsification |

`ipfrs-storage` and `ipfrs-network` also use `fastrand` — a separate, explicitly
non-cryptographic dependency — for retry jitter (`retry.rs`) and test-data
shuffling/generation (`storage_benchmark.rs`, `utils.rs`, `cache.rs`), a deliberate
choice of a faster generator where cryptographic unpredictability was never required.
`ipfrs-semantic/src/dimension_reducer.rs` uses a local, non-cryptographic `FnvPrng`
for OPQ Gaussian sampling and does not touch the `rand` crate at all.
