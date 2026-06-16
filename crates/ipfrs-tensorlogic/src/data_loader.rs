//! TensorDataLoader — batch data loading with shuffling and epoch tracking.
//!
//! Provides a deterministic, reproducible data loading pipeline for training
//! loops. Supports configurable batch sizes, Fisher-Yates shuffling with
//! FNV-1a-based PRNG, epoch tracking, and progress reporting.

/// Configuration for [`TensorDataLoader`].
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle indices at the start of each epoch.
    pub shuffle: bool,
    /// If `true`, the last batch is dropped when it has fewer samples than
    /// `batch_size`.
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
        }
    }
}

/// A single batch of data yielded by [`TensorDataLoader::next_batch`].
#[derive(Debug, Clone)]
pub struct DataBatch {
    /// Zero-based index of this batch within the current epoch.
    pub batch_index: usize,
    /// Sample feature vectors in this batch.
    pub samples: Vec<Vec<f64>>,
    /// Corresponding labels.
    pub labels: Vec<String>,
    /// The epoch during which this batch was yielded.
    pub epoch: usize,
}

/// Aggregate statistics about the data loader state.
#[derive(Debug, Clone)]
pub struct DataLoaderStats {
    pub total_samples: usize,
    pub batch_size: usize,
    pub current_epoch: usize,
    pub batches_yielded: u64,
    pub progress: f64,
}

/// Batch data loader with deterministic shuffling and epoch tracking.
///
/// Samples are stored as `(feature_vector, label)` pairs. Each epoch iterates
/// through all samples in (optionally shuffled) order, yielding fixed-size
/// batches.
pub struct TensorDataLoader {
    config: DataLoaderConfig,
    samples: Vec<(Vec<f64>, String)>,
    current_index: usize,
    current_epoch: usize,
    indices: Vec<usize>,
    batches_yielded: u64,
    seed: u64,
    /// Batch index within the current epoch (reset each epoch).
    batch_in_epoch: usize,
    /// Whether the initial shuffle for the current epoch has been performed.
    epoch_started: bool,
}

impl std::fmt::Debug for TensorDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorDataLoader")
            .field("config", &self.config)
            .field("total_samples", &self.samples.len())
            .field("current_epoch", &self.current_epoch)
            .field("current_index", &self.current_index)
            .field("batches_yielded", &self.batches_yielded)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// FNV-1a based PRNG helpers
// ---------------------------------------------------------------------------

/// FNV-1a hash of a u64 value (used as PRNG seed mixer).
fn fnv1a_hash(value: u64) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let bytes = value.to_le_bytes();
    let mut hash = FNV_OFFSET;
    for &b in &bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Simple deterministic PRNG: returns the next state and a value in `[0, bound)`.
fn prng_next(state: u64, bound: usize) -> (u64, usize) {
    let next = fnv1a_hash(state);
    let value = (next % bound as u64) as usize;
    (next, value)
}

impl TensorDataLoader {
    /// Create a new data loader with the given config and PRNG seed.
    pub fn new(config: DataLoaderConfig, seed: u64) -> Self {
        Self {
            config,
            samples: Vec::new(),
            current_index: 0,
            current_epoch: 0,
            indices: Vec::new(),
            batches_yielded: 0,
            seed,
            batch_in_epoch: 0,
            epoch_started: false,
        }
    }

    /// Add a single training sample.
    pub fn add_sample(&mut self, data: Vec<f64>, label: &str) {
        self.samples.push((data, label.to_string()));
        self.rebuild_indices();
    }

    /// Bulk-add training samples.
    pub fn add_samples(&mut self, samples: Vec<(Vec<f64>, String)>) {
        self.samples.extend(samples);
        self.rebuild_indices();
    }

    /// Return the next batch, or `None` if there are no samples.
    ///
    /// When the current epoch is exhausted the loader automatically increments
    /// the epoch counter, reshuffles (if enabled), and begins yielding batches
    /// from the new epoch.
    pub fn next_batch(&mut self) -> Option<DataBatch> {
        if self.samples.is_empty() {
            return None;
        }

        let n = self.samples.len();
        let remaining = n.saturating_sub(self.current_index);

        // If we have exhausted this epoch, advance.
        if remaining == 0 {
            self.advance_epoch();
        } else if self.config.drop_last && remaining < self.config.batch_size {
            // Drop incomplete final batch; advance to next epoch.
            self.advance_epoch();
        }

        // Ensure indices are initialised and shuffled for this epoch.
        if !self.epoch_started {
            if self.indices.is_empty() {
                self.rebuild_indices();
            }
            if self.config.shuffle {
                self.shuffle_indices();
            }
            self.epoch_started = true;
        }

        self.yield_batch()
    }

    /// Reset to the beginning of the current epoch without incrementing.
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.batch_in_epoch = 0;
        self.epoch_started = false;
    }

    /// Perform a Fisher-Yates shuffle on `self.indices` using an FNV-1a-based
    /// PRNG seeded from `seed + current_epoch`.
    pub fn shuffle_indices(&mut self) {
        let n = self.indices.len();
        if n <= 1 {
            return;
        }
        // Mix seed with epoch for per-epoch determinism.
        let mut state = fnv1a_hash(self.seed.wrapping_add(self.current_epoch as u64));
        for i in (1..n).rev() {
            let (next_state, j) = prng_next(state, i + 1);
            state = next_state;
            self.indices.swap(i, j);
        }
    }

    /// Total number of loaded samples.
    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }

    /// Total number of batches per epoch.
    ///
    /// Returns `ceil(samples / batch_size)` normally, or
    /// `floor(samples / batch_size)` when `drop_last` is enabled.
    pub fn total_batches(&self) -> usize {
        if self.samples.is_empty() || self.config.batch_size == 0 {
            return 0;
        }
        if self.config.drop_last {
            self.samples.len() / self.config.batch_size
        } else {
            self.samples.len().div_ceil(self.config.batch_size)
        }
    }

    /// Current epoch (zero-based).
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Fraction of the current epoch that has been consumed (`0.0..=1.0`).
    pub fn progress(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.current_index as f64 / self.samples.len() as f64
    }

    /// Snapshot of the loader's current statistics.
    pub fn stats(&self) -> DataLoaderStats {
        DataLoaderStats {
            total_samples: self.samples.len(),
            batch_size: self.config.batch_size,
            current_epoch: self.current_epoch,
            batches_yielded: self.batches_yielded,
            progress: self.progress(),
        }
    }

    // -- private helpers ----------------------------------------------------

    fn rebuild_indices(&mut self) {
        self.indices = (0..self.samples.len()).collect();
    }

    fn advance_epoch(&mut self) {
        self.current_epoch += 1;
        self.current_index = 0;
        self.batch_in_epoch = 0;
        self.epoch_started = false;
        self.rebuild_indices();
    }

    fn yield_batch(&mut self) -> Option<DataBatch> {
        let n = self.samples.len();
        if self.current_index >= n {
            return None;
        }

        let end = (self.current_index + self.config.batch_size).min(n);
        let batch_indices = &self.indices[self.current_index..end];

        let mut samples = Vec::with_capacity(batch_indices.len());
        let mut labels = Vec::with_capacity(batch_indices.len());
        for &idx in batch_indices {
            let (ref data, ref label) = self.samples[idx];
            samples.push(data.clone());
            labels.push(label.clone());
        }

        let batch = DataBatch {
            batch_index: self.batch_in_epoch,
            samples,
            labels,
            epoch: self.current_epoch,
        };

        self.current_index = end;
        self.batch_in_epoch += 1;
        self.batches_yielded += 1;

        Some(batch)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(n: usize) -> Vec<(Vec<f64>, String)> {
        (0..n)
            .map(|i| (vec![i as f64], format!("label_{i}")))
            .collect()
    }

    // -- basic batch size ---------------------------------------------------

    #[test]
    fn test_next_batch_correct_size() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 3,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(10));
        let batch = loader.next_batch().expect("should yield batch");
        assert_eq!(batch.samples.len(), 3);
        assert_eq!(batch.labels.len(), 3);
    }

    #[test]
    fn test_last_batch_smaller_when_not_drop_last() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 3,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(10));
        // consume 3 full batches
        for _ in 0..3 {
            let _ = loader.next_batch();
        }
        let last = loader.next_batch().expect("should yield partial batch");
        assert_eq!(last.samples.len(), 1); // 10 - 9 = 1
    }

    #[test]
    fn test_drop_last_skips_partial_batch() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 3,
                shuffle: false,
                drop_last: true,
            },
            42,
        );
        loader.add_samples(make_samples(10));
        // 10 / 3 = 3 full batches, last (1 sample) dropped
        let mut count = 0;
        let mut epoch0_batches = Vec::new();
        loop {
            let b = loader.next_batch().expect("should yield");
            if b.epoch > 0 {
                break;
            }
            epoch0_batches.push(b);
            count += 1;
            if count > 20 {
                panic!("infinite loop guard");
            }
        }
        assert_eq!(epoch0_batches.len(), 3);
        for b in &epoch0_batches {
            assert_eq!(b.samples.len(), 3);
        }
    }

    // -- epoch tracking -----------------------------------------------------

    #[test]
    fn test_epoch_increments_after_exhaustion() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 5,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(5));
        let b1 = loader.next_batch().expect("batch");
        assert_eq!(b1.epoch, 0);
        let b2 = loader.next_batch().expect("batch");
        assert_eq!(b2.epoch, 1);
    }

    #[test]
    fn test_epoch_increments_multiple_times() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 2,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(2));
        assert_eq!(loader.next_batch().expect("b").epoch, 0);
        assert_eq!(loader.next_batch().expect("b").epoch, 1);
        assert_eq!(loader.next_batch().expect("b").epoch, 2);
    }

    // -- shuffle ------------------------------------------------------------

    #[test]
    fn test_shuffle_changes_order() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 100,
                shuffle: true,
                drop_last: false,
            },
            12345,
        );
        loader.add_samples(make_samples(20));

        let b0 = loader.next_batch().expect("b");
        let order0: Vec<f64> = b0.samples.iter().map(|s| s[0]).collect();

        let b1 = loader.next_batch().expect("b");
        let order1: Vec<f64> = b1.samples.iter().map(|s| s[0]).collect();

        // With high probability the two epochs have different orderings.
        // (The probability of identical order for 20 elements is 1/20! ~ 0.)
        assert_ne!(
            order0, order1,
            "shuffled orders should differ across epochs"
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let make_loader = || {
            let mut l = TensorDataLoader::new(
                DataLoaderConfig {
                    batch_size: 100,
                    shuffle: true,
                    drop_last: false,
                },
                999,
            );
            l.add_samples(make_samples(15));
            l
        };

        let mut l1 = make_loader();
        let mut l2 = make_loader();

        for _ in 0..3 {
            let b1 = l1.next_batch().expect("b");
            let b2 = l2.next_batch().expect("b");
            assert_eq!(b1.samples, b2.samples);
            assert_eq!(b1.labels, b2.labels);
        }
    }

    #[test]
    fn test_different_seeds_different_order() {
        let make = |seed| {
            let mut l = TensorDataLoader::new(
                DataLoaderConfig {
                    batch_size: 100,
                    shuffle: true,
                    drop_last: false,
                },
                seed,
            );
            l.add_samples(make_samples(20));
            l.next_batch().expect("b").samples
        };
        let a = make(1);
        let b = make(2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_no_shuffle_preserves_order() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 100,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(10));
        let b = loader.next_batch().expect("b");
        let order: Vec<f64> = b.samples.iter().map(|s| s[0]).collect();
        let expected: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(order, expected);
    }

    // -- reset --------------------------------------------------------------

    #[test]
    fn test_reset_restarts_epoch() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 2,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(6));

        let b1 = loader.next_batch().expect("b");
        assert_eq!(b1.batch_index, 0);
        let _ = loader.next_batch(); // batch_index 1

        loader.reset();

        let b3 = loader.next_batch().expect("b after reset");
        assert_eq!(b3.batch_index, 0);
        assert_eq!(b3.epoch, 0); // still epoch 0
    }

    // -- progress -----------------------------------------------------------

    #[test]
    fn test_progress_tracking() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 5,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(10));

        assert!((loader.progress() - 0.0).abs() < f64::EPSILON);
        let _ = loader.next_batch();
        assert!((loader.progress() - 0.5).abs() < f64::EPSILON);
        let _ = loader.next_batch();
        assert!((loader.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_empty() {
        let loader = TensorDataLoader::new(DataLoaderConfig::default(), 0);
        assert!((loader.progress() - 0.0).abs() < f64::EPSILON);
    }

    // -- empty loader -------------------------------------------------------

    #[test]
    fn test_empty_loader_returns_none() {
        let mut loader = TensorDataLoader::new(DataLoaderConfig::default(), 0);
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_empty_loader_total_batches_zero() {
        let loader = TensorDataLoader::new(DataLoaderConfig::default(), 0);
        assert_eq!(loader.total_batches(), 0);
    }

    // -- single sample ------------------------------------------------------

    #[test]
    fn test_single_sample() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 1,
                shuffle: true,
                drop_last: false,
            },
            42,
        );
        loader.add_sample(vec![1.0, 2.0], "only");

        let b = loader.next_batch().expect("b");
        assert_eq!(b.samples.len(), 1);
        assert_eq!(b.labels[0], "only");
        assert_eq!(b.epoch, 0);

        let b2 = loader.next_batch().expect("b2");
        assert_eq!(b2.epoch, 1);
    }

    #[test]
    fn test_single_sample_drop_last_batch_size_2() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 2,
                shuffle: false,
                drop_last: true,
            },
            42,
        );
        loader.add_sample(vec![1.0], "a");
        // With drop_last and batch_size=2, 1 sample is always incomplete.
        assert_eq!(loader.total_batches(), 0);
    }

    // -- add_samples bulk ---------------------------------------------------

    #[test]
    fn test_add_samples_bulk() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 5,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(10));
        assert_eq!(loader.total_samples(), 10);
        assert_eq!(loader.total_batches(), 2);
    }

    #[test]
    fn test_add_sample_incremental() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 2,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_sample(vec![1.0], "a");
        loader.add_sample(vec![2.0], "b");
        loader.add_sample(vec![3.0], "c");
        assert_eq!(loader.total_samples(), 3);
        assert_eq!(loader.total_batches(), 2); // ceil(3/2)
    }

    // -- stats --------------------------------------------------------------

    #[test]
    fn test_stats_accuracy() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 4,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(10));

        let s0 = loader.stats();
        assert_eq!(s0.total_samples, 10);
        assert_eq!(s0.batch_size, 4);
        assert_eq!(s0.current_epoch, 0);
        assert_eq!(s0.batches_yielded, 0);
        assert!((s0.progress - 0.0).abs() < f64::EPSILON);

        let _ = loader.next_batch(); // 4 consumed
        let s1 = loader.stats();
        assert_eq!(s1.batches_yielded, 1);
        assert!((s1.progress - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_after_epoch_change() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 5,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(5));

        let _ = loader.next_batch(); // epoch 0
        let _ = loader.next_batch(); // epoch 1
        let s = loader.stats();
        assert_eq!(s.current_epoch, 1);
        assert_eq!(s.batches_yielded, 2);
    }

    // -- total_batches ------------------------------------------------------

    #[test]
    fn test_total_batches_ceil() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 3,
                shuffle: false,
                drop_last: false,
            },
            0,
        );
        loader.add_samples(make_samples(10));
        assert_eq!(loader.total_batches(), 4); // ceil(10/3)
    }

    #[test]
    fn test_total_batches_floor_drop_last() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 3,
                shuffle: false,
                drop_last: true,
            },
            0,
        );
        loader.add_samples(make_samples(10));
        assert_eq!(loader.total_batches(), 3); // floor(10/3)
    }

    #[test]
    fn test_total_batches_exact_division() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 5,
                shuffle: false,
                drop_last: false,
            },
            0,
        );
        loader.add_samples(make_samples(10));
        assert_eq!(loader.total_batches(), 2);
    }

    // -- batch_index within epoch -------------------------------------------

    #[test]
    fn test_batch_index_increments_within_epoch() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 2,
                shuffle: false,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(6));

        for expected in 0..3 {
            let b = loader.next_batch().expect("batch");
            assert_eq!(b.batch_index, expected);
        }
        // Next call triggers new epoch; batch_index resets.
        let b_new_epoch = loader.next_batch().expect("batch");
        assert_eq!(b_new_epoch.batch_index, 0);
        assert_eq!(b_new_epoch.epoch, 1);
    }

    // -- multiple epochs consistency ----------------------------------------

    #[test]
    fn test_multiple_epochs_all_samples_seen() {
        let n = 7;
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 3,
                shuffle: true,
                drop_last: false,
            },
            42,
        );
        loader.add_samples(make_samples(n));

        // Collect all batches for several epochs and verify each epoch covers
        // all samples exactly once.
        let num_epochs = 3;
        let batches_per_epoch = loader.total_batches(); // ceil(7/3) = 3
        for epoch in 0..num_epochs {
            let mut seen = std::collections::HashSet::new();
            for _ in 0..batches_per_epoch {
                let b = loader.next_batch().expect("should yield");
                assert_eq!(b.epoch, epoch, "unexpected epoch");
                for s in &b.samples {
                    seen.insert(s[0] as usize);
                }
            }
            assert_eq!(seen.len(), n, "epoch {epoch}: not all samples seen");
        }
    }

    // -- debug impl ---------------------------------------------------------

    #[test]
    fn test_debug_impl() {
        let loader = TensorDataLoader::new(DataLoaderConfig::default(), 0);
        let dbg = format!("{loader:?}");
        assert!(dbg.contains("TensorDataLoader"));
    }

    // -- default config -----------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = DataLoaderConfig::default();
        assert_eq!(cfg.batch_size, 32);
        assert!(cfg.shuffle);
        assert!(!cfg.drop_last);
    }

    // -- current_epoch accessor ---------------------------------------------

    #[test]
    fn test_current_epoch_accessor() {
        let mut loader = TensorDataLoader::new(
            DataLoaderConfig {
                batch_size: 1,
                shuffle: false,
                drop_last: false,
            },
            0,
        );
        loader.add_sample(vec![1.0], "a");
        assert_eq!(loader.current_epoch(), 0);
        let _ = loader.next_batch();
        assert_eq!(loader.current_epoch(), 0);
        let _ = loader.next_batch();
        assert_eq!(loader.current_epoch(), 1);
    }
}
