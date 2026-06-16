//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DropPolicy, QueuingDiscipline, ShaperError};

/// Configuration for `TrafficShaper`.
#[derive(Clone, Debug)]
pub struct ShaperConfig {
    /// The queuing discipline to use.
    pub discipline: QueuingDiscipline,
    /// Maximum total number of entries in the queue.
    pub max_queue_depth: usize,
    /// Rate limit in bits per second (0 = unlimited, used at discipline level).
    pub rate_limit_bps: u64,
    /// Allowed burst in bytes above the steady-state rate.
    pub burst_allowance_bytes: u64,
    /// What to do when the queue is full.
    pub drop_policy: DropPolicy,
}
impl ShaperConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ShaperError> {
        if self.max_queue_depth == 0 {
            return Err(ShaperError::InvalidConfig(
                "max_queue_depth must be > 0".to_string(),
            ));
        }
        if let DropPolicy::RED {
            min_threshold,
            max_threshold,
        } = &self.drop_policy
        {
            if min_threshold >= max_threshold {
                return Err(ShaperError::InvalidConfig(
                    "RED min_threshold must be < max_threshold".to_string(),
                ));
            }
            if *max_threshold > self.max_queue_depth {
                return Err(ShaperError::InvalidConfig(
                    "RED max_threshold must be <= max_queue_depth".to_string(),
                ));
            }
        }
        Ok(())
    }
}
