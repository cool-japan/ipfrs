//! Common types used across IPFRS

use serde::{Deserialize, Serialize};

/// Peer identifier in the IPFRS network
pub type PeerId = String;

/// Block size in bytes
pub type BlockSize = u64;

/// Block index in a larger data structure
pub type BlockIndex = u64;

/// Priority level for block requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Low priority - background or batch operations
    Low = 0,
    /// Normal priority - default for most operations
    #[default]
    Normal = 1,
    /// High priority - user-facing operations
    High = 2,
    /// Critical priority - system-critical operations
    Critical = 3,
}

impl Priority {
    /// Get the numeric value of the priority level
    ///
    /// # Example
    ///
    /// ```
    /// use ipfrs_core::types::Priority;
    ///
    /// assert_eq!(Priority::Low.as_u8(), 0);
    /// assert_eq!(Priority::Critical.as_u8(), 3);
    /// ```
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Create a priority from a numeric value
    ///
    /// Values >= 3 are mapped to Critical, values outside the range default to Normal.
    ///
    /// # Example
    ///
    /// ```
    /// use ipfrs_core::types::Priority;
    ///
    /// assert_eq!(Priority::from_u8(0), Priority::Low);
    /// assert_eq!(Priority::from_u8(3), Priority::Critical);
    /// assert_eq!(Priority::from_u8(10), Priority::Critical);
    /// ```
    #[inline]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Priority::Low,
            1 => Priority::Normal,
            2 => Priority::High,
            _ => Priority::Critical, // 3 and above
        }
    }

    /// Check if this is a critical priority
    ///
    /// # Example
    ///
    /// ```
    /// use ipfrs_core::types::Priority;
    ///
    /// assert!(Priority::Critical.is_critical());
    /// assert!(!Priority::Normal.is_critical());
    /// ```
    #[inline]
    pub const fn is_critical(self) -> bool {
        matches!(self, Priority::Critical)
    }

    /// Check if this priority is high or critical
    ///
    /// # Example
    ///
    /// ```
    /// use ipfrs_core::types::Priority;
    ///
    /// assert!(Priority::High.is_high_or_above());
    /// assert!(Priority::Critical.is_high_or_above());
    /// assert!(!Priority::Normal.is_high_or_above());
    /// ```
    #[inline]
    pub const fn is_high_or_above(self) -> bool {
        matches!(self, Priority::High | Priority::Critical)
    }

    /// Get the minimum of two priorities
    ///
    /// # Example
    ///
    /// ```
    /// use ipfrs_core::types::Priority;
    ///
    /// assert_eq!(Priority::High.min(Priority::Low), Priority::Low);
    /// ```
    #[inline]
    pub const fn min(self, other: Priority) -> Priority {
        if (self as u8) < (other as u8) {
            self
        } else {
            other
        }
    }

    /// Get the maximum of two priorities
    ///
    /// # Example
    ///
    /// ```
    /// use ipfrs_core::types::Priority;
    ///
    /// assert_eq!(Priority::High.max(Priority::Low), Priority::High);
    /// ```
    #[inline]
    pub const fn max(self, other: Priority) -> Priority {
        if (self as u8) > (other as u8) {
            self
        } else {
            other
        }
    }
}

impl From<u8> for Priority {
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

impl From<Priority> for u8 {
    fn from(priority: Priority) -> u8 {
        priority.as_u8()
    }
}
