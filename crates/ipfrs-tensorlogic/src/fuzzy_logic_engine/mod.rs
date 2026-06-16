//! Full-featured Fuzzy Logic Engine with Mamdani inference and multiple
//! defuzzification strategies.
//!
//! Provides membership functions (Triangle, Trapezoid, Gaussian, Bell, Sigmoid,
//! Singleton, Linear), tree-structured antecedent expressions (And / Or / Not /
//! Very / Somewhat), rule-based Mamdani inference, and five defuzzification
//! methods (Centroid, Bisector, MeanOfMaxima, LargestOfMaxima, SmallestOfMaxima).

pub mod functions;
pub mod types;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod tests;
