//! Protocol version and feature negotiation between peers.
//!
//! When two IPFRS peers connect, they exchange [`ProtocolOffer`] messages to
//! determine the best mutually supported configuration:
//! - The highest protocol version both sides understand
//! - The intersection of supported feature flags
//! - A safe chunk size (minimum of both peers' preferences)
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::protocol_negotiator::{
//!     NegotiatorConfig, ProtocolFeature, ProtocolNegotiator, NegotiationResult,
//! };
//!
//! let config = NegotiatorConfig::default();
//! let negotiator = ProtocolNegotiator::new(config);
//!
//! let local = negotiator.config.local_offer(
//!     "local-peer".to_string(),
//!     vec![ProtocolFeature::Encryption, ProtocolFeature::Compression],
//! );
//! let remote = negotiator.config.local_offer(
//!     "remote-peer".to_string(),
//!     vec![ProtocolFeature::Encryption, ProtocolFeature::Multiplexing],
//! );
//!
//! match negotiator.negotiate(&local, &remote) {
//!     NegotiationResult::Agreed { version, features, chunk_size } => {
//!         assert_eq!(version, 3);
//!         assert_eq!(features, vec![ProtocolFeature::Encryption]);
//!         assert_eq!(chunk_size, 65536);
//!     }
//!     other => panic!("unexpected result: {:?}", other),
//! }
//! ```

pub mod constants;
pub mod functions;
pub mod negotiatorconfig_traits;
pub mod peerprotocolnegotiator_traits;
pub mod peerprotocolversion_traits;
pub mod pnnegotiatorconfig_traits;
pub mod pnprotocolnegotiator_traits;
pub mod pnprotocolversion_traits;
pub mod types;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod tests;
