//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashSet;

use super::functions::{all_known_features, fnv1a_64, xorshift64};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn default_negotiator() -> ProtocolNegotiator {
        ProtocolNegotiator::new(NegotiatorConfig::default())
    }
    fn offer(
        peer_id: &str,
        min_v: u32,
        max_v: u32,
        features: Vec<ProtocolFeature>,
        chunk_size: u64,
    ) -> ProtocolOffer {
        ProtocolOffer {
            peer_id: peer_id.to_string(),
            min_version: min_v,
            max_version: max_v,
            supported_features: features,
            preferred_chunk_size: chunk_size,
        }
    }
    #[test]
    fn test_version_overlap_produces_agreed() {
        let neg = default_negotiator();
        let local = offer("local", 1, 3, vec![ProtocolFeature::Encryption], 65_536);
        let remote = offer("remote", 2, 5, vec![ProtocolFeature::Encryption], 65_536);
        let result = neg.negotiate(&local, &remote);
        assert!(
            matches!(result, NegotiationResult::Agreed { version: 3, .. }),
            "expected Agreed(v3), got {:?}",
            result
        );
    }
    #[test]
    fn test_no_version_overlap_produces_mismatch() {
        let neg = default_negotiator();
        let local = offer("local", 1, 2, vec![ProtocolFeature::Encryption], 65_536);
        let remote = offer("remote", 3, 5, vec![ProtocolFeature::Encryption], 65_536);
        let result = neg.negotiate(&local, &remote);
        assert_eq!(
            result,
            NegotiationResult::VersionMismatch {
                local_max: 2,
                remote_min: 3,
            }
        );
    }
    #[test]
    fn test_feature_intersection() {
        let neg = default_negotiator();
        let local = offer(
            "local",
            1,
            3,
            vec![
                ProtocolFeature::Encryption,
                ProtocolFeature::Compression,
                ProtocolFeature::Multiplexing,
            ],
            65_536,
        );
        let remote = offer(
            "remote",
            1,
            3,
            vec![ProtocolFeature::Encryption, ProtocolFeature::FlowControl],
            65_536,
        );
        match neg.negotiate(&local, &remote) {
            NegotiationResult::Agreed { features, .. } => {
                assert_eq!(features, vec![ProtocolFeature::Encryption])
            }
            other => panic!("expected Agreed, got {:?}", other),
        }
    }
    #[test]
    fn test_required_feature_missing_produces_rejected() {
        let config = NegotiatorConfig {
            required_features: vec![ProtocolFeature::Encryption],
            ..Default::default()
        };
        let neg = ProtocolNegotiator::new(config);
        let local = offer("local", 1, 3, vec![ProtocolFeature::Compression], 65_536);
        let remote = offer("remote", 1, 3, vec![ProtocolFeature::Compression], 65_536);
        let result = neg.negotiate(&local, &remote);
        assert!(
            matches!(result, NegotiationResult::Rejected { .. }),
            "expected Rejected, got {:?}",
            result
        );
    }
    #[test]
    fn test_no_features_in_common() {
        let neg = default_negotiator();
        let local = offer("local", 1, 3, vec![ProtocolFeature::Compression], 65_536);
        let remote = offer("remote", 1, 3, vec![ProtocolFeature::Encryption], 65_536);
        let result = neg.negotiate(&local, &remote);
        assert_eq!(result, NegotiationResult::NoFeaturesInCommon);
    }
    #[test]
    fn test_chunk_size_takes_minimum() {
        let neg = default_negotiator();
        let local = offer("local", 1, 3, vec![ProtocolFeature::Encryption], 128_000);
        let remote = offer("remote", 1, 3, vec![ProtocolFeature::Encryption], 32_768);
        match neg.negotiate(&local, &remote) {
            NegotiationResult::Agreed { chunk_size, .. } => {
                assert_eq!(chunk_size, 32_768)
            }
            other => panic!("expected Agreed, got {:?}", other),
        }
    }
    #[test]
    fn test_chunk_size_local_smaller() {
        let neg = default_negotiator();
        let local = offer("local", 1, 3, vec![ProtocolFeature::Encryption], 8_192);
        let remote = offer("remote", 1, 3, vec![ProtocolFeature::Encryption], 65_536);
        match neg.negotiate(&local, &remote) {
            NegotiationResult::Agreed { chunk_size, .. } => assert_eq!(chunk_size, 8_192),
            other => panic!("expected Agreed, got {:?}", other),
        }
    }
    #[test]
    fn test_highest_version_selected() {
        let neg = default_negotiator();
        let local = offer("local", 1, 5, vec![ProtocolFeature::Encryption], 65_536);
        let remote = offer("remote", 1, 3, vec![ProtocolFeature::Encryption], 65_536);
        match neg.negotiate(&local, &remote) {
            NegotiationResult::Agreed { version, .. } => assert_eq!(version, 3),
            other => panic!("expected Agreed, got {:?}", other),
        }
    }
    #[test]
    fn test_can_negotiate_true() {
        let neg = default_negotiator();
        let remote = offer("remote", 2, 4, vec![ProtocolFeature::Compression], 65_536);
        assert!(neg.can_negotiate(&remote));
    }
    #[test]
    fn test_can_negotiate_false_version() {
        let neg = default_negotiator();
        let remote = offer("remote", 4, 6, vec![ProtocolFeature::Compression], 65_536);
        assert!(!neg.can_negotiate(&remote));
    }
    #[test]
    fn test_supported_versions_range() {
        let neg = default_negotiator();
        assert_eq!(neg.supported_versions(), vec![1, 2, 3]);
    }
    #[test]
    fn test_supported_versions_single() {
        let config = NegotiatorConfig {
            local_min_version: 2,
            local_max_version: 2,
            ..Default::default()
        };
        assert_eq!(
            ProtocolNegotiator::new(config).supported_versions(),
            vec![2]
        );
    }
    #[test]
    fn test_all_features_optional_when_required_empty() {
        let config = NegotiatorConfig {
            required_features: vec![],
            ..Default::default()
        };
        let neg = ProtocolNegotiator::new(config);
        let local = offer("l", 1, 3, vec![ProtocolFeature::ArrowIpc], 65_536);
        let remote = offer("r", 1, 3, vec![ProtocolFeature::ArrowIpc], 65_536);
        assert!(matches!(
            neg.negotiate(&local, &remote),
            NegotiationResult::Agreed { .. }
        ));
    }
    #[test]
    fn test_boundary_version_overlap() {
        let neg = default_negotiator();
        let local = offer("local", 1, 2, vec![ProtocolFeature::Encryption], 65_536);
        let remote = offer("remote", 2, 4, vec![ProtocolFeature::Encryption], 65_536);
        match neg.negotiate(&local, &remote) {
            NegotiationResult::Agreed { version, .. } => {
                assert_eq!(version, 2);
            }
            other => panic!("expected Agreed, got {:?}", other),
        }
    }
    #[test]
    fn test_multiple_required_features_all_present() {
        let config = NegotiatorConfig {
            required_features: vec![ProtocolFeature::Encryption, ProtocolFeature::Compression],
            ..Default::default()
        };
        let neg = ProtocolNegotiator::new(config);
        let local = offer(
            "l",
            1,
            3,
            vec![ProtocolFeature::Encryption, ProtocolFeature::Compression],
            65_536,
        );
        let remote = offer(
            "r",
            1,
            3,
            vec![
                ProtocolFeature::Encryption,
                ProtocolFeature::Compression,
                ProtocolFeature::Multiplexing,
            ],
            65_536,
        );
        assert!(matches!(
            neg.negotiate(&local, &remote),
            NegotiationResult::Agreed { .. }
        ));
    }
    #[test]
    fn test_second_required_feature_missing() {
        let config = NegotiatorConfig {
            required_features: vec![ProtocolFeature::Encryption, ProtocolFeature::FlowControl],
            ..Default::default()
        };
        let neg = ProtocolNegotiator::new(config);
        let local = offer(
            "l",
            1,
            3,
            vec![ProtocolFeature::Encryption, ProtocolFeature::FlowControl],
            65_536,
        );
        let remote = offer("r", 1, 3, vec![ProtocolFeature::Encryption], 65_536);
        assert!(matches!(
            neg.negotiate(&local, &remote),
            NegotiationResult::Rejected { .. }
        ));
    }
    #[test]
    fn test_local_offer_helper() {
        let config = NegotiatorConfig {
            local_min_version: 2,
            local_max_version: 4,
            local_chunk_size: 32_000,
            required_features: vec![],
        };
        let features = vec![ProtocolFeature::Multiplexing];
        let o = config.local_offer("my-peer".to_string(), features.clone());
        assert_eq!(o.peer_id, "my-peer");
        assert_eq!(o.min_version, 2);
        assert_eq!(o.max_version, 4);
        assert_eq!(o.preferred_chunk_size, 32_000);
        assert_eq!(o.supported_features, features);
    }
    #[test]
    fn test_all_six_features_in_intersection() {
        let neg = default_negotiator();
        let all = all_known_features().to_vec();
        let local = offer("l", 1, 3, all.clone(), 65_536);
        let remote = offer("r", 1, 3, all.clone(), 65_536);
        match neg.negotiate(&local, &remote) {
            NegotiationResult::Agreed { features, .. } => {
                let feat_set: HashSet<_> = features.iter().copied().collect();
                for f in all_known_features() {
                    assert!(feat_set.contains(&f));
                }
            }
            other => panic!("expected Agreed, got {:?}", other),
        }
    }
    #[test]
    fn test_remote_version_below_local_min() {
        let config = NegotiatorConfig {
            local_min_version: 3,
            local_max_version: 5,
            ..Default::default()
        };
        let neg = ProtocolNegotiator::new(config);
        let local = offer("l", 3, 5, vec![ProtocolFeature::Encryption], 65_536);
        let remote = offer("r", 1, 2, vec![ProtocolFeature::Encryption], 65_536);
        assert!(matches!(
            neg.negotiate(&local, &remote),
            NegotiationResult::VersionMismatch {
                local_max: 5,
                remote_min: 1
            }
        ));
    }
    fn ppv(name: &str, major: u32, minor: u32) -> PeerProtocolVersion {
        PeerProtocolVersion::new(name, major, minor)
    }
    #[test]
    fn test_peer_exact_match_accepted() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 2);
        let (res, ver) = neg.negotiate(&ppv("bitswap", 1, 2));
        assert_eq!(res, PeerNegotiationResult::Accepted);
        assert_eq!(ver, Some(ppv("bitswap", 1, 2)));
    }
    #[test]
    fn test_peer_same_major_lower_minor_downgraded() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 5);
        let (res, ver) = neg.negotiate(&ppv("bitswap", 1, 3));
        assert_eq!(res, PeerNegotiationResult::Downgraded);
        assert_eq!(ver, Some(ppv("bitswap", 1, 5)));
    }
    #[test]
    fn test_peer_same_major_higher_minor_downgraded() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 2);
        let (res, ver) = neg.negotiate(&ppv("bitswap", 1, 9));
        assert_eq!(res, PeerNegotiationResult::Downgraded);
        assert_eq!(ver, Some(ppv("bitswap", 1, 2)));
    }
    #[test]
    fn test_peer_different_major_rejected() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 2);
        let (res, ver) = neg.negotiate(&ppv("bitswap", 2, 0));
        assert_eq!(res, PeerNegotiationResult::Rejected);
        assert!(ver.is_none());
    }
    #[test]
    fn test_peer_unknown_protocol_rejected() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 0);
        let (res, ver) = neg.negotiate(&ppv("graphsync", 1, 0));
        assert_eq!(res, PeerNegotiationResult::Rejected);
        assert!(ver.is_none());
    }
    #[test]
    fn test_peer_below_minimum_rejected() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 5);
        neg.set_minimum("bitswap", 1, 3);
        let (res, ver) = neg.negotiate(&ppv("bitswap", 1, 2));
        assert_eq!(res, PeerNegotiationResult::Rejected);
        assert!(ver.is_none());
    }
    #[test]
    fn test_peer_at_minimum_boundary_accepted() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 3);
        neg.set_minimum("bitswap", 1, 3);
        let (res, ver) = neg.negotiate(&ppv("bitswap", 1, 3));
        assert_eq!(res, PeerNegotiationResult::Accepted);
        assert_eq!(ver, Some(ppv("bitswap", 1, 3)));
    }
    #[test]
    fn test_peer_above_minimum_accepted() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("dht", 2, 1);
        neg.set_minimum("dht", 1, 0);
        let (res, _) = neg.negotiate(&ppv("dht", 2, 1));
        assert_eq!(res, PeerNegotiationResult::Accepted);
    }
    #[test]
    fn test_peer_minimum_major_below_rejected() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("dht", 2, 0);
        neg.set_minimum("dht", 2, 0);
        let (res, _) = neg.negotiate(&ppv("dht", 1, 9));
        assert_eq!(res, PeerNegotiationResult::Rejected);
    }
    #[test]
    fn test_peer_register_replaces_existing() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 0);
        neg.register_protocol("bitswap", 1, 5);
        assert_eq!(neg.get_version("bitswap"), Some(&ppv("bitswap", 1, 5)));
        assert_eq!(neg.supported.len(), 1);
    }
    #[test]
    fn test_peer_remove_existing_protocol() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 0);
        assert!(neg.remove_protocol("bitswap"));
        assert!(!neg.is_supported("bitswap"));
    }
    #[test]
    fn test_peer_remove_nonexistent_protocol() {
        assert!(!PeerProtocolNegotiator::new().remove_protocol("nope"));
    }
    #[test]
    fn test_peer_is_supported() {
        let mut neg = PeerProtocolNegotiator::new();
        assert!(!neg.is_supported("bitswap"));
        neg.register_protocol("bitswap", 1, 0);
        assert!(neg.is_supported("bitswap"));
    }
    #[test]
    fn test_peer_get_version_none_for_unknown() {
        assert!(PeerProtocolNegotiator::new()
            .get_version("bitswap")
            .is_none());
    }
    #[test]
    fn test_peer_supported_protocols_lists_all() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 0);
        neg.register_protocol("dht", 2, 1);
        neg.register_protocol("graphsync", 3, 0);
        assert_eq!(neg.supported_protocols().len(), 3);
    }
    #[test]
    fn test_peer_stats_tracking() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 2);
        let _ = neg.negotiate(&ppv("bitswap", 1, 2));
        let _ = neg.negotiate(&ppv("unknown", 1, 0));
        let _ = neg.negotiate(&ppv("bitswap", 1, 0));
        let s = neg.stats();
        assert_eq!(s.negotiations, 3);
        assert_eq!(s.accepted, 1);
        assert_eq!(s.rejected, 1);
        assert_eq!(s.downgraded, 1);
        assert_eq!(s.supported_count, 1);
    }
    #[test]
    fn test_peer_stats_after_remove() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("a", 1, 0);
        neg.register_protocol("b", 2, 0);
        neg.remove_protocol("a");
        assert_eq!(neg.stats().supported_count, 1);
    }
    #[test]
    fn test_peer_multiple_protocols_independent() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("bitswap", 1, 0);
        neg.register_protocol("dht", 2, 3);
        let (r1, _) = neg.negotiate(&ppv("bitswap", 1, 0));
        let (r2, v2) = neg.negotiate(&ppv("dht", 2, 1));
        assert_eq!(r1, PeerNegotiationResult::Accepted);
        assert_eq!(r2, PeerNegotiationResult::Downgraded);
        assert_eq!(v2, Some(ppv("dht", 2, 3)));
    }
    #[test]
    fn test_peer_version_zero_zero() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("exp", 0, 0);
        let (res, ver) = neg.negotiate(&ppv("exp", 0, 0));
        assert_eq!(res, PeerNegotiationResult::Accepted);
        assert_eq!(ver, Some(ppv("exp", 0, 0)));
    }
    #[test]
    fn test_peer_large_version_numbers() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("big", u32::MAX, u32::MAX);
        let (res, ver) = neg.negotiate(&ppv("big", u32::MAX, u32::MAX));
        assert_eq!(res, PeerNegotiationResult::Accepted);
        assert_eq!(ver, Some(ppv("big", u32::MAX, u32::MAX)));
    }
    #[test]
    fn test_peer_minimum_for_unregistered() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.set_minimum("ghost", 1, 0);
        let (res, _) = neg.negotiate(&ppv("ghost", 1, 0));
        assert_eq!(res, PeerNegotiationResult::Rejected);
    }
    #[test]
    fn test_peer_remove_clears_minimum() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("x", 1, 5);
        neg.set_minimum("x", 1, 3);
        neg.remove_protocol("x");
        neg.register_protocol("x", 1, 0);
        let (res, _) = neg.negotiate(&ppv("x", 1, 0));
        assert_eq!(res, PeerNegotiationResult::Accepted);
    }
    #[test]
    fn test_peer_default_trait() {
        let neg = PeerProtocolNegotiator::default();
        assert_eq!(neg.stats().supported_count, 0);
        assert_eq!(neg.stats().negotiations, 0);
    }
    #[test]
    fn test_peer_protocol_version_display() {
        assert_eq!(format!("{}", ppv("bitswap", 1, 2)), "bitswap/1.2");
    }
    #[test]
    fn test_peer_downgrade_returns_our_version() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("sync", 3, 7);
        let (res, ver) = neg.negotiate(&ppv("sync", 3, 2));
        assert_eq!(res, PeerNegotiationResult::Downgraded);
        assert_eq!(ver, Some(ppv("sync", 3, 7)));
    }
    #[test]
    fn test_peer_stats_zero_initially() {
        let neg = PeerProtocolNegotiator::new();
        let s = neg.stats();
        assert_eq!(s.negotiations, 0);
        assert_eq!(s.accepted, 0);
        assert_eq!(s.rejected, 0);
        assert_eq!(s.downgraded, 0);
    }
    #[test]
    fn test_peer_minimum_minor_boundary_downgraded() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("p", 1, 5);
        neg.set_minimum("p", 1, 3);
        let (res, ver) = neg.negotiate(&ppv("p", 1, 3));
        assert_eq!(res, PeerNegotiationResult::Downgraded);
        assert_eq!(ver, Some(ppv("p", 1, 5)));
    }
    #[test]
    fn test_peer_register_multiple_remove_one() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("a", 1, 0);
        neg.register_protocol("b", 2, 0);
        neg.register_protocol("c", 3, 0);
        neg.remove_protocol("b");
        assert!(!neg.is_supported("b"));
        let (ra, _) = neg.negotiate(&ppv("a", 1, 0));
        let (rb, _) = neg.negotiate(&ppv("b", 2, 0));
        let (rc, _) = neg.negotiate(&ppv("c", 3, 0));
        assert_eq!(ra, PeerNegotiationResult::Accepted);
        assert_eq!(rb, PeerNegotiationResult::Rejected);
        assert_eq!(rc, PeerNegotiationResult::Accepted);
    }
    #[test]
    fn test_peer_reregister_after_remove() {
        let mut neg = PeerProtocolNegotiator::new();
        neg.register_protocol("x", 1, 0);
        neg.remove_protocol("x");
        neg.register_protocol("x", 2, 0);
        assert_eq!(neg.get_version("x"), Some(&ppv("x", 2, 0)));
    }
    fn pn_id(b: u8) -> PnProtocolId {
        let mut arr = [0u8; 16];
        arr[0] = b;
        PnProtocolId(arr)
    }
    fn pn_peer(b: u8) -> [u8; 32] {
        let mut arr = [0u8; 32];
        arr[0] = b;
        arr
    }
    fn pn_ver(major: u32, minor: u32, patch: u32) -> PnProtocolVersion {
        PnProtocolVersion::new(major, minor, patch)
    }
    fn default_pn() -> PnProtocolNegotiator {
        PnProtocolNegotiator::new(PnNegotiatorConfig::default())
    }
    #[test]
    fn pn_register_and_retrieve() {
        let mut neg = default_pn();
        let id = pn_id(1);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        assert!(neg.versions_for(&id).is_some());
    }
    #[test]
    fn pn_deregister_removes() {
        let mut neg = default_pn();
        let id = pn_id(2);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        assert!(neg.deregister_protocol(&id));
        assert!(neg.versions_for(&id).is_none());
    }
    #[test]
    fn pn_deregister_nonexistent() {
        let mut neg = default_pn();
        assert!(!neg.deregister_protocol(&pn_id(99)));
    }
    #[test]
    fn pn_supported_protocols_lists_all() {
        let mut neg = default_pn();
        neg.register_protocol(pn_id(1), vec![pn_ver(1, 0, 0)]);
        neg.register_protocol(pn_id(2), vec![pn_ver(2, 0, 0)]);
        assert_eq!(neg.supported_protocols().len(), 2);
    }
    #[test]
    fn pn_initiate_handshake_success() {
        let mut neg = default_pn();
        let id = pn_id(1);
        neg.register_protocol(id, vec![pn_ver(1, 2, 0)]);
        let out = neg.initiate_handshake(pn_peer(1), id, vec![pn_ver(1, 0, 0), pn_ver(1, 2, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::Success { .. }));
    }
    #[test]
    fn pn_initiate_handshake_unknown_protocol() {
        let mut neg = default_pn();
        let out = neg.initiate_handshake(pn_peer(1), pn_id(42), vec![pn_ver(1, 0, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::ProtocolUnknown));
    }
    #[test]
    fn pn_initiate_handshake_version_mismatch() {
        let mut neg = default_pn();
        let id = pn_id(3);
        neg.register_protocol(id, vec![pn_ver(2, 0, 0)]);
        let out = neg.initiate_handshake(pn_peer(2), id, vec![pn_ver(1, 0, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::VersionMismatch { .. }));
    }
    #[test]
    fn pn_respond_to_handshake_success() {
        let mut neg = default_pn();
        let id = pn_id(4);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        let out = neg.respond_to_handshake(pn_peer(3), id, vec![pn_ver(1, 0, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::Success { .. }));
    }
    #[test]
    fn pn_respond_to_handshake_unknown_protocol() {
        let mut neg = default_pn();
        let out = neg.respond_to_handshake(pn_peer(4), pn_id(77), vec![pn_ver(1, 0, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::ProtocolUnknown));
    }
    #[test]
    fn pn_session_created_on_success() {
        let mut neg = default_pn();
        let id = pn_id(5);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(5), id, vec![pn_ver(1, 0, 0)]);
        assert_eq!(neg.session_count(), 1);
    }
    #[test]
    fn pn_no_session_on_mismatch() {
        let mut neg = default_pn();
        let id = pn_id(6);
        neg.register_protocol(id, vec![pn_ver(2, 0, 0)]);
        neg.initiate_handshake(pn_peer(6), id, vec![pn_ver(1, 0, 0)]);
        assert_eq!(neg.session_count(), 0);
    }
    #[test]
    fn pn_update_session_activity() {
        let mut neg = default_pn();
        let id = pn_id(7);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(7), id, vec![pn_ver(1, 0, 0)]);
        let sid = neg.active_session_ids()[0];
        assert!(neg.update_session_activity(sid, 1024));
        let s = neg.get_session(&sid).expect("session must exist");
        assert_eq!(s.bytes_exchanged, 1024);
    }
    #[test]
    fn pn_update_unknown_session_returns_false() {
        let mut neg = default_pn();
        assert!(!neg.update_session_activity(PnSessionId(0xdead), 100));
    }
    #[test]
    fn pn_expire_sessions_removes_expired() {
        let cfg = PnNegotiatorConfig {
            session_ttl_secs: 5,
            ..PnNegotiatorConfig::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(8);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(8), id, vec![pn_ver(1, 0, 0)]);
        assert_eq!(neg.session_count(), 1);
        for _ in 0..10 {
            neg.now_ms();
        }
        neg.expire_sessions();
        assert_eq!(neg.session_count(), 0);
    }
    #[test]
    fn pn_expire_sessions_keeps_fresh() {
        let cfg = PnNegotiatorConfig {
            session_ttl_secs: 1000,
            ..PnNegotiatorConfig::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(9);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(9), id, vec![pn_ver(1, 0, 0)]);
        neg.expire_sessions();
        assert_eq!(neg.session_count(), 1);
    }
    #[test]
    fn pn_sessions_for_peer() {
        let mut neg = default_pn();
        let id1 = pn_id(10);
        let id2 = pn_id(11);
        let peer_a = pn_peer(10);
        let peer_b = pn_peer(11);
        neg.register_protocol(id1, vec![pn_ver(1, 0, 0)]);
        neg.register_protocol(id2, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(peer_a, id1, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(peer_b, id2, vec![pn_ver(1, 0, 0)]);
        assert_eq!(neg.sessions_for_peer(peer_a).len(), 1);
        assert_eq!(neg.sessions_for_peer(peer_b).len(), 1);
    }
    #[test]
    fn pn_negotiation_stats() {
        let mut neg = default_pn();
        let id = pn_id(12);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(12), id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(13), pn_id(99), vec![pn_ver(1, 0, 0)]);
        let stats = neg.negotiation_stats();
        assert_eq!(stats.total, 2);
        assert!((stats.success_rate - 0.5).abs() < 1e-9);
    }
    #[test]
    fn pn_history_bounded() {
        let mut neg = default_pn();
        for i in 0u16..520 {
            let pid = {
                let mut a = [0u8; 16];
                a[0] = (i >> 8) as u8;
                a[1] = i as u8;
                PnProtocolId(a)
            };
            neg.initiate_handshake(pn_peer(0), pid, vec![pn_ver(1, 0, 0)]);
        }
        assert!(neg.history_len() <= 500);
    }
    #[test]
    fn pn_prefer_latest_selects_highest() {
        let cfg = PnNegotiatorConfig {
            prefer_latest: true,
            ..PnNegotiatorConfig::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(13);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0), pn_ver(1, 1, 0), pn_ver(1, 2, 0)]);
        let out = neg.initiate_handshake(pn_peer(14), id, vec![pn_ver(1, 0, 0), pn_ver(1, 2, 0)]);
        if let PnNegotiationOutcome::Success { version } = out {
            assert_eq!(version, pn_ver(1, 2, 0));
        } else {
            panic!("expected Success");
        }
    }
    #[test]
    fn pn_prefer_oldest_selects_lowest() {
        let cfg = PnNegotiatorConfig {
            prefer_latest: false,
            ..PnNegotiatorConfig::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(14);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0), pn_ver(1, 1, 0), pn_ver(1, 2, 0)]);
        let out = neg.initiate_handshake(pn_peer(15), id, vec![pn_ver(1, 0, 0), pn_ver(1, 2, 0)]);
        if let PnNegotiationOutcome::Success { version } = out {
            assert_eq!(version, pn_ver(1, 0, 0));
        } else {
            panic!("expected Success");
        }
    }
    #[test]
    fn pn_strict_compat_rejects_minor_mismatch() {
        let cfg = PnNegotiatorConfig {
            strict_compat: true,
            ..Default::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(15);
        neg.register_protocol(id, vec![pn_ver(1, 2, 0)]);
        let out = neg.initiate_handshake(pn_peer(16), id, vec![pn_ver(1, 0, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::VersionMismatch { .. }));
    }
    #[test]
    fn pn_strict_compat_accepts_exact() {
        let cfg = PnNegotiatorConfig {
            strict_compat: true,
            ..Default::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(16);
        neg.register_protocol(id, vec![pn_ver(1, 2, 0)]);
        let out = neg.initiate_handshake(pn_peer(17), id, vec![pn_ver(1, 2, 0)]);
        assert!(matches!(out, PnNegotiationOutcome::Success { .. }));
    }
    #[test]
    fn pn_max_sessions_limit() {
        let cfg = PnNegotiatorConfig {
            max_sessions: 2,
            ..Default::default()
        };
        let mut neg = PnProtocolNegotiator::new(cfg);
        let id = pn_id(17);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        for i in 0u8..5 {
            neg.initiate_handshake(pn_peer(i), id, vec![pn_ver(1, 0, 0)]);
        }
        assert!(neg.session_count() <= 2);
    }
    #[test]
    fn pn_version_compat() {
        assert!(pn_ver(1, 2, 0).is_compatible_with(&pn_ver(1, 1, 0)));
        assert!(!pn_ver(1, 1, 0).is_compatible_with(&pn_ver(1, 2, 0)));
        assert!(!pn_ver(1, 2, 0).is_compatible_with(&pn_ver(2, 0, 0)));
    }
    #[test]
    fn pn_history_records_failures() {
        let mut neg = default_pn();
        neg.initiate_handshake(pn_peer(20), pn_id(88), vec![pn_ver(1, 0, 0)]);
        assert_eq!(neg.history_len(), 1);
        let rec = neg.history().front().expect("must have record");
        assert!(matches!(rec.outcome, PnNegotiationOutcome::ProtocolUnknown));
    }
    #[test]
    fn pn_by_protocol_counter() {
        let mut neg = default_pn();
        let id = pn_id(18);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.initiate_handshake(pn_peer(18), id, vec![pn_ver(1, 0, 0)]);
        neg.respond_to_handshake(pn_peer(19), id, vec![pn_ver(1, 0, 0)]);
        let stats = neg.negotiation_stats();
        assert_eq!(*stats.by_protocol.get(&id).unwrap_or(&0), 2);
    }
    #[test]
    fn pn_config_defaults() {
        let cfg = PnNegotiatorConfig::default();
        assert_eq!(cfg.max_sessions, 1024);
        assert_eq!(cfg.session_ttl_secs, 300);
        assert!(cfg.prefer_latest);
        assert!(!cfg.strict_compat);
    }
    #[test]
    fn pn_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }
    #[test]
    fn pn_xorshift64_varies() {
        let mut s = 42u64;
        assert_ne!(xorshift64(&mut s), xorshift64(&mut s));
    }
    #[test]
    fn pn_version_display() {
        assert_eq!(format!("{}", pn_ver(2, 3, 1)), "2.3.1");
    }
    #[test]
    fn pn_empty_offered_list_mismatch() {
        let mut neg = default_pn();
        let id = pn_id(20);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        let out = neg.initiate_handshake(pn_peer(21), id, vec![]);
        assert!(matches!(out, PnNegotiationOutcome::VersionMismatch { .. }));
    }
    #[test]
    fn pn_respond_records_history() {
        let mut neg = default_pn();
        let id = pn_id(21);
        neg.register_protocol(id, vec![pn_ver(1, 0, 0)]);
        neg.respond_to_handshake(pn_peer(22), id, vec![pn_ver(1, 0, 0)]);
        assert_eq!(neg.history_len(), 1);
    }
    #[test]
    fn pn_multiple_protocols_independent() {
        let mut neg = default_pn();
        let id1 = pn_id(22);
        let id2 = pn_id(23);
        neg.register_protocol(id1, vec![pn_ver(1, 0, 0)]);
        neg.register_protocol(id2, vec![pn_ver(2, 0, 0)]);
        let o1 = neg.initiate_handshake(pn_peer(23), id1, vec![pn_ver(1, 0, 0)]);
        let o2 = neg.initiate_handshake(pn_peer(24), id2, vec![pn_ver(1, 0, 0)]);
        assert!(matches!(o1, PnNegotiationOutcome::Success { .. }));
        assert!(matches!(o2, PnNegotiationOutcome::VersionMismatch { .. }));
    }
    #[test]
    fn pn_stats_zero_initially() {
        let neg = default_pn();
        let s = neg.negotiation_stats();
        assert_eq!(s.total, 0);
        assert_eq!(s.success_rate, 0.0);
        assert!(s.by_protocol.is_empty());
    }
    #[test]
    fn pn_default_impl() {
        let neg = PnProtocolNegotiator::default();
        assert_eq!(neg.session_count(), 0);
        assert_eq!(neg.supported_protocols().len(), 0);
    }
}
