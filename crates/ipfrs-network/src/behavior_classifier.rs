//! Peer Behavior Classifier
//!
//! Classifies peers into behavioral categories based on observed interaction
//! patterns, enabling smarter routing and eviction decisions.

use std::collections::HashMap;

/// Behavioral signals observed from peer interactions.
#[derive(Clone, Debug, PartialEq)]
pub enum BehaviorSignal {
    /// Response latency < 50ms
    FastResponder,
    /// Response latency > 500ms
    SlowResponder,
    /// Transfer rate > 10 MB/s
    HighBandwidth,
    /// Transfer rate < 100 KB/s
    LowBandwidth,
    /// Disconnects/reconnects frequently
    FrequentChurner,
    /// No disconnects in observation window
    StableConnection,
    /// Downloads much more than uploads
    DataHoarder,
    /// Uploads >= downloads
    GoodContributor,
}

impl BehaviorSignal {
    /// Returns the discriminant index based on declaration order for sorting.
    fn discriminant_index(&self) -> usize {
        match self {
            BehaviorSignal::FastResponder => 0,
            BehaviorSignal::SlowResponder => 1,
            BehaviorSignal::HighBandwidth => 2,
            BehaviorSignal::LowBandwidth => 3,
            BehaviorSignal::FrequentChurner => 4,
            BehaviorSignal::StableConnection => 5,
            BehaviorSignal::DataHoarder => 6,
            BehaviorSignal::GoodContributor => 7,
        }
    }
}

/// Profile of a peer's observed behavior signals.
#[derive(Clone, Debug)]
pub struct BehaviorProfile {
    /// Peer identifier.
    pub peer_id: String,
    /// All observed signals (duplicates allowed — frequency matters).
    pub signals: Vec<BehaviorSignal>,
    /// Total number of observations recorded.
    pub observation_count: u64,
}

impl BehaviorProfile {
    /// Returns `true` if the profile contains the given signal at least once.
    pub fn has_signal(&self, signal: &BehaviorSignal) -> bool {
        self.signals.contains(signal)
    }

    /// Returns the total number of signals stored (including duplicates).
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }
}

/// Aggregate statistics over all classified peers.
#[derive(Clone, Debug, Default)]
pub struct ClassifierStats {
    /// Total number of tracked peers.
    pub total_peers: usize,
    /// Peers whose `classify()` output includes `FastResponder`.
    pub fast_responders: usize,
    /// Peers whose `classify()` output includes `HighBandwidth`.
    pub high_bandwidth_peers: usize,
    /// Peers whose `classify()` output includes `GoodContributor`.
    pub good_contributors: usize,
    /// Peers whose `classify()` output includes `FrequentChurner`.
    pub churners: usize,
}

/// Classifies peers into behavioral categories based on recorded signals.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::behavior_classifier::{PeerBehaviorClassifier, BehaviorSignal};
///
/// let mut classifier = PeerBehaviorClassifier::new();
/// classifier.record_signal("peer-1", BehaviorSignal::FastResponder);
/// classifier.record_signal("peer-1", BehaviorSignal::FastResponder);
/// let classified = classifier.classify("peer-1");
/// assert!(classified.contains(&BehaviorSignal::FastResponder));
/// ```
#[derive(Debug, Default)]
pub struct PeerBehaviorClassifier {
    /// Profiles keyed by peer_id.
    pub profiles: HashMap<String, BehaviorProfile>,
}

impl PeerBehaviorClassifier {
    /// Creates a new, empty classifier.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Records a behavioral signal for a peer.
    ///
    /// If no profile exists for `peer_id`, one is created automatically.
    /// Duplicates are allowed — signal frequency is meaningful.
    pub fn record_signal(&mut self, peer_id: &str, signal: BehaviorSignal) {
        let profile = self
            .profiles
            .entry(peer_id.to_string())
            .or_insert_with(|| BehaviorProfile {
                peer_id: peer_id.to_string(),
                signals: Vec::new(),
                observation_count: 0,
            });
        profile.signals.push(signal);
        profile.observation_count += 1;
    }

    /// Returns the "consistent" signals for a peer — those appearing more than
    /// once in the recorded history.
    ///
    /// The returned `Vec` contains each qualifying signal exactly once, sorted
    /// by declaration order (discriminant index).
    ///
    /// Returns an empty `Vec` for unknown peers.
    pub fn classify(&self, peer_id: &str) -> Vec<BehaviorSignal> {
        let profile = match self.profiles.get(peer_id) {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Count occurrences of each variant.
        let mut counts: HashMap<usize, (usize, BehaviorSignal)> = HashMap::new();
        for signal in &profile.signals {
            let idx = signal.discriminant_index();
            let entry = counts.entry(idx).or_insert_with(|| (0, signal.clone()));
            entry.0 += 1;
        }

        // Collect signals that appear more than once.
        let mut result: Vec<(usize, BehaviorSignal)> = counts
            .into_iter()
            .filter_map(
                |(idx, (count, signal))| {
                    if count > 1 {
                        Some((idx, signal))
                    } else {
                        None
                    }
                },
            )
            .collect();

        // Sort by declaration order.
        result.sort_by_key(|(idx, _)| *idx);
        result.into_iter().map(|(_, signal)| signal).collect()
    }

    /// Returns peer IDs where `classify()` includes the given signal, sorted
    /// alphabetically.
    pub fn peers_with_signal(&self, signal: &BehaviorSignal) -> Vec<&str> {
        let mut result: Vec<&str> = self
            .profiles
            .keys()
            .filter(|peer_id| self.classify(peer_id).contains(signal))
            .map(|s| s.as_str())
            .collect();
        result.sort_unstable();
        result
    }

    /// Removes a peer profile. Returns `true` if the peer existed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.profiles.remove(peer_id).is_some()
    }

    /// Returns aggregate statistics across all tracked peers.
    pub fn stats(&self) -> ClassifierStats {
        let mut stats = ClassifierStats {
            total_peers: self.profiles.len(),
            ..Default::default()
        };

        for peer_id in self.profiles.keys() {
            let classified = self.classify(peer_id);
            for signal in &classified {
                match signal {
                    BehaviorSignal::FastResponder => stats.fast_responders += 1,
                    BehaviorSignal::HighBandwidth => stats.high_bandwidth_peers += 1,
                    BehaviorSignal::GoodContributor => stats.good_contributors += 1,
                    BehaviorSignal::FrequentChurner => stats.churners += 1,
                    _ => {}
                }
            }
        }

        stats
    }

    /// Returns an immutable reference to a peer's profile, or `None` if
    /// unknown.
    pub fn profile(&self, peer_id: &str) -> Option<&BehaviorProfile> {
        self.profiles.get(peer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_new_starts_empty() {
        let classifier = PeerBehaviorClassifier::new();
        assert!(classifier.profiles.is_empty());
    }

    // -------------------------------------------------------------------------
    // record_signal
    // -------------------------------------------------------------------------

    #[test]
    fn test_record_signal_creates_profile() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        assert!(c.profile("peer-a").is_some());
    }

    #[test]
    fn test_record_signal_increments_observation_count() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        c.record_signal("peer-a", BehaviorSignal::SlowResponder);
        assert_eq!(
            c.profile("peer-a")
                .expect("test: peer-a profile should exist")
                .observation_count,
            2
        );
    }

    #[test]
    fn test_record_signal_appends_to_signals() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::HighBandwidth);
        c.record_signal("peer-a", BehaviorSignal::HighBandwidth);
        assert_eq!(
            c.profile("peer-a")
                .expect("test: peer-a profile should exist")
                .signals
                .len(),
            2
        );
    }

    #[test]
    fn test_observation_count_accumulates_across_many_signals() {
        let mut c = PeerBehaviorClassifier::new();
        for _ in 0..10 {
            c.record_signal("peer-x", BehaviorSignal::StableConnection);
        }
        assert_eq!(
            c.profile("peer-x")
                .expect("test: peer-x profile should exist")
                .observation_count,
            10
        );
    }

    // -------------------------------------------------------------------------
    // classify
    // -------------------------------------------------------------------------

    #[test]
    fn test_classify_single_signal_not_in_result() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        assert!(c.classify("peer-a").is_empty());
    }

    #[test]
    fn test_classify_signal_appearing_twice_is_included() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        assert!(c
            .classify("peer-a")
            .contains(&BehaviorSignal::FastResponder));
    }

    #[test]
    fn test_classify_multiple_distinct_signals() {
        let mut c = PeerBehaviorClassifier::new();
        for _ in 0..2 {
            c.record_signal("peer-a", BehaviorSignal::HighBandwidth);
            c.record_signal("peer-a", BehaviorSignal::GoodContributor);
        }
        let classified = c.classify("peer-a");
        assert!(classified.contains(&BehaviorSignal::HighBandwidth));
        assert!(classified.contains(&BehaviorSignal::GoodContributor));
    }

    #[test]
    fn test_classify_returns_empty_for_unknown_peer() {
        let c = PeerBehaviorClassifier::new();
        assert!(c.classify("ghost-peer").is_empty());
    }

    #[test]
    fn test_classify_returns_sorted_by_declaration_order() {
        let mut c = PeerBehaviorClassifier::new();
        // Add in reverse declaration order to confirm sorting.
        for _ in 0..2 {
            c.record_signal("peer-a", BehaviorSignal::GoodContributor); // idx 7
            c.record_signal("peer-a", BehaviorSignal::DataHoarder); // idx 6
            c.record_signal("peer-a", BehaviorSignal::FastResponder); // idx 0
        }
        let classified = c.classify("peer-a");
        assert_eq!(classified.len(), 3);
        assert_eq!(classified[0], BehaviorSignal::FastResponder);
        assert_eq!(classified[1], BehaviorSignal::DataHoarder);
        assert_eq!(classified[2], BehaviorSignal::GoodContributor);
    }

    #[test]
    fn test_classify_threshold_exactly_two_observations() {
        let mut c = PeerBehaviorClassifier::new();
        // Exactly 2 occurrences → should be included (> 1).
        c.record_signal("peer-a", BehaviorSignal::LowBandwidth);
        c.record_signal("peer-a", BehaviorSignal::LowBandwidth);
        assert!(c.classify("peer-a").contains(&BehaviorSignal::LowBandwidth));
    }

    // -------------------------------------------------------------------------
    // peers_with_signal
    // -------------------------------------------------------------------------

    #[test]
    fn test_peers_with_signal_returns_correct_peers_sorted() {
        let mut c = PeerBehaviorClassifier::new();
        for id in &["charlie", "alice", "bob"] {
            c.record_signal(id, BehaviorSignal::FastResponder);
            c.record_signal(id, BehaviorSignal::FastResponder);
        }
        let peers = c.peers_with_signal(&BehaviorSignal::FastResponder);
        assert_eq!(peers, vec!["alice", "bob", "charlie"]);
    }

    #[test]
    fn test_peers_with_signal_returns_empty_when_none_qualify() {
        let mut c = PeerBehaviorClassifier::new();
        // Only one observation — doesn't meet threshold.
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        assert!(c
            .peers_with_signal(&BehaviorSignal::FastResponder)
            .is_empty());
    }

    #[test]
    fn test_peers_with_signal_excludes_non_qualifying_peers() {
        let mut c = PeerBehaviorClassifier::new();
        // peer-a qualifies; peer-b does not.
        c.record_signal("peer-a", BehaviorSignal::HighBandwidth);
        c.record_signal("peer-a", BehaviorSignal::HighBandwidth);
        c.record_signal("peer-b", BehaviorSignal::HighBandwidth); // only once
        let peers = c.peers_with_signal(&BehaviorSignal::HighBandwidth);
        assert_eq!(peers, vec!["peer-a"]);
    }

    // -------------------------------------------------------------------------
    // remove_peer
    // -------------------------------------------------------------------------

    #[test]
    fn test_remove_peer_existing_returns_true() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        assert!(c.remove_peer("peer-a"));
    }

    #[test]
    fn test_remove_peer_nonexistent_returns_false() {
        let mut c = PeerBehaviorClassifier::new();
        assert!(!c.remove_peer("ghost"));
    }

    #[test]
    fn test_remove_peer_actually_removes_profile() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        c.remove_peer("peer-a");
        assert!(c.profile("peer-a").is_none());
    }

    // -------------------------------------------------------------------------
    // profile
    // -------------------------------------------------------------------------

    #[test]
    fn test_profile_returns_some_for_known_peer() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::FastResponder);
        assert!(c.profile("peer-a").is_some());
    }

    #[test]
    fn test_profile_returns_none_for_unknown_peer() {
        let c = PeerBehaviorClassifier::new();
        assert!(c.profile("ghost").is_none());
    }

    // -------------------------------------------------------------------------
    // stats
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats_total_peers() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-1", BehaviorSignal::FastResponder);
        c.record_signal("peer-2", BehaviorSignal::SlowResponder);
        assert_eq!(c.stats().total_peers, 2);
    }

    #[test]
    fn test_stats_fast_responders_count() {
        let mut c = PeerBehaviorClassifier::new();
        for id in &["p1", "p2"] {
            c.record_signal(id, BehaviorSignal::FastResponder);
            c.record_signal(id, BehaviorSignal::FastResponder);
        }
        c.record_signal("p3", BehaviorSignal::FastResponder); // only once
        assert_eq!(c.stats().fast_responders, 2);
    }

    #[test]
    fn test_stats_high_bandwidth_peers_count() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("p1", BehaviorSignal::HighBandwidth);
        c.record_signal("p1", BehaviorSignal::HighBandwidth);
        assert_eq!(c.stats().high_bandwidth_peers, 1);
    }

    #[test]
    fn test_stats_good_contributors_count() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("p1", BehaviorSignal::GoodContributor);
        c.record_signal("p1", BehaviorSignal::GoodContributor);
        c.record_signal("p2", BehaviorSignal::GoodContributor);
        c.record_signal("p2", BehaviorSignal::GoodContributor);
        assert_eq!(c.stats().good_contributors, 2);
    }

    #[test]
    fn test_stats_churners_count() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("p1", BehaviorSignal::FrequentChurner);
        c.record_signal("p1", BehaviorSignal::FrequentChurner);
        assert_eq!(c.stats().churners, 1);
    }

    // -------------------------------------------------------------------------
    // BehaviorProfile helpers
    // -------------------------------------------------------------------------

    #[test]
    fn test_has_signal_true() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::StableConnection);
        let profile = c
            .profile("peer-a")
            .expect("test: peer-a profile should exist");
        assert!(profile.has_signal(&BehaviorSignal::StableConnection));
    }

    #[test]
    fn test_has_signal_false() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::StableConnection);
        let profile = c
            .profile("peer-a")
            .expect("test: peer-a profile should exist");
        assert!(!profile.has_signal(&BehaviorSignal::FastResponder));
    }

    #[test]
    fn test_signal_count() {
        let mut c = PeerBehaviorClassifier::new();
        c.record_signal("peer-a", BehaviorSignal::DataHoarder);
        c.record_signal("peer-a", BehaviorSignal::DataHoarder);
        c.record_signal("peer-a", BehaviorSignal::GoodContributor);
        let profile = c
            .profile("peer-a")
            .expect("test: peer-a profile should exist");
        assert_eq!(profile.signal_count(), 3);
    }
}
