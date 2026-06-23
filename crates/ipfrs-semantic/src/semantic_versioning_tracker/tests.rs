//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{cosine_distance, cosine_similarity, xorshift64};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn tracker() -> SemanticVersioningTracker {
        SemanticVersioningTracker::with_defaults()
    }
    fn tracker_with(threshold: f64) -> SemanticVersioningTracker {
        SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: threshold,
            min_anchors: 1,
            window_size: 20,
            auto_deprecate: false,
        })
    }
    /// Returns a unit vector in dimension `dim` with 1.0 at position `pos`.
    fn unit(dim: usize, pos: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; dim];
        v[pos] = 1.0;
        v
    }
    /// Almost-unit: 1.0 at pos, small epsilon elsewhere.
    fn near_unit(dim: usize, pos: usize, eps: f64) -> Vec<f64> {
        let mut v = vec![eps; dim];
        v[pos] = 1.0;
        v
    }
    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-10);
    }
    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-10);
    }
    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = cosine_similarity(&a, &b);
        assert!((s + 1.0).abs() < 1e-10);
    }
    #[test]
    fn cosine_similarity_zero_vector() {
        let z = vec![0.0, 0.0];
        let a = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&z, &a), 0.0);
        assert_eq!(cosine_similarity(&a, &z), 0.0);
        assert_eq!(cosine_similarity(&z, &z), 0.0);
    }
    #[test]
    fn cosine_distance_identical_is_zero() {
        let v = vec![1.0, 0.5, 0.5];
        assert!((cosine_distance(&v, &v)).abs() < 1e-10);
    }
    #[test]
    fn cosine_distance_orthogonal_is_one() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn xorshift64_produces_nonzero() {
        let mut state: u64 = 42;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 42);
    }
    #[test]
    fn xorshift64_sequence_different() {
        let mut state: u64 = 0xCAFE_BABE;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }
    #[test]
    fn register_version_basic() {
        let mut t = tracker();
        let id = t
            .register_version("v1", 128)
            .expect("test: register version v1 with dim 128 should succeed");
        assert!(id > 0);
        let ver = t
            .get_version(id)
            .expect("test: get version by id should succeed for just-registered version");
        assert_eq!(ver.name, "v1");
        assert_eq!(ver.embedding_dim, 128);
        assert!(ver.is_active);
        assert_eq!(ver.anchor_count, 0);
    }
    #[test]
    fn register_version_ids_monotone() {
        let mut t = tracker();
        let a = t
            .register_version("a", 16)
            .expect("test: register version a should succeed");
        let b = t
            .register_version("b", 16)
            .expect("test: register version b should succeed");
        let c = t
            .register_version("c", 16)
            .expect("test: register version c should succeed");
        assert!(a < b);
        assert!(b < c);
    }
    #[test]
    fn register_version_zero_dim_err() {
        let mut t = tracker();
        let res = t.register_version("bad", 0);
        assert!(res.is_err());
    }
    #[test]
    fn deprecate_and_activate() {
        let mut t = tracker();
        let id = t
            .register_version("v1", 8)
            .expect("test: register version v1 should succeed");
        t.deprecate_version(id)
            .expect("test: deprecate version should succeed for known id");
        assert!(
            !t.get_version(id)
                .expect("test: get version should succeed for known id")
                .is_active
        );
        t.activate_version(id)
            .expect("test: activate version should succeed for known id");
        assert!(
            t.get_version(id)
                .expect("test: get version should succeed for known id")
                .is_active
        );
    }
    #[test]
    fn deprecate_nonexistent_err() {
        let mut t = tracker();
        assert!(t.deprecate_version(9999).is_err());
    }
    #[test]
    fn activate_nonexistent_err() {
        let mut t = tracker();
        assert!(t.activate_version(9999).is_err());
    }
    #[test]
    fn list_versions_sorted() {
        let mut t = tracker();
        let c = t
            .register_version("c", 4)
            .expect("test: register version c should succeed");
        let a = t
            .register_version("a", 4)
            .expect("test: register version a should succeed");
        let b = t
            .register_version("b", 4)
            .expect("test: register version b should succeed");
        let listed: Vec<SvtVersionId> = t.list_versions().iter().map(|v| v.id).collect();
        assert!(listed.contains(&a));
        assert!(listed.contains(&b));
        assert!(listed.contains(&c));
        for w in listed.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
    #[test]
    fn active_versions_filters_deprecated() {
        let mut t = tracker();
        let a = t
            .register_version("a", 4)
            .expect("test: register version a should succeed");
        let b = t
            .register_version("b", 4)
            .expect("test: register version b should succeed");
        t.deprecate_version(a)
            .expect("test: deprecate version a should succeed");
        let active_ids: Vec<SvtVersionId> = t.active_versions().iter().map(|v| v.id).collect();
        assert!(!active_ids.contains(&a));
        assert!(active_ids.contains(&b));
    }
    #[test]
    fn add_anchor_basic() {
        let mut t = tracker();
        let v = t
            .register_version("v", 3)
            .expect("test: register version v should succeed");
        t.add_anchor("cat", v, unit(3, 0))
            .expect("test: add anchor cat to version v should succeed");
        let emb = t
            .get_anchor("cat", v)
            .expect("test: get anchor cat for version v should succeed");
        assert_eq!(emb, &unit(3, 0));
    }
    #[test]
    fn add_anchor_replaces() {
        let mut t = tracker();
        let v = t
            .register_version("v", 3)
            .expect("test: register version v should succeed");
        t.add_anchor("dog", v, unit(3, 0))
            .expect("test: add anchor dog to version v should succeed");
        t.add_anchor("dog", v, unit(3, 1))
            .expect("test: replace anchor dog for version v should succeed");
        let emb = t
            .get_anchor("dog", v)
            .expect("test: get replaced anchor dog should succeed");
        assert_eq!(emb, &unit(3, 1));
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed")
                .anchor_count,
            1
        );
    }
    #[test]
    fn add_anchor_dim_mismatch_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 4)
            .expect("test: register version v should succeed");
        assert!(t.add_anchor("x", v, vec![1.0, 0.0]).is_err());
    }
    #[test]
    fn add_anchor_empty_concept_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.add_anchor("", v, vec![1.0, 0.0]).is_err());
    }
    #[test]
    fn add_anchor_unknown_version_err() {
        let mut t = tracker();
        assert!(t.add_anchor("x", 999, vec![1.0]).is_err());
    }
    #[test]
    fn get_anchor_missing_concept_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.get_anchor("missing", v).is_err());
    }
    #[test]
    fn get_anchor_missing_version_for_concept_err() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, vec![1.0, 0.0])
            .expect("test: add anchor x to v1 should succeed");
        assert!(t.get_anchor("x", v2).is_err());
    }
    #[test]
    fn anchor_count_increments() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed")
                .anchor_count,
            0
        );
        t.add_anchor("a", v, vec![1.0, 0.0])
            .expect("test: add anchor a to version v should succeed");
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed after anchor a")
                .anchor_count,
            1
        );
        t.add_anchor("b", v, vec![0.0, 1.0])
            .expect("test: add anchor b to version v should succeed");
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed after anchor b")
                .anchor_count,
            2
        );
    }
    #[test]
    fn concepts_for_version_correct() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("cat", v1, vec![1.0, 0.0])
            .expect("test: add anchor cat to v1 should succeed");
        t.add_anchor("dog", v1, vec![0.0, 1.0])
            .expect("test: add anchor dog to v1 should succeed");
        t.add_anchor("cat", v2, vec![1.0, 0.0])
            .expect("test: add anchor cat to v2 should succeed");
        let v1_concepts = t.concepts_for_version(v1);
        assert!(v1_concepts.contains(&"cat"));
        assert!(v1_concepts.contains(&"dog"));
        let v2_concepts = t.concepts_for_version(v2);
        assert!(v2_concepts.contains(&"cat"));
        assert!(!v2_concepts.contains(&"dog"));
    }
    #[test]
    fn compute_drift_zero_for_identical_embeddings() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 3)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 3)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, vec![1.0, 0.0, 0.0])
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, vec![1.0, 0.0, 0.0])
            .expect("test: add anchor x to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift with identical embeddings should succeed");
        assert!(report.overall_drift < 1e-10);
        assert!(report.drifted_concepts.is_empty());
        assert!(!report.stable_concepts.is_empty());
    }
    #[test]
    fn compute_drift_max_for_orthogonal() {
        let mut t = tracker_with(0.5);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift for orthogonal embeddings should succeed");
        assert!((report.overall_drift - 1.0).abs() < 1e-10);
        assert_eq!(report.drifted_concepts.len(), 1);
    }
    #[test]
    fn compute_drift_unknown_version_err() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        assert!(t.compute_drift(v1, 9999).is_err());
        assert!(t.compute_drift(9999, v1).is_err());
    }
    #[test]
    fn compute_drift_insufficient_anchors_err() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.1,
            min_anchors: 3,
            window_size: 10,
            auto_deprecate: false,
        });
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, vec![1.0, 0.0])
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, vec![1.0, 0.0])
            .expect("test: add anchor x to v2 should succeed");
        assert!(t.compute_drift(v1, v2).is_err());
    }
    #[test]
    fn compute_drift_records_to_log() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("a", v1, unit(2, 0))
            .expect("test: add anchor a to v1 should succeed");
        t.add_anchor("a", v2, unit(2, 0))
            .expect("test: add anchor a to v2 should succeed");
        t.add_anchor("b", v1, unit(2, 1))
            .expect("test: add anchor b to v1 should succeed");
        t.add_anchor("b", v2, unit(2, 0))
            .expect("test: add anchor b to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert_eq!(t.drift_log().len(), 2);
    }
    #[test]
    fn compute_drift_report_has_recommendation() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(!report.recommendation.is_empty());
    }
    #[test]
    fn compute_drift_drifted_sorted_descending() {
        let mut t = tracker_with(0.01);
        let v1 = t
            .register_version("v1", 4)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 4)
            .expect("test: register version v2 should succeed");
        t.add_anchor("a", v1, unit(4, 0))
            .expect("test: add anchor a to v1 should succeed");
        t.add_anchor("a", v2, unit(4, 1))
            .expect("test: add anchor a to v2 should succeed");
        t.add_anchor("b", v1, near_unit(4, 0, 0.5))
            .expect("test: add anchor b to v1 should succeed");
        t.add_anchor("b", v2, unit(4, 0))
            .expect("test: add anchor b to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(report.drifted_concepts.len() >= 2);
        for w in report.drifted_concepts.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }
    #[test]
    fn find_drifted_concepts_above_threshold() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("same", v1, unit(2, 0))
            .expect("test: add anchor same to v1 should succeed");
        t.add_anchor("same", v2, unit(2, 0))
            .expect("test: add anchor same to v2 should succeed");
        t.add_anchor("diff", v1, unit(2, 0))
            .expect("test: add anchor diff to v1 should succeed");
        t.add_anchor("diff", v2, unit(2, 1))
            .expect("test: add anchor diff to v2 should succeed");
        let drifted = t
            .find_drifted_concepts(v1, v2, 0.3)
            .expect("test: find drifted concepts should succeed");
        assert_eq!(drifted.len(), 1);
        assert_eq!(drifted[0].0, "diff");
    }
    #[test]
    fn find_drifted_concepts_unknown_version_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.find_drifted_concepts(v, 999, 0.1).is_err());
    }
    #[test]
    fn semantic_similarity_over_time_basic() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 3)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 3)
            .expect("test: register version v2 should succeed");
        let v3 = t
            .register_version("v3", 3)
            .expect("test: register version v3 should succeed");
        t.add_anchor("cat", v1, unit(3, 0))
            .expect("test: add anchor cat to v1 should succeed");
        t.add_anchor("cat", v2, near_unit(3, 0, 0.01))
            .expect("test: add anchor cat to v2 should succeed");
        t.add_anchor("cat", v3, unit(3, 0))
            .expect("test: add anchor cat to v3 should succeed");
        let pairs = t
            .semantic_similarity_over_time("cat")
            .expect("test: semantic similarity over time for cat should succeed");
        assert_eq!(pairs.len(), 2);
        for (_, _, sim) in &pairs {
            assert!(*sim > 0.9);
        }
    }
    #[test]
    fn semantic_similarity_over_time_empty_concept_err() {
        let t = tracker();
        assert!(t.semantic_similarity_over_time("").is_err());
    }
    #[test]
    fn semantic_similarity_over_time_missing_concept_err() {
        let t = tracker();
        assert!(t.semantic_similarity_over_time("ghost").is_err());
    }
    #[test]
    fn semantic_similarity_over_time_single_version_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        t.add_anchor("x", v, unit(2, 0))
            .expect("test: add anchor x to v should succeed");
        assert!(t.semantic_similarity_over_time("x").is_err());
    }
    #[test]
    fn semantic_similarity_over_time_window_limit() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.1,
            min_anchors: 1,
            window_size: 3,
            auto_deprecate: false,
        });
        for i in 0..10usize {
            let v = t
                .register_version(format!("v{i}"), 2)
                .expect("test: register version should succeed");
            t.add_anchor("x", v, vec![1.0, i as f64])
                .expect("test: add anchor x to version should succeed");
        }
        let pairs = t
            .semantic_similarity_over_time("x")
            .expect("test: semantic similarity over time for x should succeed");
        assert!(pairs.len() <= 3);
    }
    #[test]
    fn stability_score_no_anchors_returns_one() {
        let t = tracker();
        let s = t
            .stability_score("ghost")
            .expect("test: stability score for unregistered concept should return Ok(1.0)");
        assert!((s - 1.0).abs() < 1e-10);
    }
    #[test]
    fn stability_score_one_version_returns_one() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        t.add_anchor("x", v, unit(2, 0))
            .expect("test: add anchor x to v should succeed");
        let s = t
            .stability_score("x")
            .expect("test: stability score should succeed");
        assert!((s - 1.0).abs() < 1e-10);
    }
    #[test]
    fn stability_score_identical_embeddings_is_one() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let s = t
            .stability_score("x")
            .expect("test: stability score for identical embeddings should succeed");
        assert!((s - 1.0).abs() < 1e-10);
    }
    #[test]
    fn stability_score_orthogonal_embeddings_is_zero() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let s = t
            .stability_score("x")
            .expect("test: stability score for orthogonal embeddings should succeed");
        assert!(s.abs() < 1e-10);
    }
    #[test]
    fn stability_score_empty_concept_err() {
        let t = tracker();
        assert!(t.stability_score("").is_err());
    }
    #[test]
    fn recommend_migration_returns_drifted() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("stable", v1, unit(2, 0))
            .expect("test: add anchor stable to v1 should succeed");
        t.add_anchor("stable", v2, unit(2, 0))
            .expect("test: add anchor stable to v2 should succeed");
        t.add_anchor("drifted", v1, unit(2, 0))
            .expect("test: add anchor drifted to v1 should succeed");
        t.add_anchor("drifted", v2, unit(2, 1))
            .expect("test: add anchor drifted to v2 should succeed");
        let recs = t
            .recommend_migration(v1, v2)
            .expect("test: recommend migration should succeed");
        assert!(recs.contains(&"drifted".to_string()));
        assert!(!recs.contains(&"stable".to_string()));
    }
    #[test]
    fn recommend_migration_unknown_version_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.recommend_migration(v, 9999).is_err());
    }
    #[test]
    fn tracker_stats_empty() {
        let t = tracker();
        let stats = t.tracker_stats();
        assert_eq!(stats.total_versions, 0);
        assert_eq!(stats.active_versions, 0);
        assert_eq!(stats.total_anchors, 0);
        assert_eq!(stats.distinct_concepts, 0);
        assert_eq!(stats.drift_events, 0);
        assert!((stats.mean_logged_drift).abs() < 1e-10);
    }
    #[test]
    fn tracker_stats_after_operations() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.deprecate_version(v1)
            .expect("test: deprecate version v1 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        t.add_anchor("y", v1, unit(2, 1))
            .expect("test: add anchor y to v1 should succeed");
        t.add_anchor("y", v2, unit(2, 1))
            .expect("test: add anchor y to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        let stats = t.tracker_stats();
        assert_eq!(stats.total_versions, 2);
        assert_eq!(stats.active_versions, 1);
        assert_eq!(stats.distinct_concepts, 2);
        assert_eq!(stats.total_anchors, 4);
        assert_eq!(stats.drift_events, 2);
    }
    #[test]
    fn drift_log_bounded_at_500() {
        let mut t = tracker_with(0.0);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        for i in 0usize..600 {
            let c = format!("c{i}");
            t.add_anchor(&c, v1, unit(2, 0))
                .expect("test: add anchor to v1 should succeed");
            t.add_anchor(&c, v2, unit(2, 1))
                .expect("test: add anchor to v2 should succeed");
        }
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift with 600 concepts should succeed");
        assert!(t.drift_log().len() <= 500);
    }
    #[test]
    fn clear_drift_log() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(!t.drift_log().is_empty());
        t.clear_drift_log();
        assert!(t.drift_log().is_empty());
    }
    #[test]
    fn add_anchors_batch_partial_failure() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        let items = vec![
            ("good".to_string(), v, vec![1.0, 0.0]),
            ("bad".to_string(), v, vec![1.0, 0.0, 0.0]),
        ];
        let errors = t.add_anchors_batch(items);
        assert_eq!(errors.len(), 1);
        assert!(t.get_anchor("good", v).is_ok());
    }
    #[test]
    fn compute_all_consecutive_drifts() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        let v3 = t
            .register_version("v3", 2)
            .expect("test: register version v3 should succeed");
        for v in [v1, v2, v3] {
            t.add_anchor("x", v, unit(2, 0))
                .expect("test: add anchor x to version should succeed");
        }
        let results = t.compute_all_consecutive_drifts();
        assert_eq!(results.len(), 2);
        for r in results {
            assert!(r.is_ok());
        }
    }
    #[test]
    fn global_stability_no_concepts_is_one() {
        let t = tracker();
        assert!((t.global_stability() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn global_stability_with_perfect_concepts() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let gs = t.global_stability();
        assert!((gs - 1.0).abs() < 1e-10);
    }
    #[test]
    fn concepts_by_stability_sorted() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("stable", v1, unit(2, 0))
            .expect("test: add anchor stable to v1 should succeed");
        t.add_anchor("stable", v2, unit(2, 0))
            .expect("test: add anchor stable to v2 should succeed");
        t.add_anchor("unstable", v1, unit(2, 0))
            .expect("test: add anchor unstable to v1 should succeed");
        t.add_anchor("unstable", v2, unit(2, 1))
            .expect("test: add anchor unstable to v2 should succeed");
        let sorted = t.concepts_by_stability();
        assert_eq!(sorted[0].0, "stable");
        assert_eq!(sorted[1].0, "unstable");
    }
    #[test]
    fn top_drifted_concepts_limits_result() {
        let mut t = tracker_with(0.0);
        let v1 = t
            .register_version("v1", 4)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 4)
            .expect("test: register version v2 should succeed");
        for i in 0..4usize {
            let c = format!("c{i}");
            t.add_anchor(&c, v1, unit(4, 0))
                .expect("test: add anchor to v1 should succeed");
            t.add_anchor(&c, v2, unit(4, (i + 1) % 4))
                .expect("test: add anchor to v2 should succeed");
        }
        let top2 = t
            .top_drifted_concepts(v1, v2, 2)
            .expect("test: top drifted concepts should succeed");
        assert!(top2.len() <= 2);
    }
    #[test]
    fn are_compatible_identical_true() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        assert!(t
            .are_compatible(v1, v2)
            .expect("test: are_compatible should succeed"));
    }
    #[test]
    fn are_compatible_orthogonal_false() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        assert!(!t
            .are_compatible(v1, v2)
            .expect("test: are_compatible should succeed"));
    }
    #[test]
    fn remove_version_anchors_clears_entries() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let removed = t
            .remove_version_anchors(v1)
            .expect("test: remove version anchors should succeed");
        assert_eq!(removed, 1);
        assert!(t.get_anchor("x", v1).is_err());
        assert!(t.get_anchor("x", v2).is_ok());
    }
    #[test]
    fn remove_version_anchors_unknown_version_err() {
        let mut t = tracker();
        assert!(t.remove_version_anchors(9999).is_err());
    }
    #[test]
    fn concept_drift_matrix_correct() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("same", v1, unit(2, 0))
            .expect("test: add anchor same to v1 should succeed");
        t.add_anchor("same", v2, unit(2, 0))
            .expect("test: add anchor same to v2 should succeed");
        t.add_anchor("diff", v1, unit(2, 0))
            .expect("test: add anchor diff to v1 should succeed");
        t.add_anchor("diff", v2, unit(2, 1))
            .expect("test: add anchor diff to v2 should succeed");
        let matrix = t
            .concept_drift_matrix(v1, v2)
            .expect("test: concept drift matrix should succeed");
        assert!((matrix["same"]).abs() < 1e-10);
        assert!((matrix["diff"] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn concept_drift_matrix_does_not_log() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .concept_drift_matrix(v1, v2)
            .expect("test: concept drift matrix should succeed");
        assert!(t.drift_log().is_empty());
    }
    #[test]
    fn auto_deprecate_deactivates_old_version() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.1,
            min_anchors: 1,
            window_size: 10,
            auto_deprecate: true,
        });
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(
            !t.get_version(v1)
                .expect("test: get version v1 after auto-deprecate should succeed")
                .is_active
        );
    }
    #[test]
    fn auto_deprecate_does_not_deactivate_when_drift_low() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.5,
            min_anchors: 1,
            window_size: 10,
            auto_deprecate: true,
        });
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(
            t.get_version(v1)
                .expect("test: get version v1 should succeed")
                .is_active
        );
    }
    #[test]
    fn recommendation_transparent_when_very_low_drift() {
        let mut t = tracker_with(0.5);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let r = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(
            r.recommendation.contains("transparent") || r.recommendation.contains("compatible")
        );
    }
    #[test]
    fn recommendation_full_reindex_when_very_high_drift() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let r = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(r.recommendation.contains("incompatible") || r.recommendation.contains("re-index"));
    }
    #[test]
    fn drift_event_significant_flag() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        let event = t
            .drift_log()
            .back()
            .expect("test: drift log should have at least one event after compute_drift");
        assert!(event.is_significant);
        assert!((event.drift_score - 1.0).abs() < 1e-10);
    }
    #[test]
    fn drift_event_not_significant_for_identical() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        let event = t
            .drift_log()
            .back()
            .expect("test: drift log should have at least one event after compute_drift");
        assert!(!event.is_significant);
    }
    #[test]
    fn config_accessor_returns_correct_threshold() {
        let t = tracker_with(0.42);
        assert!((t.config().drift_threshold - 0.42).abs() < 1e-10);
    }
    #[test]
    fn config_mut_modifies_threshold() {
        let mut t = tracker_with(0.1);
        t.config_mut().drift_threshold = 0.9;
        assert!((t.config().drift_threshold - 0.9).abs() < 1e-10);
    }
}
