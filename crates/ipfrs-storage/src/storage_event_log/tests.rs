//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use std::collections::HashMap;
    /// Deterministic xorshift64 PRNG (no rand crate).
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }
    fn default_log() -> SelStorageEventLog {
        SelStorageEventLog::new(EventLogConfig {
            max_events: 10_000,
            retention_policy: RetentionPolicy::KeepAll,
            enable_checksums: true,
            batch_flush_size: 0,
        })
    }
    fn log_event(
        log: &mut SelStorageEventLog,
        et: SelEventType,
        oid: &str,
        uid: &str,
        sz: u64,
        ts: u64,
    ) -> EventId {
        log.log(et, oid.to_string(), uid.to_string(), sz, vec![], None, ts)
    }
    fn log_event_corr(
        log: &mut SelStorageEventLog,
        et: SelEventType,
        oid: &str,
        uid: &str,
        sz: u64,
        ts: u64,
        corr: &str,
    ) -> EventId {
        log.log(
            et,
            oid.to_string(),
            uid.to_string(),
            sz,
            vec![],
            Some(corr.to_string()),
            ts,
        )
    }
    #[test]
    fn test_first_id_is_one() {
        let mut log = default_log();
        let id = log_event(&mut log, SelEventType::Created, "obj1", "user1", 100, 1000);
        assert_eq!(id.0, 1, "first id must be 1");
    }
    #[test]
    fn test_ids_monotonically_increasing() {
        let mut log = default_log();
        let a = log_event(&mut log, SelEventType::Created, "obj1", "u1", 10, 1);
        let b = log_event(&mut log, SelEventType::Read, "obj1", "u1", 10, 2);
        let c = log_event(&mut log, SelEventType::Deleted, "obj1", "u1", 10, 3);
        assert!(a < b && b < c, "ids must be strictly increasing");
    }
    #[test]
    fn test_get_returns_correct_event() {
        let mut log = default_log();
        let id = log_event(&mut log, SelEventType::Created, "obj42", "alice", 512, 9999);
        let evt = log.get(id).expect("event must exist");
        assert_eq!(evt.id, id);
        assert_eq!(evt.object_id, "obj42");
        assert_eq!(evt.user_id, "alice");
        assert_eq!(evt.size_bytes, 512);
        assert_eq!(evt.timestamp, 9999);
    }
    #[test]
    fn test_get_event_not_found() {
        let log = default_log();
        let err = log.get(EventId(999)).unwrap_err();
        assert_eq!(err, EventLogError::EventNotFound(EventId(999)));
    }
    #[test]
    fn test_event_count() {
        let mut log = default_log();
        assert_eq!(log.event_count(), 0);
        log_event(&mut log, SelEventType::Created, "o", "u", 0, 1);
        assert_eq!(log.event_count(), 1);
        log_event(&mut log, SelEventType::Read, "o", "u", 0, 2);
        assert_eq!(log.event_count(), 2);
    }
    #[test]
    fn test_checksum_stored() {
        let mut log = default_log();
        let id = log_event(&mut log, SelEventType::Created, "cobj", "cuser", 64, 5000);
        let evt = log.get(id).expect("must exist");
        assert_ne!(
            evt.checksum, 0,
            "checksum should be non-zero for non-trivial input"
        );
        assert_eq!(evt.checksum, event_checksum(evt));
    }
    #[test]
    fn test_checksum_disabled() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            enable_checksums: false,
            ..EventLogConfig::default()
        });
        let id = log.log(
            SelEventType::Created,
            "o".to_string(),
            "u".to_string(),
            0,
            vec![],
            None,
            1,
        );
        let evt = log.get(id).expect("must exist");
        assert_eq!(evt.checksum, 0);
    }
    #[test]
    fn test_verify_integrity_clean() {
        let mut log = default_log();
        for i in 0..10u64 {
            log_event(
                &mut log,
                SelEventType::Created,
                &format!("obj{i}"),
                "u",
                i * 100,
                i,
            );
        }
        assert!(log.verify_integrity().is_empty());
    }
    #[test]
    fn test_verify_integrity_detects_corruption() {
        let mut log = default_log();
        let id = log_event(&mut log, SelEventType::Created, "obj", "u", 100, 1000);
        if let Some(evt) = log.events.iter_mut().find(|e| e.id == id) {
            evt.checksum = evt.checksum.wrapping_add(1);
        }
        let corrupt = log.verify_integrity();
        assert_eq!(corrupt.len(), 1);
        assert_eq!(corrupt[0], id);
    }
    #[test]
    fn test_verify_integrity_disabled() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            enable_checksums: false,
            retention_policy: RetentionPolicy::KeepAll,
            ..EventLogConfig::default()
        });
        let id = log.log(
            SelEventType::Created,
            "o".to_string(),
            "u".to_string(),
            0,
            vec![],
            None,
            1,
        );
        if let Some(evt) = log.events.iter_mut().find(|e| e.id == id) {
            evt.checksum = 0xDEAD_BEEF;
        }
        assert!(log.verify_integrity().is_empty());
    }
    #[test]
    fn test_query_by_event_type() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o1", "u", 1, 1);
        log_event(&mut log, SelEventType::Read, "o1", "u", 1, 2);
        log_event(&mut log, SelEventType::Created, "o2", "u", 1, 3);
        let q = EventQuery {
            event_types: vec![SelEventType::Created],
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("query ok");
        assert_eq!(res.len(), 2);
        assert!(res
            .iter()
            .all(|e| matches!(e.event_type, SelEventType::Created)));
    }
    #[test]
    fn test_query_by_object_id() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "target", "u", 1, 1);
        log_event(&mut log, SelEventType::Created, "other", "u", 1, 2);
        log_event(&mut log, SelEventType::Read, "target", "u", 1, 3);
        let q = EventQuery {
            object_id: Some("target".to_string()),
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("query ok");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|e| e.object_id == "target"));
    }
    #[test]
    fn test_query_by_user_id() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "alice", 1, 1);
        log_event(&mut log, SelEventType::Created, "o", "bob", 1, 2);
        log_event(&mut log, SelEventType::Read, "o", "alice", 1, 3);
        let q = EventQuery {
            user_id: Some("alice".to_string()),
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("query ok");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|e| e.user_id == "alice"));
    }
    #[test]
    fn test_query_by_time_range() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 100);
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 200);
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 300);
        let q = EventQuery {
            time_range: Some((150, 250)),
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("query ok");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].timestamp, 200);
    }
    #[test]
    fn test_query_time_range_inclusive() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 100);
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 200);
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 300);
        let q = EventQuery {
            time_range: Some((100, 300)),
            ..EventQuery::default()
        };
        assert_eq!(log.query(&q).expect("ok").len(), 3);
    }
    #[test]
    fn test_query_by_correlation_id() {
        let mut log = default_log();
        log_event_corr(&mut log, SelEventType::Created, "o", "u", 1, 1, "batch-1");
        log_event(&mut log, SelEventType::Read, "o", "u", 1, 2);
        log_event_corr(&mut log, SelEventType::Updated, "o", "u", 1, 3, "batch-1");
        let q = EventQuery {
            correlation_id: Some("batch-1".to_string()),
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("ok");
        assert_eq!(res.len(), 2);
    }
    #[test]
    fn test_query_limit() {
        let mut log = default_log();
        for i in 0..10u64 {
            log_event(&mut log, SelEventType::Read, "o", "u", i, i);
        }
        let q = EventQuery {
            limit: 3,
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("ok");
        assert_eq!(res.len(), 3);
    }
    #[test]
    fn test_query_offset() {
        let mut log = default_log();
        for i in 0..5u64 {
            log_event(&mut log, SelEventType::Read, "o", "u", i, i);
        }
        let q = EventQuery {
            offset: 3,
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("ok");
        assert_eq!(res.len(), 2);
    }
    #[test]
    fn test_query_offset_limit() {
        let mut log = default_log();
        for i in 0u64..10 {
            log_event(&mut log, SelEventType::Read, "o", "u", i, i);
        }
        let q = EventQuery {
            offset: 2,
            limit: 3,
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("ok");
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].timestamp, 2);
    }
    #[test]
    fn test_query_no_match() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 1);
        let q = EventQuery {
            event_types: vec![SelEventType::Deleted],
            ..EventQuery::default()
        };
        assert!(log.query(&q).expect("ok").is_empty());
    }
    #[test]
    fn test_aggregate_count_and_bytes() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o1", "u", 100, 1);
        log_event(&mut log, SelEventType::Created, "o2", "u", 200, 2);
        log_event(&mut log, SelEventType::Read, "o1", "u", 50, 3);
        let agg = log.aggregate(&[SelEventType::Created], None);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].count, 2);
        assert_eq!(agg[0].total_bytes, 300);
    }
    #[test]
    fn test_aggregate_first_last_seen() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Read, "o", "u", 10, 100);
        log_event(&mut log, SelEventType::Read, "o", "u", 10, 500);
        log_event(&mut log, SelEventType::Read, "o", "u", 10, 300);
        let agg = log.aggregate(&[SelEventType::Read], None);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].first_seen, 100);
        assert_eq!(agg[0].last_seen, 500);
    }
    #[test]
    fn test_aggregate_unique_objects_users() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Read, "obj1", "alice", 10, 1);
        log_event(&mut log, SelEventType::Read, "obj2", "alice", 10, 2);
        log_event(&mut log, SelEventType::Read, "obj1", "bob", 10, 3);
        let agg = log.aggregate(&[SelEventType::Read], None);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].unique_objects, 2);
        assert_eq!(agg[0].unique_users, 2);
    }
    #[test]
    fn test_aggregate_time_range() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 100, 10);
        log_event(&mut log, SelEventType::Created, "o", "u", 200, 20);
        log_event(&mut log, SelEventType::Created, "o", "u", 300, 30);
        let agg = log.aggregate(&[], Some((15, 25)));
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].count, 1);
        assert_eq!(agg[0].total_bytes, 200);
    }
    #[test]
    fn test_aggregate_all_types() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 10, 1);
        log_event(&mut log, SelEventType::Read, "o", "u", 10, 2);
        log_event(&mut log, SelEventType::Deleted, "o", "u", 10, 3);
        let agg = log.aggregate(&[], None);
        assert_eq!(agg.len(), 3);
    }
    #[test]
    fn test_correlate() {
        let mut log = default_log();
        log_event_corr(
            &mut log,
            SelEventType::BatchStart("b1".to_string()),
            "sys",
            "system",
            0,
            1,
            "b1",
        );
        log_event_corr(&mut log, SelEventType::Created, "o1", "u", 100, 2, "b1");
        log_event(&mut log, SelEventType::Read, "o1", "u", 100, 3);
        log_event_corr(
            &mut log,
            SelEventType::BatchEnd("b1".to_string()),
            "sys",
            "system",
            0,
            4,
            "b1",
        );
        let related = log.correlate("b1");
        assert_eq!(related.len(), 3);
    }
    #[test]
    fn test_correlate_empty() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 1);
        assert!(log.correlate("none").is_empty());
    }
    #[test]
    fn test_retention_keep_all() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::KeepAll,
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 0u64..50 {
            log.log(
                SelEventType::Created,
                "o".to_string(),
                "u".to_string(),
                i,
                vec![],
                None,
                i,
            );
        }
        let purged = log.apply_retention(100);
        assert_eq!(purged, 0);
        assert_eq!(log.event_count(), 50);
    }
    #[test]
    fn test_retention_keep_last() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::KeepLast(3),
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 0u64..10 {
            log.log(
                SelEventType::Created,
                "o".to_string(),
                "u".to_string(),
                i,
                vec![],
                None,
                i,
            );
        }
        let purged = log.apply_retention(100);
        assert_eq!(purged, 7);
        assert_eq!(log.event_count(), 3);
        let s = log.stats();
        assert_eq!(s.oldest_event_ts, Some(7));
        assert_eq!(s.newest_event_ts, Some(9));
    }
    #[test]
    fn test_retention_keep_for_duration() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::KeepForDuration(50),
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 1u64..=10 {
            log.log(
                SelEventType::Read,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i * 10,
            );
        }
        let purged = log.apply_retention(100);
        assert_eq!(purged, 4);
        assert_eq!(log.event_count(), 6);
    }
    #[test]
    fn test_retention_keep_by_type() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::KeepByType({
                let mut m = HashMap::new();
                m.insert("Read".to_string(), 2);
                m.insert("Created".to_string(), 1);
                m
            }),
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 0u64..5 {
            log.log(
                SelEventType::Read,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i,
            );
        }
        for i in 5u64..8 {
            log.log(
                SelEventType::Created,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i,
            );
        }
        log.apply_retention(100);
        let s = log.stats();
        let reads = s
            .events_by_type
            .iter()
            .find(|(t, _)| t == "Read")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        let creates = s
            .events_by_type
            .iter()
            .find(|(t, _)| t == "Created")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert!(reads <= 2, "reads={reads}");
        assert!(creates <= 1, "creates={creates}");
    }
    #[test]
    fn test_retention_size_limited() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::SizeLimited(1000),
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 0u64..10 {
            log.log(
                SelEventType::Read,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i,
            );
        }
        log.apply_retention(100);
        assert!(log.event_count() <= 4, "count={}", log.event_count());
    }
    #[test]
    fn test_retention_increments_purge_counter() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::KeepLast(2),
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 0u64..5 {
            log.log(
                SelEventType::Read,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i,
            );
        }
        log.apply_retention(100);
        assert_eq!(log.stats().retention_purges, 1);
        log.apply_retention(100);
        assert_eq!(log.stats().retention_purges, 1);
    }
    #[test]
    fn test_export() {
        let mut log = default_log();
        for i in 0u64..5 {
            log_event(
                &mut log,
                SelEventType::Created,
                &format!("o{i}"),
                "u",
                i * 10,
                i,
            );
        }
        let q = EventQuery::default();
        let exported = log.export(&q);
        assert_eq!(exported.len(), 5);
        log_event(&mut log, SelEventType::Deleted, "o_extra", "u", 1, 99);
        assert_eq!(exported.len(), 5);
    }
    #[test]
    fn test_export_limit_offset() {
        let mut log = default_log();
        for i in 0u64..10 {
            log_event(&mut log, SelEventType::Read, "o", "u", i, i);
        }
        let q = EventQuery {
            offset: 2,
            limit: 4,
            ..EventQuery::default()
        };
        let exported = log.export(&q);
        assert_eq!(exported.len(), 4);
        assert_eq!(exported[0].timestamp, 2);
    }
    #[test]
    fn test_stats_total_events() {
        let mut log = default_log();
        for _ in 0..7 {
            log_event(&mut log, SelEventType::Read, "o", "u", 10, 1);
        }
        assert_eq!(log.stats().total_events, 7);
    }
    #[test]
    fn test_stats_events_by_type() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 1);
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 2);
        log_event(&mut log, SelEventType::Read, "o", "u", 1, 3);
        let s = log.stats();
        let created = s
            .events_by_type
            .iter()
            .find(|(t, _)| t == "Created")
            .map(|(_, c)| *c);
        let read = s
            .events_by_type
            .iter()
            .find(|(t, _)| t == "Read")
            .map(|(_, c)| *c);
        assert_eq!(created, Some(2));
        assert_eq!(read, Some(1));
    }
    #[test]
    fn test_stats_total_bytes() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 100, 1);
        log_event(&mut log, SelEventType::Created, "o", "u", 250, 2);
        assert_eq!(log.stats().total_bytes_logged, 350);
    }
    #[test]
    fn test_stats_timestamps() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Read, "o", "u", 1, 100);
        log_event(&mut log, SelEventType::Read, "o", "u", 1, 50);
        log_event(&mut log, SelEventType::Read, "o", "u", 1, 200);
        let s = log.stats();
        assert_eq!(s.oldest_event_ts, Some(50));
        assert_eq!(s.newest_event_ts, Some(200));
    }
    #[test]
    fn test_stats_empty() {
        let log = default_log();
        let s = log.stats();
        assert_eq!(s.total_events, 0);
        assert!(s.oldest_event_ts.is_none());
        assert!(s.newest_event_ts.is_none());
        assert_eq!(s.total_bytes_logged, 0);
    }
    #[test]
    fn test_max_events_eviction() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 5,
            retention_policy: RetentionPolicy::KeepAll,
            enable_checksums: false,
            batch_flush_size: 0,
        });
        for i in 0u64..8 {
            log.log(
                SelEventType::Read,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i,
            );
        }
        assert_eq!(log.event_count(), 5);
        assert!(log.get(EventId(1)).is_err());
        assert!(log.get(EventId(2)).is_err());
        assert!(log.get(EventId(3)).is_err());
        assert!(log.get(EventId(4)).is_ok());
    }
    #[test]
    fn test_metadata_preserved() {
        let mut log = default_log();
        let meta = vec![
            ("codec".to_string(), "dag-pb".to_string()),
            ("region".to_string(), "eu-west".to_string()),
        ];
        let id = log.log(
            SelEventType::Created,
            "obj-meta".to_string(),
            "user-meta".to_string(),
            1024,
            meta.clone(),
            None,
            42,
        );
        let evt = log.get(id).expect("must exist");
        assert_eq!(evt.metadata, meta);
    }
    #[test]
    fn test_copied_variant() {
        let mut log = default_log();
        let id = log_event(
            &mut log,
            SelEventType::Copied {
                dest: "dest-obj".to_string(),
            },
            "src-obj",
            "u",
            512,
            1,
        );
        let evt = log.get(id).expect("must exist");
        if let SelEventType::Copied { dest } = &evt.event_type {
            assert_eq!(dest, "dest-obj");
        } else {
            panic!("expected Copied variant");
        }
    }
    #[test]
    fn test_moved_variant() {
        let mut log = default_log();
        let id = log_event(
            &mut log,
            SelEventType::Moved {
                dest: "new-loc".to_string(),
            },
            "old-loc",
            "u",
            0,
            2,
        );
        let evt = log.get(id).expect("ok");
        assert!(matches!(& evt.event_type, SelEventType::Moved { dest } if dest == "new-loc"));
    }
    #[test]
    fn test_versioned_variant() {
        let mut log = default_log();
        let id = log_event(
            &mut log,
            SelEventType::Versioned { version: 7 },
            "obj",
            "u",
            0,
            3,
        );
        let evt = log.get(id).expect("ok");
        assert!(matches!(
            &evt.event_type,
            SelEventType::Versioned { version: 7 }
        ));
    }
    #[test]
    fn test_tiered_variant() {
        let mut log = default_log();
        let id = log_event(
            &mut log,
            SelEventType::Tiered {
                from: "hot".to_string(),
                to: "cold".to_string(),
            },
            "big-obj",
            "admin",
            1_000_000,
            999,
        );
        let evt = log.get(id).expect("ok");
        if let SelEventType::Tiered { from, to } = &evt.event_type {
            assert_eq!(from, "hot");
            assert_eq!(to, "cold");
        } else {
            panic!("expected Tiered");
        }
    }
    #[test]
    fn test_batch_start_end() {
        let mut log = default_log();
        let bid = "batch-99".to_string();
        let id_start = log_event(
            &mut log,
            SelEventType::BatchStart(bid.clone()),
            "sys",
            "sys",
            0,
            1,
        );
        let id_end = log_event(
            &mut log,
            SelEventType::BatchEnd(bid.clone()),
            "sys",
            "sys",
            0,
            100,
        );
        let e_start = log.get(id_start).expect("ok");
        let e_end = log.get(id_end).expect("ok");
        assert!(
            matches!(& e_start.event_type, SelEventType::BatchStart(b) if b ==
            "batch-99")
        );
        assert!(matches!(& e_end.event_type, SelEventType::BatchEnd(b) if b == "batch-99"));
    }
    #[test]
    fn test_system_error_variant() {
        let mut log = default_log();
        let id = log_event(
            &mut log,
            SelEventType::SystemError("disk full".to_string()),
            "sys",
            "storage",
            0,
            1,
        );
        let evt = log.get(id).expect("ok");
        assert!(
            matches!(& evt.event_type, SelEventType::SystemError(msg) if msg ==
            "disk full")
        );
    }
    #[test]
    fn test_query_multiple_event_types() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Created, "o", "u", 1, 1);
        log_event(&mut log, SelEventType::Read, "o", "u", 1, 2);
        log_event(&mut log, SelEventType::Deleted, "o", "u", 1, 3);
        log_event(&mut log, SelEventType::Updated, "o", "u", 1, 4);
        let q = EventQuery {
            event_types: vec![SelEventType::Created, SelEventType::Deleted],
            ..EventQuery::default()
        };
        let res = log.query(&q).expect("ok");
        assert_eq!(res.len(), 2);
    }
    #[test]
    fn test_verify_integrity_multiple_corrupt() {
        let mut log = default_log();
        let id1 = log_event(&mut log, SelEventType::Read, "o1", "u", 10, 1);
        let id2 = log_event(&mut log, SelEventType::Read, "o2", "u", 20, 2);
        let id3 = log_event(&mut log, SelEventType::Read, "o3", "u", 30, 3);
        for evt in log.events.iter_mut() {
            if evt.id == id1 || evt.id == id3 {
                evt.checksum ^= 0xFFFF_FFFF;
            }
        }
        let corrupt = log.verify_integrity();
        assert_eq!(corrupt.len(), 2);
        assert!(corrupt.contains(&id1));
        assert!(corrupt.contains(&id3));
        assert!(!corrupt.contains(&id2));
    }
    #[test]
    fn test_fnv1a_stable() {
        let h1 = sel_fnv1a_64(b"hello");
        let h2 = sel_fnv1a_64(b"hello");
        assert_eq!(h1, h2);
        assert_ne!(sel_fnv1a_64(b"hello"), sel_fnv1a_64(b"world"));
    }
    #[test]
    fn test_event_checksum_deterministic() {
        let mut log = default_log();
        let id = log_event(
            &mut log,
            SelEventType::Created,
            "stable",
            "usr",
            999,
            42_000,
        );
        let evt = log.get(id).expect("ok");
        let c1 = event_checksum(evt);
        let c2 = event_checksum(evt);
        assert_eq!(c1, c2);
        assert_eq!(c1, evt.checksum);
    }
    #[test]
    fn test_export_type_filter() {
        let mut log = default_log();
        for i in 0u64..5 {
            log_event(&mut log, SelEventType::Created, "o", "u", i, i);
            log_event(&mut log, SelEventType::Read, "o", "u", i, i + 100);
        }
        let q = EventQuery {
            event_types: vec![SelEventType::Created],
            ..EventQuery::default()
        };
        let exported = log.export(&q);
        assert_eq!(exported.len(), 5);
        assert!(exported
            .iter()
            .all(|e| matches!(e.event_type, SelEventType::Created)));
    }
    #[test]
    fn test_aggregate_no_match() {
        let mut log = default_log();
        log_event(&mut log, SelEventType::Read, "o", "u", 10, 1);
        let agg = log.aggregate(&[SelEventType::Deleted], None);
        assert!(agg.is_empty());
    }
    #[test]
    fn test_large_log_stress() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 500,
            retention_policy: RetentionPolicy::KeepAll,
            enable_checksums: true,
            batch_flush_size: 0,
        });
        let mut rng = 0xDEAD_BEEF_u64;
        for i in 0u64..1000 {
            let sz = xorshift64(&mut rng) % 4096;
            let ts = i * 100;
            let et = match i % 4 {
                0 => SelEventType::Created,
                1 => SelEventType::Read,
                2 => SelEventType::Updated,
                _ => SelEventType::Deleted,
            };
            log.log(
                et,
                format!("obj-{}", i % 50),
                format!("user-{}", i % 10),
                sz,
                vec![],
                None,
                ts,
            );
        }
        assert_eq!(log.event_count(), 500);
        assert!(log.verify_integrity().is_empty());
    }
    #[test]
    fn test_query_combined_filters() {
        let mut log = default_log();
        log_event_corr(
            &mut log,
            SelEventType::Read,
            "important",
            "alice",
            100,
            50,
            "tx-1",
        );
        log_event_corr(
            &mut log,
            SelEventType::Read,
            "important",
            "alice",
            200,
            150,
            "tx-1",
        );
        log_event_corr(
            &mut log,
            SelEventType::Updated,
            "important",
            "alice",
            300,
            250,
            "tx-1",
        );
        log_event(&mut log, SelEventType::Read, "other", "bob", 400, 100);
        let q = EventQuery {
            event_types: vec![SelEventType::Read],
            object_id: Some("important".to_string()),
            user_id: Some("alice".to_string()),
            time_range: Some((0, 200)),
            correlation_id: Some("tx-1".to_string()),
            limit: 10,
            offset: 0,
        };
        let res = log.query(&q).expect("ok");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|e| e.object_id == "important"));
        assert!(res.iter().all(|e| e.user_id == "alice"));
        assert!(res.iter().all(|e| e.timestamp <= 200));
    }
    #[test]
    fn test_error_display() {
        let e1 = EventLogError::EventNotFound(EventId(42));
        assert!(e1.to_string().contains("42"));
        let e2 = EventLogError::QueryTooExpensive {
            estimated: 2_000_000,
        };
        assert!(e2.to_string().contains("2000000"));
        let e3 = EventLogError::RetentionError("oops".to_string());
        assert!(e3.to_string().contains("oops"));
        let e4 = EventLogError::CorruptedEvent(EventId(7));
        assert!(e4.to_string().contains("7"));
    }
    #[test]
    fn test_event_type_names() {
        assert_eq!(SelEventType::Created.type_name(), "Created");
        assert_eq!(SelEventType::Read.type_name(), "Read");
        assert_eq!(SelEventType::Updated.type_name(), "Updated");
        assert_eq!(SelEventType::Deleted.type_name(), "Deleted");
        assert_eq!(
            SelEventType::Copied {
                dest: "x".to_string()
            }
            .type_name(),
            "Copied"
        );
        assert_eq!(
            SelEventType::Moved {
                dest: "x".to_string()
            }
            .type_name(),
            "Moved"
        );
        assert_eq!(
            SelEventType::Versioned { version: 1 }.type_name(),
            "Versioned"
        );
        assert_eq!(SelEventType::Compressed.type_name(), "Compressed");
        assert_eq!(SelEventType::Encrypted.type_name(), "Encrypted");
        assert_eq!(
            SelEventType::Tiered {
                from: "h".to_string(),
                to: "c".to_string()
            }
            .type_name(),
            "Tiered"
        );
        assert_eq!(
            SelEventType::BatchStart("b".to_string()).type_name(),
            "BatchStart"
        );
        assert_eq!(
            SelEventType::BatchEnd("b".to_string()).type_name(),
            "BatchEnd"
        );
        assert_eq!(
            SelEventType::SystemError("e".to_string()).type_name(),
            "SystemError"
        );
    }
    #[test]
    fn test_compressed_encrypted_variants() {
        let mut log = default_log();
        let ic = log_event(&mut log, SelEventType::Compressed, "obj-c", "u", 512, 1);
        let ie = log_event(&mut log, SelEventType::Encrypted, "obj-e", "u", 512, 2);
        assert!(matches!(
            log.get(ic).expect("ok").event_type,
            SelEventType::Compressed
        ));
        assert!(matches!(
            log.get(ie).expect("ok").event_type,
            SelEventType::Encrypted
        ));
    }
    #[test]
    fn test_batch_flush_auto_retention() {
        let mut log = SelStorageEventLog::new(EventLogConfig {
            max_events: 0,
            retention_policy: RetentionPolicy::KeepLast(5),
            enable_checksums: false,
            batch_flush_size: 10,
        });
        for i in 0u64..20 {
            log.log(
                SelEventType::Read,
                "o".to_string(),
                "u".to_string(),
                0,
                vec![],
                None,
                i,
            );
        }
        assert!(log.event_count() <= 10, "count={}", log.event_count());
    }
}
