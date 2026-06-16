//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_text_doc(id: &str, text: &str) -> MmiIndexedDocument {
        MmiIndexedDocument::new(
            id,
            vec![("body".to_string(), ModalityData::Text(text.to_string()))],
            vec![],
            1_000_000,
        )
    }
    fn make_vec_doc(id: &str, v: Vec<f64>) -> MmiIndexedDocument {
        MmiIndexedDocument::new(
            id,
            vec![("embed".to_string(), ModalityData::Vector(v))],
            vec![],
            1_000_000,
        )
    }
    fn make_struct_doc(id: &str, pairs: Vec<(&str, &str)>) -> MmiIndexedDocument {
        let owned: Vec<(String, String)> = pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        MmiIndexedDocument::new(
            id,
            vec![("meta".to_string(), ModalityData::Structured(owned))],
            vec![],
            1_000_000,
        )
    }
    fn make_multi_doc(
        id: &str,
        text: &str,
        v: Vec<f64>,
        pairs: Vec<(&str, &str)>,
    ) -> MmiIndexedDocument {
        let owned: Vec<(String, String)> = pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        MmiIndexedDocument::new(
            id,
            vec![
                ("body".to_string(), ModalityData::Text(text.to_string())),
                ("embed".to_string(), ModalityData::Vector(v)),
                ("meta".to_string(), ModalityData::Structured(owned)),
            ],
            vec![("author".to_string(), "test".to_string())],
            1_000_000,
        )
    }
    #[test]
    fn test_index_and_get() {
        let mut idx = MultiModalIndexer::with_defaults();
        let doc = make_text_doc("d1", "hello world");
        idx.index_document(doc)
            .expect("test: index_document failed");
        let got = idx.get_document("d1").expect("test: get_document failed");
        assert_eq!(got.id, "d1");
        assert_eq!(got.version, 1);
    }
    #[test]
    fn test_get_missing_returns_error() {
        let idx = MultiModalIndexer::with_defaults();
        match idx.get_document("nope") {
            Err(MmiIndexError::DocumentNotFound(id)) => assert_eq!(id, "nope"),
            other => panic!("unexpected: {other:?}"),
        }
    }
    #[test]
    fn test_remove_existing() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "foo"))
            .expect("test: index_document failed");
        idx.remove_document("d1")
            .expect("test: remove_document failed");
        assert!(matches!(
            idx.get_document("d1"),
            Err(MmiIndexError::DocumentNotFound(_))
        ));
    }
    #[test]
    fn test_remove_missing_returns_error() {
        let mut idx = MultiModalIndexer::with_defaults();
        assert!(matches!(
            idx.remove_document("ghost"),
            Err(MmiIndexError::DocumentNotFound(_))
        ));
    }
    #[test]
    fn test_index_multiple_documents() {
        let mut idx = MultiModalIndexer::with_defaults();
        for i in 0..5 {
            idx.index_document(make_text_doc(&format!("d{i}"), "some text"))
                .expect("test: index_document failed");
        }
        assert_eq!(idx.stats().total_documents, 5);
    }
    #[test]
    fn test_version_increments_on_update() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: first index_document failed");
        idx.index_document(make_text_doc("d1", "world"))
            .expect("test: second index_document failed");
        assert_eq!(
            idx.get_document("d1")
                .expect("test: get_document failed")
                .version,
            2
        );
    }
    #[test]
    fn test_version_increments_multiple_times() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "v1"))
            .expect("test: index v1 failed");
        idx.index_document(make_text_doc("d1", "v2"))
            .expect("test: index v2 failed");
        idx.index_document(make_text_doc("d1", "v3"))
            .expect("test: index v3 failed");
        assert_eq!(
            idx.get_document("d1")
                .expect("test: get_document failed")
                .version,
            3
        );
    }
    fn text_query(q: &str, top_k: usize) -> MmiSearchQuery {
        MmiSearchQuery {
            text_query: Some(q.to_string()),
            top_k,
            ..Default::default()
        }
    }
    #[test]
    fn test_text_search_basic() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "rust systems programming"))
            .expect("test: index d1 failed");
        idx.index_document(make_text_doc("d2", "python machine learning"))
            .expect("test: index d2 failed");
        let results = idx
            .search(&text_query("rust", 5))
            .expect("test: search failed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_text_search_no_match() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello world"))
            .expect("test: index_document failed");
        let results = idx
            .search(&text_query("zzz", 5))
            .expect("test: search failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_text_search_top_k_respected() {
        let mut idx = MultiModalIndexer::with_defaults();
        for i in 0..10 {
            idx.index_document(make_text_doc(&format!("d{i}"), "common term query token"))
                .expect("test: index_document failed");
        }
        let results = idx
            .search(&text_query("common", 3))
            .expect("test: search failed");
        assert!(results.len() <= 3);
    }
    #[test]
    fn test_text_search_scores_descending() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "rust programming"))
            .expect("test: index d1 failed");
        idx.index_document(make_text_doc("d2", "rust rust systems"))
            .expect("test: index d2 failed");
        let results = idx
            .search(&text_query("rust", 5))
            .expect("test: search failed");
        assert!(results.len() >= 2);
        assert!(results[0].score >= results[1].score);
    }
    #[test]
    fn test_text_search_multi_term() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "deep learning neural networks"))
            .expect("test: index d1 failed");
        idx.index_document(make_text_doc("d2", "deep blue chess"))
            .expect("test: index d2 failed");
        let results = idx
            .search(&text_query("deep learning", 5))
            .expect("test: search failed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_text_search_min_score_filters() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "rust programming"))
            .expect("test: index_document failed");
        let query = MmiSearchQuery {
            text_query: Some("rust".to_string()),
            top_k: 5,
            min_score: 999.0,
            ..Default::default()
        };
        let results = idx
            .search(&query)
            .expect("test: search with min_score failed");
        assert!(results.is_empty());
    }
    fn vec_query(v: Vec<f64>, top_k: usize) -> MmiSearchQuery {
        MmiSearchQuery {
            vector_query: Some(v),
            top_k,
            ..Default::default()
        }
    }
    #[test]
    fn test_vector_search_basic() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0, 0.0]))
            .expect("test: index d1 failed");
        idx.index_document(make_vec_doc("d2", vec![0.0, 1.0, 0.0]))
            .expect("test: index d2 failed");
        let results = idx
            .search(&vec_query(vec![1.0, 0.0, 0.0], 5))
            .expect("test: vector search failed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "d1");
        assert!((results[0].score - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_vector_search_near_perfect() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_vec_doc("d1", vec![0.6, 0.8, 0.0]))
            .expect("test: index d1 failed");
        idx.index_document(make_vec_doc("d2", vec![0.0, 0.0, 1.0]))
            .expect("test: index d2 failed");
        let results = idx
            .search(&vec_query(vec![0.6, 0.8, 0.0], 5))
            .expect("test: vector search failed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "d1");
        assert!((results[0].score - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_vector_search_top_k() {
        let mut state = 0xdeadbeef_u64;
        let mut idx = MultiModalIndexer::with_defaults();
        for i in 0..20usize {
            let v: Vec<f64> = (0..4).map(|_| xorshift_f64(&mut state)).collect();
            idx.index_document(make_vec_doc(&format!("d{i}"), v))
                .expect("test: index_document failed");
        }
        let query_v: Vec<f64> = (0..4).map(|_| xorshift_f64(&mut state)).collect();
        let results = idx
            .search(&vec_query(query_v, 5))
            .expect("test: vector search failed");
        assert!(results.len() <= 5);
    }
    #[test]
    fn test_vector_search_scores_descending() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0]))
            .expect("test: index d1 failed");
        idx.index_document(make_vec_doc(
            "d2",
            vec![
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
        ))
        .expect("test: index d2 failed");
        idx.index_document(make_vec_doc("d3", vec![0.0, 1.0]))
            .expect("test: index d3 failed");
        let results = idx
            .search(&vec_query(vec![1.0, 0.0], 5))
            .expect("test: vector search failed");
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
    #[test]
    fn test_vector_dim_mismatch() {
        let cfg = MmiIndexConfig {
            vector_dim: Some(3),
            ..MmiIndexConfig::default()
        };
        let mut idx = MultiModalIndexer::new(cfg);
        let doc = make_vec_doc("d1", vec![1.0, 2.0]);
        assert!(matches!(
            idx.index_document(doc),
            Err(MmiIndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }
    #[test]
    fn test_vector_search_dim_mismatch_on_query() {
        let cfg = MmiIndexConfig {
            vector_dim: Some(3),
            ..MmiIndexConfig::default()
        };
        let mut idx = MultiModalIndexer::new(cfg);
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0, 0.0]))
            .expect("test: index_document failed");
        let result = idx.search(&vec_query(vec![1.0, 0.0], 5));
        assert!(matches!(
            result,
            Err(MmiIndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }
    fn struct_query(filters: Vec<(&str, &str)>, top_k: usize) -> MmiSearchQuery {
        MmiSearchQuery {
            structured_filters: filters
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            top_k,
            ..Default::default()
        }
    }
    #[test]
    fn test_structured_filter_exact_match() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_struct_doc(
            "d1",
            vec![("type", "image"), ("lang", "en")],
        ))
        .expect("test: index d1 failed");
        idx.index_document(make_struct_doc(
            "d2",
            vec![("type", "video"), ("lang", "en")],
        ))
        .expect("test: index d2 failed");
        let results = idx
            .search(&struct_query(vec![("type", "image")], 5))
            .expect("test: structured search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_structured_filter_and_semantics() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_struct_doc("d1", vec![("type", "img"), ("lang", "en")]))
            .expect("test: index d1 failed");
        idx.index_document(make_struct_doc("d2", vec![("type", "img"), ("lang", "fr")]))
            .expect("test: index d2 failed");
        let results = idx
            .search(&struct_query(vec![("type", "img"), ("lang", "en")], 5))
            .expect("test: structured search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_structured_filter_no_match() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_struct_doc("d1", vec![("type", "image")]))
            .expect("test: index_document failed");
        let results = idx
            .search(&struct_query(vec![("type", "audio")], 5))
            .expect("test: structured search failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_structured_filter_multiple_values() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_struct_doc(
            "d1",
            vec![("a", "1"), ("b", "2"), ("c", "3")],
        ))
        .expect("test: index d1 failed");
        idx.index_document(make_struct_doc("d2", vec![("a", "1"), ("b", "9")]))
            .expect("test: index d2 failed");
        let results = idx
            .search(&struct_query(vec![("a", "1"), ("b", "2"), ("c", "3")], 5))
            .expect("test: structured search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_combined_text_and_vector() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(MmiIndexedDocument::new(
            "d1",
            vec![
                (
                    "body".to_string(),
                    ModalityData::Text("rust programming language".to_string()),
                ),
                ("embed".to_string(), ModalityData::Vector(vec![0.9, 0.1])),
            ],
            vec![],
            0,
        ))
        .expect("test: index d1 failed");
        idx.index_document(MmiIndexedDocument::new(
            "d2",
            vec![
                (
                    "body".to_string(),
                    ModalityData::Text("java language".to_string()),
                ),
                ("embed".to_string(), ModalityData::Vector(vec![1.0, 0.0])),
            ],
            vec![],
            0,
        ))
        .expect("test: index d2 failed");
        let query = MmiSearchQuery {
            text_query: Some("rust".to_string()),
            vector_query: Some(vec![1.0, 0.0]),
            top_k: 5,
            ..Default::default()
        };
        let results = idx.search(&query).expect("test: combined search failed");
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.doc_id == "d1"));
        assert!(results.iter().any(|r| r.doc_id == "d2"));
    }
    #[test]
    fn test_combined_all_three_modalities() {
        let mut idx = MultiModalIndexer::with_defaults();
        let doc = make_multi_doc("d1", "rust systems", vec![1.0, 0.0], vec![("lang", "en")]);
        idx.index_document(doc)
            .expect("test: index_document failed");
        let query = MmiSearchQuery {
            text_query: Some("rust".to_string()),
            vector_query: Some(vec![1.0, 0.0]),
            structured_filters: vec![("lang".to_string(), "en".to_string())],
            top_k: 5,
            ..Default::default()
        };
        let results = idx
            .search(&query)
            .expect("test: combined three-modality search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
        assert_eq!(results[0].score_breakdown.len(), 3);
    }
    #[test]
    fn test_combined_structured_prunes_text_results() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_multi_doc(
            "d1",
            "rust programming",
            vec![1.0, 0.0],
            vec![("lang", "en")],
        ))
        .expect("test: index d1 failed");
        idx.index_document(make_multi_doc(
            "d2",
            "rust programming",
            vec![1.0, 0.0],
            vec![("lang", "fr")],
        ))
        .expect("test: index d2 failed");
        let query = MmiSearchQuery {
            text_query: Some("rust".to_string()),
            structured_filters: vec![("lang".to_string(), "en".to_string())],
            top_k: 5,
            ..Default::default()
        };
        let results = idx
            .search(&query)
            .expect("test: structured-pruned text search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_update_modality_text() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello world"))
            .expect("test: index_document failed");
        assert!(!idx
            .search(&text_query("hello", 5))
            .expect("test: search hello failed")
            .is_empty());
        idx.update_modality(
            "d1",
            "body".to_string(),
            ModalityData::Text("foo bar baz".to_string()),
        )
        .expect("test: update_modality failed");
        assert!(idx
            .search(&text_query("hello", 5))
            .expect("test: search hello after update failed")
            .is_empty());
        assert!(!idx
            .search(&text_query("foo", 5))
            .expect("test: search foo after update failed")
            .is_empty());
    }
    #[test]
    fn test_update_modality_version_increments() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index_document failed");
        idx.update_modality(
            "d1",
            "body".to_string(),
            ModalityData::Text("world".to_string()),
        )
        .expect("test: update_modality failed");
        assert_eq!(
            idx.get_document("d1")
                .expect("test: get_document failed")
                .version,
            2
        );
    }
    #[test]
    fn test_update_modality_adds_new_slot() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index_document failed");
        idx.update_modality(
            "d1",
            "embed".to_string(),
            ModalityData::Vector(vec![1.0, 0.0]),
        )
        .expect("test: update_modality failed");
        let doc = idx.get_document("d1").expect("test: get_document failed");
        assert_eq!(doc.modalities.len(), 2);
    }
    #[test]
    fn test_update_modality_missing_doc() {
        let mut idx = MultiModalIndexer::with_defaults();
        assert!(matches!(
            idx.update_modality(
                "ghost",
                "body".to_string(),
                ModalityData::Text("x".to_string())
            ),
            Err(MmiIndexError::DocumentNotFound(_))
        ));
    }
    #[test]
    fn test_update_modality_vector() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0]))
            .expect("test: index_document failed");
        idx.update_modality(
            "d1",
            "embed".to_string(),
            ModalityData::Vector(vec![0.0, 1.0]),
        )
        .expect("test: update_modality failed");
        let results = idx
            .search(&vec_query(vec![0.0, 1.0], 5))
            .expect("test: vector search failed");
        assert_eq!(results[0].doc_id, "d1");
        assert!((results[0].score - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_documents_with_modality_basic() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index d1 failed");
        idx.index_document(make_vec_doc("d2", vec![1.0, 0.0]))
            .expect("test: index d2 failed");
        let text_docs = idx.documents_with_modality("body");
        assert_eq!(text_docs.len(), 1);
        assert_eq!(text_docs[0].id, "d1");
        let vec_docs = idx.documents_with_modality("embed");
        assert_eq!(vec_docs.len(), 1);
        assert_eq!(vec_docs[0].id, "d2");
    }
    #[test]
    fn test_documents_with_modality_none() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index_document failed");
        let docs = idx.documents_with_modality("nonexistent_modality");
        assert!(docs.is_empty());
    }
    #[test]
    fn test_documents_with_modality_multiple() {
        let mut idx = MultiModalIndexer::with_defaults();
        for i in 0..5 {
            idx.index_document(make_text_doc(&format!("d{i}"), "text"))
                .expect("test: index_document failed");
        }
        let docs = idx.documents_with_modality("body");
        assert_eq!(docs.len(), 5);
    }
    #[test]
    fn test_modalities_required_filter() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(MmiIndexedDocument::new(
            "d1",
            vec![
                ("body".to_string(), ModalityData::Text("rust".to_string())),
                ("embed".to_string(), ModalityData::Vector(vec![1.0, 0.0])),
            ],
            vec![],
            0,
        ))
        .expect("test: index d1 failed");
        idx.index_document(make_text_doc("d2", "rust"))
            .expect("test: index d2 failed");
        let query = MmiSearchQuery {
            text_query: Some("rust".to_string()),
            modalities_required: vec!["embed".to_string()],
            top_k: 5,
            ..Default::default()
        };
        let results = idx
            .search(&query)
            .expect("test: modalities_required search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_stats_empty() {
        let idx = MultiModalIndexer::with_defaults();
        let s = idx.stats();
        assert_eq!(s.total_documents, 0);
        assert_eq!(s.search_count, 0);
    }
    #[test]
    fn test_stats_after_indexing() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index_document failed");
        idx.index_document(make_vec_doc("d2", vec![1.0, 0.0]))
            .expect("test: index d2 failed");
        let s = idx.stats();
        assert_eq!(s.total_documents, 2);
        assert!(s.index_size_estimate_bytes > 0);
    }
    #[test]
    fn test_stats_search_count() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index_document failed");
        idx.search(&text_query("hello", 5))
            .expect("test: first search failed");
        idx.search(&text_query("world", 5))
            .expect("test: second search failed");
        assert_eq!(idx.stats().search_count, 2);
    }
    #[test]
    fn test_stats_modality_counts() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "a"))
            .expect("test: index d1 failed");
        idx.index_document(make_text_doc("d2", "b"))
            .expect("test: index d2 failed");
        idx.index_document(make_vec_doc("d3", vec![1.0]))
            .expect("test: index d3 failed");
        let s = idx.stats();
        let body_count = s
            .modality_counts
            .iter()
            .find(|(name, _)| name == "body")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        let embed_count = s
            .modality_counts
            .iter()
            .find(|(name, _)| name == "embed")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(body_count, 2);
        assert_eq!(embed_count, 1);
    }
    #[test]
    fn test_stats_avg_modalities() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_multi_doc("d1", "text", vec![1.0], vec![("k", "v")]))
            .expect("test: index_document failed");
        let s = idx.stats();
        assert!((s.avg_modalities_per_doc - 3.0).abs() < 1e-9);
    }
    #[test]
    fn test_max_documents_exceeded() {
        let cfg = MmiIndexConfig {
            max_documents: 2,
            ..MmiIndexConfig::default()
        };
        let mut idx = MultiModalIndexer::new(cfg);
        idx.index_document(make_text_doc("d1", "a"))
            .expect("test: index d1 failed");
        idx.index_document(make_text_doc("d2", "b"))
            .expect("test: index d2 failed");
        assert!(matches!(
            idx.index_document(make_text_doc("d3", "c")),
            Err(MmiIndexError::MaxDocumentsExceeded)
        ));
    }
    #[test]
    fn test_max_documents_allows_update() {
        let cfg = MmiIndexConfig {
            max_documents: 1,
            ..MmiIndexConfig::default()
        };
        let mut idx = MultiModalIndexer::new(cfg);
        idx.index_document(make_text_doc("d1", "a"))
            .expect("test: first index d1 failed");
        idx.index_document(make_text_doc("d1", "b"))
            .expect("test: second index d1 failed");
        assert_eq!(idx.stats().total_documents, 1);
    }
    #[test]
    fn test_index_error_display_document_not_found() {
        let e = MmiIndexError::DocumentNotFound("x".to_string());
        assert!(e.to_string().contains("x"));
    }
    #[test]
    fn test_index_error_display_dimension_mismatch() {
        let e = MmiIndexError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        let s = e.to_string();
        assert!(s.contains("128") && s.contains("64"));
    }
    #[test]
    fn test_index_error_display_max_exceeded() {
        let e = MmiIndexError::MaxDocumentsExceeded;
        assert!(!e.to_string().is_empty());
    }
    #[test]
    fn test_empty_query_returns_empty() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello"))
            .expect("test: index_document failed");
        let query = MmiSearchQuery {
            top_k: 5,
            ..Default::default()
        };
        let results = idx.search(&query).expect("test: empty query search failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_search_empty_index() {
        let mut idx = MultiModalIndexer::with_defaults();
        let results = idx
            .search(&text_query("hello", 5))
            .expect("test: search on empty index failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_binary_modality_stored() {
        let mut idx = MultiModalIndexer::with_defaults();
        let doc = MmiIndexedDocument::new(
            "d1",
            vec![("blob".to_string(), ModalityData::Binary(vec![0xde, 0xad]))],
            vec![],
            0,
        );
        idx.index_document(doc)
            .expect("test: index_document failed");
        let got = idx.get_document("d1").expect("test: get_document failed");
        assert!(
            matches!(& got.modalities[0].1, ModalityData::Binary(b) if b == & [0xde,
            0xad])
        );
    }
    #[test]
    fn test_numeric_modality_indexed_as_structured() {
        let mut idx = MultiModalIndexer::with_defaults();
        let doc = MmiIndexedDocument::new(
            "d1",
            vec![("score".to_string(), ModalityData::Numeric(42.0))],
            vec![],
            0,
        );
        idx.index_document(doc)
            .expect("test: index_document failed");
        let results = idx
            .search(&struct_query(vec![("score", "42")], 5))
            .expect("test: numeric structured search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }
    #[test]
    fn test_score_breakdown_present() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello world"))
            .expect("test: index_document failed");
        let results = idx
            .search(&text_query("hello", 5))
            .expect("test: search failed");
        assert!(!results.is_empty());
        assert!(!results[0].score_breakdown.is_empty());
        assert!(!results[0].matched_modalities.is_empty());
    }
    #[test]
    fn test_vector_score_breakdown_present() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0]))
            .expect("test: index_document failed");
        let results = idx
            .search(&vec_query(vec![1.0, 0.0], 5))
            .expect("test: vector search failed");
        assert!(!results.is_empty());
        assert!(results[0]
            .score_breakdown
            .iter()
            .any(|(name, _)| name == "vector"));
    }
    #[test]
    fn test_remove_clears_text_index() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_text_doc("d1", "hello world"))
            .expect("test: index_document failed");
        idx.remove_document("d1")
            .expect("test: remove_document failed");
        let results = idx
            .search(&text_query("hello", 5))
            .expect("test: search after remove failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_remove_clears_vector_index() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0]))
            .expect("test: index_document failed");
        idx.remove_document("d1")
            .expect("test: remove_document failed");
        let results = idx
            .search(&vec_query(vec![1.0, 0.0], 5))
            .expect("test: vector search after remove failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_remove_clears_structured_index() {
        let mut idx = MultiModalIndexer::with_defaults();
        idx.index_document(make_struct_doc("d1", vec![("type", "image")]))
            .expect("test: index_document failed");
        idx.remove_document("d1")
            .expect("test: remove_document failed");
        let results = idx
            .search(&struct_query(vec![("type", "image")], 5))
            .expect("test: structured search after remove failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![0.6, 0.8];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-12);
    }
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-12);
    }
    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    #[test]
    fn test_cosine_similarity_dim_mismatch() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    #[test]
    fn test_large_scale_vector_search() {
        let mut state = 0xc0ffee_u64;
        let mut idx = MultiModalIndexer::with_defaults();
        let dim = 16;
        for i in 0..200usize {
            let v: Vec<f64> = (0..dim).map(|_| xorshift_f64(&mut state)).collect();
            idx.index_document(make_vec_doc(&format!("d{i}"), v))
                .expect("test: index_document failed");
        }
        let query_v: Vec<f64> = (0..dim).map(|_| xorshift_f64(&mut state)).collect();
        let results = idx
            .search(&vec_query(query_v, 10))
            .expect("test: large-scale vector search failed");
        assert!(results.len() <= 10);
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0 + 1e-9);
        }
    }
    #[test]
    fn test_large_scale_text_search() {
        let corpus = [
            "rust systems programming fast",
            "python machine learning data",
            "java enterprise backend services",
            "javascript frontend browser dom",
            "haskell functional type theory",
        ];
        let mut idx = MultiModalIndexer::with_defaults();
        for (i, text) in corpus.iter().enumerate() {
            idx.index_document(make_text_doc(&format!("d{i}"), text))
                .expect("test: index_document failed");
        }
        let results = idx
            .search(&text_query("rust", 5))
            .expect("test: text search failed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "d0");
    }
    #[test]
    fn test_disabled_text_index() {
        let cfg = MmiIndexConfig {
            enable_text_index: false,
            ..Default::default()
        };
        let mut idx = MultiModalIndexer::new(cfg);
        idx.index_document(make_text_doc("d1", "hello world"))
            .expect("test: index_document failed");
        let results = idx
            .search(&text_query("hello", 5))
            .expect("test: search with disabled text index failed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_disabled_vector_index() {
        let cfg = MmiIndexConfig {
            enable_vector_index: false,
            ..Default::default()
        };
        let mut idx = MultiModalIndexer::new(cfg);
        idx.index_document(make_vec_doc("d1", vec![1.0, 0.0]))
            .expect("test: index_document failed");
        let results = idx
            .search(&vec_query(vec![1.0, 0.0], 5))
            .expect("test: vector search with disabled vector index failed");
        assert!(results.is_empty());
    }
}
