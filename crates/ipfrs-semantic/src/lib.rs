#![doc = include_str!("CRATE_DOCS.md")]

pub mod adapters;
pub mod analytics;
pub mod auto_scaling;
pub mod benchmark_comparison;
pub mod cache;
pub mod cross_encoder;
pub mod dht;
pub mod dht_node;
pub mod diagnostics;
pub mod diskann;
pub mod drift_detector;
pub mod dynamic;
pub mod federated;
pub mod hnsw;
pub mod hybrid;
pub mod kb_query;
pub mod learned;
pub mod metadata;
pub mod migration;
pub mod multimodal;
pub mod optimization;
pub mod persistence;
pub mod privacy;
pub mod prod_tests;
pub mod provenance;
pub mod query_cache;
pub use query_cache::{CachedQueryResult, QueryCacheConfig, QueryCacheStats, SemanticQueryCache};
pub mod query_planner;
pub mod query_rewriter;
pub use query_rewriter::{
    QueryRewriter, QueryRewriterConfig, QueryRewriterStats, RewriteResult, RewriteRule,
    RewriteRuleType, RewrittenTerm,
};
pub mod federated_search;
pub mod index_compactor;
pub mod index_merger;
pub mod index_partitioner;
pub mod index_rebalancer;
pub mod partial_sync;
pub mod quantization;
pub mod regression;
pub mod reranking;
pub mod result_aggregator;
pub mod router;
pub mod shard_balancer;
pub mod shard_coordinator;
pub mod simd;
pub mod solver;
pub mod stats;
pub mod utils;
pub mod vector_quality;

// Core vector index exports
pub use hnsw::{
    BuildHealthStats, DistanceMetric, IncrementalBuildStats, ParameterRecommendation,
    ParameterTuner, RebuildStats, SearchResult, UseCase, VectorIndex,
};

// Router exports
pub use router::{
    BatchStats, CacheStats, IndexBackend, QueryFilter, RouterConfig, RouterStats, SemanticRouter,
};

// Hybrid search exports
pub use hybrid::{
    FilterStrategy, HybridConfig, HybridIndex, HybridQuery, HybridResponse, HybridResult,
    PruningStats,
};

// Metadata exports
pub use metadata::{Metadata, MetadataFilter, MetadataStore, MetadataValue, TemporalOptions};

// Quantization exports
pub use quantization::{
    dequantize_i8_to_f32, quantize_f32_to_i8, BinaryVectorStore, OptimizedProductQuantizer, PQCode,
    ProductQuantizer, QuantizationBenchmark, QuantizationBenchmarker, QuantizationComparison,
    QuantizedVector, QuantizedVectorStore, ScalarQuantizer,
};

// Statistics exports
pub use stats::{IndexHealth, IndexStats, MemoryUsage, PerfTimer, StatsSnapshot};

// Result aggregator exports
pub use result_aggregator::{
    AggregatedResult, AggregationStrategy as AggAggregationStrategy, AggregatorConfig,
    AggregatorStats, ResultAggregator, SearchResult as AggSearchResult,
};

// DiskANN exports
pub use diskann::SearchResult as DiskANNSearchResult;
pub use diskann::{CompactionStats, DiskANNConfig, DiskANNIndex, DiskANNStats};

// Solver exports (Logic Integration)
pub use solver::{
    LogicSolver, PredicateEmbedder, ProofSearch, ProofTreeNode, SolverConfig, SolverStats,
};

// Knowledge Base Query exports
pub use kb_query::{
    BooleanQuery, FilterExpr, Query, QueryExecutor, QueryPattern, QueryResult, QueryStats,
    TermPattern, TermType,
};

// Provenance exports
pub use provenance::{
    AuditLogEntry, AuditOperation, EmbeddingMetadata, EmbeddingSource, EmbeddingVersion,
    FeatureAttribution, ProvenanceStats, ProvenanceTracker, SearchExplanation, VersionHistory,
};

// SIMD exports (Performance optimization)
pub use simd::{cosine_distance, dot_product, l2_distance};

// Cache exports (Advanced caching)
pub use cache::{
    AdaptiveCacheStrategy, AlignedVector, CacheInvalidator, HotCacheStats, HotEmbeddingCache,
    InvalidationPolicy,
};

// Multi-modal exports (Multi-Modal Search)
pub use multimodal::{
    Modality, ModalityAlignment, ModalityStats, MultiModalConfig, MultiModalEmbedding,
    MultiModalIndex,
};

// Privacy exports (Differential Privacy)
pub use privacy::{
    NoiseDistribution, PrivacyBudget, PrivacyBudgetStats, PrivacyMechanism, PrivateEmbedding,
    QueryRecord, TradeoffAnalyzer, TradeoffPoint,
};

// Dynamic embedding exports (Dynamic Updates)
pub use dynamic::{
    DynamicIndex, EmbeddingTransform, ModelVersion, OnlineUpdater, OnlineUpdaterStats, VersionStats,
};

// Distributed DHT exports (Distributed Semantic Search)
pub use dht::{
    DHTQuery, DHTQueryResponse, ReplicationStrategy, SemanticDHTConfig, SemanticDHTStats,
    SemanticPeer, SemanticRoutingTable,
};
pub use dht_node::{SemanticDHTNode, SyncStats};

// Federated Query exports (Multi-Index Search)
pub use federated::{
    AggregationStrategy, FederatedConfig, FederatedQueryExecutor, FederatedQueryStats,
    FederatedSearchResult, LocalIndexAdapter, QueryableIndex,
};

// Re-ranking exports (Query Result Re-ranking)
pub use reranking::{ReRanker, ReRankingConfig, ReRankingStrategy, ScoreComponent, ScoredResult};

// Analytics exports (Query Analytics and Performance Tracking)
pub use analytics::{
    AnalyticsSummary, AnalyticsTracker, DetectedPattern, QueryMetrics, QueryTimer,
};

// Auto-Scaling Advisor exports (Production Operations)
pub use auto_scaling::{
    ActionType, AdvisorConfig, AutoScalingAdvisor, ScalingAction, ScalingRecommendations,
    TrendReport, WorkloadMetrics,
};

// Learned Index exports (ML-Based Indexing)
pub use learned::{LearnedIndex, LearnedIndexStats, ModelType, RMIConfig};

// Vector Database Adapter exports (External Integration)
pub use adapters::{
    BackendConfig, BackendMigration, BackendRegistry, BackendSearchResult, BackendStats,
    IpfrsBackend, MigrationStats, VectorBackend,
};

// Vector Quality Analysis exports (Quality Validation and Anomaly Detection)
pub use vector_quality::{
    analyze_quality, compute_batch_stats, compute_diversity, compute_stats, cosine_similarity,
    detect_anomaly, find_outliers, AnomalyReport, AnomalyType, VectorQuality, VectorStats,
};

// Diagnostics exports (Index Health Monitoring and Performance Profiling)
pub use diagnostics::{
    diagnose_index, DiagnosticIssue, DiagnosticReport, HealthMonitor, HealthStatus, IssueCategory,
    IssueSeverity, PerformanceMetrics, ProfilerStats, SearchProfiler,
};

// Optimization exports (Index Optimization and Resource Management)
pub use optimization::{
    analyze_optimization, MemoryOptimizer, OptimizationGoal, OptimizationResult, QueryOptimizer,
};

// Utility exports (Helper Functions and Common Workflows)
pub use utils::{
    average_embedding, create_hybrid_index_from_map, health_check, index_with_quality_check,
    normalize_vector, normalize_vectors, validate_embeddings, BatchEmbeddingStats,
    BatchIndexResult, HealthCheckResult,
};

// Production Testing exports (Stress Testing and Endurance Testing)
pub use prod_tests::{
    EnduranceTest, EnduranceTestConfig, EnduranceTestResults, StressTest, StressTestConfig,
    StressTestResults,
};

// Performance Regression Detection exports (Regression Testing)
pub use regression::{
    MetricSummary, RegressionConfig, RegressionDetector, RegressionIssue, RegressionReport,
};

// Benchmark Comparison exports (Configuration Comparison and Parameter Tuning)
pub use benchmark_comparison::{
    BenchmarkResult, BenchmarkSuite, ComparisonReport, IndexConfig, ParameterSweep,
};

// Index Migration exports (Index Type Migration and Configuration Updates)
pub use migration::{
    BatchMigration, ConfigMigration, DimensionMigration, IndexMigration, MetricMigration,
    MigrationConfig, MigrationProgress,
};

// Shard balancer exports (HNSW-on-DHT shard balancing)
pub use shard_balancer::{DhtShardRouter, ShardAssignment, ShardBalancer, ShardConfig};

// Shard coordinator exports (consistent-hash vector distribution for 1M+ vectors)
pub use shard_coordinator::{
    ConsistentHashRing, ShardCoordinator, ShardError, ShardId, ShardStats, ShardStatsSnapshot,
    VectorShard,
};

// Index Persistence exports (HNSW Snapshot Serialization + incremental snapshots)
pub use persistence::{
    IncrementalSnapshot, IncrementalTracker, IndexEntry, IndexPersistence, IndexSnapshot,
};

// Partial sync / dirty region tracking
pub use partial_sync::{DirtyRegionTracker, EmbeddingDelta, EmbeddingRegion, PartialSyncManager};

// Index Compactor exports (HNSW fragmentation detection and rebuild coordination)
pub use index_compactor::{
    CompactionPlan, CompactionPolicy, CompactionPriority, CompactionReason, CompactorStats,
    CompactorStatsSnapshot, IndexCompactor, IndexFragmentStats,
};

// Federated Search Coordinator — cross-node vector similarity search
pub use federated_search::{
    CachedSearchResult, FederatedSearchCoordinator, FederatedSearchStats,
    FederatedSearchStatsSnapshot, QueryKey, SearchPeer, SearchResult as PeerSearchResult,
};

pub mod embedding_normalizer;
pub use embedding_normalizer::{
    EmbeddingNormalizer, NormStats, NormalizationType, NormalizerConfig, NormalizerStats,
};

// Embedding Pipeline — preprocess raw content into normalised vectors
pub mod embedding_pipeline;
pub use embedding_pipeline::{
    fnv1a_hash_f32, EmbeddingInput, EmbeddingPipeline, EmbeddingPipelineConfig,
    NormalizationStrategy, PipelineError, PipelineResult, PipelineStage, PipelineStats,
    PipelineStatsSnapshot, SemanticEmbeddingPipeline, SemanticPipelineStats,
};

pub mod quantization_error;
pub use quantization_error::{QErrorError, QuantizationError, QuantizationErrorTracker};

// Search Quality Evaluation — Recall@K, Precision@K, NDCG@K, AP, RR
pub mod search_quality;
pub use search_quality::{
    EvalError, EvaluatorStats, EvaluatorStatsSnapshot, GroundTruth, QualityMetrics,
    SearchQualityEvaluator, SearchResultSet,
};

pub mod search_explainer;
pub use search_explainer::{
    ExplainerConfig, ExplainerStats, ExplanationNode, QueryContext, ScoreContribution,
    SearchExplainer,
};

// Vector Search Re-Ranker — multi-signal scoring (similarity, recency, tag overlap, peer reliability)
pub mod search_ranker;
pub use search_ranker::{
    // SemanticSearchRanker and associated types
    RankSignal,
    RankedResult,
    RankerConfig,
    RankerStats,
    RankingSignal,
    RawCandidate,
    SearchCandidate,
    SemanticRankedResult,
    SemanticRankerConfig,
    SemanticSearchRanker,
    VectorSearchRanker,
};

// Two-level LFU/TTL similarity score cache for k-NN searches
pub mod similarity_cache;

// Pairwise cosine-similarity cache with LFU eviction and tick-based TTL
pub mod similarity_cache_v2;

// Vector Anomaly Detector — z-score and isolation-score detection
pub mod anomaly_detector;
pub use anomaly_detector::{
    AnomalyConfig, AnomalyDetectorStats, AnomalyMethod, AnomalyResult, DetectorConfig,
    DetectorStats, SemanticAnomalyDetector, SemanticAnomalyMethod, SemanticAnomalyResult,
    VectorAnomalyDetector,
};

// Embedding Drift Monitor — concept drift detection via normalised deviation
pub mod drift_monitor;
pub use drift_monitor::{
    BaselineStats, DriftMonitorConfig, DriftMonitorStats, DriftSignal, EmbeddingDriftMonitor,
};

// Semantic Cluster Analyzer — k-means++ style cluster analysis over embedding vectors
pub mod cluster_analyzer;
pub use cluster_analyzer::{
    AnalyzerConfig, Cluster, ClusterPoint, ClusterStats, SemanticClusterAnalyzer,
};

// Product Quantization for compressing high-dimensional vectors into compact codes
pub mod vector_quantizer;

// HNSW index structure analysis and parameter tuning recommendations
pub mod index_optimizer;

// Relevance feedback loop: signal collection and score boosting
pub mod feedback_loop;

// Multi-Modal Search Coordinator — cross-modality result fusion and deduplication
pub mod multimodal_search;
pub use multimodal_search::{
    CoordinatorStats, FusedResult, FusionStrategy, Modality as SearchModality, ModalityResult,
    MultiModalSearchCoordinator, SearchQuery,
};
pub use vector_quantizer::{
    Codebook, QuantizationConfig, QuantizationStats, QuantizerCode, VectorQuantizer, VqError,
};

// Semantic Personalizer — per-user interest profile management and search result biasing
pub mod personalizer;

// Embedding Composer — late-fusion of multiple embeddings into a single representation
pub mod embedding_composer;
pub use personalizer::{
    InteractionRecord, InteractionType, PersonalizationBias, SemanticPersonalizer, UserProfile,
};

// Semantic Tag Extractor — similarity-based tag assignment with TF-IDF-like scoring
pub mod tag_extractor;
pub use tag_extractor::{
    ExtractionConfig, ExtractorStats, SemanticTagExtractor, Tag, TagAssignment,
};

// Semantic Graph Linker — builds a similarity graph over embeddings
pub mod graph_linker;
pub use graph_linker::{
    EdgeType, GraphLinkerStats, GraphNode, LinkerConfig, SemanticEdge, SemanticGraphLinker,
};

// Semantic Content Router — routes queries to most relevant nodes/shards
pub mod content_router;
pub use content_router::{
    RouteScore, RouterConfig as ContentRouterConfig, RouterStats as ContentRouterStats,
    RoutingDecision, SemanticContentRouter, TopicEmbedding,
};

// Semantic Hotspot Detector — detects frequently queried regions in embedding space
pub mod hotspot_detector;
pub use hotspot_detector::{
    cosine_sim as hotspot_cosine_sim, HotspotConfig, HotspotRegion, HotspotStats, QueryHit,
    SemanticHotspotDetector,
};

// Semantic Query Expander — generates synonyms, paraphrases, and sub-queries to improve recall
pub mod query_expander;
pub use query_expander::{
    ExpandedQuery, ExpanderStats, ExpansionStrategy, SemanticQueryExpander, TermEntry,
    TermRelation, VectorExpandedQuery, VectorExpanderConfig, VectorExpanderStats,
    VectorQueryExpander, VectorQueryExpansion,
};

// Semantic Near-Duplicate Detector — LSH-based sub-linear near-dup detection
pub mod near_dup_detector;
pub use near_dup_detector::{
    cosine_sim as near_dup_cosine_sim, DupCandidate, DupDetectorStats, DuplicatePair, LshBand,
    MinHashConfig, MinHashNearDupDetector, MinHashSignature, NearDupConfig, NearDupDetectorStats,
    SemanticNearDupDetector,
};

// Semantic Concept Hierarchy — DAG-based concept ontology with IsA / RelatedTo / OppositeOf edges
pub mod concept_hierarchy;
pub use concept_hierarchy::{
    ConceptEdge, ConceptNode, ConceptRelation, HierarchyStats, SemanticConceptHierarchy,
};

// Concept and Keyword Extraction — TF-IDF and frequency-based concept extraction
pub mod concept_extractor;
pub use concept_extractor::{
    Concept, ConceptExtractor, ConceptType, ExtractorConfig as ConceptExtractorConfig,
    ExtractorStats as ConceptExtractorStats,
};

// Semantic Topic Modeller — online clustering for latent topic modelling
pub mod topic_modeler;
pub use topic_modeler::{
    cosine_sim as topic_cosine_sim,
    // LDA-based TopicModeler
    DocumentTopics,
    LdaTopic,
    ModelDocument,
    ModellerConfig,
    SemanticTopicModeller,
    TopicAssignment,
    TopicModel,
    TopicModelConfig,
    TopicModelError,
    TopicModelResult,
    TopicModeler,
    TopicModelerStats,
    TopicModellerStats,
    TopicWord,
};

// Semantic Query Pipeline — composable multi-stage query processing
pub mod query_pipeline;
pub use query_pipeline::{
    PipelineConfig, PipelineRun, PipelineStageKind, PipelineStats as QueryPipelineStats,
    QueryResult as PipelineQueryResult, SemanticQueryPipeline, StageMetrics,
};

// Semantic Knowledge Graph — multi-hop semantic reasoning over entity/concept graphs
pub mod knowledge_graph;
pub use knowledge_graph::{
    cosine_sim as knowledge_graph_cosine_sim, EntityKind, GraphEdge, GraphEntity, GraphQuery,
    KnowledgeGraphStats, SemanticKnowledgeGraph,
};

pub mod entity_linker;
pub use entity_linker::{
    cosine_sim, KbEntity, LinkedMention, LinkerConfig as EntityLinkerConfig, LinkerStats,
    MentionKind, SemanticEntityLinker,
};

pub mod entity_resolution;
pub use entity_resolution::{
    CanonicalEntity, EntityMention, EntityResolver, EntityType, ResolutionMethod, ResolutionResult,
    ResolverConfig, ResolverStats,
};

pub mod relevance_feedback;
pub use relevance_feedback::{
    cosine_similarity as relevance_cosine_similarity, FeedbackItem, FeedbackLabel, FeedbackSession,
    FeedbackStats, RocchioConfig, SemanticRelevanceFeedback,
};

// Semantic Diversifier — Maximal Marginal Relevance (MMR) result diversification
pub mod diversifier;
pub use diversifier::{
    cosine_similarity as diversifier_cosine_similarity, DiversificationCandidate,
    DiversifiedResult, DiversifierConfig, DiversifierStats, SemanticDiversifier,
};

// Semantic Synonym Expander — weighted synonym graph for vocabulary expansion in semantic search
pub mod synonym_expander;
pub use synonym_expander::{
    ExpandedTerm, ExpanderConfig as SynonymExpanderConfig, SemanticSynonymExpander, SynonymEdge,
    SynonymExpanderStats, SynonymRelation,
};

// Semantic Cluster Manager — online k-means-style document clustering with drift detection
pub mod cluster_manager;
pub use cluster_manager::{
    euclidean_distance as cluster_euclidean_distance,
    vec_mean as cluster_vec_mean,
    BatchCluster,
    // Batch k-means clustering
    BatchClusterConfig,
    BatchClusterManagerStats,
    BatchSemanticClusterManager,
    ClusterAssignment,
    ClusterManagerConfig,
    ClusterManagerStats,
    SemanticCluster,
    SemanticClusterManager,
};

pub mod document_summarizer;
pub use document_summarizer::{
    cosine_similarity as ds_cosine_similarity,
    split_sentences as ds_split_sentences,
    tf_idf as ds_tf_idf,
    tokenize as ds_tokenize,
    xorshift64 as ds_xorshift64,
    DocumentChunk,
    DocumentSummarizer,
    // Renamed to avoid collision with text_summarizer::{SentenceScore, SummarizerConfig, SummarizerError}
    SentenceScore as DsSentenceScore,
    SummarizerConfig as DsSummarizerConfig,
    SummarizerError as DsSummarizerError,
    SummarizerStats,
    SummaryResult,
    SummaryStyle,
};

pub mod intent_classifier;
pub use intent_classifier::{
    ClassifierConfig as IntentClassifierConfig, ClassifierStats as IntentClassifierStats,
    IntentClassification, IntentKind, IntentPrototype, SemanticIntentClassifier,
};

// Semantic Context Window — sliding window of recent interactions for session-aware personalization
pub mod context_window;
pub use context_window::{ContextEntry, ContextStats, SemanticContextWindow, WindowConfig};

// Semantic Multilingual Index — language-organised embedding index for cross-lingual search
pub mod multilingual_index;
pub use multilingual_index::{
    CrossLingualQuery, Language, MultilingualDoc, MultilingualIndexStats, MultilingualResult,
    SemanticMultilingualIndex,
};

// Semantic Attribution Tracker — attribution chains for explainability and audit
pub mod attribution_tracker;
pub use attribution_tracker::{
    AttributionRecord, AttributionSource, AttributionStats, SemanticAttributionTracker,
};

pub mod embedding_pool;
pub use embedding_pool::{EmbeddingBuffer, PoolConfig, PoolStats, SemanticEmbeddingPool};

// Semantic Document Graph — graph structure for document relationships based on semantic similarity
pub mod document_graph;
pub use document_graph::{
    cosine_sim as doc_graph_cosine_sim, DocGraphEdge, DocGraphNode, DocumentGraphStats,
    EdgeKind as DocEdgeKind, SemanticDocumentGraph,
};

// Multi-factor document ranking combining BM25 lexical scoring with semantic similarity
pub mod document_ranker;
pub use document_ranker::{
    DocumentIndex,
    DocumentRanker,
    RankedDocument,
    // RankerStats collides with search_ranker::RankerStats — alias for disambiguation
    RankerStats as DrRankerStats,
    RankingConfig,
};

// Semantic Vocabulary Index — token-to-ID mapping with frequency / TF-IDF tracking
pub mod vocab_index;
pub use vocab_index::{SemanticVocabIndex, VocabConfig, VocabEntry, VocabIndexStats};

// Semantic Summary Extractor — extractive summarization via embedding similarity
pub mod summary_extractor;
pub use summary_extractor::{
    ExtractionResult, ExtractorScoredSentence, ExtractorSummaryConfig, SemanticSummaryExtractor,
    SummaryExtractorStats,
};

// Semantic Term Weighter — TF-IDF and BM25 term weighting for semantic search
pub mod term_weighter;
pub use term_weighter::{
    DocumentProfile, SemanticTermWeighter, TermWeight, TermWeighterStats, WeighterConfig,
    WeightingScheme,
};

// Semantic Dimension Reducer — dimensionality reduction for embeddings
pub mod dimension_reducer;
pub use dimension_reducer::{
    ReducerConfig, ReducerStats, ReductionMethod, ReductionResult, SemanticDimensionReducer,
};

// Semantic Tokenizer — text tokenization for semantic search indexing
pub mod tokenizer;
pub use tokenizer::{
    SemanticTokenizer, Token as SemanticToken, TokenizerConfig, TokenizerMode, TokenizerStats,
};

pub use feedback_loop::{FeedbackEntry, FeedbackLoopStats, FeedbackType, QueryFeedbackSummary};

pub mod embedding_cache;
pub use embedding_cache::{
    CachedEmbedding, EmbeddingCacheConfig, EmbeddingCacheStats, SemanticEmbeddingCache,
};

pub use cross_encoder::{
    CandidateDoc, CrossEncoder, CrossEncoderConfig, CrossEncoderStats, RerankedDoc, ScoringModel,
};

// Multi-algorithm semantic vector clustering engine
pub mod semantic_clusterer;
pub use semantic_clusterer::{
    ClusterAlgorithm, ClusterError, Linkage, ScCluster, ScClusterPoint, ScClustererStats,
    ScClusteringResult, SemanticClusterer,
};

// Lexicon-based sentiment analysis engine with aspect-level detection
pub mod sentiment_analyzer;
pub use sentiment_analyzer::{
    AspectSentiment, LexiconEntry, SentimentAnalyzer, SentimentAnalyzerStats, SentimentConfig,
    SentimentPolarity, SentimentResult, SentimentScore,
};

// TF-IDF + TextRank extractive text summarization engine
pub mod text_summarizer;
pub use text_summarizer::{
    SentenceScore,
    SummarizationMethod,
    SummarizerConfig,
    SummarizerError,
    TextSummarizer,
    TextSummarizerStats as TsSummarizerStats,
    // Renamed to avoid collision with document_summarizer::{SummaryResult, SummarizerStats}
    TextSummaryResult as TsSummaryResult,
};

// End-to-end semantic search pipeline (vector + BM25 + fusion + re-ranking)
pub mod search_pipeline;
pub use search_pipeline::{
    FusionMethod,
    SearchDocument,
    SearchHit,
    SearchPipelineResult,
    SemanticSearchPipeline,
    // Renamed to avoid collision with query_pipeline::PipelineConfig
    SpPipelineConfig,
    // Renamed to avoid collision with embedding_pipeline::PipelineStats
    SpPipelineStats,
    // Renamed to avoid collision with multimodal_search::SearchQuery
    SpSearchQuery,
};

// Knowledge Base Builder — incremental semantic knowledge base with entities, relations, and concept graphs
pub mod knowledge_base_builder;
pub use knowledge_base_builder::{
    // KbBuilderEntity instead of KbEntity to avoid collision with entity_linker::KbEntity
    KbBuilderEntity,
    // KbConceptNode instead of ConceptNode to avoid collision with concept_hierarchy::ConceptNode
    KbConceptNode,
    KbDocument,
    KbError,
    KbRelation as KbBuilderRelation,
    KbStats as KbBuilderStats,
    KbTriple,
    KnowledgeBaseBuilder,
};

// Multilingual Normalizer — Unicode normalization, script detection, and script-aware tokenization
pub mod multilingual_normalizer;
pub use multilingual_normalizer::{
    LanguageHint, MultilingualNormalizer, NormalizationOptions, NormalizedText,
    NormalizerStats as MlnNormalizerStats, Script, TokenizationStrategy,
};

// Inverted index corpus indexer with BM25 scoring and faceted filtering
pub mod corpus_indexer;
pub use corpus_indexer::{
    CorpusIndexer,
    FacetFilter,
    IndexError,
    IndexQuery,
    // IndexStats aliased to avoid collision with stats::IndexStats
    IndexStats as CiIndexStats,
    IndexedDocument,
    InvertedIndex,
    PostingEntry,
    // SearchResult aliased to avoid collision with hnsw::SearchResult
    SearchResult as CiSearchResult,
};

// Embedding Pipeline Manager — multi-stage text-to-vector transformation engine
pub mod embedding_pipeline_manager;
pub use embedding_pipeline_manager::{
    l2_normalize as epm_l2_normalize, mean_pool as epm_mean_pool,
    random_projection as epm_random_projection, EmbeddingBatch, EmbeddingPipelineManager,
    EpmPipelineConfig, EpmPipelineError, EpmPipelineStage, EpmPipelineStats, EpmReductionMethod,
    StageTiming,
};

pub mod semantic_versioning;
pub use semantic_versioning::{
    BumpType, ChangeRecord, ChangeType, CompatibilityLevel, CompatibilityMatrix, SemVer,
    SemVerError, SemanticVersioningEngine, VersionedArtifact, VersioningStats,
};

pub mod similarity_graph;
pub use similarity_graph::{
    GraphConfig, SemanticSimilarityGraph, SgCommunity, SgEdge, SgNode, SgStats,
};

pub mod embedding_aggregator;
pub use embedding_aggregator::{
    AggregationInput, AggregationMethod, AggregationResult as EaAggregationResult, AggregatorError,
    EaAggregatorStats, EmbeddingAggregator, EmbeddingAggregatorConfig,
};

// Semantic reranker (cross-encoder-style query-document pair scoring)
pub mod semantic_reranker;
pub use semantic_reranker::{
    RerankCandidate, RerankConfig, RerankFeature, RerankQuery, RerankResult, RerankStats,
    SemanticReranker,
};

// Multimodal index (cross-modal unified index with fusion strategies)
// Document Chunker — splits text into semantically coherent chunks for embedding and retrieval
pub mod document_chunker;
pub use document_chunker::{
    ChunkStats, ChunkStrategy, DocumentChunker, DocumentChunkerConfig, TextChunk,
};

pub mod multimodal_index;
pub use multimodal_index::{
    CrossModalQuery, CrossModalResult, FusionStrategy as MmiFusionStrategy, MmiError, MmiStats,
    Modality as MmiModality, ModalityEmbedding, MultiModalDocument,
    MultiModalIndex as MmiMultiModalIndex, MultiModalIndexConfig,
};

// Vector-similarity-based semantic cache (avoids redundant computation for close queries)
pub mod semantic_cache;
pub use semantic_cache::{
    CacheConfig, CacheEntry, CacheEvictionPolicy, CacheKey, CacheLookupResult, ScCacheStats,
    SemanticCacheLayer,
};

// Query expansion engine — enriches queries with synonyms, hypernyms, hyponyms,
// and contextual terms to improve search recall.
pub mod query_expansion;
pub use query_expansion::{
    ExpansionConfig, ExpansionSource, ExpansionStats, QeExpandedQuery, QeExpansionTerm,
    QueryExpansionEngine, SynonymEntry,
};

pub mod embedding_finetuner;
pub use embedding_finetuner::{
    cosine_similarity as ef_cosine_similarity, l2_distance_sq as ef_l2_distance_sq,
    EmbeddingFinetuner, FinetunerConfig, FinetunerError, ProjectionLayer, TrainingPair,
    TrainingStats, TripletLoss,
};

// Dense retriever — hybrid exact cosine + BM25 sparse retrieval with min-max score fusion
pub mod dense_retriever;
pub use dense_retriever::{
    BM25Index, DenseRetriever, Document as RetrieverDocument, RetrievalQuery, RetrievalResult,
    RetrieverConfig, RetrieverError, RetrieverStats,
};

// Concept Graph Builder — semantic concept graph with weighted edges, BFS path finding,
// co-occurrence mining, and embedding-based similarity search.
pub mod concept_graph;
pub use concept_graph::{
    canonize_key_test as cg_canonize_key_test,
    cosine_similarity as cg_cosine_similarity,
    tokenize as cg_tokenize,
    // Aliased to avoid collision with concept_extractor::Concept
    CgConcept,
    // Aliased to avoid collision with concept_hierarchy::ConceptEdge
    CgConceptEdge,
    // Aliased to avoid collision with concept_hierarchy::ConceptRelation
    CgConceptRelation,
    // Aliased to avoid collision with similarity_graph::GraphConfig
    CgGraphConfig,
    ConceptGraphBuilder,
    ConceptGraphStats,
    ConceptId,
};

// SemanticRouterV2 — advanced semantic routing with fallback chains and analytics
pub mod semantic_router_v2;
pub use semantic_router_v2::{
    FallbackStrategy, RouteDefinition, RouteHandlerId, RouteStats as Srv2RouteStats,
    RouterV2Config, RouterV2Error, RouterV2Stats, SemanticRouterV2, V2RoutingDecision,
};

pub mod text_similarity_scorer;
pub use text_similarity_scorer::{
    ScorerConfig, SimilarityMetric, SimilarityScore, TextPair, TextSimilarityResult,
    TextSimilarityScorer,
};

// Embedding Cluster Analyzer — comprehensive cluster analysis for embedding spaces
pub mod embedding_cluster_analyzer;
pub use embedding_cluster_analyzer::{
    ClusterDescriptor, ClusterId, ClusterQuality, EcaAnalyzerConfig, EcaAnalyzerStats,
    EcaClusterPoint, EmbeddingClusterAnalyzer, OutlierReason, OutlierScore,
};

// Semantic Federated Search Coordinator — cross-node result merging with quorum and re-ranking
pub mod semantic_federated_search;
pub use semantic_federated_search::{
    FederatedQuery, FederatedResult, FederatedStats, MergeStrategy, NodeResponse, RemoteNode,
    RemoteResult, SemanticFederatedSearch,
};

// Topic Model Extractor — collapsed Gibbs sampling LDA
pub mod topic_model_extractor;
pub use topic_model_extractor::{
    ExtractorConfig, ExtractorDocumentTopics, ExtractorError, ExtractorTopic, ExtractorTopicWord,
    ModelStats as TopicModelStats, TmeDocumentTopics, TmeError, TmeTopic, TmeTopicWord,
    TopicModelExtractor,
};

// Cross-Modal Reranker — fuses BM25 text and dense vector signals for unified reranking
pub mod cross_modal_reranker;
pub use cross_modal_reranker::{
    CmrFusionStrategy, CrossModalReranker, ModalityScore, RerankerCandidate, RerankerConfig,
    RerankerError, RerankerStats, TextFeatures, VectorFeatures,
};

pub mod semantic_graph_builder;
pub use semantic_graph_builder::{
    BuilderConfig, BuilderError, EdgeRelation, GraphStats, NodeType, SemanticGraphBuilder,
    SgbGraphEdge, SgbGraphNode, SgbGraphQuery,
};

// Embedding Drift Detector — statistical concept drift detection in embedding spaces
pub mod embedding_drift_detector;
pub use embedding_drift_detector::{
    DetectionMethod,
    // DetectorConfig aliases to avoid collision with anomaly_detector::DetectorConfig
    DetectorConfig as EddDetectorConfig,
    DetectorError,
    // DriftSignal aliases to avoid collision with drift_monitor::DriftSignal
    DriftSignal as EddDriftSignal,
    DriftSnapshot,
    DriftStats as EddDriftStats,
    DriftType,
    EmbeddingDriftDetector as EddEmbeddingDriftDetector,
};
/// Type alias: `EddDriftSignal` is the production drift signal from [`embedding_drift_detector`].
pub type EddDriftSignalAlias = EddDriftSignal;
/// Type alias: `EddDetectorConfig` is the config for [`EddEmbeddingDriftDetector`].
pub type EddDetectorConfigAlias = EddDetectorConfig;

// Multi-Modal Indexer — unified index for text, vector, and structured data
pub mod multi_modal_indexer;
pub use multi_modal_indexer::{
    cosine_similarity as mmi_cosine_similarity, IndexedDocument as MmiIndexedDocument,
    MmiIndexConfig, MmiIndexConfigAlias, MmiIndexError, MmiIndexErrorAlias, MmiIndexStats,
    MmiIndexStatsAlias, MmiSearchQuery, MmiSearchQueryAlias, MmiSearchResult, MmiSearchResultAlias,
    ModalityData, MultiModalIndexer,
};

// Contextual Embedding Search — context-aware vector search with query expansion,
// negative suppression, and diversity-aware re-ranking.
pub mod contextual_embedding_search;
pub use contextual_embedding_search::{
    cosine_similarity as ces_cosine_similarity, weighted_sum as ces_weighted_sum, CesExpandedQuery,
    ContextualEmbeddingSearch, ContextualResult, DiversityStrategy,
    SearchConfig as CesSearchConfig, SearchContext, SearchDoc, SearchError as CesSearchError,
    SearchStats as CesSearchStats,
};

// Semantic Cache Manager — similarity-aware cache with multiple eviction strategies.
pub mod semantic_cache_manager;
pub use semantic_cache_manager::{
    ScmCacheConfig, ScmCacheEntry, ScmCacheError, ScmCacheHit, ScmCacheKey, ScmCacheStats,
    ScmEntryAlias, ScmErrorAlias, ScmEvictionStrategy, ScmHitAlias, ScmKeyAlias, ScmStatsAlias,
    SemanticCacheManager,
};

pub mod semantic_query_optimizer;
pub use semantic_query_optimizer::{
    ExecutionStep, FilterOp as SqoFilterOp, IndexHints, JoinType, OptimizationRule,
    OptimizerConfig, OptimizerError, OptimizerStats, QueryNode as SqoQueryNode,
    QueryPlan as SqoQueryPlan, SemanticQueryOptimizer, StepType,
};

// Vector Index Optimizer — workload-driven index structure selection and maintenance
pub mod vector_index_optimizer;
pub use vector_index_optimizer::{
    IndexRecommendation, IndexStats as VioIndexStats, IndexStructure, MaintenanceAction,
    OptimizationCriterion, OptimizerConfig as VioOptimizerConfig,
    OptimizerError as VioOptimizerError, OptimizerStats as VioOptimizerStats, VectorIndexOptimizer,
    WorkloadProfile,
};

// Semantic Anomaly Detector — production-grade anomaly detection for embedding corpora
// using CentroidDistance, MahalanobisApprox, LOF, IsolationForest, and EnsembleVote
pub mod semantic_anomaly_detector;
pub use semantic_anomaly_detector::{
    cosine_similarity as sad_cosine_similarity,
    AnomalyRecord as SadAnomalyRecord,
    ReferencePoint as SadReferencePoint,
    SadAnomalyScore,
    SadDetectionMethod,
    SadDetectorConfig,
    SadDetectorStats,
    SadDriftReport,
    // SemanticAnomalyDetector collides with anomaly_detector::SemanticAnomalyDetector → alias
    SemanticAnomalyDetector as SadSemanticAnomalyDetector,
};

pub mod hierarchical_topic_model;
pub use hierarchical_topic_model::{
    HierarchicalTopicModel, HtmDocument, HtmModelConfig, HtmModelStats, HtmTopic, HtmTopicNode,
};

pub mod multilingual_embedding_aligner;
pub use multilingual_embedding_aligner::{
    MeaAlignerConfig, MeaAlignerStats, MeaAlignmentMatrix, MeaAlignmentMethod, MeaLanguageSpace,
    MultilingualEmbeddingAligner,
};

// Embedding Compression Codec — multi-method lossy/lossless codec for dense embedding vectors
pub mod embedding_compression_codec;
pub use embedding_compression_codec::{
    EccCodecConfig, EccCodecStats, EccCompressed, EccError, EccMethod, EmbeddingCompressionCodec,
};

// Semantic Cluster Labeler — automatic human-readable label assignment for embedding clusters
pub mod semantic_cluster_labeler;
pub use semantic_cluster_labeler::{
    SclCluster, SclError, SclLabelCandidate, SclLabelerConfig, SclLabelerStats, SclLabelingMethod,
    SemanticClusterLabeler,
};

// Semantic Versioning Tracker — detects semantic drift in embedding spaces across model versions
pub mod semantic_versioning_tracker;
pub use semantic_versioning_tracker::{
    SemanticVersioningTracker, SvtDriftEvent, SvtDriftReport, SvtError,
    SvtSemanticVersioningTracker, SvtTrackerConfig, SvtTrackerStats, SvtVersion, SvtVersionId,
};

// Semantic Search Pipeline — full-stack pipeline with preprocessing, retrieval, and postprocessing
pub mod semantic_search_pipeline;
pub use semantic_search_pipeline::{
    SemanticSearchPipeline as SspSemanticSearchPipelineExport, SspDocId, SspDocument,
    SspPipelineConfig, SspPipelineStats, SspQueryRecord, SspRerankMethod, SspSearchResult,
    SspSemanticSearchPipeline, SspStage,
};
