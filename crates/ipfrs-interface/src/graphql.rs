//! GraphQL API for IPFRS
//!
//! Provides a modern GraphQL interface for all IPFRS operations

use async_graphql::{Context, EmptySubscription, Object, Result, Schema, SimpleObject};
use ipfrs_core::Cid;
use ipfrs_semantic::{QueryFilter, SemanticRouter};
use ipfrs_storage::{BlockStoreTrait, SledBlockStore};
use ipfrs_tensorlogic::{Predicate, TensorLogicStore, Term};
use std::sync::Arc;

/// GraphQL schema type
pub type IpfrsSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

/// Root query type
pub struct QueryRoot;

/// Root mutation type
pub struct MutationRoot;

/// Block information
#[derive(SimpleObject, Clone)]
pub struct BlockInfo {
    /// Content ID
    pub cid: String,
    /// Block size in bytes
    pub size: u64,
    /// Block data (base64 encoded)
    pub data: Option<String>,
}

/// Semantic search result
#[derive(SimpleObject, Clone)]
pub struct SemanticSearchResult {
    /// Content ID
    pub cid: String,
    /// Similarity score
    pub score: f32,
}

/// Logic inference result
#[derive(SimpleObject, Clone)]
pub struct InferenceResult {
    /// Number of solutions found
    pub solution_count: usize,
    /// Solutions as JSON string
    pub solutions: String,
}

/// Proof information
#[derive(SimpleObject, Clone)]
pub struct ProofInfo {
    /// Proof exists
    pub exists: bool,
    /// Proof CID if stored
    pub cid: Option<String>,
    /// Proof goal
    pub goal: String,
}

/// Router statistics
#[derive(SimpleObject, Clone)]
pub struct RouterStats {
    /// Number of indexed vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: String,
}

/// Knowledge base statistics
#[derive(SimpleObject, Clone)]
pub struct KbStats {
    /// Number of facts
    pub num_facts: usize,
    /// Number of rules
    pub num_rules: usize,
}

#[Object]
impl QueryRoot {
    /// Get a block by CID
    async fn block(&self, ctx: &Context<'_>, cid: String) -> Result<Option<BlockInfo>> {
        let store = ctx.data::<Arc<SledBlockStore>>()?;

        let cid_parsed = cid
            .parse::<Cid>()
            .map_err(|e| format!("Invalid CID: {}", e))?;

        match store.get(&cid_parsed).await? {
            Some(block) => {
                let data_base64 = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    block.data(),
                );

                Ok(Some(BlockInfo {
                    cid: cid.clone(),
                    size: block.size(),
                    data: Some(data_base64),
                }))
            }
            None => Ok(None),
        }
    }

    /// Check if a block exists
    async fn has_block(&self, ctx: &Context<'_>, cid: String) -> Result<bool> {
        let store = ctx.data::<Arc<SledBlockStore>>()?;

        let cid_parsed = cid
            .parse::<Cid>()
            .map_err(|e| format!("Invalid CID: {}", e))?;

        Ok(store.has(&cid_parsed).await?)
    }

    /// Get block statistics
    async fn block_stats(&self, ctx: &Context<'_>, cid: String) -> Result<Option<BlockInfo>> {
        let store = ctx.data::<Arc<SledBlockStore>>()?;

        let cid_parsed = cid
            .parse::<Cid>()
            .map_err(|e| format!("Invalid CID: {}", e))?;

        match store.get(&cid_parsed).await? {
            Some(block) => {
                Ok(Some(BlockInfo {
                    cid: cid.clone(),
                    size: block.size(),
                    data: None, // Don't include data in stats
                }))
            }
            None => Ok(None),
        }
    }

    /// Search for similar content
    async fn semantic_search(
        &self,
        ctx: &Context<'_>,
        query: Vec<f32>,
        k: Option<usize>,
        min_score: Option<f32>,
    ) -> Result<Vec<SemanticSearchResult>> {
        let router = ctx.data::<Arc<SemanticRouter>>()?;

        let k = k.unwrap_or(10);

        let mut filter = QueryFilter::default();
        if let Some(min) = min_score {
            filter.min_score = Some(min);
        }
        filter.max_results = Some(k);

        let results = router.query_with_filter(&query, k, filter).await?;

        Ok(results
            .into_iter()
            .map(|r| SemanticSearchResult {
                cid: r.cid.to_string(),
                score: r.score,
            })
            .collect())
    }

    /// Get semantic router statistics
    async fn semantic_stats(&self, ctx: &Context<'_>) -> Result<RouterStats> {
        let router = ctx.data::<Arc<SemanticRouter>>()?;
        let stats = router.stats();

        let metric = match stats.metric {
            ipfrs_semantic::DistanceMetric::Cosine => "cosine",
            ipfrs_semantic::DistanceMetric::L2 => "l2",
            ipfrs_semantic::DistanceMetric::DotProduct => "dotproduct",
        };

        Ok(RouterStats {
            num_vectors: stats.num_vectors,
            dimension: stats.dimension,
            metric: metric.to_string(),
        })
    }

    /// Run inference query
    async fn infer(
        &self,
        ctx: &Context<'_>,
        predicate: String,
        terms: Vec<String>,
    ) -> Result<InferenceResult> {
        let tensorlogic = ctx.data::<Arc<TensorLogicStore<SledBlockStore>>>()?;

        // Parse terms (simplified - in production would use Datalog parser)
        let parsed_terms: Vec<Term> = terms
            .iter()
            .map(|t| {
                if t.starts_with('?') || t.chars().next().is_some_and(char::is_uppercase) {
                    Term::Var(t.to_string())
                } else {
                    Term::Const(ipfrs_tensorlogic::Constant::String(t.to_string()))
                }
            })
            .collect();

        let goal = Predicate::new(predicate, parsed_terms);
        let solutions = tensorlogic.infer(&goal)?;

        let solutions_json = serde_json::to_string(&solutions)
            .map_err(|e| format!("Failed to serialize solutions: {}", e))?;

        Ok(InferenceResult {
            solution_count: solutions.len(),
            solutions: solutions_json,
        })
    }

    /// Generate proof for a goal
    async fn prove(
        &self,
        ctx: &Context<'_>,
        predicate: String,
        terms: Vec<String>,
    ) -> Result<ProofInfo> {
        let tensorlogic = ctx.data::<Arc<TensorLogicStore<SledBlockStore>>>()?;

        // Parse terms
        let parsed_terms: Vec<Term> = terms
            .iter()
            .map(|t| {
                if t.starts_with('?') || t.chars().next().is_some_and(char::is_uppercase) {
                    Term::Var(t.to_string())
                } else {
                    Term::Const(ipfrs_tensorlogic::Constant::String(t.to_string()))
                }
            })
            .collect();

        let goal = Predicate::new(predicate.clone(), parsed_terms);

        match tensorlogic.prove(&goal)? {
            Some(proof) => {
                let proof_cid = tensorlogic.store_proof(&proof).await?;
                Ok(ProofInfo {
                    exists: true,
                    cid: Some(proof_cid.to_string()),
                    goal: predicate,
                })
            }
            None => Ok(ProofInfo {
                exists: false,
                cid: None,
                goal: predicate,
            }),
        }
    }

    /// Get knowledge base statistics
    async fn kb_stats(&self, ctx: &Context<'_>) -> Result<KbStats> {
        let tensorlogic = ctx.data::<Arc<TensorLogicStore<SledBlockStore>>>()?;
        let stats = tensorlogic.kb_stats();

        Ok(KbStats {
            num_facts: stats.num_facts,
            num_rules: stats.num_rules,
        })
    }

    /// Get system version
    async fn version(&self) -> Result<String> {
        Ok(env!("CARGO_PKG_VERSION").to_string())
    }
}

#[Object]
impl MutationRoot {
    /// Add a block to storage
    async fn add_block(&self, ctx: &Context<'_>, data: String) -> Result<BlockInfo> {
        use ipfrs_core::Block;

        let store = ctx.data::<Arc<SledBlockStore>>()?;

        // Decode base64 data
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
            .map_err(|e| format!("Invalid base64: {}", e))?;

        // Create block
        let block = Block::new(bytes::Bytes::from(bytes))
            .map_err(|e| format!("Failed to create block: {}", e))?;

        let cid = *block.cid();
        let size = block.size();

        // Store block
        store.put(&block).await?;

        Ok(BlockInfo {
            cid: cid.to_string(),
            size,
            data: None,
        })
    }

    /// Index content for semantic search
    async fn index_content(
        &self,
        ctx: &Context<'_>,
        cid: String,
        embedding: Vec<f32>,
    ) -> Result<bool> {
        let router = ctx.data::<Arc<SemanticRouter>>()?;

        let cid_parsed = cid
            .parse::<Cid>()
            .map_err(|e| format!("Invalid CID: {}", e))?;

        router.add(&cid_parsed, &embedding)?;

        Ok(true)
    }

    /// Add a fact to the knowledge base
    async fn add_fact(
        &self,
        ctx: &Context<'_>,
        predicate: String,
        terms: Vec<String>,
    ) -> Result<bool> {
        let tensorlogic = ctx.data::<Arc<TensorLogicStore<SledBlockStore>>>()?;

        // Parse terms
        let parsed_terms: Vec<Term> = terms
            .iter()
            .map(|t| Term::Const(ipfrs_tensorlogic::Constant::String(t.to_string())))
            .collect();

        let fact = Predicate::new(predicate, parsed_terms);
        tensorlogic.add_fact(fact)?;

        Ok(true)
    }

    /// Add a rule to the knowledge base (simplified - takes Datalog string)
    async fn add_rule(&self, ctx: &Context<'_>, datalog: String) -> Result<bool> {
        let tensorlogic = ctx.data::<Arc<TensorLogicStore<SledBlockStore>>>()?;

        // Parse Datalog rule
        let rule = ipfrs_tensorlogic::parse_rule(&datalog)
            .map_err(|e| format!("Failed to parse rule: {}", e))?;

        tensorlogic.add_rule(rule)?;

        Ok(true)
    }

    /// Delete a block
    async fn delete_block(&self, ctx: &Context<'_>, cid: String) -> Result<bool> {
        let store = ctx.data::<Arc<SledBlockStore>>()?;

        let cid_parsed = cid
            .parse::<Cid>()
            .map_err(|e| format!("Invalid CID: {}", e))?;

        store.delete(&cid_parsed).await?;

        Ok(true)
    }
}

/// Create GraphQL schema
pub fn create_schema(
    store: Arc<SledBlockStore>,
    semantic: Option<Arc<SemanticRouter>>,
    tensorlogic: Option<Arc<TensorLogicStore<SledBlockStore>>>,
) -> IpfrsSchema {
    let mut schema = Schema::build(QueryRoot, MutationRoot, EmptySubscription).data(store);

    if let Some(router) = semantic {
        schema = schema.data(router);
    }

    if let Some(tl) = tensorlogic {
        schema = schema.data(tl);
    }

    schema.finish()
}
