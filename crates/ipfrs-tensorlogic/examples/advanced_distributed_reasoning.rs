//! Advanced Distributed Reasoning Example
//!
//! This example demonstrates a complete distributed reasoning workflow including:
//! - Remote knowledge retrieval with caching
//! - Recursive query handling with tabling
//! - Distributed goal resolution
//! - Proof construction and verification
//! - Performance monitoring with cache statistics
//!
//! Run with: cargo run --example advanced_distributed_reasoning

use ipfrs_tensorlogic::{
    CacheManager, Constant, DistributedGoalResolver, DistributedReasoner, FactDiscoveryRequest,
    IncrementalLoadRequest, KnowledgeBase, MockRemoteKnowledgeProvider, Predicate,
    RemoteKnowledgeProvider, Rule, Substitution, TabledInferenceEngine, Term,
};
use std::collections::HashSet;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Distributed Reasoning Example ===\n");

    // Step 1: Set up local and remote knowledge bases
    println!("1. Setting up knowledge bases...");
    let (local_kb, remote_kb) = setup_knowledge_bases();
    println!(
        "   ✓ Local KB: {} facts, {} rules",
        local_kb.facts.len(),
        local_kb.rules.len()
    );
    println!(
        "   ✓ Remote KB: {} facts, {} rules",
        remote_kb.facts.len(),
        remote_kb.rules.len()
    );

    // Step 2: Create distributed reasoning infrastructure
    println!("\n2. Creating distributed reasoning infrastructure...");
    let cache_manager = Arc::new(CacheManager::new());
    let remote_provider = Arc::new(MockRemoteKnowledgeProvider::new(Arc::new(remote_kb)));

    let reasoner = DistributedReasoner::with_cache(cache_manager.clone())?;
    let mut goal_resolver = DistributedGoalResolver::new(Arc::new(local_kb.clone()))
        .with_provider(remote_provider.clone())
        .with_timeout(5000);

    println!("   ✓ Distributed reasoner created with cache");
    println!("   ✓ Goal resolver configured");

    // Step 3: Demonstrate local reasoning
    println!("\n3. Local reasoning (cached)...");
    let local_goal = Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Var("X".to_string()),
        ],
    );

    let local_solutions = reasoner.query(&local_goal, &local_kb).await?;
    println!("   ✓ Found {} local solutions", local_solutions.len());
    print_solutions("parent(alice, X)", &local_solutions);

    // Step 4: Demonstrate remote fact discovery
    println!("\n4. Remote fact discovery...");
    let discovery_request = FactDiscoveryRequest {
        predicate_name: "knows".to_string(),
        arg_patterns: vec![],
        max_hops: 3,
        ttl: 30,
        exclude_peers: HashSet::new(),
    };

    let discovery_response = remote_provider.discover_facts(discovery_request).await?;
    println!(
        "   ✓ Discovered {} facts from {} peer(s)",
        discovery_response.facts.len(),
        discovery_response.peers_queried
    );

    for (i, fact) in discovery_response.facts.iter().take(3).enumerate() {
        println!("     {}. {}", i + 1, fact);
    }

    // Step 5: Demonstrate incremental loading
    println!("\n5. Incremental fact loading...");
    let load_request = IncrementalLoadRequest {
        predicate_name: "knows".to_string(),
        batch_size: 2,
        offset: 0,
        filter: None,
    };

    let load_response = remote_provider.load_incremental(load_request).await?;
    println!(
        "   ✓ Loaded batch: {} of {} total facts",
        load_response.batch.len(),
        load_response.total_count
    );
    println!("   ✓ More available: {}", !load_response.is_last);

    // Step 6: Prefetch remote facts
    println!("\n6. Prefetching remote facts...");
    let prefetch_count = goal_resolver.prefetch_facts("knows").await?;
    println!("   ✓ Prefetched {} facts", prefetch_count);

    // Step 7: Recursive query with tabling
    println!("\n7. Recursive query (ancestor relation)...");
    let ancestor_goal = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Var("Z".to_string()),
        ],
    );

    let tabled_engine = TabledInferenceEngine::new();
    let ancestor_solutions = tabled_engine.query(&ancestor_goal, &local_kb)?;
    println!(
        "   ✓ Found {} ancestors using tabling",
        ancestor_solutions.len()
    );
    print_solutions("ancestor(alice, Z)", &ancestor_solutions);

    let table_stats = tabled_engine.table_stats();
    println!(
        "   ✓ Table statistics: {} entries, {} complete",
        table_stats.entries, table_stats.complete_entries
    );

    // Step 8: Distributed goal resolution
    println!("\n8. Distributed goal resolution...");
    let distributed_goal = Predicate::new(
        "knows".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Var("Y".to_string()),
        ],
    );

    let distributed_solutions = goal_resolver
        .resolve(&distributed_goal, &Substitution::new())
        .await?;

    println!(
        "   ✓ Resolved {} solutions using distributed reasoning",
        distributed_solutions.len()
    );
    print_solutions("knows(alice, Y)", &distributed_solutions);

    // Step 9: Cache statistics
    println!("\n9. Cache performance analysis...");
    if let Some(stats) = reasoner.cache_stats() {
        println!("   Query Cache:");
        println!("     • Hits: {}", stats.query_stats.hits);
        println!("     • Misses: {}", stats.query_stats.misses);
        println!("     • Evictions: {}", stats.query_stats.evictions);
        println!(
            "     • Hit Rate: {:.2}%",
            stats.query_stats.hits as f64
                / (stats.query_stats.hits + stats.query_stats.misses).max(1) as f64
                * 100.0
        );

        println!("   Fact Cache:");
        println!("     • Hits: {}", stats.fact_stats.hits);
        println!("     • Misses: {}", stats.fact_stats.misses);
        println!("     • Evictions: {}", stats.fact_stats.evictions);
    }

    // Step 10: Proof construction
    println!("\n10. Proof construction and verification...");
    let proof = reasoner.prove(&local_goal, &local_kb).await?;
    if let Some(proof) = proof {
        println!("   ✓ Proof constructed:");
        println!("     • Goal: {}", proof.goal);
        println!("     • Is fact: {}", proof.is_fact());
        println!("     • Depth: {}", proof.depth());
        println!("     • Size: {} nodes", proof.size());

        // Verify proof
        use ipfrs_tensorlogic::InferenceEngine;
        let engine = InferenceEngine::new();
        let valid = engine.verify(&proof, &local_kb)?;
        println!(
            "   ✓ Proof verification: {}",
            if valid { "VALID ✓" } else { "INVALID ✗" }
        );
    } else {
        println!("   ✗ No proof found");
    }

    println!("\n=== Summary ===");
    println!("✓ Demonstrated local reasoning with caching");
    println!("✓ Performed remote fact discovery");
    println!("✓ Used incremental loading for large datasets");
    println!("✓ Applied recursive queries with tabling");
    println!("✓ Executed distributed goal resolution");
    println!("✓ Constructed and verified proofs");
    println!("\nThe distributed reasoning system is fully operational!");

    Ok(())
}

/// Set up local and remote knowledge bases with sample data
fn setup_knowledge_bases() -> (KnowledgeBase, KnowledgeBase) {
    let mut local_kb = KnowledgeBase::new();
    let mut remote_kb = KnowledgeBase::new();

    // Local KB: Family relationships
    // Facts: parent relationships
    local_kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    ));
    local_kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("bob".to_string())),
            Term::Const(Constant::String("charlie".to_string())),
        ],
    ));
    local_kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("charlie".to_string())),
            Term::Const(Constant::String("david".to_string())),
        ],
    ));

    // Rules: Recursive ancestor relation
    // ancestor(X, Y) :- parent(X, Y)
    local_kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ));

    // ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
    local_kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    ));

    // Remote KB: Social network
    let people = ["alice", "bob", "charlie", "david", "eve", "frank"];

    // Create a small social network
    for i in 0..people.len() {
        for j in (i + 1)..people.len() {
            if (i + j) % 3 == 0 {
                remote_kb.add_fact(Predicate::new(
                    "knows".to_string(),
                    vec![
                        Term::Const(Constant::String(people[i].to_string())),
                        Term::Const(Constant::String(people[j].to_string())),
                    ],
                ));
            }
        }
    }

    (local_kb, remote_kb)
}

/// Helper function to print solutions
fn print_solutions(query: &str, solutions: &[Substitution]) {
    if solutions.is_empty() {
        println!("     No solutions found for {}", query);
        return;
    }

    for (i, solution) in solutions.iter().take(5).enumerate() {
        let bindings: Vec<String> = solution
            .iter()
            .map(|(var, term)| format!("{} = {}", var, term))
            .collect();
        println!("     {}. {{ {} }}", i + 1, bindings.join(", "));
    }

    if solutions.len() > 5 {
        println!("     ... and {} more", solutions.len() - 5);
    }
}
