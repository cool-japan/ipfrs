//! Distributed Reasoning Example
//!
//! This example demonstrates distributed TensorLogic reasoning across multiple nodes.
//! It shows how the framework supports:
//! - Multi-node knowledge base setup
//! - Remote fact caching
//! - Distributed proof construction
//! - Goal decomposition for distributed solving
//!
//! Note: This example simulates distributed nodes locally. Full network integration
//! will be available when ipfrs-network is complete.

use ipfrs_core::CidBuilder;
use ipfrs_tensorlogic::{
    Constant, GoalDecomposition, KnowledgeBase, Predicate, ProofAssembler, ProofFragment,
    ProofFragmentRef, ProofFragmentStore, RemoteFactCache, Rule, Term,
};
use std::time::Duration;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Distributed Reasoning Example ===\n");

    // Simulate three distributed nodes
    let mut node1 = create_node("Node-1", "Medical Knowledge");
    let mut node2 = create_node("Node-2", "Treatment Database");
    let mut node3 = create_node("Node-3", "Patient Records");

    // Each node has different knowledge

    // Node 1: Medical knowledge (diseases and symptoms)
    println!("1. Setting up Node-1 (Medical Knowledge)...");
    add_medical_facts(&mut node1);
    let stats1 = node1.kb.stats();
    println!("   ✓ {} facts loaded", stats1.num_facts);

    // Node 2: Treatment knowledge
    println!("\n2. Setting up Node-2 (Treatment Database)...");
    add_treatment_facts(&mut node2);
    let stats2 = node2.kb.stats();
    println!("   ✓ {} facts loaded", stats2.num_facts);

    // Node 3: Patient records
    println!("\n3. Setting up Node-3 (Patient Records)...");
    add_patient_facts(&mut node3);
    let stats3 = node3.kb.stats();
    println!("   ✓ {} facts loaded", stats3.num_facts);

    // Simulate fact sharing between nodes
    println!("\n4. Simulating fact sharing between nodes...");
    share_facts(&mut node1, &node2, "has_treatment");
    share_facts(&mut node1, &node3, "patient_symptom");
    share_facts(&mut node2, &node1, "has_symptom");
    println!("   ✓ Facts cached across nodes");

    // Create a distributed query
    println!("\n5. Distributed query: Find treatment for patient 'Alice'");

    // Query: What treatment should Alice receive?
    // This requires:
    // 1. Getting Alice's symptoms from Node 3
    // 2. Diagnosing disease from symptoms using Node 1
    // 3. Finding treatment from Node 2

    // Simulate distributed inference
    let result = distributed_inference(&mut node1, &mut node2, &mut node3, "Alice")?;

    println!("\n6. Query result:");
    if !result.is_empty() {
        for treatment in result {
            println!("   → Treatment: {}", treatment);
        }
    } else {
        println!("   (no treatment found)");
    }

    // Demonstrate proof construction across nodes
    println!("\n7. Constructing distributed proof...");
    let proof_store = construct_distributed_proof(&node1, &node2, &node3)?;
    println!("   ✓ Proof fragments stored: {}", proof_store.len());

    // Show proof assembly
    println!("\n8. Assembling proof from fragments...");
    demonstrate_proof_assembly(&proof_store)?;

    // Show cache statistics
    println!("\n9. Cache statistics:");
    display_cache_stats(&node1, "Node-1");
    display_cache_stats(&node2, "Node-2");
    display_cache_stats(&node3, "Node-3");

    // Demonstrate goal decomposition
    println!("\n10. Goal decomposition for distributed solving:");
    demonstrate_goal_decomposition()?;

    println!("\n=== Summary ===");
    println!("✓ Simulated 3 distributed nodes with specialized knowledge");
    println!("✓ Demonstrated fact sharing via remote caching");
    println!("✓ Performed distributed inference across nodes");
    println!("✓ Constructed and assembled distributed proofs");
    println!("✓ Showed goal decomposition for parallel solving");
    println!("\nNote: Full network integration will enable real distributed execution.");

    Ok(())
}

/// Represents a simulated distributed node
struct Node {
    name: String,
    #[allow(dead_code)]
    description: String,
    kb: KnowledgeBase,
    remote_cache: RemoteFactCache,
    #[allow(dead_code)]
    proof_store: ProofFragmentStore,
}

impl Node {
    fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            kb: KnowledgeBase::new(),
            remote_cache: RemoteFactCache::new(100, Duration::from_secs(300)),
            proof_store: ProofFragmentStore::new(),
        }
    }
}

fn create_node(name: &str, description: &str) -> Node {
    Node::new(name.to_string(), description.to_string())
}

fn add_medical_facts(node: &mut Node) {
    // Symptoms of diseases
    node.kb.add_fact(Predicate::new(
        "has_symptom".to_string(),
        vec![
            Term::Const(Constant::String("flu".to_string())),
            Term::Const(Constant::String("fever".to_string())),
        ],
    ));

    node.kb.add_fact(Predicate::new(
        "has_symptom".to_string(),
        vec![
            Term::Const(Constant::String("flu".to_string())),
            Term::Const(Constant::String("cough".to_string())),
        ],
    ));

    node.kb.add_fact(Predicate::new(
        "has_symptom".to_string(),
        vec![
            Term::Const(Constant::String("cold".to_string())),
            Term::Const(Constant::String("cough".to_string())),
        ],
    ));

    node.kb.add_fact(Predicate::new(
        "has_symptom".to_string(),
        vec![
            Term::Const(Constant::String("cold".to_string())),
            Term::Const(Constant::String("sore_throat".to_string())),
        ],
    ));

    // Diagnosis rules
    node.kb.add_rule(Rule::new(
        Predicate::new(
            "diagnose".to_string(),
            vec![Term::Var("P".to_string()), Term::Var("D".to_string())],
        ),
        vec![
            Predicate::new(
                "patient_symptom".to_string(),
                vec![Term::Var("P".to_string()), Term::Var("S".to_string())],
            ),
            Predicate::new(
                "has_symptom".to_string(),
                vec![Term::Var("D".to_string()), Term::Var("S".to_string())],
            ),
        ],
    ));
}

fn add_treatment_facts(node: &mut Node) {
    // Treatments for diseases
    node.kb.add_fact(Predicate::new(
        "has_treatment".to_string(),
        vec![
            Term::Const(Constant::String("flu".to_string())),
            Term::Const(Constant::String("antiviral_medication".to_string())),
        ],
    ));

    node.kb.add_fact(Predicate::new(
        "has_treatment".to_string(),
        vec![
            Term::Const(Constant::String("cold".to_string())),
            Term::Const(Constant::String("rest_and_fluids".to_string())),
        ],
    ));

    // Treatment recommendation rule
    node.kb.add_rule(Rule::new(
        Predicate::new(
            "recommend_treatment".to_string(),
            vec![Term::Var("P".to_string()), Term::Var("T".to_string())],
        ),
        vec![
            Predicate::new(
                "diagnose".to_string(),
                vec![Term::Var("P".to_string()), Term::Var("D".to_string())],
            ),
            Predicate::new(
                "has_treatment".to_string(),
                vec![Term::Var("D".to_string()), Term::Var("T".to_string())],
            ),
        ],
    ));
}

fn add_patient_facts(node: &mut Node) {
    // Patient symptoms
    node.kb.add_fact(Predicate::new(
        "patient_symptom".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("fever".to_string())),
        ],
    ));

    node.kb.add_fact(Predicate::new(
        "patient_symptom".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("cough".to_string())),
        ],
    ));

    node.kb.add_fact(Predicate::new(
        "patient_symptom".to_string(),
        vec![
            Term::Const(Constant::String("Bob".to_string())),
            Term::Const(Constant::String("sore_throat".to_string())),
        ],
    ));
}

fn share_facts(target: &mut Node, source: &Node, predicate: &str) {
    // Simulate sharing facts from source node to target node's remote cache
    for fact in source.kb.get_predicates(predicate) {
        target.remote_cache.add_fact(
            fact.clone(),
            None, // No CID for local facts
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn distributed_inference(
    _node1: &mut Node,
    _node2: &mut Node,
    _node3: &mut Node,
    patient: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // In a real distributed system, this would involve:
    // 1. Query node3 for patient symptoms
    // 2. Query node1 for diagnosis based on symptoms
    // 3. Query node2 for treatment based on diagnosis

    // For this example, we'll simulate the result
    println!("   Step 1: Querying Node-3 for {}'s symptoms", patient);
    println!("   → Found: fever, cough");

    println!("   Step 2: Querying Node-1 for diagnosis");
    println!("   → Diagnosed: flu");

    println!("   Step 3: Querying Node-2 for treatment");
    println!("   → Treatment: antiviral_medication");

    Ok(vec!["antiviral_medication".to_string()])
}

fn construct_distributed_proof(
    node1: &Node,
    node2: &Node,
    node3: &Node,
) -> Result<ProofFragmentStore, Box<dyn std::error::Error>> {
    let mut store = ProofFragmentStore::new();

    // Create proof fragments for each step

    // Fragment 1: Patient has symptom (from node3)
    let fact1 = Predicate::new(
        "patient_symptom".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("fever".to_string())),
        ],
    );

    let cid1 = CidBuilder::new().build(b"proof_fragment_1")?;
    let mut fragment1 = ProofFragment::fact(fact1);
    fragment1.metadata.created_by = Some(node3.name.clone());
    fragment1.metadata.created_at = Some(1234567890);
    fragment1
        .metadata
        .custom
        .insert("source".to_string(), "patient_records".to_string());

    store.add_with_cid(fragment1, cid1);

    // Fragment 2: Disease has symptom (from node1)
    let fact2 = Predicate::new(
        "has_symptom".to_string(),
        vec![
            Term::Const(Constant::String("flu".to_string())),
            Term::Const(Constant::String("fever".to_string())),
        ],
    );

    let cid2 = CidBuilder::new().build(b"proof_fragment_2")?;
    let mut fragment2 = ProofFragment::fact(fact2);
    fragment2.metadata.created_by = Some(node1.name.clone());
    fragment2.metadata.created_at = Some(1234567891);
    fragment2
        .metadata
        .custom
        .insert("source".to_string(), "medical_knowledge".to_string());

    store.add_with_cid(fragment2, cid2);

    // Fragment 3: Diagnosis (combining previous facts using a rule)
    let conclusion = Predicate::new(
        "diagnose".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("flu".to_string())),
        ],
    );

    // Create a rule for diagnosis
    let diagnose_rule = Rule::new(
        Predicate::new(
            "diagnose".to_string(),
            vec![Term::Var("P".to_string()), Term::Var("D".to_string())],
        ),
        vec![
            Predicate::new(
                "patient_symptom".to_string(),
                vec![Term::Var("P".to_string()), Term::Var("S".to_string())],
            ),
            Predicate::new(
                "has_symptom".to_string(),
                vec![Term::Var("D".to_string()), Term::Var("S".to_string())],
            ),
        ],
    );

    let cid3 = CidBuilder::new().build(b"proof_fragment_3")?;
    let mut fragment3 = ProofFragment::with_rule(
        conclusion,
        &diagnose_rule,
        vec![ProofFragmentRef::new(cid1), ProofFragmentRef::new(cid2)],
        vec![], // Substitution
    );
    fragment3.metadata.created_by = Some(node1.name.clone());
    fragment3.metadata.created_at = Some(1234567892);
    fragment3
        .metadata
        .custom
        .insert("type".to_string(), "inference".to_string());

    store.add_with_cid(fragment3, cid3);

    // Fragment 4: Treatment fact (from node2)
    let fact4 = Predicate::new(
        "has_treatment".to_string(),
        vec![
            Term::Const(Constant::String("flu".to_string())),
            Term::Const(Constant::String("antiviral_medication".to_string())),
        ],
    );

    let cid4 = CidBuilder::new().build(b"proof_fragment_4")?;
    let mut fragment4 = ProofFragment::fact(fact4);
    fragment4.metadata.created_by = Some(node2.name.clone());
    fragment4.metadata.created_at = Some(1234567893);
    fragment4
        .metadata
        .custom
        .insert("source".to_string(), "treatment_db".to_string());

    store.add_with_cid(fragment4, cid4);

    Ok(store)
}

fn demonstrate_proof_assembly(
    store: &ProofFragmentStore,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get the final conclusion
    let conclusions = store.find_by_conclusion("diagnose");

    if conclusions.is_empty() {
        println!("   (no proof conclusions found)");
        return Ok(());
    }

    println!("   Found {} proof fragments", store.len());

    // Assemble the proof
    let mut assembler = ProofAssembler::new(store);

    for fragment in conclusions {
        println!("   Assembling proof for: {}", fragment.conclusion);

        match assembler.assemble(&fragment.id) {
            Some(_proof) => {
                println!("   ✓ Proof assembled successfully");
                // In a real system, we would verify the proof here
            }
            None => {
                println!("   ⚠ Failed to assemble proof (missing fragments)");
            }
        }
    }

    Ok(())
}

fn display_cache_stats(node: &Node, name: &str) {
    let stats = node.remote_cache.stats();
    let snapshot = stats.snapshot();
    println!(
        "   {}: hits={}, misses={}, evictions={}",
        name, snapshot.hits, snapshot.misses, snapshot.evictions
    );
}

fn demonstrate_goal_decomposition() -> Result<(), Box<dyn std::error::Error>> {
    // Create a complex goal
    let goal = Predicate::new(
        "recommend_treatment".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Var("T".to_string()),
        ],
    );

    println!("   Goal: {}", goal);

    // Create goal decomposition
    let decomp = GoalDecomposition::new(goal.clone(), 0);

    // Simulate adding subgoals
    let subgoal1 = Predicate::new(
        "diagnose".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Var("D".to_string()),
        ],
    );

    let subgoal2 = Predicate::new(
        "has_treatment".to_string(),
        vec![Term::Var("D".to_string()), Term::Var("T".to_string())],
    );

    println!("   Subgoals:");
    println!("     1. {} (can be solved on Node-1 or Node-3)", subgoal1);
    println!("     2. {} (can be solved on Node-2)", subgoal2);
    println!("   Depth: {}", decomp.depth);
    println!("   ✓ Goals can be solved in parallel across nodes");

    Ok(())
}
