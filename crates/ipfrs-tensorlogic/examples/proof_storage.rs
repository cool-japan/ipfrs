//! Proof storage and compression example
//!
//! Demonstrates:
//! - Creating proof fragments
//! - Storing proofs in a fragment store
//! - Assembling proofs from fragments
//! - Compressing proofs to reduce redundancy
//! - Delta encoding for incremental proofs

use ipfrs_tensorlogic::ir::{Constant, Predicate, Rule, Term};
use ipfrs_tensorlogic::proof_storage::{
    ProofAssembler, ProofCompressor, ProofFragment, ProofFragmentStore, ProofMetadata,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Proof Storage and Compression Example ===\n");

    // Create a proof fragment store
    let mut store = ProofFragmentStore::new();

    println!("--- Creating Proof Fragments ---");

    // Create some fact proofs (leaf nodes)
    println!("Adding fact fragments...");

    let fact1 = ProofFragment::fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    ));

    let fact2 = ProofFragment::fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("bob".to_string())),
            Term::Const(Constant::String("charlie".to_string())),
        ],
    ));

    let fact3 = ProofFragment::fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("david".to_string())),
        ],
    ));

    // Add duplicate fact (for compression demo)
    let fact4 = ProofFragment::fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    ));

    let id1 = store.add(fact1);
    let id2 = store.add(fact2);
    let id3 = store.add(fact3);
    let id4 = store.add(fact4);

    println!("Added {} fact fragments", store.len());
    println!("  Fragment IDs: {}, {}, {}, {}", id1, id2, id3, id4);

    // Create a rule-based proof
    println!("\nCreating rule-based proof fragment...");

    let grandparent_rule = Rule::new(
        Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    );

    let mut grandparent_proof = ProofFragment::with_rule(
        Predicate::new(
            "grandparent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("charlie".to_string())),
            ],
        ),
        &grandparent_rule,
        vec![], // Would contain CIDs of premise proofs in a real implementation
        vec![
            (
                "X".to_string(),
                Term::Const(Constant::String("alice".to_string())),
            ),
            (
                "Y".to_string(),
                Term::Const(Constant::String("bob".to_string())),
            ),
            (
                "Z".to_string(),
                Term::Const(Constant::String("charlie".to_string())),
            ),
        ],
    );

    // Add metadata before storing
    println!("\n--- Adding Metadata ---");

    grandparent_proof.metadata = ProofMetadata::new()
        .with_created_at(1234567890)
        .with_created_by("example_user".to_string())
        .with_complexity(3)
        .with_depth(2);

    println!("Adding metadata to grandparent fragment:");
    println!("  Created at: {:?}", grandparent_proof.metadata.created_at);
    println!("  Created by: {:?}", grandparent_proof.metadata.created_by);
    println!("  Complexity: {:?}", grandparent_proof.metadata.complexity);
    println!("  Depth: {}", grandparent_proof.metadata.depth);

    let id5 = store.add(grandparent_proof);
    println!("\nAdded rule-based fragment with metadata: {}", id5);
    println!("Total fragments in store: {}", store.len());

    // Query fragments by predicate
    println!("\n--- Querying Fragments ---");

    let parent_fragments = store.find_by_conclusion("parent");
    println!(
        "Found {} fragments with 'parent' conclusion",
        parent_fragments.len()
    );

    let grandparent_fragments = store.find_by_conclusion("grandparent");
    println!(
        "Found {} fragments with 'grandparent' conclusion",
        grandparent_fragments.len()
    );

    // Proof verification
    println!("\n--- Proof Verification ---");

    let assembler = ProofAssembler::new(&store);

    // Verify a fact proof
    if let Some(fragment) = store.get(&id1) {
        let proof = fragment.to_proof(vec![]);
        let is_valid = assembler.verify(&proof);
        println!(
            "Fact proof verification: {}",
            if is_valid { "✓ Valid" } else { "✗ Invalid" }
        );
    }

    // Compression
    println!("\n--- Proof Compression ---");

    println!("Before compression:");
    println!("  Total fragments: {}", store.len());

    let mut compressor = ProofCompressor::new();
    let stats = compressor.compress(&mut store);

    println!("\nCompression statistics:");
    println!("  Original count: {}", stats.original_count);
    println!("  Compressed count: {}", stats.compressed_count);
    println!("  Shared subproofs: {}", stats.shared_subproofs);
    println!("  Size reduction: {} bytes", stats.size_reduction);
    println!(
        "  Compression ratio: {:.2}%",
        stats.compression_ratio() * 100.0
    );
    println!("  Space savings: {:.2}%", stats.space_savings());

    println!("\nAfter compression:");
    println!("  Total fragments: {}", store.len());

    // Delta encoding
    println!("\n--- Delta Encoding ---");

    let base_proof = ProofFragment::fact(Predicate::new(
        "likes".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("pizza".to_string())),
        ],
    ));

    let new_proof = ProofFragment::fact(Predicate::new(
        "likes".to_string(),
        vec![
            Term::Const(Constant::String("bob".to_string())),
            Term::Const(Constant::String("pizza".to_string())),
        ],
    ));

    let delta = compressor.compute_delta(&base_proof, &new_proof);
    println!(
        "Delta between base and new proof: {} fragments",
        delta.len()
    );

    // Same proof should have no delta
    let same_delta = compressor.compute_delta(&base_proof, &base_proof);
    println!(
        "Delta between identical proofs: {} fragments (should be 0)",
        same_delta.len()
    );

    // Storage statistics
    println!("\n--- Storage Statistics ---");

    println!("Final store statistics:");
    println!("  Total fragments: {}", store.len());

    let fact_count =
        store.find_by_conclusion("parent").len() + store.find_by_conclusion("likes").len();
    let derived_count = store.find_by_conclusion("grandparent").len();

    println!("  Fact proofs: ~{}", fact_count);
    println!("  Derived proofs: {}", derived_count);

    // Demonstrate indexing
    println!("\n--- Fragment Indexing ---");

    println!("Fragments by predicate:");
    let predicates = vec!["parent", "grandparent", "likes"];

    for pred in &predicates {
        let count = store.find_by_conclusion(pred).len();
        if count > 0 {
            println!("  {}: {} fragments", pred, count);
        }
    }

    println!("\n--- Summary ---");
    println!("✓ Created {} proof fragments", stats.original_count);
    println!("✓ Added metadata to proofs");
    println!("✓ Verified proof validity");
    println!(
        "✓ Compressed proofs with {:.1}% space savings",
        stats.space_savings()
    );
    println!("✓ Implemented delta encoding");
    println!("✓ Indexed fragments by conclusion");
    println!("\n✓ Example completed successfully!");

    Ok(())
}
