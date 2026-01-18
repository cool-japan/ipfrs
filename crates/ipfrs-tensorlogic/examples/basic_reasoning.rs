//! Basic TensorLogic reasoning example
//!
//! Demonstrates:
//! - Creating facts and rules
//! - Building a knowledge base
//! - Performing inference
//! - Backward chaining
//! - Query optimization

use ipfrs_tensorlogic::ir::{Constant, KnowledgeBase, Predicate, Rule, Term};
use ipfrs_tensorlogic::optimizer::QueryOptimizer;
use ipfrs_tensorlogic::reasoning::InferenceEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TensorLogic Basic Reasoning Example ===\n");

    // Create a knowledge base
    let mut kb = KnowledgeBase::new();

    println!("--- Building Knowledge Base ---");

    // Add facts: parent(X, Y) means X is a parent of Y
    println!("Adding parent facts...");
    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("charlie".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("bob".to_string())),
            Term::Const(Constant::String("david".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("charlie".to_string())),
            Term::Const(Constant::String("eve".to_string())),
        ],
    ));

    // Add facts: gender(X, G) means X has gender G
    println!("Adding gender facts...");
    kb.add_fact(Predicate::new(
        "gender".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("female".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "gender".to_string(),
        vec![
            Term::Const(Constant::String("bob".to_string())),
            Term::Const(Constant::String("male".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "gender".to_string(),
        vec![
            Term::Const(Constant::String("charlie".to_string())),
            Term::Const(Constant::String("male".to_string())),
        ],
    ));

    println!("Total facts: {}\n", kb.facts.len());

    // Add rules
    println!("--- Adding Rules ---");

    // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    println!("Adding grandparent rule...");
    kb.add_rule(Rule::new(
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
    ));

    // Rule: mother(X, Y) :- parent(X, Y), gender(X, female)
    println!("Adding mother rule...");
    kb.add_rule(Rule::new(
        Predicate::new(
            "mother".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "gender".to_string(),
                vec![
                    Term::Var("X".to_string()),
                    Term::Const(Constant::String("female".to_string())),
                ],
            ),
        ],
    ));

    // Rule: father(X, Y) :- parent(X, Y), gender(X, male)
    println!("Adding father rule...");
    kb.add_rule(Rule::new(
        Predicate::new(
            "father".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "gender".to_string(),
                vec![
                    Term::Var("X".to_string()),
                    Term::Const(Constant::String("male".to_string())),
                ],
            ),
        ],
    ));

    println!("Total rules: {}\n", kb.rules.len());

    // Create inference engine
    let engine = InferenceEngine::new();

    println!("--- Performing Inference ---");

    // Query 1: Find all mothers
    println!("\nQuery: mother(X, Y)");
    let query1 = Predicate::new(
        "mother".to_string(),
        vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
    );

    match engine.query(&query1, &kb) {
        Ok(solutions) => {
            println!(
                "✓ Found {} solution(s) for mother relationship",
                solutions.len()
            );
            if let Some(first) = solutions.first() {
                println!("  First solution: {:?}", first);
            }
        }
        Err(e) => println!("✗ Query failed: {}", e),
    }

    // Query 2: Find grandparents
    println!("\nQuery: grandparent(X, Y)");
    let query2 = Predicate::new(
        "grandparent".to_string(),
        vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
    );

    match engine.query(&query2, &kb) {
        Ok(solutions) => {
            println!(
                "✓ Found {} solution(s) for grandparent relationship",
                solutions.len()
            );
            for (i, solution) in solutions.iter().take(3).enumerate() {
                println!("  Solution {}: {:?}", i + 1, solution);
            }
        }
        Err(e) => println!("✗ Query failed: {}", e),
    }

    // Query 3: Specific query - Is alice a mother of bob?
    println!("\nQuery: mother(alice, bob)");
    let query3 = Predicate::new(
        "mother".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    );

    match engine.query(&query3, &kb) {
        Ok(solutions) if !solutions.is_empty() => {
            println!("✓ Yes, alice is a mother of bob");
        }
        Ok(_) => println!("✗ No solution found"),
        Err(e) => println!("✗ Query failed: {}", e),
    }

    // Query Optimization
    println!("\n--- Query Optimization ---");

    let mut optimizer = QueryOptimizer::new();
    optimizer.update_statistics(&kb);

    println!("Knowledge base statistics:");
    println!("  Total facts: {}", optimizer.total_facts());

    for (pred_name, stats) in optimizer.all_stats() {
        println!(
            "  Predicate '{}': {} facts, selectivity: {:.3}",
            pred_name, stats.fact_count, stats.selectivity
        );
    }

    // Optimize a rule
    println!("\nOptimizing grandparent rule...");
    let grandparent_rule = kb
        .rules
        .iter()
        .find(|r| r.head.name == "grandparent")
        .unwrap();

    let optimized_rule = optimizer.optimize_rule(grandparent_rule, &kb);
    println!(
        "  Original order: {:?}",
        grandparent_rule
            .body
            .iter()
            .map(|p| &p.name)
            .collect::<Vec<_>>()
    );
    println!(
        "  Optimized order: {:?}",
        optimized_rule
            .body
            .iter()
            .map(|p| &p.name)
            .collect::<Vec<_>>()
    );

    // Query planning
    println!("\nCreating query plan for grandparent query...");
    let plan = optimizer.plan_query(std::slice::from_ref(&query2), &kb);
    println!("  Estimated cost: {:.2}", plan.estimated_cost);
    println!("  Estimated rows: {:.2}", plan.estimated_rows);

    println!("\n--- Summary ---");
    println!(
        "✓ Created knowledge base with {} facts and {} rules",
        kb.facts.len(),
        kb.rules.len()
    );
    println!("✓ Performed backward chaining inference");
    println!("✓ Applied query optimization");
    println!("\n✓ Example completed successfully!");

    Ok(())
}
