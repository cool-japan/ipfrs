//! Logic Programming Example
//!
//! This example demonstrates IPFRS logic programming capabilities:
//! - Storing logical terms
//! - Working with predicates
//! - Defining inference rules
//! - Content-addressed reasoning

use ipfrs::{Constant, Node, NodeConfig};
use ipfrs_tensorlogic::{Predicate, Rule, Term};

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFRS Logic Programming Example ===\n");

    // Create and start node
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;
    println!("✓ Node started");

    // Verify TensorLogic is enabled
    if !node.is_tensorlogic_enabled() {
        eprintln!("ERROR: TensorLogic not enabled!");
        return Ok(());
    }

    // Example 1: Store logical terms
    println!("\n--- Example 1: Logical Terms ---");

    let var_x = Term::Var("X".to_string());
    let var_y = Term::Var("Y".to_string());
    let const_alice = Term::Const(Constant::String("alice".to_string()));
    let const_bob = Term::Const(Constant::String("bob".to_string()));

    let var_x_cid = node.put_term(&var_x).await?;
    let const_alice_cid = node.put_term(&const_alice).await?;

    println!("Stored term Variable(X): {}", var_x_cid);
    println!("Stored term Constant(alice): {}", const_alice_cid);

    // Retrieve a term
    if let Some(retrieved_term) = node.get_term(&var_x_cid).await? {
        println!("Retrieved term: {:?}", retrieved_term);
    }

    // Example 2: Store predicates (facts)
    println!("\n--- Example 2: Predicates (Facts) ---");

    // parent(alice, bob) - Alice is parent of Bob
    let parent_alice_bob = Predicate {
        name: "parent".to_string(),
        args: vec![const_alice.clone(), const_bob.clone()],
    };

    // parent(bob, charlie) - Bob is parent of Charlie
    let parent_bob_charlie = Predicate {
        name: "parent".to_string(),
        args: vec![
            const_bob.clone(),
            Term::Const(Constant::String("charlie".to_string())),
        ],
    };

    let pred1_cid = node.store_predicate(&parent_alice_bob).await?;
    let pred2_cid = node.store_predicate(&parent_bob_charlie).await?;

    println!("Stored fact: parent(alice, bob) → {}", pred1_cid);
    println!("Stored fact: parent(bob, charlie) → {}", pred2_cid);

    // Retrieve a predicate
    if let Some(retrieved_pred) = node.get_predicate(&pred1_cid).await? {
        println!(
            "Retrieved fact: {}({}, {})",
            retrieved_pred.name,
            format_term(&retrieved_pred.args[0]),
            format_term(&retrieved_pred.args[1])
        );
    }

    // Example 3: Store inference rules
    println!("\n--- Example 3: Inference Rules ---");

    // Rule: ancestor(X, Y) :- parent(X, Y)
    // (If X is parent of Y, then X is ancestor of Y)
    let ancestor_rule1 = Rule {
        head: Predicate {
            name: "ancestor".to_string(),
            args: vec![var_x.clone(), var_y.clone()],
        },
        body: vec![Predicate {
            name: "parent".to_string(),
            args: vec![var_x.clone(), var_y.clone()],
        }],
    };

    // Rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
    // (If X is parent of Y and Y is ancestor of Z, then X is ancestor of Z)
    let var_z = Term::Var("Z".to_string());
    let ancestor_rule2 = Rule {
        head: Predicate {
            name: "ancestor".to_string(),
            args: vec![var_x.clone(), var_z.clone()],
        },
        body: vec![
            Predicate {
                name: "parent".to_string(),
                args: vec![var_x.clone(), var_y.clone()],
            },
            Predicate {
                name: "ancestor".to_string(),
                args: vec![var_y.clone(), var_z.clone()],
            },
        ],
    };

    let rule1_cid = node.store_rule(&ancestor_rule1).await?;
    let rule2_cid = node.store_rule(&ancestor_rule2).await?;

    println!("Stored rule 1: ancestor(X,Y) :- parent(X,Y)");
    println!("  CID: {}", rule1_cid);
    println!("Stored rule 2: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)");
    println!("  CID: {}", rule2_cid);

    // Retrieve a rule
    if let Some(retrieved_rule) = node.get_rule(&rule1_cid).await? {
        println!("\nRetrieved rule:");
        println!(
            "  Head: {}({})",
            retrieved_rule.head.name,
            retrieved_rule
                .head
                .args
                .iter()
                .map(format_term)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("  Body:");
        for pred in &retrieved_rule.body {
            println!(
                "    {}({})",
                pred.name,
                pred.args
                    .iter()
                    .map(format_term)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    // Example 4: Knowledge base overview
    println!("\n--- Example 4: Knowledge Base Overview ---");

    println!("Facts stored:");
    println!("  parent(alice, bob)");
    println!("  parent(bob, charlie)");

    println!("\nRules stored:");
    println!("  ancestor(X,Y) :- parent(X,Y)");
    println!("  ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)");

    println!("\nInference (conceptual):");
    println!("  ancestor(alice, bob)   ← from rule 1 + fact");
    println!("  ancestor(bob, charlie) ← from rule 1 + fact");
    println!("  ancestor(alice, charlie) ← from rule 2 + facts");

    println!("\nNote: Actual inference engine will be implemented in 0.2.0");

    // Example 5: TensorLogic statistics
    println!("\n--- Example 5: TensorLogic Statistics ---");

    let stats = node.tensorlogic_stats()?;
    println!("TensorLogic enabled: {}", stats.enabled);

    // Clean shutdown
    println!("\n--- Shutting Down ---");
    node.stop().await?;
    println!("✓ Node stopped");

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Helper function to format terms for display
fn format_term(term: &Term) -> String {
    match term {
        Term::Var(name) => name.clone(),
        Term::Const(Constant::String(s)) => s.clone(),
        Term::Const(Constant::Int(i)) => i.to_string(),
        Term::Const(Constant::Float(f)) => f.to_string(),
        Term::Const(Constant::Bool(b)) => b.to_string(),
        Term::Fun(functor, args) => {
            format!(
                "{}({})",
                functor,
                args.iter().map(format_term).collect::<Vec<_>>().join(", ")
            )
        }
        Term::Ref(_) => "<ref>".to_string(),
    }
}
