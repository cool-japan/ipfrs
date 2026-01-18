//! Logic programming commands
//!
//! This module provides logic operations:
//! - `logic_infer` - Run inference query
//! - `logic_prove` - Generate proof
//! - `logic_kb_stats` - Knowledge base statistics
//! - `logic_kb_save` - Save knowledge base
//! - `logic_kb_load` - Load knowledge base

use anyhow::Result;

/// Run inference query
pub async fn logic_infer(predicate: &str, terms: &[String], format: &str) -> Result<()> {
    use ipfrs::{Constant, Node, NodeConfig, Predicate, Term};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Parse terms from JSON strings
    let mut parsed_terms = Vec::new();
    for term_str in terms {
        if term_str.starts_with('"') && term_str.ends_with('"') {
            // String constant
            let s = term_str.trim_matches('"');
            parsed_terms.push(Term::Const(Constant::String(s.to_string())));
        } else if term_str.parse::<i64>().is_ok() {
            // Integer constant
            let n = term_str.parse::<i64>()?;
            parsed_terms.push(Term::Const(Constant::Int(n)));
        } else if term_str.starts_with('?') || term_str.chars().next().unwrap().is_uppercase() {
            // Variable
            parsed_terms.push(Term::Var(term_str.to_string()));
        } else {
            return Err(anyhow::anyhow!("Invalid term: {}", term_str));
        }
    }

    let goal = Predicate::new(predicate.to_string(), parsed_terms);

    println!("Running inference query: {}", goal);
    let solutions = node.infer(&goal)?;

    match format {
        "json" => {
            println!("{{");
            println!("  \"goal\": \"{}\",", goal);
            println!("  \"solutions\": [");
            for (i, solution) in solutions.iter().enumerate() {
                print!("    {{");
                for (j, (var, term)) in solution.iter().enumerate() {
                    print!("\"{}\": \"{}\"", var, term);
                    if j < solution.len() - 1 {
                        print!(", ");
                    }
                }
                print!("}}");
                if i < solutions.len() - 1 {
                    println!(",");
                } else {
                    println!();
                }
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            if solutions.is_empty() {
                println!("No solutions found");
            } else {
                println!("Found {} solution(s):", solutions.len());
                for (i, solution) in solutions.iter().enumerate() {
                    println!("  Solution {}:", i + 1);
                    for (var, term) in solution {
                        println!("    {} = {}", var, term);
                    }
                }
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Generate proof for a goal
pub async fn logic_prove(predicate: &str, terms: &[String], format: &str) -> Result<()> {
    use ipfrs::{Constant, Node, NodeConfig, Predicate, Term};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Parse terms from JSON strings (same as infer)
    let mut parsed_terms = Vec::new();
    for term_str in terms {
        if term_str.starts_with('"') && term_str.ends_with('"') {
            let s = term_str.trim_matches('"');
            parsed_terms.push(Term::Const(Constant::String(s.to_string())));
        } else if term_str.parse::<i64>().is_ok() {
            let n = term_str.parse::<i64>()?;
            parsed_terms.push(Term::Const(Constant::Int(n)));
        } else if term_str.starts_with('?') || term_str.chars().next().unwrap().is_uppercase() {
            parsed_terms.push(Term::Var(term_str.to_string()));
        } else {
            return Err(anyhow::anyhow!("Invalid term: {}", term_str));
        }
    }

    let goal = Predicate::new(predicate.to_string(), parsed_terms);

    println!("Generating proof for: {}", goal);
    let proof = node.prove(&goal)?;

    match format {
        "json" => {
            println!("{{");
            println!("  \"goal\": \"{}\",", goal);
            if let Some(p) = &proof {
                println!("  \"proof_found\": true,");
                println!("  \"proof\": {{");
                println!("    \"goal\": \"{}\",", p.goal);
                if let Some(rule) = &p.rule {
                    println!("    \"is_fact\": {},", rule.is_fact);
                    println!("    \"subproofs\": {}", p.subproofs.len());
                } else {
                    println!("    \"is_fact\": true,");
                    println!("    \"subproofs\": 0");
                }
                println!("  }}");
            } else {
                println!("  \"proof_found\": false");
            }
            println!("}}");
        }
        _ => {
            if let Some(p) = &proof {
                println!("Proof found!");
                println!("Goal: {}", p.goal);
                if let Some(rule) = &p.rule {
                    if rule.is_fact {
                        println!("Proved by fact");
                    } else {
                        println!("Proved by rule: {} :- {:?}", rule.head, rule.body);
                        println!("Number of subproofs: {}", p.subproofs.len());
                    }
                }
            } else {
                println!("No proof found");
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Show knowledge base statistics
pub async fn logic_kb_stats(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let stats = node.kb_stats()?;

    match format {
        "json" => {
            println!("{{");
            println!("  \"num_facts\": {},", stats.num_facts);
            println!("  \"num_rules\": {}", stats.num_rules);
            println!("}}");
        }
        _ => {
            println!("Knowledge Base Statistics");
            println!("=========================");
            println!("Facts: {}", stats.num_facts);
            println!("Rules: {}", stats.num_rules);
        }
    }

    node.stop().await?;
    Ok(())
}

/// Save knowledge base to file
pub async fn logic_kb_save(path: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Saving knowledge base to {}...", path);
    node.save_knowledge_base(path).await?;
    println!("Knowledge base saved successfully");

    node.stop().await?;
    Ok(())
}

/// Load knowledge base from file
pub async fn logic_kb_load(path: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Loading knowledge base from {}...", path);
    node.load_knowledge_base(path).await?;
    println!("Knowledge base loaded successfully");

    let stats = node.kb_stats()?;
    println!(
        "Loaded {} facts and {} rules",
        stats.num_facts, stats.num_rules
    );

    node.stop().await?;
    Ok(())
}
