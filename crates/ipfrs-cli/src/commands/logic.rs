//! Logic programming commands
//!
//! This module provides logic operations:
//! - `logic_infer` - Run inference query
//! - `logic_prove` - Generate proof
//! - `logic_kb_stats` - Knowledge base statistics
//! - `logic_kb_save` - Save knowledge base
//! - `logic_kb_load` - Load knowledge base

use anyhow::Result;

/// Parse a goal string like `ancestor(X, bob)` into a `Predicate`.
///
/// Returns an error with a descriptive message if the string is malformed.
fn parse_goal(goal: &str) -> Result<ipfrs::Predicate> {
    use ipfrs::{Constant, Predicate, Term};

    let goal = goal.trim();
    let paren_open = goal.find('(').ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid goal '{}': expected 'predicate(args...)' syntax",
            goal
        )
    })?;
    let paren_close = goal
        .rfind(')')
        .ok_or_else(|| anyhow::anyhow!("Invalid goal '{}': missing closing ')'", goal))?;

    if paren_close <= paren_open {
        return Err(anyhow::anyhow!(
            "Invalid goal '{}': closing ')' appears before or at '('",
            goal
        ));
    }

    let predicate_name = goal[..paren_open].trim().to_string();
    if predicate_name.is_empty() {
        return Err(anyhow::anyhow!(
            "Invalid goal '{}': predicate name is empty",
            goal
        ));
    }

    let args_str = &goal[paren_open + 1..paren_close];
    let mut terms = Vec::new();

    for raw in args_str.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let term = if token.starts_with('"') && token.ends_with('"') {
            let s = token.trim_matches('"');
            Term::Const(Constant::String(s.to_string()))
        } else if let Ok(n) = token.parse::<i64>() {
            Term::Const(Constant::Int(n))
        } else if token.starts_with('?')
            || token
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        {
            Term::Var(token.to_string())
        } else {
            // Treat bare atoms as string constants
            Term::Const(Constant::String(token.to_string()))
        };
        terms.push(term);
    }

    Ok(Predicate::new(predicate_name, terms))
}

/// Logic Datalog-style query with streaming output and indicatif spinner.
///
/// Identical to [`logic_query`] in terms of logic execution, but shows a
/// "Searching…" spinner on TTY while inference is running, then prints each
/// result binding as it arrives.  JSON output uses newline-delimited records.
pub async fn logic_query_streaming(
    goal: &str,
    max_depth: usize,
    json_output: bool,
    timeout_secs: u64,
) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use std::time::Instant;

    let goal_pred = parse_goal(goal)?;

    let mut node = Node::new(NodeConfig::default().with_tensorlogic())?;
    node.start().await?;

    // Show a spinner while inference runs — hidden on non-TTY automatically
    // because indicatif ProgressBar uses atty detection internally.
    let spinner = {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::time::Duration;
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        pb.set_message("Searching...");
        pb.enable_steady_tick(Duration::from_millis(80));
        pb
    };

    let start = Instant::now();

    let solutions = match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        async { node.infer(&goal_pred) },
    )
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(_)) => {
            spinner.finish_and_clear();
            crate::output::warning(
                "TensorLogic not initialized. Use 'ipfrs logic' commands to load a knowledge base first.",
            );
            node.stop().await?;
            if json_output {
                println!(
                    "{{\"goal\": \"{}\", \"proved\": false, \"solutions\": [], \"time_ms\": 0}}",
                    goal
                );
            } else {
                println!("Logic query: {}", goal);
                println!("Proved: No");
                println!();
                println!("Bindings:");
                println!("  (none)");
                println!();
                println!("Time: 0ms | Depth: {}", max_depth);
            }
            return Ok(());
        }
        Err(_) => {
            spinner.finish_and_clear();
            node.stop().await?;
            return Err(anyhow::anyhow!(
                "Logic query timed out after {} seconds",
                timeout_secs
            ));
        }
    };

    let elapsed_ms = start.elapsed().as_millis();
    spinner.finish_and_clear();
    node.stop().await?;

    let proved = !solutions.is_empty();

    if json_output {
        // Newline-delimited JSON — emit header record then one record per solution.
        println!(
            "{{\"goal\": \"{}\", \"proved\": {}, \"time_ms\": {}}}",
            goal, proved, elapsed_ms
        );
        for solution in &solutions {
            let binding_pairs: Vec<String> = solution
                .iter()
                .map(|(var, term)| format!("\"{}\": \"{}\"", var, term))
                .collect();
            println!("{{{}}}", binding_pairs.join(", "));
        }
    } else {
        println!("Logic query: {}", goal);
        println!("Proved: {}", if proved { "Yes" } else { "No" });
        println!();
        println!("Bindings:");
        if solutions.is_empty() {
            println!("  (none)");
        } else {
            // Stream each solution as it would arrive.
            for (i, solution) in solutions.iter().enumerate() {
                let bindings: Vec<String> = solution
                    .iter()
                    .map(|(var, term)| format!("{} = {}", var, term))
                    .collect();
                println!("  Solution {}: {}", i + 1, bindings.join(", "));
            }
        }
        println!();
        println!("Time: {}ms | Depth: {}", elapsed_ms, max_depth);
    }

    Ok(())
}

/// Filter an explicit list of CIDs through the logic engine.
///
/// This is the programmatic counterpart to [`logic_filter`], accepting an
/// already-collected `Vec<String>` of CIDs instead of reading from stdin.
/// Used by the hybrid pipeline to apply a logic predicate to semantic results.
pub async fn logic_filter_cids(
    cids: &[String],
    predicate_template: &str,
    json_output: bool,
) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    if cids.is_empty() {
        if json_output {
            println!("[]");
        }
        return Ok(());
    }

    let mut node = Node::new(NodeConfig::default().with_tensorlogic())?;
    node.start().await?;

    let mut matched_cids: Vec<String> = Vec::new();

    for cid in cids {
        let goal_str = predicate_template.replace('X', cid);
        let goal_pred = match parse_goal(&goal_str) {
            Ok(p) => p,
            Err(e) => {
                crate::output::warning(&format!("Skipping CID {}: {}", cid, e));
                continue;
            }
        };
        let solutions = node.infer(&goal_pred).unwrap_or_default();
        if !solutions.is_empty() {
            matched_cids.push(cid.clone());
        }
    }

    node.stop().await.ok();

    if json_output {
        let json_items: Vec<String> = matched_cids.iter().map(|c| format!("\"{}\"", c)).collect();
        println!("[{}]", json_items.join(", "));
    } else {
        if matched_cids.is_empty() {
            println!("  (no CIDs matched predicate: {})", predicate_template);
        } else {
            for cid in &matched_cids {
                println!("  {}", cid);
            }
        }
    }

    Ok(())
}

/// Logic Datalog-style query: `ipfrs logic query "ancestor(X, bob)"`
pub async fn logic_query(
    goal: &str,
    max_depth: usize,
    json_output: bool,
    timeout_secs: u64,
) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use std::time::Instant;

    let goal_pred = parse_goal(goal)?;

    let mut node = Node::new(NodeConfig::default().with_tensorlogic())?;
    node.start().await?;

    let start = Instant::now();

    let solutions = match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        async { node.infer(&goal_pred) },
    )
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(_)) => {
            crate::output::warning(
                "TensorLogic not initialized. Use 'ipfrs logic' commands to load a knowledge base first.",
            );
            node.stop().await?;
            if json_output {
                println!(
                    "{{\"goal\": \"{}\", \"proved\": false, \"solutions\": [], \"time_ms\": 0}}",
                    goal
                );
            } else {
                println!("Logic query: {}", goal);
                println!("Proved: No");
                println!();
                println!("Bindings:");
                println!("  (none)");
                println!();
                println!("Time: 0ms | Depth: {}", max_depth);
            }
            return Ok(());
        }
        Err(_) => {
            node.stop().await?;
            return Err(anyhow::anyhow!(
                "Logic query timed out after {} seconds",
                timeout_secs
            ));
        }
    };

    let elapsed_ms = start.elapsed().as_millis();
    node.stop().await?;

    let proved = !solutions.is_empty();

    if json_output {
        println!("{{");
        println!("  \"goal\": \"{}\",", goal);
        println!("  \"proved\": {},", proved);
        println!("  \"solutions\": [");
        for (i, solution) in solutions.iter().enumerate() {
            let comma = if i + 1 < solutions.len() { "," } else { "" };
            let binding_pairs: Vec<String> = solution
                .iter()
                .map(|(var, term)| format!("\"{}\": \"{}\"", var, term))
                .collect();
            println!("    {{{}}} {}", binding_pairs.join(", "), comma);
        }
        println!("  ],");
        println!("  \"time_ms\": {}", elapsed_ms);
        println!("}}");
    } else {
        println!("Logic query: {}", goal);
        println!("Proved: {}", if proved { "Yes" } else { "No" });
        println!();
        println!("Bindings:");
        if solutions.is_empty() {
            println!("  (none)");
        } else {
            for (i, solution) in solutions.iter().enumerate() {
                let bindings: Vec<String> = solution
                    .iter()
                    .map(|(var, term)| format!("{} = {}", var, term))
                    .collect();
                println!("  Solution {}: {}", i + 1, bindings.join(", "));
            }
        }
        println!();
        println!("Time: {}ms | Depth: {}", elapsed_ms, max_depth);
    }

    Ok(())
}

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
        } else if term_str.starts_with('?')
            || term_str
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        {
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
        } else if term_str.starts_with('?')
            || term_str
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        {
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

/// Read CIDs from stdin and filter by logic predicate.
///
/// Each non-empty line on stdin is treated as a CID. The predicate template
/// uses `X` as a placeholder that is replaced by the CID before inference.
///
/// # Example
/// ```text
/// echo "bafkrei123" | ipfrs logic filter "indexed(X)"
/// ```
pub async fn logic_filter(
    predicate_template: &str,
    json_output: bool,
    data_dir: &str,
) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use std::io::{self, BufRead};

    let _ = data_dir; // reserved for future per-repo node config

    let stdin = io::stdin();
    let mut matched_cids: Vec<String> = Vec::new();

    // Collect all CIDs first so we can start the node once.
    let cids: Vec<String> = stdin
        .lock()
        .lines()
        .filter_map(|line_result| line_result.ok().map(|l| l.trim().to_string()))
        .filter(|l| !l.is_empty())
        .collect();

    if cids.is_empty() {
        // Nothing on stdin — emit empty output and exit cleanly.
        if json_output {
            println!("[]");
        }
        return Ok(());
    }

    let mut node = Node::new(NodeConfig::default().with_tensorlogic())?;
    node.start().await?;

    for cid in &cids {
        // Instantiate the predicate template: "valid(X)" -> "valid(bafkrei...)"
        let goal_str = predicate_template.replace('X', cid);

        let goal_pred = match parse_goal(&goal_str) {
            Ok(p) => p,
            Err(e) => {
                crate::output::warning(&format!("Skipping CID {}: {}", cid, e));
                continue;
            }
        };

        let solutions = node.infer(&goal_pred).unwrap_or_default();
        if !solutions.is_empty() {
            matched_cids.push(cid.clone());
        }
    }

    node.stop().await.ok();

    if json_output {
        let json_items: Vec<String> = matched_cids.iter().map(|c| format!("\"{}\"", c)).collect();
        println!("[{}]", json_items.join(", "));
    } else {
        for cid in &matched_cids {
            println!("{}", cid);
        }
    }

    Ok(())
}

#[cfg(test)]
mod filter_tests {
    #[test]
    fn test_predicate_instantiation() {
        // "valid(X)" -> "valid(bafkrei123)"
        let template = "valid(X)";
        let cid = "bafkrei123";
        let goal = template.replace('X', cid);
        assert_eq!(goal, "valid(bafkrei123)");
    }

    #[test]
    fn test_predicate_instantiation_multiple_vars() {
        // "related(X, topic)" — X is substituted, other tokens untouched
        let goal = "related(X, topic)".replace('X', "cid456");
        assert_eq!(goal, "related(cid456, topic)");
    }

    #[test]
    fn test_predicate_instantiation_no_placeholder() {
        // Template without X is left unchanged
        let goal = "indexed(item)".replace('X', "cid789");
        assert_eq!(goal, "indexed(item)");
    }
}
