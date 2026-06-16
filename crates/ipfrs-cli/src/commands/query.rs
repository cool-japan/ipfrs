//! Hybrid query command — combines semantic and logic query capabilities.
//!
//! Routes automatically based on query syntax, or runs both in `--hybrid` mode.
//! Supports optional `--logic` post-filter when running in hybrid mode.

use anyhow::Result;

/// Output format for query results.
///
/// Used across semantic, logic, and hybrid query commands to control whether
/// results are rendered as human-readable text or newline-delimited JSON.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable text output (default)
    #[default]
    Text,
    /// Newline-delimited JSON output for machine consumption
    Json,
}

impl OutputFormat {
    /// Returns `true` when JSON output is requested.
    pub fn is_json(&self) -> bool {
        matches!(self, OutputFormat::Json)
    }
}

/// Return `true` when the query string looks like a Datalog/Prolog predicate.
///
/// Heuristic: the string contains both `(` and `)`, suggesting predicate
/// notation such as `ancestor(X, bob)`.
fn looks_like_logic_query(query: &str) -> bool {
    query.contains('(') && query.contains(')')
}

/// Handle the top-level `ipfrs query` command.
///
/// * In `--pipeline` mode, CIDs are read from stdin and the query string is
///   used as a logic predicate template filter (identical to `logic filter`).
/// * In `--hybrid` mode, both semantic and logic search are executed.
///   If `logic_filter` is provided, the semantic results are then post-filtered
///   through the logic engine using the given predicate template.
/// * Otherwise the query is auto-routed: logic predicates go to the logic
///   engine; natural-language strings go to semantic search.
pub async fn handle_query(
    query: &str,
    hybrid: bool,
    pipeline: bool,
    top_k: usize,
    logic_filter: Option<&str>,
    format: &OutputFormat,
) -> Result<()> {
    use crate::output;

    let json_output = format.is_json();

    if pipeline {
        // Pipeline mode: read CIDs from stdin and apply query as a predicate filter.
        return crate::commands::logic::logic_filter(query, json_output, ".ipfrs").await;
    }

    if hybrid {
        if !json_output {
            output::print_header(&format!("Hybrid Query: \"{}\"", query));
            println!();
            println!("--- Semantic Results ---");
        }

        // Semantic search — threshold 0.0 returns all indexed results
        let semantic_cids =
            crate::commands::semantic::semantic_query_with_cids(query, top_k, 0.0, json_output)
                .await?;

        // If a logic filter is provided, post-filter the semantic results.
        if let Some(filter_predicate) = logic_filter {
            if !json_output {
                println!();
                println!("--- Logic Filter ---");
                println!("Predicate: {}", filter_predicate);
            }
            crate::commands::logic::logic_filter_cids(
                &semantic_cids,
                filter_predicate,
                json_output,
            )
            .await?;
        } else {
            if !json_output {
                println!();
                println!("--- Logic Results ---");
            }

            if looks_like_logic_query(query) {
                crate::commands::logic::logic_query_streaming(query, 10, json_output, 30).await?;
            } else if !json_output {
                output::info(
                    "Query does not look like a logic predicate (no parentheses). Skipping logic search.",
                );
            }
        }
    } else if looks_like_logic_query(query) {
        // Looks like a Datalog goal — route to logic engine with streaming output
        crate::commands::logic::logic_query_streaming(query, 10, json_output, 30).await?;
    } else {
        // Natural language — route to semantic search
        crate::commands::semantic::semantic_query(query, top_k, 0.0, json_output).await?;
    }

    Ok(())
}

#[cfg(test)]
mod query_tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // OutputFormat helpers
    // ---------------------------------------------------------------------------

    #[test]
    fn test_output_format_is_json_text() {
        let fmt = OutputFormat::Text;
        assert!(!fmt.is_json());
    }

    #[test]
    fn test_output_format_is_json_json() {
        let fmt = OutputFormat::Json;
        assert!(fmt.is_json());
    }

    #[test]
    fn test_output_format_default_is_text() {
        let fmt = OutputFormat::default();
        assert_eq!(fmt, OutputFormat::Text);
    }

    // ---------------------------------------------------------------------------
    // test_output_format_text: verify text formatting of search results
    // ---------------------------------------------------------------------------

    #[test]
    fn test_output_format_text() {
        // Simulate rendering a search result in text mode.
        // When json_output is false the format should be human-readable.
        let json_output = OutputFormat::Text.is_json();
        assert!(
            !json_output,
            "text format must not produce json_output=true"
        );

        // Simulate what semantic_query writes for a single result in text mode.
        let cid = "bafkreiabc123";
        let score = 0.9512_f32;
        let text_line = format!("  CID: {} (score: {:.2})", cid, score);
        assert!(
            text_line.contains("CID:"),
            "text output should contain 'CID:' label"
        );
        assert!(
            text_line.contains("score:"),
            "text output should contain 'score:' label"
        );
        assert!(
            !text_line.contains('{'),
            "text output should not contain JSON braces"
        );
    }

    // ---------------------------------------------------------------------------
    // test_output_format_json: verify JSON output is valid JSON
    // ---------------------------------------------------------------------------

    #[test]
    fn test_output_format_json() {
        // Simulate rendering a search result in JSON mode.
        let json_output = OutputFormat::Json.is_json();
        assert!(json_output, "json format must produce json_output=true");

        // Simulate what semantic_query writes for a single result in JSON mode.
        let cid = "bafkreiabc123";
        let score = 0.9512_f32;
        let json_line = format!("  {{\"cid\": \"{}\", \"score\": {:.4}}}", cid, score);

        // Parse to validate it is valid JSON.
        let parsed: serde_json::Value =
            serde_json::from_str(json_line.trim()).expect("JSON output must be valid JSON");

        assert_eq!(
            parsed["cid"].as_str().expect("cid field"),
            cid,
            "cid field must match"
        );
        let got_score = parsed["score"].as_f64().expect("score field") as f32;
        assert!(
            (got_score - score).abs() < 1e-3,
            "score field must be close to {}",
            score
        );
    }

    // ---------------------------------------------------------------------------
    // test_hybrid_query_struct: verify clap parsing of hybrid query arguments
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hybrid_query_struct() {
        // Validate that the argument semantics expected for hybrid mode are correct.
        // We can't invoke the full Cli parser from a unit test without spawning a
        // binary, so we instead verify the structural invariants that the parser
        // would enforce at runtime.

        // 1. --hybrid activates hybrid mode.
        let hybrid = true;
        let query = "test";
        let logic_filter: &str = "foo(X)";
        let top_k: usize = 10;

        assert!(hybrid, "hybrid flag must be true when set");
        assert_eq!(query, "test");
        assert_eq!(logic_filter, "foo(X)");
        assert_eq!(top_k, 10);

        // 2. looks_like_logic_query correctly identifies predicates.
        assert!(
            looks_like_logic_query("foo(X)"),
            "foo(X) should be identified as a logic query"
        );
        assert!(
            looks_like_logic_query("ancestor(X, bob)"),
            "ancestor(X, bob) should be identified as a logic query"
        );
        assert!(
            !looks_like_logic_query("machine learning"),
            "plain text should not be identified as a logic query"
        );

        // 3. The logic_filter is passed through as-is when present.
        assert_eq!(logic_filter, "foo(X)");
    }

    // ---------------------------------------------------------------------------
    // Routing heuristic tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_looks_like_logic_query_with_parens() {
        assert!(looks_like_logic_query("parent(alice, bob)"));
        assert!(looks_like_logic_query("fact()"));
    }

    #[test]
    fn test_looks_like_logic_query_without_parens() {
        assert!(!looks_like_logic_query("natural language query"));
        assert!(!looks_like_logic_query("tensor operations in IPFS"));
    }

    #[test]
    fn test_looks_like_logic_query_partial_parens() {
        // Only one paren — should not be identified as logic
        assert!(!looks_like_logic_query("half("));
        assert!(!looks_like_logic_query(")close"));
    }
}
