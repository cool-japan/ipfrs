//! Query optimization with materialized views example
//!
//! Demonstrates:
//! - Query planning and cost estimation
//! - Creating materialized views
//! - TTL-based view refresh
//! - View eviction policies
//! - Performance tracking

use ipfrs_tensorlogic::ir::{Constant, KnowledgeBase, Predicate, Term};
use ipfrs_tensorlogic::optimizer::{MaterializedViewManager, QueryOptimizer};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Query Optimization with Materialized Views ===\n");

    // Create a large knowledge base
    let mut kb = KnowledgeBase::new();

    println!("--- Building Large Knowledge Base ---");

    // Add many facts to simulate a real scenario
    println!("Adding 1000 user facts...");
    for i in 0..1000 {
        kb.add_fact(Predicate::new(
            "user".to_string(),
            vec![
                Term::Const(Constant::Int(i)),
                Term::Const(Constant::String(format!("user{}", i))),
            ],
        ));
    }

    println!("Adding 500 role facts...");
    for i in 0..500 {
        kb.add_fact(Predicate::new(
            "role".to_string(),
            vec![
                Term::Const(Constant::Int(i)),
                Term::Const(Constant::String(if i % 3 == 0 {
                    "admin".to_string()
                } else if i % 3 == 1 {
                    "editor".to_string()
                } else {
                    "viewer".to_string()
                })),
            ],
        ));
    }

    println!("Adding 2000 activity facts...");
    for i in 0..2000 {
        kb.add_fact(Predicate::new(
            "activity".to_string(),
            vec![
                Term::Const(Constant::Int(i % 1000)), // user_id
                Term::Const(Constant::String(format!("action_{}", i % 10))),
                Term::Const(Constant::Int(i)), // timestamp
            ],
        ));
    }

    println!("Total facts in KB: {}\n", kb.facts.len());

    // Initialize query optimizer
    println!("--- Query Optimizer Setup ---");
    let mut optimizer = QueryOptimizer::new();
    optimizer.update_statistics(&kb);

    println!("Statistics for predicates:");
    for (name, stats) in optimizer.all_stats() {
        println!(
            "  {}: {} facts, selectivity: {:.4}",
            name, stats.fact_count, stats.selectivity
        );
    }

    // Create materialized view manager
    println!("\n--- Materialized View Manager ---");
    let mut view_manager = MaterializedViewManager::new(5); // Max 5 views

    // Define common queries
    let query1 = vec![Predicate::new(
        "user".to_string(),
        vec![Term::Var("ID".to_string()), Term::Var("Name".to_string())],
    )];

    let query2 = vec![Predicate::new(
        "role".to_string(),
        vec![
            Term::Var("ID".to_string()),
            Term::Const(Constant::String("admin".to_string())),
        ],
    )];

    let query3 = vec![Predicate::new(
        "activity".to_string(),
        vec![
            Term::Var("UserID".to_string()),
            Term::Var("Action".to_string()),
            Term::Var("Time".to_string()),
        ],
    )];

    // Create materialized views for common queries
    println!("Creating materialized view for user query...");
    view_manager.create_view(
        "user_view".to_string(),
        query1.clone(),
        None, // No TTL
    )?;

    println!("Creating materialized view for admin role query with TTL...");
    view_manager.create_view(
        "admin_view".to_string(),
        query2.clone(),
        Some(Duration::from_secs(60)), // 60 second TTL
    )?;

    println!("Creating materialized view for activity query...");
    view_manager.create_view("activity_view".to_string(), query3.clone(), None)?;

    println!("Total views created: {}\n", view_manager.all_views().len());

    // Simulate view usage and populate results
    println!("--- Populating View Results ---");

    // Populate user view
    let user_results: Vec<Vec<Term>> = (0..10)
        .map(|i| {
            vec![
                Term::Const(Constant::Int(i)),
                Term::Const(Constant::String(format!("user{}", i))),
            ]
        })
        .collect();

    view_manager.refresh_view("user_view", user_results)?;
    println!("Refreshed user_view with 10 results");

    // Populate admin view
    let admin_results: Vec<Vec<Term>> = (0..5)
        .map(|i| {
            vec![
                Term::Const(Constant::Int(i * 3)),
                Term::Const(Constant::String("admin".to_string())),
            ]
        })
        .collect();

    view_manager.refresh_view("admin_view", admin_results)?;
    println!("Refreshed admin_view with 5 results");

    // Simulate view accesses
    println!("\n--- Simulating View Accesses ---");

    if let Some(view) = view_manager.get_view_mut("user_view") {
        for i in 0..20 {
            view.record_access(10.0 + i as f64); // Simulated cost saved
        }
        println!(
            "user_view: {} accesses, cost saved: {:.2}",
            view.access_count, view.total_cost_saved
        );
    }

    if let Some(view) = view_manager.get_view_mut("admin_view") {
        for i in 0..15 {
            view.record_access(8.0 + i as f64);
        }
        println!(
            "admin_view: {} accesses, cost saved: {:.2}",
            view.access_count, view.total_cost_saved
        );
    }

    if let Some(view) = view_manager.get_view_mut("activity_view") {
        for i in 0..5 {
            view.record_access(5.0 + i as f64);
        }
        println!(
            "activity_view: {} accesses, cost saved: {:.2}",
            view.access_count, view.total_cost_saved
        );
    }

    // View matching
    println!("\n--- View Matching ---");

    println!("Searching for view matching user query...");
    if let Some(matched_view) = view_manager.find_matching_view(&query1) {
        println!("✓ Found matching view: {}", matched_view.name);
        println!("  Results available: {}", matched_view.results.len());
    } else {
        println!("✗ No matching view found");
    }

    println!("Searching for view matching admin query...");
    if let Some(matched_view) = view_manager.find_matching_view(&query2) {
        println!("✓ Found matching view: {}", matched_view.name);
        println!("  TTL refresh needed: {}", matched_view.needs_refresh());
    } else {
        println!("✗ No matching view found");
    }

    // Statistics
    println!("\n--- View Manager Statistics ---");
    let stats = view_manager.get_statistics();
    println!("Total views: {}", stats.total_views);
    println!("Total accesses: {}", stats.total_accesses);
    println!("Total cost saved: {:.2}", stats.total_cost_saved);
    println!("Average accesses per view: {:.2}", stats.avg_access_count);

    // Test eviction by creating more views
    println!("\n--- Testing View Eviction ---");
    println!("Creating additional views to trigger eviction...");

    for i in 4..8 {
        let query = vec![Predicate::new(
            format!("test{}", i),
            vec![Term::Var("X".to_string())],
        )];

        view_manager.create_view(format!("test_view{}", i), query, None)?;
    }

    println!("Views after eviction: {}", view_manager.all_views().len());
    println!("Remaining views:");
    for (name, view) in view_manager.all_views() {
        println!("  - {}: {} accesses", name, view.access_count);
    }

    // Cleanup stale views
    println!("\n--- Cleanup Stale Views ---");
    view_manager.set_min_access_threshold(10);
    view_manager.cleanup_stale_views();

    println!("Views after cleanup: {}", view_manager.all_views().len());

    let final_stats = view_manager.get_statistics();
    println!("Final statistics:");
    println!("  Active views: {}", final_stats.total_views);
    println!("  Total cost saved: {:.2}", final_stats.total_cost_saved);

    // Query planning with optimizer
    println!("\n--- Query Planning ---");

    let complex_query = vec![
        Predicate::new(
            "user".to_string(),
            vec![Term::Var("ID".to_string()), Term::Var("Name".to_string())],
        ),
        Predicate::new(
            "role".to_string(),
            vec![Term::Var("ID".to_string()), Term::Var("Role".to_string())],
        ),
    ];

    let plan = optimizer.plan_query(&complex_query, &kb);
    println!("Query plan for join query:");
    println!("  Estimated cost: {:.2}", plan.estimated_cost);
    println!("  Estimated rows: {:.2}", plan.estimated_rows);

    println!("\n--- Summary ---");
    println!("✓ Created knowledge base with {} facts", kb.facts.len());
    println!("✓ Optimized queries with statistics");
    println!("✓ Created and managed materialized views");
    println!("✓ Implemented view eviction and cleanup");
    println!("✓ Tracked performance metrics");
    println!("\n✓ Example completed successfully!");

    Ok(())
}
