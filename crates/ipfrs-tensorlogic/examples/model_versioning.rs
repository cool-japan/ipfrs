//! Model Versioning Example
//!
//! This example demonstrates the Git-like version control system for ML models.
//! It shows how to:
//! - Create commits and track model versions
//! - Create and manage branches
//! - Checkout different versions
//! - Compare model versions with diffs
//! - Merge branches (fast-forward)

use ipfrs_core::{Cid, CidBuilder};
use ipfrs_tensorlogic::{
    ModelCommit, ModelDiff, ModelDiffer, ModelRepository, VersionControlError,
};
use std::collections::HashMap;

fn main() -> Result<(), VersionControlError> {
    println!("=== Model Versioning Example ===\n");

    // 1. Initialize a new repository
    println!("1. Initializing repository...");
    let mut repo = ModelRepository::new();

    // Create some fake model CIDs (in reality these would be actual model content CIDs)
    let model_v1 = create_fake_cid("model_v1");
    let model_v2 = create_fake_cid("model_v2");
    let model_v3 = create_fake_cid("model_v3");

    // 2. Create initial commit on main branch
    println!("\n2. Creating initial commit...");
    let commit1_id = create_fake_cid("commit1");
    let commit1 = ModelCommit::new(
        commit1_id,
        vec![],
        model_v1,
        "Initial model: baseline CNN".to_string(),
        "alice@example.com".to_string(),
    )
    .with_metadata("learning_rate".to_string(), "0.001".to_string())
    .with_metadata("batch_size".to_string(), "32".to_string());

    repo.init(commit1.clone())?;

    println!("   ✓ Created commit: {}", commit1.message);
    println!("   ✓ Branch: main");
    println!(
        "   ✓ Metadata: learning_rate={}, batch_size={}",
        commit1.metadata.get("learning_rate").unwrap(),
        commit1.metadata.get("batch_size").unwrap()
    );

    // 3. Make another commit on main
    println!("\n3. Creating second commit on main...");
    let commit2 = repo.commit(
        model_v2,
        "Improved accuracy: added batch normalization".to_string(),
        "alice@example.com".to_string(),
    )?;

    println!("   ✓ Created commit: {}", commit2.message);
    if !commit2.parents.is_empty() {
        println!("   ✓ Parent: {}", commit2.parents[0]);
    }

    // 4. Create a branch for experiments
    println!("\n4. Creating experimental branch...");
    let experiment_start = repo.head_commit().unwrap().id;
    repo.create_branch(
        "experiment/larger-model".to_string(),
        Some(experiment_start),
    )?;
    repo.checkout("experiment/larger-model")?;

    println!("   ✓ Created branch: experiment/larger-model");
    println!("   ✓ Starting from commit: {}", commit2.message);

    // 5. Make changes on experimental branch
    println!("\n5. Making changes on experimental branch...");
    let commit3 = repo.commit(
        model_v3,
        "Experiment: doubled model size".to_string(),
        "alice@example.com".to_string(),
    )?;

    println!("   ✓ Created commit: {}", commit3.message);

    // 6. Create a model diff
    println!("\n6. Comparing models (diff between different versions)...");
    let model_a = create_sample_model("baseline");
    let model_b = create_sample_model("improved");
    let diff = ModelDiffer::diff(&model_a, &model_b);
    display_model_diff(&diff);

    // 7. Switch back to main branch
    println!("\n7. Switching back to main branch...");
    repo.checkout("main")?;
    println!("   ✓ Checked out main branch");
    if let Some(head) = repo.head_commit() {
        println!("   ✓ Current HEAD: {}", head.id);
    }

    // 8. List all branches
    println!("\n8. Listing all branches...");
    let branches = repo.list_branches();
    for branch in branches {
        let is_current = repo.current_branch() == Some(&branch.name);
        let marker = if is_current { "*" } else { " " };
        println!("   {} {}", marker, branch.name);
    }

    // 9. Check if we can fast-forward merge
    println!("\n9. Checking merge possibility...");
    let can_merge = repo.can_fast_forward("experiment/larger-model")?;
    if can_merge {
        println!("   ✓ Can fast-forward merge experiment/larger-model into main");

        // 10. Perform fast-forward merge
        println!("\n10. Merging experimental branch into main...");
        repo.merge_fast_forward("experiment/larger-model")?;
        println!("   ✓ Successfully merged!");
        if let Some(head) = repo.head_commit() {
            println!("   ✓ Main branch now points to: {}", head.id);
        }
    } else {
        println!("   ⚠ Cannot fast-forward merge (would require 3-way merge)");
    }

    // 11. Create another branch from current state
    println!("\n11. Creating production branch...");
    repo.create_branch("production".to_string(), None)?;
    println!("   ✓ Created production branch");

    // 12. Display commit history
    println!("\n12. Commit history:");
    if let Some(head) = repo.head_commit() {
        display_commit_history(&repo, &head.id);
    }

    // 13. Checkout a specific commit (detached HEAD)
    println!("\n13. Checking out specific commit (detached HEAD)...");
    let commit_to_checkout = commit2.id.to_string();
    repo.checkout(&commit_to_checkout)?;
    println!("   ✓ Checked out commit: {}", commit2.id);
    println!("   ✓ State: Detached HEAD");
    println!("   ⚠ Changes made here won't be associated with any branch");

    // 14. Return to a branch
    println!("\n14. Returning to main branch...");
    repo.checkout("main")?;
    println!("   ✓ Back on main branch");

    println!("\n=== Summary ===");
    println!("Total branches: {}", repo.list_branches().len());
    println!(
        "Current branch: {}",
        repo.current_branch().unwrap_or("detached")
    );
    if let Some(head) = repo.head_commit() {
        println!("HEAD: {}", head.id);
    }

    println!("\n✓ Model versioning example completed successfully!");

    Ok(())
}

/// Helper function to create a fake CID for demonstration
fn create_fake_cid(data: &str) -> Cid {
    CidBuilder::new()
        .build(data.as_bytes())
        .expect("Failed to build CID")
}

/// Create a sample model for demonstration
fn create_sample_model(variant: &str) -> HashMap<String, Vec<f32>> {
    let mut model = HashMap::new();

    match variant {
        "baseline" => {
            model.insert("layer1".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
            model.insert("layer2".to_string(), vec![0.5, 0.6, 0.7]);
            model.insert("layer3".to_string(), vec![0.1, 0.2]);
        }
        "improved" => {
            model.insert("layer1".to_string(), vec![1.1, 2.1, 3.1, 4.1]);
            model.insert("layer2".to_string(), vec![0.55, 0.65, 0.75]);
            model.insert("layer4".to_string(), vec![0.8, 0.9]); // New layer
                                                                // layer3 removed
        }
        _ => {}
    }

    model
}

/// Display a model diff in a readable format
fn display_model_diff(diff: &ModelDiff) {
    if !diff.added_layers.is_empty() {
        println!("   Added layers:");
        for name in &diff.added_layers {
            println!("     + {}", name);
        }
    }

    if !diff.removed_layers.is_empty() {
        println!("   Removed layers:");
        for name in &diff.removed_layers {
            println!("     - {}", name);
        }
    }

    if !diff.modified_layers.is_empty() {
        println!("   Modified layers:");
        for layer in &diff.modified_layers {
            println!(
                "     ~ {} (L2: {:.4}, max: {:.4})",
                layer.name, layer.l2_diff, layer.max_diff
            );
        }
    }

    if diff.added_layers.is_empty()
        && diff.removed_layers.is_empty()
        && diff.modified_layers.is_empty()
    {
        println!("   (no differences)");
    }
}

/// Display commit history by traversing parent links
fn display_commit_history(repo: &ModelRepository, start_cid: &Cid) {
    let history = repo.get_history(start_cid, Some(10));

    if history.is_empty() {
        println!("   (no commits)");
        return;
    }

    for commit in history {
        println!("  * {} ({})", commit.message, commit.id);
        println!("    Author: {}", commit.author);
        println!(
            "    Date: {}",
            chrono::DateTime::from_timestamp(commit.timestamp, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );

        if !commit.metadata.is_empty() {
            println!("    Metadata:");
            for (key, value) in &commit.metadata {
                println!("      {}: {}", key, value);
            }
        }
        println!();
    }
}
