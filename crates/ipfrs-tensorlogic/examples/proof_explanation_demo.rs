//! Automatic Proof Explanation Example
//!
//! This example demonstrates the automatic proof explanation system that generates
//! natural language explanations of logical proofs in multiple styles.
//!
//! Features demonstrated:
//! - Concise explanations (one-liners)
//! - Detailed explanations (step-by-step)
//! - Pedagogical explanations (educational style)
//! - Formal explanations (mathematical notation)
//! - Proof explanation builder (fluent API)
//! - Fragment-based proof explanation

use ipfrs_tensorlogic::{
    Constant, ExplanationStyle, FragmentProofExplainer, Predicate, Proof, ProofExplainer,
    ProofExplanationBuilder, ProofFragment, ProofFragmentRef, ProofMetadata, ProofRule, RuleRef,
    Term,
};

fn main() {
    println!("=== Automatic Proof Explanation Demo ===\n");

    // Example 1: Simple fact proof with different styles
    simple_fact_example();

    // Example 2: Rule-based proof with subproofs
    rule_based_proof_example();

    // Example 3: Complex multi-step proof
    complex_proof_example();

    // Example 4: Using the builder API
    builder_api_example();

    // Example 5: Fragment-based proof explanation
    fragment_explanation_example();

    // Example 6: Custom naturalization for domain-specific predicates
    custom_naturalization_example();
}

fn simple_fact_example() {
    println!("--- Example 1: Simple Fact with Different Styles ---\n");

    let proof = Proof {
        goal: Predicate {
            name: "mortal".to_string(),
            args: vec![Term::Const(Constant::String("socrates".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "mortal".to_string(),
                args: vec![Term::Const(Constant::String("socrates".to_string()))],
            },
            body: vec![],
            is_fact: true,
        }),
        subproofs: vec![],
    };

    let explainer = ProofExplainer::new();

    println!("Concise Style:");
    println!("{}", explainer.explain(&proof, ExplanationStyle::Concise));
    println!();

    println!("Detailed Style:");
    println!("{}", explainer.explain(&proof, ExplanationStyle::Detailed));
    println!();

    println!("Pedagogical Style:");
    println!(
        "{}",
        explainer.explain(&proof, ExplanationStyle::Pedagogical)
    );
    println!();

    println!("Formal Style:");
    println!("{}", explainer.explain(&proof, ExplanationStyle::Formal));
    println!();
}

fn rule_based_proof_example() {
    println!("--- Example 2: Rule-Based Proof with Subproofs ---\n");
    println!("Rule: All humans are mortal");
    println!("Fact: Socrates is a human");
    println!("Goal: Prove Socrates is mortal\n");

    let premise = Proof {
        goal: Predicate {
            name: "human".to_string(),
            args: vec![Term::Const(Constant::String("socrates".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "human".to_string(),
                args: vec![Term::Const(Constant::String("socrates".to_string()))],
            },
            body: vec![],
            is_fact: true,
        }),
        subproofs: vec![],
    };

    let proof = Proof {
        goal: Predicate {
            name: "mortal".to_string(),
            args: vec![Term::Const(Constant::String("socrates".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "mortal".to_string(),
                args: vec![Term::Var("X".to_string())],
            },
            body: vec![Predicate {
                name: "human".to_string(),
                args: vec![Term::Var("X".to_string())],
            }],
            is_fact: false,
        }),
        subproofs: vec![premise],
    };

    let explainer = ProofExplainer::new();

    println!("Detailed Explanation:");
    println!("{}", explainer.explain(&proof, ExplanationStyle::Detailed));
    println!();

    println!("Pedagogical Explanation:");
    println!(
        "{}",
        explainer.explain(&proof, ExplanationStyle::Pedagogical)
    );
    println!();
}

fn complex_proof_example() {
    println!("--- Example 3: Complex Multi-Step Proof ---\n");
    println!("Rule 1: All humans are mortal");
    println!("Rule 2: All philosophers are humans");
    println!("Fact: Socrates is a philosopher");
    println!("Goal: Prove Socrates is mortal\n");

    // Bottom level: fact
    let philosopher_fact = Proof {
        goal: Predicate {
            name: "philosopher".to_string(),
            args: vec![Term::Const(Constant::String("socrates".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "philosopher".to_string(),
                args: vec![Term::Const(Constant::String("socrates".to_string()))],
            },
            body: vec![],
            is_fact: true,
        }),
        subproofs: vec![],
    };

    // Middle level: philosopher -> human
    let human_proof = Proof {
        goal: Predicate {
            name: "human".to_string(),
            args: vec![Term::Const(Constant::String("socrates".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "human".to_string(),
                args: vec![Term::Var("X".to_string())],
            },
            body: vec![Predicate {
                name: "philosopher".to_string(),
                args: vec![Term::Var("X".to_string())],
            }],
            is_fact: false,
        }),
        subproofs: vec![philosopher_fact],
    };

    // Top level: human -> mortal
    let mortal_proof = Proof {
        goal: Predicate {
            name: "mortal".to_string(),
            args: vec![Term::Const(Constant::String("socrates".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "mortal".to_string(),
                args: vec![Term::Var("X".to_string())],
            },
            body: vec![Predicate {
                name: "human".to_string(),
                args: vec![Term::Var("X".to_string())],
            }],
            is_fact: false,
        }),
        subproofs: vec![human_proof],
    };

    let explainer = ProofExplainer::new();

    println!("Detailed Explanation:");
    println!(
        "{}",
        explainer.explain(&mortal_proof, ExplanationStyle::Detailed)
    );
    println!();

    println!("Pedagogical Explanation:");
    println!(
        "{}",
        explainer.explain(&mortal_proof, ExplanationStyle::Pedagogical)
    );
    println!();
}

fn builder_api_example() {
    println!("--- Example 4: Using the Builder API ---\n");

    let proof = Proof {
        goal: Predicate {
            name: "mortal".to_string(),
            args: vec![Term::Const(Constant::String("plato".to_string()))],
        },
        rule: Some(ProofRule {
            head: Predicate {
                name: "mortal".to_string(),
                args: vec![Term::Const(Constant::String("plato".to_string()))],
            },
            body: vec![],
            is_fact: true,
        }),
        subproofs: vec![],
    };

    println!("Concise with natural language:");
    let explanation = ProofExplanationBuilder::new(proof.clone())
        .style(ExplanationStyle::Concise)
        .natural_language()
        .build();
    println!("{}\n", explanation);

    println!("Detailed with formal notation:");
    let explanation = ProofExplanationBuilder::new(proof.clone())
        .style(ExplanationStyle::Detailed)
        .formal_notation()
        .with_premises()
        .build();
    println!("{}\n", explanation);

    println!("Pedagogical with max depth limit:");
    let explanation = ProofExplanationBuilder::new(proof)
        .style(ExplanationStyle::Pedagogical)
        .with_premises()
        .with_substitutions()
        .max_depth(5)
        .build();
    println!("{}\n", explanation);
}

fn fragment_explanation_example() {
    println!("--- Example 5: Fragment-Based Proof Explanation ---\n");

    let fragment = ProofFragment {
        id: "proof_fragment_001".to_string(),
        conclusion: Predicate {
            name: "mortal".to_string(),
            args: vec![Term::Const(Constant::String("aristotle".to_string()))],
        },
        rule_applied: Some(RuleRef {
            rule_id: "mortality_rule".to_string(),
            rule_cid: None,
            rule: None,
        }),
        premise_refs: vec![ProofFragmentRef {
            cid: ipfrs_core::Cid::default(),
            conclusion_hint: Some("human(aristotle)".to_string()),
        }],
        substitution: vec![(
            "X".to_string(),
            Term::Const(Constant::String("aristotle".to_string())),
        )],
        metadata: ProofMetadata {
            created_at: Some(1704067200),
            created_by: Some("reasoning_engine".to_string()),
            complexity: Some(2),
            depth: 1,
            custom: std::collections::HashMap::new(),
        },
    };

    let explainer = FragmentProofExplainer::new();

    println!("Concise fragment explanation:");
    println!(
        "{}\n",
        explainer.explain_fragment(&fragment, ExplanationStyle::Concise)
    );

    println!("Detailed fragment explanation:");
    println!(
        "{}\n",
        explainer.explain_fragment(&fragment, ExplanationStyle::Detailed)
    );

    println!("Metadata explanation:");
    println!("{}\n", explainer.explain_metadata(&fragment.metadata));
}

fn custom_naturalization_example() {
    println!("--- Example 6: Custom Naturalization Examples ---\n");

    let explainer = ProofExplainer::new();

    // Parent relationship
    let parent_pred = Predicate {
        name: "parent".to_string(),
        args: vec![
            Term::Const(Constant::String("john".to_string())),
            Term::Const(Constant::String("mary".to_string())),
        ],
    };
    println!(
        "parent(john, mary) → {}",
        explainer.naturalize_predicate(&parent_pred)
    );

    // Ancestor relationship
    let ancestor_pred = Predicate {
        name: "ancestor".to_string(),
        args: vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    };
    println!(
        "ancestor(alice, bob) → {}",
        explainer.naturalize_predicate(&ancestor_pred)
    );

    // Comparison predicates
    let gt_pred = Predicate {
        name: "greater_than".to_string(),
        args: vec![
            Term::Const(Constant::Int(10)),
            Term::Const(Constant::Int(5)),
        ],
    };
    println!(
        "greater_than(10, 5) → {}",
        explainer.naturalize_predicate(&gt_pred)
    );

    let eq_pred = Predicate {
        name: "equal".to_string(),
        args: vec![
            Term::Const(Constant::String("x".to_string())),
            Term::Const(Constant::String("x".to_string())),
        ],
    };
    println!(
        "equal(x, x) → {}\n",
        explainer.naturalize_predicate(&eq_pred)
    );
}
