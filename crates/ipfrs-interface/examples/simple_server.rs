//! Simple IPFRS Server Example
//!
//! This example demonstrates the easiest way to start an IPFRS server
//! using the new configuration presets.
//!
//! # Usage
//!
//! Development mode (localhost, fast compression):
//! ```bash
//! cargo run --example simple_server -- dev
//! ```
//!
//! Production mode (all interfaces, maximum compression):
//! ```bash
//! cargo run --example simple_server -- prod
//! ```
//!
//! Testing mode (minimal overhead):
//! ```bash
//! cargo run --example simple_server -- test
//! ```

use ipfrs_interface::{Gateway, GatewayConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().init();

    // Get mode from command line args
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("dev");

    // Select configuration based on mode
    let config = match mode {
        "prod" | "production" => {
            println!("🚀 Starting in PRODUCTION mode");
            GatewayConfig::production()
        }
        "test" | "testing" => {
            println!("🧪 Starting in TESTING mode");
            GatewayConfig::testing()
        }
        _ => {
            println!("🔧 Starting in DEVELOPMENT mode");
            GatewayConfig::development()
        }
    };

    // Validate configuration
    config.validate()?;

    println!("\n📋 Configuration:");
    println!("  Listen: {}", config.listen_addr);
    println!("  Storage: {}", config.storage_config.path.display());
    println!(
        "  Cache: {}MB",
        config.storage_config.cache_size / (1024 * 1024)
    );
    println!("\n");

    // Create and start gateway with GraphQL enabled
    let gateway = Gateway::new(config)?.with_graphql();

    gateway.start().await?;

    Ok(())
}
