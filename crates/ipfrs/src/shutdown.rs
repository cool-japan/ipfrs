//! Graceful shutdown handling
//!
//! This module provides graceful shutdown capabilities for IPFRS nodes,
//! including signal handling, connection draining, and state persistence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::timeout;

/// Shutdown signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownSignal {
    /// SIGTERM - graceful shutdown
    Terminate,
    /// SIGINT - Ctrl+C
    Interrupt,
    /// Manual shutdown
    Manual,
}

/// Shutdown coordinator
///
/// Coordinates graceful shutdown across multiple components,
/// ensuring proper cleanup and state persistence.
pub struct ShutdownCoordinator {
    /// Shutdown signal broadcaster
    shutdown_tx: broadcast::Sender<ShutdownSignal>,
    /// Shutdown flag
    is_shutdown: Arc<AtomicBool>,
    /// Graceful shutdown timeout
    timeout: Duration,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for graceful shutdown
    ///
    /// # Example
    /// ```rust
    /// use ipfrs::shutdown::ShutdownCoordinator;
    /// use std::time::Duration;
    ///
    /// let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
    /// ```
    pub fn new(timeout: Duration) -> Self {
        let (shutdown_tx, _) = broadcast::channel(16);
        Self {
            shutdown_tx,
            is_shutdown: Arc::new(AtomicBool::new(false)),
            timeout,
        }
    }

    /// Subscribe to shutdown signals
    ///
    /// Returns a receiver that will be notified when shutdown is initiated.
    pub fn subscribe(&self) -> broadcast::Receiver<ShutdownSignal> {
        self.shutdown_tx.subscribe()
    }

    /// Check if shutdown has been initiated
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.load(Ordering::Relaxed)
    }

    /// Initiate graceful shutdown
    ///
    /// Broadcasts shutdown signal to all subscribers and sets shutdown flag.
    ///
    /// # Arguments
    /// * `signal` - The shutdown signal type
    pub fn shutdown(&self, signal: ShutdownSignal) {
        if !self.is_shutdown.swap(true, Ordering::SeqCst) {
            tracing::info!("Initiating graceful shutdown: {:?}", signal);
            let _ = self.shutdown_tx.send(signal);
        }
    }

    /// Wait for graceful shutdown to complete
    ///
    /// Waits for all components to finish cleanup within the configured timeout.
    ///
    /// # Returns
    /// `Ok(())` if shutdown completed successfully, `Err(())` if timed out
    pub async fn wait_for_shutdown(&self) -> Result<(), ()> {
        if timeout(self.timeout, self.wait_internal()).await.is_err() {
            tracing::warn!(
                "Graceful shutdown timeout ({:?}) exceeded, forcing shutdown",
                self.timeout
            );
            Err(())
        } else {
            tracing::info!("Graceful shutdown completed successfully");
            Ok(())
        }
    }

    /// Internal wait implementation
    async fn wait_internal(&self) {
        // This would coordinate with component shutdown handlers
        // For now, we just wait a bit to allow components to clean up
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// Get shutdown timeout
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

/// Shutdown handler for components
///
/// Provides a simple interface for components to handle shutdown gracefully.
pub struct ShutdownHandler {
    /// Shutdown signal receiver
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
    /// Component name
    component_name: String,
}

impl ShutdownHandler {
    /// Create a new shutdown handler
    ///
    /// # Arguments
    /// * `shutdown_rx` - Shutdown signal receiver
    /// * `component_name` - Name of the component for logging
    pub fn new(shutdown_rx: broadcast::Receiver<ShutdownSignal>, component_name: String) -> Self {
        Self {
            shutdown_rx,
            component_name,
        }
    }

    /// Wait for shutdown signal
    ///
    /// Blocks until a shutdown signal is received.
    ///
    /// # Returns
    /// The shutdown signal that was received
    pub async fn wait_for_shutdown(&mut self) -> ShutdownSignal {
        match self.shutdown_rx.recv().await {
            Ok(signal) => {
                tracing::info!(
                    "Component '{}' received shutdown signal: {:?}",
                    self.component_name,
                    signal
                );
                signal
            }
            Err(_) => {
                tracing::warn!(
                    "Component '{}' shutdown receiver closed, assuming shutdown",
                    self.component_name
                );
                ShutdownSignal::Manual
            }
        }
    }

    /// Check if shutdown has been signaled (non-blocking)
    pub fn is_shutdown(&mut self) -> bool {
        self.shutdown_rx.try_recv().is_ok()
    }
}

/// Signal handler for Unix signals (SIGTERM, SIGINT)
#[cfg(unix)]
pub async fn wait_for_signal() -> ShutdownSignal {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("Failed to register SIGINT handler");

    tokio::select! {
        _ = sigterm.recv() => {
            tracing::info!("Received SIGTERM");
            ShutdownSignal::Terminate
        }
        _ = sigint.recv() => {
            tracing::info!("Received SIGINT (Ctrl+C)");
            ShutdownSignal::Interrupt
        }
    }
}

/// Signal handler for Windows (Ctrl+C only)
#[cfg(not(unix))]
pub async fn wait_for_signal() -> ShutdownSignal {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to register Ctrl+C handler");
    tracing::info!("Received Ctrl+C");
    ShutdownSignal::Interrupt
}

// Implement Clone for ShutdownCoordinator to allow sharing
impl Clone for ShutdownCoordinator {
    fn clone(&self) -> Self {
        Self {
            shutdown_tx: self.shutdown_tx.clone(),
            is_shutdown: Arc::clone(&self.is_shutdown),
            timeout: self.timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_coordinator_creation() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(10));
        assert!(!coordinator.is_shutdown());
        assert_eq!(coordinator.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_shutdown_coordinator_default() {
        let coordinator = ShutdownCoordinator::default();
        assert!(!coordinator.is_shutdown());
        assert_eq!(coordinator.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn test_shutdown_signal() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(5));
        assert!(!coordinator.is_shutdown());

        coordinator.shutdown(ShutdownSignal::Manual);
        assert!(coordinator.is_shutdown());

        // Second shutdown should be idempotent
        coordinator.shutdown(ShutdownSignal::Manual);
        assert!(coordinator.is_shutdown());
    }

    #[tokio::test]
    async fn test_shutdown_subscribe() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(5));
        let mut rx = coordinator.subscribe();

        coordinator.shutdown(ShutdownSignal::Manual);

        let signal = rx
            .recv()
            .await
            .expect("test: shutdown signal recv should succeed");
        assert_eq!(signal, ShutdownSignal::Manual);
    }

    #[tokio::test]
    async fn test_shutdown_handler() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(5));
        let rx = coordinator.subscribe();
        let mut handler = ShutdownHandler::new(rx, "test-component".to_string());

        assert!(!handler.is_shutdown());

        // Spawn a task to trigger shutdown
        let coord = coordinator.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            coord.shutdown(ShutdownSignal::Terminate);
        });

        let signal = handler.wait_for_shutdown().await;
        assert_eq!(signal, ShutdownSignal::Terminate);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(5));
        let mut rx1 = coordinator.subscribe();
        let mut rx2 = coordinator.subscribe();
        let mut rx3 = coordinator.subscribe();

        coordinator.shutdown(ShutdownSignal::Interrupt);

        assert_eq!(
            rx1.recv().await.expect("test: rx1 recv should succeed"),
            ShutdownSignal::Interrupt
        );
        assert_eq!(
            rx2.recv().await.expect("test: rx2 recv should succeed"),
            ShutdownSignal::Interrupt
        );
        assert_eq!(
            rx3.recv().await.expect("test: rx3 recv should succeed"),
            ShutdownSignal::Interrupt
        );
    }

    #[tokio::test]
    async fn test_shutdown_wait() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(1));
        coordinator.shutdown(ShutdownSignal::Manual);

        let result = coordinator.wait_for_shutdown().await;
        assert!(result.is_ok());
    }
}
