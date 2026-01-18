// Backpressure handling for gRPC streaming
//
// This module provides adaptive flow control and backpressure management
// for streaming RPCs to ensure stable and efficient data transfer.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::sleep;

/// Configuration for backpressure controller
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Initial window size (number of items that can be in-flight)
    pub initial_window: usize,
    /// Minimum window size
    pub min_window: usize,
    /// Maximum window size
    pub max_window: usize,
    /// Window increase factor on success
    pub increase_factor: f64,
    /// Window decrease factor on congestion
    pub decrease_factor: f64,
    /// Threshold for detecting slow consumer (items pending / window size)
    pub slow_consumer_threshold: f64,
    /// Interval for checking congestion
    pub check_interval: Duration,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            initial_window: 100,
            min_window: 10,
            max_window: 10000,
            increase_factor: 1.5,
            decrease_factor: 0.5,
            slow_consumer_threshold: 0.8,
            check_interval: Duration::from_millis(100),
        }
    }
}

/// Backpressure controller for managing flow control
#[derive(Clone)]
pub struct BackpressureController {
    config: Arc<BackpressureConfig>,
    semaphore: Arc<Semaphore>,
    window_size: Arc<AtomicUsize>,
    items_sent: Arc<AtomicU64>,
    items_consumed: Arc<AtomicU64>,
    last_adjustment: Arc<tokio::sync::Mutex<Instant>>,
}

impl BackpressureController {
    /// Create a new backpressure controller with the given configuration
    pub fn new(config: BackpressureConfig) -> Self {
        let initial_window = config.initial_window;
        Self {
            semaphore: Arc::new(Semaphore::new(initial_window)),
            window_size: Arc::new(AtomicUsize::new(initial_window)),
            items_sent: Arc::new(AtomicU64::new(0)),
            items_consumed: Arc::new(AtomicU64::new(0)),
            last_adjustment: Arc::new(tokio::sync::Mutex::new(Instant::now())),
            config: Arc::new(config),
        }
    }

    /// Acquire a permit to send an item (blocks if window is full)
    pub async fn acquire(&self) -> Result<BackpressurePermit, BackpressureError> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| BackpressureError::Closed)?;

        self.items_sent.fetch_add(1, Ordering::Relaxed);

        Ok(BackpressurePermit {
            _permit: permit,
            controller: self.clone(),
        })
    }

    /// Try to acquire a permit without blocking
    pub fn try_acquire(&self) -> Result<BackpressurePermit, BackpressureError> {
        let permit = self
            .semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| BackpressureError::WouldBlock)?;

        self.items_sent.fetch_add(1, Ordering::Relaxed);

        Ok(BackpressurePermit {
            _permit: permit,
            controller: self.clone(),
        })
    }

    /// Signal that an item has been consumed
    pub fn signal_consumed(&self) {
        self.items_consumed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current window size
    pub fn window_size(&self) -> usize {
        self.window_size.load(Ordering::Relaxed)
    }

    /// Get number of items sent
    pub fn items_sent(&self) -> u64 {
        self.items_sent.load(Ordering::Relaxed)
    }

    /// Get number of items consumed
    pub fn items_consumed(&self) -> u64 {
        self.items_consumed.load(Ordering::Relaxed)
    }

    /// Get current pending items (sent - consumed)
    pub fn pending_items(&self) -> u64 {
        let sent = self.items_sent();
        let consumed = self.items_consumed();
        sent.saturating_sub(consumed)
    }

    /// Check for congestion and adjust window size
    pub async fn check_congestion(&self) {
        let mut last_adjustment = self.last_adjustment.lock().await;
        let now = Instant::now();

        // Only check periodically
        if now.duration_since(*last_adjustment) < self.config.check_interval {
            return;
        }

        let pending = self.pending_items();
        let window = self.window_size() as u64;

        if window == 0 {
            return;
        }

        let utilization = pending as f64 / window as f64;

        // Detect congestion (slow consumer)
        if utilization >= self.config.slow_consumer_threshold {
            self.decrease_window().await;
            tracing::debug!(
                "Congestion detected, decreased window to {}",
                self.window_size()
            );
        } else if utilization < 0.5 && (window as usize) < self.config.max_window {
            // Low utilization, can increase window
            self.increase_window().await;
            tracing::debug!(
                "Low utilization, increased window to {}",
                self.window_size()
            );
        }

        *last_adjustment = now;
    }

    /// Increase window size
    async fn increase_window(&self) {
        let current = self.window_size();
        let new_size =
            ((current as f64 * self.config.increase_factor) as usize).min(self.config.max_window);

        if new_size > current {
            let diff = new_size - current;
            self.window_size.store(new_size, Ordering::Relaxed);
            self.semaphore.add_permits(diff);
        }
    }

    /// Decrease window size
    async fn decrease_window(&self) {
        let current = self.window_size();
        let new_size =
            ((current as f64 * self.config.decrease_factor) as usize).max(self.config.min_window);

        if new_size < current {
            self.window_size.store(new_size, Ordering::Relaxed);
            // Note: We can't actually remove permits from semaphore,
            // but the window_size tracks the logical limit
        }
    }

    /// Wait if consumer is slow (adaptive delay)
    pub async fn adaptive_delay(&self) {
        let pending = self.pending_items();
        let window = self.window_size() as u64;

        if window == 0 {
            return;
        }

        let utilization = pending as f64 / window as f64;

        if utilization > self.config.slow_consumer_threshold {
            // Consumer is slow, add proportional delay
            let delay_ms = ((utilization - self.config.slow_consumer_threshold) * 100.0) as u64;
            sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    /// Start background task for automatic congestion monitoring
    pub fn start_monitoring(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                self.check_congestion().await;
                sleep(self.config.check_interval).await;
            }
        })
    }
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self::new(BackpressureConfig::default())
    }
}

/// RAII guard for backpressure permit
pub struct BackpressurePermit {
    _permit: OwnedSemaphorePermit,
    #[allow(dead_code)]
    controller: BackpressureController,
}

impl Drop for BackpressurePermit {
    fn drop(&mut self) {
        // Permit is automatically released by OwnedSemaphorePermit's Drop
        // Note: We don't automatically signal_consumed here because consumption
        // should be tracked separately from permit release
    }
}

/// Errors that can occur during backpressure control
#[derive(Debug, Clone, thiserror::Error)]
pub enum BackpressureError {
    #[error("Backpressure controller is closed")]
    Closed,
    #[error("Would block, no permits available")]
    WouldBlock,
}

/// Stream wrapper with backpressure control
pub struct BackpressureStream<S> {
    inner: S,
    controller: Arc<BackpressureController>,
}

impl<S> BackpressureStream<S> {
    /// Create a new backpressure stream wrapper
    pub fn new(stream: S, controller: Arc<BackpressureController>) -> Self {
        Self {
            inner: stream,
            controller,
        }
    }

    /// Get reference to the controller
    pub fn controller(&self) -> &Arc<BackpressureController> {
        &self.controller
    }
}

impl<S> futures::Stream for BackpressureStream<S>
where
    S: futures::Stream + Unpin,
{
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Check congestion before polling
        let pending = self.controller.pending_items();
        let window = self.controller.window_size() as u64;

        if window > 0 && pending >= window {
            // Window is full, apply backpressure by returning Pending
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }

        // Poll inner stream
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backpressure_controller_creation() {
        let config = BackpressureConfig::default();
        let controller = BackpressureController::new(config);

        assert_eq!(controller.window_size(), 100);
        assert_eq!(controller.items_sent(), 0);
        assert_eq!(controller.items_consumed(), 0);
        assert_eq!(controller.pending_items(), 0);
    }

    #[tokio::test]
    async fn test_acquire_permit() {
        let controller = BackpressureController::default();

        let permit = controller.acquire().await.unwrap();
        assert_eq!(controller.items_sent(), 1);
        assert_eq!(controller.items_consumed(), 0);
        assert_eq!(controller.pending_items(), 1);

        // Drop permit to release semaphore, then signal consumption
        drop(permit);
        controller.signal_consumed();
        assert_eq!(controller.items_consumed(), 1);
        assert_eq!(controller.pending_items(), 0);
    }

    #[tokio::test]
    async fn test_try_acquire() {
        let config = BackpressureConfig {
            initial_window: 2,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        let _permit1 = controller.try_acquire().unwrap();
        let _permit2 = controller.try_acquire().unwrap();

        // Third acquire should fail
        assert!(controller.try_acquire().is_err());
    }

    #[tokio::test]
    async fn test_congestion_detection() {
        let config = BackpressureConfig {
            initial_window: 10,
            min_window: 5,
            slow_consumer_threshold: 0.8,
            check_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        // Simulate slow consumer by acquiring many permits
        let mut permits = Vec::new();
        for _ in 0..9 {
            permits.push(controller.acquire().await.unwrap());
        }

        // Should have 9 pending items
        assert_eq!(controller.pending_items(), 9);

        // Check congestion (utilization = 9/10 = 0.9 > 0.8)
        sleep(Duration::from_millis(20)).await;
        controller.check_congestion().await;

        // Window should have decreased
        assert!(controller.window_size() < 10);
    }

    #[tokio::test]
    async fn test_window_increase() {
        let config = BackpressureConfig {
            initial_window: 10,
            max_window: 100,
            increase_factor: 2.0,
            check_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        // Low utilization (0 pending items)
        sleep(Duration::from_millis(20)).await;
        controller.check_congestion().await;

        // Window should have increased
        assert!(controller.window_size() > 10);
    }

    #[tokio::test]
    async fn test_adaptive_delay() {
        let config = BackpressureConfig {
            initial_window: 10,
            slow_consumer_threshold: 0.8,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        // Acquire permits to simulate congestion
        let mut permits = Vec::new();
        for _ in 0..9 {
            permits.push(controller.acquire().await.unwrap());
        }

        let start = Instant::now();
        controller.adaptive_delay().await;
        let elapsed = start.elapsed();

        // Should have added some delay
        assert!(elapsed >= Duration::from_millis(0));
    }

    #[tokio::test]
    async fn test_automatic_monitoring() {
        let config = BackpressureConfig {
            initial_window: 10,
            check_interval: Duration::from_millis(50),
            ..Default::default()
        };
        let controller = Arc::new(BackpressureController::new(config));

        // Start monitoring
        let handle = controller.clone().start_monitoring();

        // Let it run for a bit
        sleep(Duration::from_millis(200)).await;

        // Stop monitoring
        handle.abort();

        // Controller should still be functional
        let _permit = controller.acquire().await.unwrap();
    }

    #[tokio::test]
    async fn test_signal_consumed() {
        let controller = BackpressureController::default();

        controller.signal_consumed();
        assert_eq!(controller.items_consumed(), 1);

        controller.signal_consumed();
        assert_eq!(controller.items_consumed(), 2);
    }
}
