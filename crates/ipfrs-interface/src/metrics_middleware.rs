//! Prometheus metrics middleware for Axum
//!
//! This middleware automatically tracks HTTP request metrics including:
//! - Request count by endpoint and status
//! - Request duration
//! - Request/response sizes
//! - Active connections

use axum::{
    body::Body,
    extract::MatchedPath,
    http::{Request, Response, StatusCode},
    middleware::Next,
    response::IntoResponse,
};
use http_body_util::BodyExt;
use std::time::Instant;

use crate::metrics::{
    record_http_request, record_http_request_size, record_http_response_size,
    HTTP_CONNECTIONS_ACTIVE, HTTP_REQUEST_DURATION_SECONDS,
};

/// Middleware that records Prometheus metrics for HTTP requests
pub async fn metrics_middleware(
    req: Request<Body>,
    next: Next,
) -> Result<Response<Body>, StatusCode> {
    let start = Instant::now();

    // Extract path and method
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let method = req.method().to_string();

    // Get request size
    let (parts, body) = req.into_parts();
    let body_bytes = body
        .collect()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .to_bytes();
    let request_size = body_bytes.len();

    // Reconstruct request
    let req = Request::from_parts(parts, Body::from(body_bytes));

    // Increment active connections
    HTTP_CONNECTIONS_ACTIVE.with_label_values(&[&path]).inc();

    // Call the next middleware/handler
    let response = next.run(req).await;

    // Get response status and size
    let status = response.status();
    let (parts, body) = response.into_parts();
    let body_bytes = body
        .collect()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .to_bytes();
    let response_size = body_bytes.len();

    // Reconstruct response
    let response = Response::from_parts(parts, Body::from(body_bytes));

    // Record metrics
    let duration = start.elapsed().as_secs_f64();

    HTTP_REQUEST_DURATION_SECONDS
        .with_label_values(&[&path, &method])
        .observe(duration);

    record_http_request(&path, &method, status.as_u16());
    record_http_request_size(&path, &method, request_size);
    record_http_response_size(&path, &method, response_size);

    HTTP_CONNECTIONS_ACTIVE.with_label_values(&[&path]).dec();

    Ok(response)
}

/// Lightweight metrics middleware that doesn't buffer bodies
/// Use this for streaming endpoints to avoid memory issues
pub async fn metrics_middleware_streaming(req: Request<Body>, next: Next) -> impl IntoResponse {
    let start = Instant::now();

    // Extract path and method
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let method = req.method().to_string();

    // Increment active connections
    HTTP_CONNECTIONS_ACTIVE.with_label_values(&[&path]).inc();

    // Call the next middleware/handler
    let response = next.run(req).await;

    // Get response status
    let status = response.status();

    // Record metrics (without body sizes for streaming)
    let duration = start.elapsed().as_secs_f64();

    HTTP_REQUEST_DURATION_SECONDS
        .with_label_values(&[&path, &method])
        .observe(duration);

    record_http_request(&path, &method, status.as_u16());

    HTTP_CONNECTIONS_ACTIVE.with_label_values(&[&path]).dec();

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Router};
    use tower::ServiceExt;

    async fn test_handler() -> &'static str {
        "Hello, World!"
    }

    #[tokio::test]
    async fn test_metrics_middleware_streaming() {
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(axum::middleware::from_fn(metrics_middleware_streaming));

        let request = Request::builder()
            .uri("/test")
            .body(Body::empty())
            .expect("test: request construction should succeed");

        let response = app
            .oneshot(request)
            .await
            .expect("test: handler should respond without error");

        assert_eq!(response.status(), StatusCode::OK);

        // Verify metrics were recorded
        let metrics =
            crate::metrics::encode_metrics().expect("test: metrics encoding should succeed");
        assert!(metrics.contains("ipfrs_http_requests_total"));
        assert!(metrics.contains("ipfrs_http_request_duration_seconds"));
    }
}
