//! HTTP gzip response-compression middleware for the IPFRS gateway.
//!
//! This module implements *real* gzip content-coding for gateway responses,
//! backed by the pure-Rust [`oxiarc_deflate`] implementation of RFC 1952
//! (gzip) / RFC 1951 (DEFLATE). It replaces the previously dead
//! [`CompressionConfig`] plumbing, which advertised compression the gateway
//! never performed.
//!
//! # Design
//!
//! Compression is applied as an [`axum::middleware::from_fn_with_state`] async
//! function ([`compress_response`]) rather than a hand-written `tower::Layer`.
//! [`axum::middleware::Next::run`] hands back a fully materialized
//! [`Response`], which is exactly what a *buffer-then-compress* operation
//! needs. This matches every other middleware in this crate
//! (`cors_middleware`, `rate_limit_middleware`, `metrics_middleware`).
//!
//! # HTTP content-coding semantics
//!
//! The middleware negotiates the `gzip` coding per **RFC 9110 §12.5.3**
//! (`Accept-Encoding`), and only ever produces a compressed representation when
//! doing so is both *permitted* and *beneficial*:
//!
//! * `Accept-Encoding` q-values are honored ([`gzip_quality`]). An explicit
//!   `gzip` (or legacy `x-gzip`) entry always overrides a `*` wildcard; a
//!   `q=0` disqualifies the coding; an absent header means *no* compression.
//! * On success the middleware sets `Content-Encoding: gzip`, rewrites
//!   `Content-Length` to the compressed size, **appends** `Accept-Encoding` to
//!   `Vary` (so shared caches key on it), drops `Accept-Ranges` (the gzip
//!   representation cannot honor identity byte offsets), and **weakens** any
//!   strong `ETag` to `W/"…"` (a gzipped body is a distinct representation and
//!   must not share a strong validator with the identity body).
//!
//! # Streaming safety (the crux)
//!
//! Two gateway endpoints serve bodies that must **never** be buffered:
//! [`stream_download`](crate::streaming::stream_download) (a potentially huge
//! `Body::from_stream`) and
//! [`progress_stream`](crate::streaming::progress_stream) (an effectively
//! infinite `Sse` stream — buffering it would hang forever).
//!
//! `stream_download` sets an *explicit* `Content-Length` header over an
//! *unsized* body, so the `Content-Length` **header cannot** be used to decide
//! "buffered vs streaming". Instead the middleware inspects the body's own
//! size hint (`http_body::Body::size_hint`, reachable as the
//! [`axum::body::HttpBody`] re-export): `Full` bodies (`Body::from(bytes)`) report an
//! exact `upper` bound, whereas `StreamBody` (`from_stream`) and `SseBody`
//! (`Sse`) do not override `size_hint` and therefore report `upper == None`.
//! A `None` upper bound is the streaming guard: the response is passed through
//! untouched and is **never** collected.

use axum::{
    body::{Body, HttpBody},
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::Response,
};
use bytes::Bytes;
use http_body_util::BodyExt;
use tracing::warn;

use crate::middleware::{CompressionConfig, CompressionLevel};

/// Hard upper bound on the response size (in bytes) the middleware is willing
/// to buffer and compress in memory.
///
/// Beyond this the response is passed through uncompressed: buffering it would
/// risk unbounded memory/CPU spikes even though its size hint is finite. 10 MiB
/// comfortably covers HTML/JSON/tensor payloads while capping worst-case cost.
const MAX_COMPRESS_BYTES: u64 = 10 * 1024 * 1024;

/// Threshold (in bytes) at or above which gzip runs on
/// [`tokio::task::spawn_blocking`] instead of inline on the async task.
///
/// DEFLATE is CPU-bound; running a large compression inline would stall the
/// runtime worker. Small bodies compress fast enough that the hop to a blocking
/// thread would cost more than it saves.
const SPAWN_BLOCKING_THRESHOLD: usize = 256 * 1024;

/// State carried by the compression middleware.
///
/// Cloned once per request by [`axum::middleware::from_fn_with_state`]; the
/// inner [`CompressionConfig`] is cheap to clone.
#[derive(Clone)]
pub struct CompressionState {
    /// The active compression configuration.
    pub config: CompressionConfig,
}

/// Axum middleware that gzip-compresses eligible responses.
///
/// The response is served **untouched** whenever any skip predicate holds (see
/// the module docs and the inline predicate comments). Only when compression is
/// permitted, safe, and actually shrinks the body is a `Content-Encoding: gzip`
/// representation returned. Compression failures never break a response: they
/// degrade gracefully to the identity body.
///
/// See the [module documentation](self) for the full HTTP and streaming-safety
/// semantics.
pub async fn compress_response(
    State(state): State<CompressionState>,
    request: Request,
    next: Next,
) -> Response {
    // Capture request-derived facts *before* the request is consumed by the
    // downstream stack. A response alone cannot reveal the request method or
    // which codings the client is willing to accept.
    let is_head = request.method() == Method::HEAD;
    let accept_encoding = request
        .headers()
        .get(header::ACCEPT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);

    let response = next.run(request).await;
    let config = &state.config;

    // Skip predicates, evaluated cheapest-first. If ANY holds, serve the
    // original response untouched.

    // (1) Compression globally disabled. (2) HEAD carries no body — a
    // `Content-Encoding` without a body is meaningless. (3) The client does not
    // accept gzip (RFC 9110 §12.5.3 proactive negotiation).
    if !config.enable_gzip || is_head || !accepts_gzip(accept_encoding.as_deref()) {
        return response;
    }

    let status = response.status();

    // (4) 1xx / 204 No Content / 304 Not Modified carry no representation body.
    if status.is_informational()
        || status == StatusCode::NO_CONTENT
        || status == StatusCode::NOT_MODIFIED
    {
        return response;
    }

    // (5) Never double-encode: a non-identity `Content-Encoding` is already set.
    if has_non_identity_encoding(response.headers()) {
        return response;
    }

    // (6) A 206, or any `Content-Range`, describes a byte range of the identity
    //     representation; compressing it would invalidate the offsets.
    if status == StatusCode::PARTIAL_CONTENT
        || response.headers().contains_key(header::CONTENT_RANGE)
    {
        return response;
    }

    // Split the response so the body's size hint can be inspected. From here the
    // body is moved out, so every early return rebuilds a response from `parts`.
    let (mut parts, body) = response.into_parts();

    // (7) STREAMING-SAFETY GUARD. Trust the body's size hint, never the
    //     `Content-Length` header (which `stream_download` sets over an unsized
    //     body). `Full` bodies report an exact `upper`; `StreamBody`/`SseBody`
    //     report `None`. `None` ⇒ pass through untouched, never collected.
    let upper = match body.size_hint().upper() {
        Some(upper) => upper,
        None => return Response::from_parts(parts, body),
    };

    // (8) Too small to be worth it, or too large to buffer safely. `upper` is
    //     only an upper bound; the real length is re-checked after collecting.
    if upper < config.min_size as u64 || upper > MAX_COMPRESS_BYTES {
        return Response::from_parts(parts, body);
    }

    // (9) The `Content-Type` already denotes a compressed payload.
    if let Some(content_type) = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    {
        if is_incompressible_content_type(content_type) {
            return Response::from_parts(parts, body);
        }
    }

    // Buffer the (now known-bounded) body.
    let original = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            // `into_parts()` already moved the body out, and a `collect()` error
            // means the underlying stream failed partway through, so the original
            // bytes cannot be replayed. Surface an honest 500 rather than a
            // truncated/corrupt 200. This path is unreachable for the gateway's
            // own `Body::from(bytes)` (`Full`) responses — whose collect is
            // infallible — and only guards a hypothetical fallible *sized* body.
            warn!(error = %error, "compression: failed to buffer response body; returning 500");
            parts.status = StatusCode::INTERNAL_SERVER_ERROR;
            parts.headers.remove(header::CONTENT_LENGTH);
            parts.headers.remove(header::CONTENT_ENCODING);
            parts.headers.remove(header::CONTENT_TYPE);
            return Response::from_parts(parts, Body::empty());
        }
    };

    let original_len = original.len();

    // Re-check the *real* length against `min_size`: the size hint above was
    // only an upper bound and may have overestimated.
    if original_len < config.min_size {
        return Response::from_parts(parts, Body::from(original));
    }

    // Compress, offloading large payloads to a blocking thread. Any failure
    // (compressor error or task-join error) is logged and degrades to identity.
    let level = compression_level(config.level);
    let compressed = match gzip_compress_maybe_blocking(original.clone(), level, original_len).await
    {
        Some(compressed) => compressed,
        None => return Response::from_parts(parts, Body::from(original)),
    };

    // Only-if-smaller: never ship a "compressed" body that grew (e.g. already
    // high-entropy data). Serve the identity representation instead.
    if compressed.len() >= original_len {
        return Response::from_parts(parts, Body::from(original));
    }

    // Success: rewrite headers to describe the gzip representation.
    let headers = &mut parts.headers;
    headers.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    // `From<usize>` for `HeaderValue` is infallible.
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(compressed.len()));
    append_vary_accept_encoding(headers);
    // The gzip representation cannot honor the identity byte offsets.
    headers.remove(header::ACCEPT_RANGES);
    // A gzipped body is a *different* representation; weakening a strong ETag
    // prevents a cache from serving it under the identity validator.
    weaken_strong_etag(headers);

    Response::from_parts(parts, Body::from(compressed))
}

/// Compute the effective quality assigned to the `gzip` coding by an
/// `Accept-Encoding` header value, following RFC 9110 §12.5.3.
///
/// The header is a comma-separated list of `coding [ ";" "q=" qvalue ]`. Coding
/// names are case-insensitive and an absent `q` parameter means `q=1.0`. This
/// function:
///
/// * tracks the maximum explicit quality for `gzip` / `x-gzip`, and separately
///   for the `*` wildcard;
/// * returns the `gzip` quality when `gzip` is listed (an explicit entry always
///   overrides `*`), else the wildcard quality, else `0.0` (unlisted with no
///   wildcard ⇒ not acceptable);
/// * treats a malformed or non-finite `q` value as `0.0` (conservative reject)
///   and clamps parsed values to `[0.0, 1.0]`.
pub fn gzip_quality(header: &str) -> f32 {
    let mut gzip_q: Option<f32> = None;
    let mut wildcard_q: Option<f32> = None;

    for element in header.split(',') {
        let element = element.trim();
        if element.is_empty() {
            continue;
        }

        let mut segments = element.split(';');
        let coding = match segments.next() {
            Some(coding) => coding.trim(),
            None => continue,
        };
        if coding.is_empty() {
            continue;
        }

        // Resolve the q-value from the first `q=` parameter, if any.
        let mut quality = 1.0_f32;
        for parameter in segments {
            if let Some((key, value)) = parameter.split_once('=') {
                if key.trim().eq_ignore_ascii_case("q") {
                    quality = parse_qvalue(value.trim());
                    break;
                }
            }
        }

        if coding.eq_ignore_ascii_case("gzip") || coding.eq_ignore_ascii_case("x-gzip") {
            gzip_q = Some(gzip_q.map_or(quality, |previous| previous.max(quality)));
        } else if coding == "*" {
            wildcard_q = Some(wildcard_q.map_or(quality, |previous| previous.max(quality)));
        }
    }

    // Explicit gzip wins; fall back to the wildcard only when gzip is unlisted;
    // default to "not acceptable" when neither appears.
    gzip_q.or(wildcard_q).unwrap_or(0.0)
}

/// Parse an RFC 9110 q-value, rejecting malformed or non-finite input as `0.0`
/// and clamping the result to `[0.0, 1.0]`.
fn parse_qvalue(value: &str) -> f32 {
    match value.parse::<f32>() {
        Ok(quality) if quality.is_finite() => quality.clamp(0.0, 1.0),
        _ => 0.0,
    }
}

/// Return whether the client accepts the `gzip` coding.
///
/// An absent header (`None`) means the client expressed no preference and, per
/// this gateway's conservative policy, gzip is **not** applied. A present header
/// is acceptable only when [`gzip_quality`] yields a strictly positive value.
pub fn accepts_gzip(header: Option<&str>) -> bool {
    match header {
        Some(header) => gzip_quality(header) > 0.0,
        None => false,
    }
}

/// Return `true` when a `Content-Type` denotes an already-compressed (or
/// otherwise poorly-compressible) payload that should be served as-is.
///
/// Media-type parameters (e.g. `; charset=utf-8`) are ignored and matching is
/// case-insensitive. `image/svg+xml` is a deliberate exception to the `image/*`
/// rule (SVG is text and compresses well), and `text/event-stream` is skipped
/// as belt-and-braces for Server-Sent Events. `application/octet-stream` is
/// deliberately **not** treated as incompressible — the gateway's `get_content`
/// and tensor endpoints use it and it is frequently compressible; the
/// only-if-smaller guard prevents waste when it is not.
pub fn is_incompressible_content_type(content_type: &str) -> bool {
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let media_type = media_type.as_str();

    // SVG is textual XML — compress it despite the `image/` prefix.
    if media_type == "image/svg+xml" {
        return false;
    }

    if media_type.starts_with("image/")
        || media_type.starts_with("video/")
        || media_type.starts_with("audio/")
    {
        return true;
    }

    if media_type == "font/woff" || media_type == "font/woff2" {
        return true;
    }

    // Belt-and-braces for SSE, which is also caught by the size-hint guard.
    if media_type == "text/event-stream" {
        return true;
    }

    matches!(
        media_type,
        "application/zip"
            | "application/gzip"
            | "application/x-gzip"
            | "application/zstd"
            | "application/x-zstd"
            | "application/x-xz"
            | "application/x-bzip2"
            | "application/x-7z-compressed"
            | "application/pdf"
    )
}

/// Return `true` when the response already carries a non-`identity`
/// `Content-Encoding` (so re-encoding would double-compress).
///
/// A header consisting solely of `identity` tokens (or absent) is treated as
/// unencoded. A value that is not valid ASCII cannot be verified as `identity`
/// and is conservatively treated as already-encoded.
fn has_non_identity_encoding(headers: &HeaderMap) -> bool {
    for value in headers.get_all(header::CONTENT_ENCODING) {
        match value.to_str() {
            Ok(text) => {
                let has_real_coding = text.split(',').any(|token| {
                    let token = token.trim();
                    !token.is_empty() && !token.eq_ignore_ascii_case("identity")
                });
                if has_real_coding {
                    return true;
                }
            }
            Err(_) => return true,
        }
    }
    false
}

/// Append `Accept-Encoding` to the response's `Vary` header without clobbering
/// existing values.
///
/// A new `Vary` field line is appended (per RFC 9110 §5.3 multiple field lines
/// are equivalent to one comma-separated value), so an existing `Vary: Origin`
/// is preserved. Nothing is added when `Accept-Encoding` is already listed, or
/// when `Vary: *` is present (which already declares dependence on every
/// request header).
fn append_vary_accept_encoding(headers: &mut HeaderMap) {
    let mut has_wildcard = false;
    let mut has_accept_encoding = false;

    for value in headers.get_all(header::VARY) {
        if let Ok(text) = value.to_str() {
            for token in text.split(',') {
                let token = token.trim();
                if token == "*" {
                    has_wildcard = true;
                } else if token.eq_ignore_ascii_case("accept-encoding") {
                    has_accept_encoding = true;
                }
            }
        }
    }

    if has_wildcard || has_accept_encoding {
        return;
    }

    headers.append(header::VARY, HeaderValue::from_static("accept-encoding"));
}

/// Weaken a strong `ETag` (`"…"`) to a weak validator (`W/"…"`).
///
/// A gzipped body is a different representation of the same resource, so it must
/// not be served under the identity representation's strong validator. Already
/// weak, empty, or non-ASCII ETags are left untouched.
fn weaken_strong_etag(headers: &mut HeaderMap) {
    let weakened = match headers
        .get(header::ETAG)
        .and_then(|value| value.to_str().ok())
    {
        Some(etag) => {
            let etag = etag.trim();
            if etag.is_empty() || etag.starts_with("W/") {
                return;
            }
            format!("W/{etag}")
        }
        None => return,
    };

    if let Ok(value) = HeaderValue::from_str(&weakened) {
        headers.insert(header::ETAG, value);
    }
}

/// Map a [`CompressionLevel`] to the `u8` level oxiarc-deflate expects.
///
/// [`CompressionLevel::to_level`] yields a `u32` already constrained to
/// `0..=9`; the extra `.min(9)` is defensive and the subsequent cast is
/// therefore always lossless.
fn compression_level(level: CompressionLevel) -> u8 {
    level.to_level().min(9) as u8
}

/// Gzip-compress `data`, offloading to a blocking thread for large inputs.
///
/// Returns `Some(compressed)` on success, or `None` (after logging a warning)
/// when the compressor errors or a spawned task fails to join — so the caller
/// can degrade gracefully to the identity body.
async fn gzip_compress_maybe_blocking(data: Bytes, level: u8, len: usize) -> Option<Vec<u8>> {
    if len >= SPAWN_BLOCKING_THRESHOLD {
        match tokio::task::spawn_blocking(move || oxiarc_deflate::gzip::gzip_compress(&data, level))
            .await
        {
            Ok(Ok(compressed)) => Some(compressed),
            Ok(Err(error)) => {
                warn!(error = %error, "gzip compression failed; serving identity response");
                None
            }
            Err(join_error) => {
                warn!(error = %join_error, "gzip compression task failed to join; serving identity response");
                None
            }
        }
    } else {
        match oxiarc_deflate::gzip::gzip_compress(&data, level) {
            Ok(compressed) => Some(compressed),
            Err(error) => {
                warn!(error = %error, "gzip compression failed; serving identity response");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request, routing::get, Router};
    use futures::stream;
    use oxiarc_deflate::gzip::gzip_decompress;
    use tower::ServiceExt;

    // ---- Fixtures --------------------------------------------------------

    /// Large, highly-compressible payload (crosses the spawn-blocking
    /// threshold so the round-trip test also exercises that path).
    const COMPRESSIBLE_LEN: usize = 512 * 1024;
    /// High-entropy payload used to prove the only-if-smaller guard.
    const INCOMPRESSIBLE_LEN: usize = 4096;

    fn compressible_bytes(len: usize) -> Vec<u8> {
        let pattern = b"The quick brown fox jumps over the lazy dog. 0123456789. ";
        pattern.iter().copied().cycle().take(len).collect()
    }

    fn incompressible_bytes(len: usize) -> Vec<u8> {
        // Deterministic xorshift64 PRNG — no `rand` dependency needed.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 33) as u8
            })
            .collect()
    }

    fn config(min_size: usize, enable_gzip: bool) -> CompressionConfig {
        CompressionConfig {
            enable_gzip,
            level: CompressionLevel::Balanced,
            min_size,
        }
    }

    fn octet_response(body: Vec<u8>) -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(body))
            .expect("test: building octet-stream response is infallible")
    }

    fn build_router(
        path: &str,
        route: axum::routing::MethodRouter,
        cfg: CompressionConfig,
    ) -> Router {
        Router::new()
            .route(path, route)
            .layer(axum::middleware::from_fn_with_state(
                CompressionState { config: cfg },
                compress_response,
            ))
    }

    fn vary_tokens(headers: &HeaderMap) -> Vec<String> {
        headers
            .get_all(header::VARY)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .flat_map(|text| {
                text.split(',')
                    .map(|token| token.trim().to_ascii_lowercase())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn content_encoding(headers: &HeaderMap) -> Option<String> {
        headers
            .get(header::CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    }

    // ---- Handlers --------------------------------------------------------

    async fn compressible_handler() -> Response {
        octet_response(compressible_bytes(COMPRESSIBLE_LEN))
    }

    async fn small_handler() -> Response {
        octet_response(b"tiny body".to_vec())
    }

    async fn png_handler() -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "image/png")
            .body(Body::from(compressible_bytes(COMPRESSIBLE_LEN)))
            .expect("test: building png response is infallible")
    }

    async fn pre_encoded_handler() -> Response {
        // Handler claims the body is already gzipped (it is not); the
        // middleware must not touch it.
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::CONTENT_ENCODING, "gzip")
            .body(Body::from(compressible_bytes(COMPRESSIBLE_LEN)))
            .expect("test: building pre-encoded response is infallible")
    }

    async fn vary_origin_handler() -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::VARY, "Origin")
            .body(Body::from(compressible_bytes(COMPRESSIBLE_LEN)))
            .expect("test: building vary-origin response is infallible")
    }

    async fn stream_handler() -> Response {
        // A small FINITE stream — an unsized body whose `size_hint().upper()`
        // is `None`. The test cannot hang.
        let chunks: Vec<Result<Bytes, std::convert::Infallible>> = vec![
            Ok(Bytes::from_static(b"chunk-a-")),
            Ok(Bytes::from_static(b"chunk-b-")),
            Ok(Bytes::from_static(b"chunk-c")),
        ];
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from_stream(stream::iter(chunks)))
            .expect("test: building streaming response is infallible")
    }

    async fn no_content_handler() -> Response {
        Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .expect("test: building 204 response is infallible")
    }

    async fn incompressible_handler() -> Response {
        octet_response(incompressible_bytes(INCOMPRESSIBLE_LEN))
    }

    fn gzip_request(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("test: building request is infallible")
    }

    // ---- Unit tests: Accept-Encoding q-value parsing ---------------------

    #[test]
    fn test_accepts_gzip_table() {
        // The full RFC 9110 §12.5.3 negotiation table from the spec.
        let cases: &[(&str, bool)] = &[
            ("gzip", true),
            ("gzip, deflate, br", true),
            ("gzip;q=0", false),
            ("gzip;q=0.0", false),
            ("gzip;q=0.001", true),
            ("identity, gzip;q=0", false),
            ("*", true),
            ("*;q=0", false),
            ("deflate, *;q=0", false),
            ("deflate", false),
            ("GZIP", true),
            (" gzip ; q=1 ", true),
            ("gzip;q=0, *;q=1", false),
            ("gzip;q=0.5, *;q=0", true),
            ("x-gzip", true),
            ("gzip;q=bad", false),
        ];
        for (header, expected) in cases {
            assert_eq!(
                accepts_gzip(Some(header)),
                *expected,
                "accept-encoding header {header:?}"
            );
        }
        // Absent header ⇒ no compression.
        assert!(!accepts_gzip(None));
    }

    #[test]
    fn test_gzip_quality_values() {
        assert_eq!(gzip_quality("gzip"), 1.0);
        assert_eq!(gzip_quality("*"), 1.0);
        assert_eq!(gzip_quality("deflate"), 0.0);
        assert_eq!(gzip_quality("gzip;q=0"), 0.0);
        assert_eq!(gzip_quality("gzip;q=0.5, *;q=0"), 0.5);
        // Explicit gzip;q=0 overrides *;q=1.
        assert_eq!(gzip_quality("gzip;q=0, *;q=1"), 0.0);
        // Malformed q ⇒ conservative reject.
        assert_eq!(gzip_quality("gzip;q=bad"), 0.0);
        // Out-of-range q is clamped.
        assert_eq!(gzip_quality("gzip;q=5"), 1.0);
    }

    // ---- Unit tests: content-type classification -------------------------

    #[test]
    fn test_is_incompressible_content_type() {
        assert!(!is_incompressible_content_type("application/json"));
        assert!(!is_incompressible_content_type("text/html; charset=utf-8"));
        assert!(is_incompressible_content_type("image/png"));
        assert!(!is_incompressible_content_type("image/svg+xml"));
        assert!(is_incompressible_content_type("video/mp4"));
        assert!(is_incompressible_content_type("application/gzip"));
        assert!(!is_incompressible_content_type("application/octet-stream"));
        assert!(is_incompressible_content_type("font/woff2"));
        // Additional coverage: case-insensitivity, parameters, more families.
        assert!(is_incompressible_content_type("IMAGE/PNG"));
        assert!(is_incompressible_content_type("audio/mpeg"));
        assert!(is_incompressible_content_type("application/pdf"));
        assert!(is_incompressible_content_type("text/event-stream"));
        assert!(!is_incompressible_content_type(
            "image/svg+xml; charset=utf-8"
        ));
    }

    // ---- Unit tests: Vary append + ETag weaken + level mapping -----------

    #[test]
    fn test_vary_append_behavior() {
        // Empty ⇒ adds accept-encoding.
        let mut headers = HeaderMap::new();
        append_vary_accept_encoding(&mut headers);
        assert_eq!(vary_tokens(&headers), vec!["accept-encoding".to_string()]);

        // Vary: Origin ⇒ both present.
        let mut headers = HeaderMap::new();
        headers.insert(header::VARY, HeaderValue::from_static("Origin"));
        append_vary_accept_encoding(&mut headers);
        let tokens = vary_tokens(&headers);
        assert!(tokens.iter().any(|token| token == "origin"), "{tokens:?}");
        assert!(
            tokens.iter().any(|token| token == "accept-encoding"),
            "{tokens:?}"
        );

        // Already Accept-Encoding ⇒ no duplicate.
        let mut headers = HeaderMap::new();
        headers.insert(header::VARY, HeaderValue::from_static("Accept-Encoding"));
        append_vary_accept_encoding(&mut headers);
        assert_eq!(headers.get_all(header::VARY).iter().count(), 1);

        // Vary: * ⇒ no add.
        let mut headers = HeaderMap::new();
        headers.insert(header::VARY, HeaderValue::from_static("*"));
        append_vary_accept_encoding(&mut headers);
        assert_eq!(vary_tokens(&headers), vec!["*".to_string()]);
    }

    #[test]
    fn test_weaken_strong_etag() {
        // Strong ⇒ weakened.
        let mut headers = HeaderMap::new();
        headers.insert(header::ETAG, HeaderValue::from_static("\"abc\""));
        weaken_strong_etag(&mut headers);
        assert_eq!(
            headers
                .get(header::ETAG)
                .and_then(|value| value.to_str().ok()),
            Some("W/\"abc\"")
        );

        // Already weak ⇒ unchanged.
        let mut headers = HeaderMap::new();
        headers.insert(header::ETAG, HeaderValue::from_static("W/\"abc\""));
        weaken_strong_etag(&mut headers);
        assert_eq!(
            headers
                .get(header::ETAG)
                .and_then(|value| value.to_str().ok()),
            Some("W/\"abc\"")
        );

        // Absent ⇒ nothing inserted.
        let mut headers = HeaderMap::new();
        weaken_strong_etag(&mut headers);
        assert!(headers.get(header::ETAG).is_none());
    }

    #[test]
    fn test_compression_level_mapping() {
        assert_eq!(compression_level(CompressionLevel::Fastest), 1);
        assert_eq!(compression_level(CompressionLevel::Balanced), 5);
        assert_eq!(compression_level(CompressionLevel::Best), 9);
        // Out-of-range custom levels are clamped to 9.
        assert_eq!(compression_level(CompressionLevel::Custom(20)), 9);
        assert_eq!(compression_level(CompressionLevel::Custom(3)), 3);
        // Level 0 is store-only and panic-free.
        assert_eq!(compression_level(CompressionLevel::Custom(0)), 0);
    }

    // ---- Integration tests via `tower::ServiceExt::oneshot` --------------

    /// (1) THE CORE TEST: large body + `Accept-Encoding: gzip` round-trips.
    #[tokio::test]
    async fn test_roundtrip_gzip_compression() {
        let original = compressible_bytes(COMPRESSIBLE_LEN);
        let router = build_router("/data", get(compressible_handler), config(64, true));

        let response = router
            .oneshot(gzip_request("/data"))
            .await
            .expect("test: router responds");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            content_encoding(response.headers()).as_deref(),
            Some("gzip")
        );
        assert!(
            vary_tokens(response.headers())
                .iter()
                .any(|token| token == "accept-encoding"),
            "Vary must contain accept-encoding"
        );

        let declared_len: usize = response
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|text| text.parse().ok())
            .expect("test: numeric Content-Length present");

        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test: collect body");

        assert_eq!(
            body.len(),
            declared_len,
            "Content-Length matches body length"
        );
        assert!(body.len() < original.len(), "compressed body is smaller");

        let decoded = gzip_decompress(&body).expect("test: gzip stream decompresses");
        assert_eq!(
            decoded, original,
            "round-trip reproduces the original bytes"
        );
    }

    /// (2) No `Accept-Encoding` ⇒ identity.
    #[tokio::test]
    async fn test_no_accept_encoding_is_identity() {
        let original = compressible_bytes(COMPRESSIBLE_LEN);
        let router = build_router("/data", get(compressible_handler), config(64, true));

        let request = Request::builder()
            .uri("/data")
            .body(Body::empty())
            .expect("test: build request");
        let response = router
            .oneshot(request)
            .await
            .expect("test: router responds");

        assert!(content_encoding(response.headers()).is_none());
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test: collect body");
        assert_eq!(body.as_ref(), original.as_slice());
    }

    /// (3) `Accept-Encoding: gzip;q=0` ⇒ not compressed.
    #[tokio::test]
    async fn test_gzip_qvalue_zero_is_identity() {
        let router = build_router("/data", get(compressible_handler), config(64, true));

        let request = Request::builder()
            .uri("/data")
            .header(header::ACCEPT_ENCODING, "gzip;q=0")
            .body(Body::empty())
            .expect("test: build request");
        let response = router
            .oneshot(request)
            .await
            .expect("test: router responds");

        assert!(content_encoding(response.headers()).is_none());
    }

    /// (4) Body below `min_size` ⇒ not compressed.
    #[tokio::test]
    async fn test_below_min_size_is_identity() {
        // min_size far larger than the tiny body.
        let router = build_router("/small", get(small_handler), config(1024 * 1024, true));

        let response = router
            .oneshot(gzip_request("/small"))
            .await
            .expect("test: router responds");

        assert!(content_encoding(response.headers()).is_none());
    }

    /// (5) Handler pre-sets `Content-Encoding: gzip` ⇒ untouched.
    #[tokio::test]
    async fn test_pre_encoded_is_untouched() {
        let router = build_router("/data", get(pre_encoded_handler), config(64, true));

        let response = router
            .oneshot(gzip_request("/data"))
            .await
            .expect("test: router responds");

        // The handler's single gzip Content-Encoding is preserved (not doubled).
        assert_eq!(
            content_encoding(response.headers()).as_deref(),
            Some("gzip")
        );
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test: collect body");
        // Body was NOT recompressed by the middleware: it is the original bytes.
        assert_eq!(body.len(), COMPRESSIBLE_LEN);
    }

    /// (6) Handler sets `Vary: Origin` ⇒ compressed response has BOTH.
    #[tokio::test]
    async fn test_vary_origin_preserved_and_appended() {
        let router = build_router("/data", get(vary_origin_handler), config(64, true));

        let response = router
            .oneshot(gzip_request("/data"))
            .await
            .expect("test: router responds");

        assert_eq!(
            content_encoding(response.headers()).as_deref(),
            Some("gzip")
        );
        let tokens = vary_tokens(response.headers());
        assert!(tokens.iter().any(|token| token == "origin"), "{tokens:?}");
        assert!(
            tokens.iter().any(|token| token == "accept-encoding"),
            "{tokens:?}"
        );
    }

    /// (7) `image/png` above `min_size` ⇒ not compressed.
    #[tokio::test]
    async fn test_incompressible_content_type_is_identity() {
        let router = build_router("/img", get(png_handler), config(64, true));

        let response = router
            .oneshot(gzip_request("/img"))
            .await
            .expect("test: router responds");

        assert!(content_encoding(response.headers()).is_none());
    }

    /// (8) Unsized `Body::from_stream` ⇒ NOT compressed (size-hint guard).
    #[tokio::test]
    async fn test_streaming_body_is_never_buffered() {
        let router = build_router("/stream", get(stream_handler), config(1, true));

        let response = router
            .oneshot(gzip_request("/stream"))
            .await
            .expect("test: router responds");

        // The streaming guard fired: no compression was applied.
        assert!(content_encoding(response.headers()).is_none());

        // The (finite) stream still flows through untouched.
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test: collect body");
        assert_eq!(body.as_ref(), b"chunk-a-chunk-b-chunk-c");
    }

    /// (9) `HEAD` ⇒ not compressed.
    #[tokio::test]
    async fn test_head_request_is_identity() {
        let router = build_router("/data", get(compressible_handler), config(64, true));

        let request = Request::builder()
            .method(Method::HEAD)
            .uri("/data")
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("test: build HEAD request");
        let response = router
            .oneshot(request)
            .await
            .expect("test: router responds");

        assert!(content_encoding(response.headers()).is_none());
    }

    /// (10) `enable_gzip: false` ⇒ not compressed.
    #[tokio::test]
    async fn test_disabled_is_identity() {
        let router = build_router("/data", get(compressible_handler), config(64, false));

        let response = router
            .oneshot(gzip_request("/data"))
            .await
            .expect("test: router responds");

        assert!(content_encoding(response.headers()).is_none());
    }

    /// (11) `204 No Content` ⇒ not compressed, empty body preserved.
    #[tokio::test]
    async fn test_no_content_is_identity() {
        let router = build_router("/none", get(no_content_handler), config(1, true));

        let response = router
            .oneshot(gzip_request("/none"))
            .await
            .expect("test: router responds");

        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(content_encoding(response.headers()).is_none());
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test: collect body");
        assert!(body.is_empty());
    }

    /// (12) Only-if-smaller: above-min_size incompressible bytes ⇒ identity.
    #[tokio::test]
    async fn test_only_if_smaller_serves_identity() {
        let original = incompressible_bytes(INCOMPRESSIBLE_LEN);
        let router = build_router("/rand", get(incompressible_handler), config(64, true));

        let response = router
            .oneshot(gzip_request("/rand"))
            .await
            .expect("test: router responds");

        // gzip of high-entropy data does not shrink it, so identity is served.
        assert!(content_encoding(response.headers()).is_none());
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test: collect body");
        assert_eq!(body.as_ref(), original.as_slice());
    }
}
