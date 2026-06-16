//! Network Utilities
//!
//! This module provides common utility functions for network operations.

use libp2p::{Multiaddr, PeerId};
use std::time::Duration;

/// Format bytes in human-readable format (B, KB, MB, GB, TB)
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::format_bytes;
///
/// assert_eq!(format_bytes(1024), "1.00 KB");
/// assert_eq!(format_bytes(1_048_576), "1.00 MB");
/// assert_eq!(format_bytes(500), "500 B");
/// ```
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    const TB: usize = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format bytes per second in human-readable format (B/s, KB/s, MB/s, GB/s)
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::format_bandwidth;
///
/// assert_eq!(format_bandwidth(1024), "1.00 KB/s");
/// assert_eq!(format_bandwidth(1_048_576), "1.00 MB/s");
/// ```
pub fn format_bandwidth(bytes_per_sec: usize) -> String {
    format!("{}/s", format_bytes(bytes_per_sec))
}

/// Format duration in human-readable format
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use ipfrs_network::utils::format_duration;
///
/// assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
/// assert_eq!(format_duration(Duration::from_secs(3665)), "1h 1m 5s");
/// assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
/// ```
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();

    if total_secs == 0 {
        if millis == 0 {
            return format!("{}µs", duration.subsec_micros());
        }
        return format!("{}ms", millis);
    }

    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    let mut parts = Vec::new();
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{}m", minutes));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{}s", seconds));
    }

    parts.join(" ")
}

/// Parse a multiaddress string
///
/// # Errors
///
/// Returns an error if the address cannot be parsed
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::parse_multiaddr;
///
/// let addr = parse_multiaddr("/ip4/127.0.0.1/tcp/4001").unwrap();
/// ```
pub fn parse_multiaddr(addr: &str) -> Result<Multiaddr, String> {
    addr.parse::<Multiaddr>()
        .map_err(|e| format!("Failed to parse multiaddress: {}", e))
}

/// Parse multiple multiaddress strings
///
/// # Errors
///
/// Returns an error if any address cannot be parsed
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::parse_multiaddrs;
///
/// let addrs = parse_multiaddrs(&[
///     "/ip4/127.0.0.1/tcp/4001".to_string(),
///     "/ip6/::1/tcp/4001".to_string(),
/// ]).unwrap();
/// assert_eq!(addrs.len(), 2);
/// ```
pub fn parse_multiaddrs(addrs: &[String]) -> Result<Vec<Multiaddr>, String> {
    addrs.iter().map(|s| parse_multiaddr(s)).collect()
}

/// Check if a multiaddress is a local address (loopback or link-local)
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::{parse_multiaddr, is_local_addr};
///
/// let local = parse_multiaddr("/ip4/127.0.0.1/tcp/4001").unwrap();
/// assert!(is_local_addr(&local));
///
/// let public = parse_multiaddr("/ip4/8.8.8.8/tcp/4001").unwrap();
/// assert!(!is_local_addr(&public));
/// ```
pub fn is_local_addr(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;

    for proto in addr.iter() {
        match proto {
            Protocol::Ip4(ip) => {
                return ip.is_loopback() || ip.is_link_local() || ip.is_private();
            }
            Protocol::Ip6(ip) => {
                return ip.is_loopback() || ip.is_unicast_link_local();
            }
            _ => continue,
        }
    }
    false
}

/// Check if a multiaddress is a public address
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::{parse_multiaddr, is_public_addr};
///
/// let public = parse_multiaddr("/ip4/8.8.8.8/tcp/4001").unwrap();
/// assert!(is_public_addr(&public));
///
/// let local = parse_multiaddr("/ip4/127.0.0.1/tcp/4001").unwrap();
/// assert!(!is_public_addr(&local));
/// ```
pub fn is_public_addr(addr: &Multiaddr) -> bool {
    !is_local_addr(addr)
}

/// Calculate exponential backoff duration
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use ipfrs_network::utils::exponential_backoff;
///
/// assert_eq!(exponential_backoff(0, Duration::from_secs(1), Duration::from_secs(60)),
///            Duration::from_secs(1));
/// assert_eq!(exponential_backoff(1, Duration::from_secs(1), Duration::from_secs(60)),
///            Duration::from_secs(2));
/// assert_eq!(exponential_backoff(2, Duration::from_secs(1), Duration::from_secs(60)),
///            Duration::from_secs(4));
/// ```
pub fn exponential_backoff(attempt: u32, base: Duration, max: Duration) -> Duration {
    let backoff = base.saturating_mul(2_u32.saturating_pow(attempt));
    backoff.min(max)
}

/// Calculate jittered exponential backoff duration
///
/// Adds random jitter (±25%) to prevent thundering herd problem
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use ipfrs_network::utils::jittered_backoff;
///
/// let backoff = jittered_backoff(2, Duration::from_secs(1), Duration::from_secs(60));
/// // Should be roughly 4 seconds ± 25%
/// assert!(backoff >= Duration::from_secs(3));
/// assert!(backoff <= Duration::from_secs(5));
/// ```
pub fn jittered_backoff(attempt: u32, base: Duration, max: Duration) -> Duration {
    use rand::Rng;
    let backoff = exponential_backoff(attempt, base, max);
    let mut rng = rand::rng();
    let random_value = rng.next_u64() as f64 / u64::MAX as f64;
    let jitter = 0.75 + (random_value * 0.5); // Maps [0, 1] to [0.75, 1.25]
    Duration::from_secs_f64(backoff.as_secs_f64() * jitter)
}

/// Truncate a peer ID for display purposes
///
/// # Examples
///
/// ```
/// use libp2p::PeerId;
/// use ipfrs_network::utils::truncate_peer_id;
///
/// let peer_id = PeerId::random();
/// let truncated = truncate_peer_id(&peer_id, 8);
/// assert_eq!(truncated.len(), 11); // "12..." + 8 chars
/// ```
pub fn truncate_peer_id(peer_id: &PeerId, length: usize) -> String {
    let s = peer_id.to_string();
    if s.len() <= length + 3 {
        s
    } else {
        format!("{}...{}", &s[..length / 2], &s[s.len() - length / 2..])
    }
}

/// Calculate percentage with proper rounding
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::percentage;
///
/// assert_eq!(percentage(25, 100), 25.0);
/// assert_eq!(percentage(1, 3), 33.33);
/// assert_eq!(percentage(0, 0), 0.0); // Handles division by zero
/// ```
pub fn percentage(value: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        ((value as f64 / total as f64) * 10000.0).round() / 100.0
    }
}

/// Calculate moving average
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::moving_average;
///
/// let current = 10.0;
/// let new_value = 20.0;
/// let alpha = 0.5;
///
/// assert_eq!(moving_average(current, new_value, alpha), 15.0);
/// ```
pub fn moving_average(current: f64, new_value: f64, alpha: f64) -> f64 {
    alpha * new_value + (1.0 - alpha) * current
}

/// Validate alpha value for exponential moving average
///
/// # Panics
///
/// Panics if alpha is not in range [0.0, 1.0]
///
/// # Examples
///
/// ```
/// use ipfrs_network::utils::validate_alpha;
///
/// validate_alpha(0.5); // OK
/// validate_alpha(0.0); // OK
/// validate_alpha(1.0); // OK
/// ```
///
/// ```should_panic
/// use ipfrs_network::utils::validate_alpha;
///
/// validate_alpha(1.5); // Panics
/// ```
pub fn validate_alpha(alpha: f64) {
    assert!(
        (0.0..=1.0).contains(&alpha),
        "Alpha must be in range [0.0, 1.0], got {}",
        alpha
    );
}

/// Check if two peer IDs match
///
/// # Examples
///
/// ```
/// use libp2p::PeerId;
/// use ipfrs_network::utils::peers_match;
///
/// let peer1 = PeerId::random();
/// let peer2 = peer1;
/// let peer3 = PeerId::random();
///
/// assert!(peers_match(&peer1, &peer2));
/// assert!(!peers_match(&peer1, &peer3));
/// ```
pub fn peers_match(peer1: &PeerId, peer2: &PeerId) -> bool {
    peer1 == peer2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_bytes(1_099_511_627_776), "1.00 TB");
    }

    #[test]
    fn test_format_bandwidth() {
        assert_eq!(format_bandwidth(1024), "1.00 KB/s");
        assert_eq!(format_bandwidth(1_048_576), "1.00 MB/s");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3665)), "1h 1m 5s");
        assert_eq!(format_duration(Duration::from_secs(7200)), "2h");
    }

    #[test]
    fn test_parse_multiaddr() {
        let addr =
            parse_multiaddr("/ip4/127.0.0.1/tcp/4001").expect("test: valid multiaddr should parse");
        assert!(addr.to_string().contains("127.0.0.1"));
    }

    #[test]
    fn test_parse_multiaddrs() {
        let addrs = parse_multiaddrs(&[
            "/ip4/127.0.0.1/tcp/4001".to_string(),
            "/ip6/::1/tcp/4001".to_string(),
        ])
        .expect("test: valid multiaddrs should parse");
        assert_eq!(addrs.len(), 2);
    }

    #[test]
    fn test_is_local_addr() {
        let local =
            parse_multiaddr("/ip4/127.0.0.1/tcp/4001").expect("test: valid multiaddr should parse");
        assert!(is_local_addr(&local));

        let local_ipv6 =
            parse_multiaddr("/ip6/::1/tcp/4001").expect("test: valid multiaddr should parse");
        assert!(is_local_addr(&local_ipv6));

        let private = parse_multiaddr("/ip4/192.168.1.1/tcp/4001")
            .expect("test: valid multiaddr should parse");
        assert!(is_local_addr(&private));

        let public =
            parse_multiaddr("/ip4/8.8.8.8/tcp/4001").expect("test: valid multiaddr should parse");
        assert!(!is_local_addr(&public));
    }

    #[test]
    fn test_is_public_addr() {
        let public =
            parse_multiaddr("/ip4/8.8.8.8/tcp/4001").expect("test: valid multiaddr should parse");
        assert!(is_public_addr(&public));

        let local =
            parse_multiaddr("/ip4/127.0.0.1/tcp/4001").expect("test: valid multiaddr should parse");
        assert!(!is_public_addr(&local));
    }

    #[test]
    fn test_exponential_backoff() {
        let base = Duration::from_secs(1);
        let max = Duration::from_secs(60);

        assert_eq!(exponential_backoff(0, base, max), Duration::from_secs(1));
        assert_eq!(exponential_backoff(1, base, max), Duration::from_secs(2));
        assert_eq!(exponential_backoff(2, base, max), Duration::from_secs(4));
        assert_eq!(exponential_backoff(3, base, max), Duration::from_secs(8));
        assert_eq!(exponential_backoff(10, base, max), Duration::from_secs(60));
        // Capped at max
    }

    #[test]
    fn test_jittered_backoff() {
        let base = Duration::from_secs(1);
        let max = Duration::from_secs(60);

        for attempt in 0..5 {
            let backoff = jittered_backoff(attempt, base, max);
            let expected = exponential_backoff(attempt, base, max);
            // Jitter should be within ±25%
            assert!(backoff.as_secs_f64() >= expected.as_secs_f64() * 0.75);
            assert!(backoff.as_secs_f64() <= expected.as_secs_f64() * 1.25);
        }
    }

    #[test]
    fn test_truncate_peer_id() {
        let peer_id = PeerId::random();
        let truncated = truncate_peer_id(&peer_id, 8);
        assert!(truncated.len() <= peer_id.to_string().len());
        assert!(truncated.contains("..."));
    }

    #[test]
    fn test_percentage() {
        assert_eq!(percentage(25, 100), 25.0);
        assert_eq!(percentage(1, 3), 33.33);
        assert_eq!(percentage(2, 3), 66.67);
        assert_eq!(percentage(0, 0), 0.0);
        assert_eq!(percentage(5, 0), 0.0);
    }

    #[test]
    fn test_moving_average() {
        assert_eq!(moving_average(10.0, 20.0, 0.5), 15.0);
        assert_eq!(moving_average(10.0, 20.0, 0.0), 10.0);
        assert_eq!(moving_average(10.0, 20.0, 1.0), 20.0);
    }

    #[test]
    fn test_validate_alpha() {
        validate_alpha(0.0);
        validate_alpha(0.5);
        validate_alpha(1.0);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in range")]
    fn test_validate_alpha_too_high() {
        validate_alpha(1.5);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in range")]
    fn test_validate_alpha_negative() {
        validate_alpha(-0.1);
    }

    #[test]
    fn test_peers_match() {
        let peer1 = PeerId::random();
        let peer2 = peer1;
        let peer3 = PeerId::random();

        assert!(peers_match(&peer1, &peer2));
        assert!(!peers_match(&peer1, &peer3));
    }
}
