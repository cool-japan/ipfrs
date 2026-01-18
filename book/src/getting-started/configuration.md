# Configuration

Learn how to configure IPFRS for your specific needs.

## Configuration File

IPFRS uses TOML format for configuration. The default location is `~/.ipfrs/config.toml`.

### Full Configuration Example

```toml
# IPFRS Configuration File

[storage]
# Path to block storage directory
path = "~/.ipfrs/blocks"

# Cache size in MB
cache_size_mb = 512

# Enable bloom filter for faster existence checks
enable_bloom_filter = true

[network]
# Listen addresses for libp2p
listen_addresses = [
    "/ip4/0.0.0.0/tcp/4001",
    "/ip4/0.0.0.0/udp/4001/quic"
]

# Enable DHT for content discovery
enable_dht = true

# Enable mDNS for local peer discovery
enable_mdns = true

# Bootstrap peers
bootstrap_peers = [
    "/dnsaddr/bootstrap.ipfrs.io/p2p/12D3KooW...",
]

# Maximum number of connections
max_connections = 100

[http]
# HTTP API listen address
address = "127.0.0.1:8080"

# Enable CORS
enable_cors = true

# Allowed CORS origins
cors_origins = ["*"]

# Request timeout in seconds
request_timeout_secs = 30

# Enable TLS
enable_tls = false

# TLS certificate path
tls_cert_path = ""

# TLS key path
tls_key_path = ""

[graphql]
# Enable GraphQL API
enable = true

# GraphQL endpoint path
endpoint = "/graphql"

# Enable GraphQL playground
enable_playground = true

[semantic]
# Enable semantic search
enable = true

# Vector dimension
dimension = 384

# HNSW parameters
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 100

# Index persistence path
index_path = "~/.ipfrs/semantic.index"

# Auto-save interval (seconds, 0 = disabled)
auto_save_interval = 300

[tensorlogic]
# Enable logic programming
enable = true

# Maximum inference depth
max_depth = 100

# Maximum solutions to return
max_solutions = 100

# Knowledge base persistence path
kb_path = "~/.ipfrs/knowledge_base.kb"

# Auto-save interval (seconds, 0 = disabled)
auto_save_interval = 300

[auth]
# Enable authentication
enable = false

# JWT secret (change in production!)
jwt_secret = "CHANGE_THIS_SECRET_IN_PRODUCTION"

# Token expiration (seconds)
token_expiration_secs = 86400

# Default admin user
admin_username = "admin"
admin_password_hash = ""

[metrics]
# Enable Prometheus metrics
enable = true

# Metrics listen address
address = "127.0.0.1:9000"

[tracing]
# Enable OpenTelemetry tracing
enable = false

# OTLP endpoint
otlp_endpoint = "http://localhost:4317"

# Service name
service_name = "ipfrs"

# Log level (trace, debug, info, warn, error)
log_level = "info"

# Enable JSON logging
json_logging = false

[reliability]
# Enable health checks
enable_health_checks = true

# Graceful shutdown timeout (seconds)
shutdown_timeout_secs = 30

# Enable circuit breaker
enable_circuit_breaker = true

# Circuit breaker failure threshold
circuit_breaker_threshold = 5

# Circuit breaker timeout (seconds)
circuit_breaker_timeout_secs = 60
```

## Environment Variables

Configuration can also be set via environment variables:

```bash
# Storage
export IPFRS_STORAGE_PATH=~/.ipfrs/blocks
export IPFRS_STORAGE_CACHE_SIZE_MB=512

# Network
export IPFRS_NETWORK_LISTEN_ADDRESSES='/ip4/0.0.0.0/tcp/4001,/ip4/0.0.0.0/udp/4001/quic'
export IPFRS_NETWORK_ENABLE_DHT=true
export IPFRS_NETWORK_ENABLE_MDNS=true

# HTTP
export IPFRS_HTTP_ADDRESS=127.0.0.1:8080
export IPFRS_HTTP_ENABLE_CORS=true

# Semantic
export IPFRS_SEMANTIC_ENABLE=true
export IPFRS_SEMANTIC_DIMENSION=384

# TensorLogic
export IPFRS_TENSORLOGIC_ENABLE=true
export IPFRS_TENSORLOGIC_MAX_DEPTH=100

# Auth
export IPFRS_AUTH_ENABLE=true
export IPFRS_AUTH_JWT_SECRET=your_secret_here

# Metrics
export IPFRS_METRICS_ENABLE=true
export IPFRS_METRICS_ADDRESS=127.0.0.1:9000

# Tracing
export IPFRS_TRACING_ENABLE=true
export IPFRS_TRACING_OTLP_ENDPOINT=http://localhost:4317
export IPFRS_TRACING_LOG_LEVEL=info
```

Environment variables take precedence over configuration file values.

## Command-Line Flags

Override configuration with CLI flags:

```bash
ipfrs daemon \
  --http-addr 0.0.0.0:8080 \
  --storage-path /data/ipfrs \
  --cache-size 1024 \
  --enable-dht \
  --enable-mdns \
  --log-level debug
```

## Configuration Precedence

Configuration is loaded in this order (later sources override earlier):

1. Default values
2. Configuration file (`~/.ipfrs/config.toml`)
3. Environment variables
4. Command-line flags

## Common Configurations

### Development

```toml
[http]
address = "127.0.0.1:8080"
enable_cors = true

[metrics]
enable = true
address = "127.0.0.1:9000"

[tracing]
enable = true
log_level = "debug"

[auth]
enable = false  # Disable auth for development
```

### Production

```toml
[http]
address = "0.0.0.0:8080"
enable_cors = true
cors_origins = ["https://app.example.com"]
enable_tls = true
tls_cert_path = "/etc/ipfrs/cert.pem"
tls_key_path = "/etc/ipfrs/key.pem"

[network]
listen_addresses = [
    "/ip4/0.0.0.0/tcp/4001",
    "/ip4/0.0.0.0/udp/4001/quic"
]
enable_dht = true
max_connections = 500

[auth]
enable = true
jwt_secret = "GENERATE_STRONG_SECRET"
token_expiration_secs = 3600

[metrics]
enable = true
address = "127.0.0.1:9000"

[tracing]
enable = true
otlp_endpoint = "http://jaeger:4317"
log_level = "info"
json_logging = true

[reliability]
enable_health_checks = true
shutdown_timeout_secs = 30
enable_circuit_breaker = true
```

### High Performance

```toml
[storage]
cache_size_mb = 4096  # Large cache
enable_bloom_filter = true

[semantic]
hnsw_m = 32  # Higher connectivity for better recall
hnsw_ef_construction = 400
hnsw_ef_search = 200

[network]
max_connections = 1000
```

### Low Resource

```toml
[storage]
cache_size_mb = 128  # Minimal cache

[network]
max_connections = 20

[semantic]
hnsw_m = 8  # Lower connectivity
hnsw_ef_construction = 100
hnsw_ef_search = 50
```

## Security Considerations

### Authentication

Always enable authentication in production:

```toml
[auth]
enable = true
jwt_secret = "GENERATE_STRONG_RANDOM_SECRET_HERE"
```

Generate a secure secret:

```bash
openssl rand -base64 32
```

### TLS/SSL

Enable TLS for HTTPS:

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure IPFRS
[http]
enable_tls = true
tls_cert_path = "/path/to/cert.pem"
tls_key_path = "/path/to/key.pem"
```

For production, use certificates from a trusted CA (Let's Encrypt, etc.).

### Network Security

Restrict network access:

```toml
[http]
address = "127.0.0.1:8080"  # Only localhost

[metrics]
address = "127.0.0.1:9000"  # Only localhost
```

Use a reverse proxy (nginx, Caddy) for public access.

## Next Steps

- [Quick Start](./quick-start.md) - Start using IPFRS
- [Security Guide](../advanced/security.md) - Secure your deployment
- [Performance Tuning](../advanced/performance.md) - Optimize for your workload
