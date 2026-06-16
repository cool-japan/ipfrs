# Contributing to IPFRS

Thank you for your interest in contributing to IPFRS! This document provides guidelines for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How Can I Contribute?

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Relevant logs

### Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Clear feature description
- Motivation and use cases
- Proposed implementation
- Alternatives considered

### Improving Documentation

Use the [Documentation template](.github/ISSUE_TEMPLATE/documentation.md) or submit a PR directly.

### Submitting Code

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `cargo test`
5. **Format code**: `cargo fmt`
6. **Check lints**: `cargo clippy`
7. **Commit**: `git commit -m "Add amazing feature"`
8. **Push**: `git push origin feature/amazing-feature`
9. **Open a Pull Request**

## Development Setup

### Prerequisites

- Rust 1.70 or later
- Git
- cargo-fmt, cargo-clippy

### Building

```bash
git clone https://github.com/ipfrs/ipfrs.git
cd ipfrs
cargo build
```

### Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p ipfrs-core

# Integration tests
cargo test --test '*'

# With output
cargo test -- --nocapture
```

### Running Benchmarks

```bash
cargo bench
```

### Running Fuzzing

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Run a fuzz target
cd crates/ipfrs
cargo fuzz run fuzz_block_operations
```

## Code Style

### Rust Style Guide

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Address all `cargo clippy` warnings

### Code Organization

```rust
// Module-level documentation
//! Brief module description
//!
//! Detailed explanation...

use std::...;  // Standard library
use external_crate::...;  // External crates
use crate::...;  // Internal crates

/// Function documentation
///
/// # Examples
/// ```
/// # use ipfrs::...;
/// let result = function();
/// ```
pub fn function() -> Result<()> {
    // Implementation
}
```

### Documentation

- Document all public APIs
- Include examples in doc comments
- Update relevant documentation files

### Testing

- Write unit tests for new functionality
- Add integration tests for complex features
- Ensure all tests pass before submitting
- Aim for high test coverage

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Examples:

```
feat(semantic): add batch indexing support

Implement batch indexing to improve performance when indexing
multiple items at once.

Closes #123
```

```
fix(network): resolve DHT connection timeout

The DHT was timing out when connecting to bootstrap peers.
Increased timeout from 5s to 30s.

Fixes #456
```

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Update CHANGELOG.md** under "Unreleased"
4. **Ensure CI passes** (all checks must be green)
5. **Request review** from maintainers
6. **Address feedback** promptly
7. **Squash commits** if requested

## Review Process

- Maintainers will review within 1-2 weeks
- Address all feedback before merge
- At least one approval required
- CI must pass

## Release Process

(For maintainers)

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create release commit: `git commit -m "Release v0.x.0"`
4. Tag release: `git tag -a v0.x.0 -m "Release v0.x.0"`
5. Push: `git push origin main --tags`
6. Publish to crates.io: `cargo publish`
7. Create GitHub release

## Project Structure

```
ipfrs/
├── crates/
│   ├── ipfrs/           # Main crate
│   ├── ipfrs-core/      # Core functionality
│   ├── ipfrs-storage/   # Storage layer
│   ├── ipfrs-semantic/  # Semantic search
│   ├── ipfrs-tensorlogic/ # Logic programming
│   ├── ipfrs-network/   # Networking
│   ├── ipfrs-transport/ # Transport protocols
│   └── ipfrs-interface/ # HTTP/GraphQL APIs
├── bindings/
│   ├── python/          # Python bindings
│   ├── javascript/      # JS bindings
│   └── wasm/            # WASM bindings
├── book/                # Documentation
├── examples/            # Example code
└── benches/             # Benchmarks
```

## Questions?

- Open a [Discussion](https://github.com/ipfrs/ipfrs/discussions)
- Ask in issue comments
- Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.

Thank you for contributing to IPFRS! 🚀
