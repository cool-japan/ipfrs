# Installation

This guide walks you through installing IPFRS on your system.

## Prerequisites

- **Rust**: 1.70 or later
- **Cargo**: Rust's package manager
- **Git**: For cloning the repository

### Installing Rust

If you don't have Rust installed, use [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Installation Methods

### From Source

Clone and build from source:

```bash
# Clone the repository
git clone https://github.com/ipfrs/ipfrs.git
cd ipfrs

# Build the project
cargo build --release

# The binary will be at target/release/ipfrs-cli
```

### Using Cargo

Install directly from crates.io (once published):

```bash
cargo install ipfrs-cli
```

### Pre-built Binaries

Download pre-built binaries from the [releases page](https://github.com/ipfrs/ipfrs/releases):

```bash
# Linux (x86_64)
wget https://github.com/ipfrs/ipfrs/releases/download/v0.1.0/ipfrs-linux-x86_64.tar.gz
tar -xzf ipfrs-linux-x86_64.tar.gz
sudo mv ipfrs /usr/local/bin/

# macOS (x86_64)
wget https://github.com/ipfrs/ipfrs/releases/download/v0.1.0/ipfrs-macos-x86_64.tar.gz
tar -xzf ipfrs-macos-x86_64.tar.gz
sudo mv ipfrs /usr/local/bin/

# macOS (ARM64)
wget https://github.com/ipfrs/ipfrs/releases/download/v0.1.0/ipfrs-macos-arm64.tar.gz
tar -xzf ipfrs-macos-arm64.tar.gz
sudo mv ipfrs /usr/local/bin/
```

## Verify Installation

Check that IPFRS is installed correctly:

```bash
ipfrs --version
```

You should see output like:

```
ipfrs 0.1.0
```

## Language Bindings

### Python

Install the Python bindings using pip:

```bash
pip install ipfrs
```

Or from source:

```bash
cd bindings/python
pip install maturin
maturin develop --release
```

### JavaScript/TypeScript

Install the npm package:

```bash
npm install @ipfrs/core
```

Or using yarn:

```bash
yarn add @ipfrs/core
```

### WebAssembly

For browser usage:

```bash
npm install @ipfrs/wasm
```

## Configuration

Create a configuration directory:

```bash
mkdir -p ~/.ipfrs
```

IPFRS will automatically create default configuration files on first run.

## Next Steps

- [Quick Start Guide](./quick-start.md) - Get up and running
- [Configuration](./configuration.md) - Configure IPFRS for your needs
- [Basic Concepts](./concepts.md) - Learn core concepts

## Troubleshooting

### Build Errors

If you encounter build errors, ensure you have the latest Rust toolchain:

```bash
rustup update
```

### Missing Dependencies

On Linux, you may need additional system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# Fedora
sudo dnf install gcc pkg-config openssl-devel

# Arch Linux
sudo pacman -S base-devel openssl
```

### Permission Issues

If you get permission errors when installing globally:

```bash
# Linux/macOS: Use sudo or install to user directory
cargo install --root ~/.local ipfrs-cli
```

Then add `~/.local/bin` to your PATH.
