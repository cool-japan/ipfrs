# Changelog

All notable changes to ipfrs-cli will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (2026-01-09)

#### Gateway TLS/HTTPS Support
- TLS/HTTPS support for HTTP gateway with `--tls-cert` and `--tls-key` CLI flags
- Automatic TLS configuration validation
- Protocol detection (HTTP vs HTTPS)
- Comprehensive error handling for certificate/key files
- Updated documentation with HTTPS examples

Example:
```bash
ipfrs gateway -l 0.0.0.0:8443 --tls-cert cert.pem --tls-key key.pem
```

#### Comprehensive User Guide
- Created `USER_GUIDE.md` with 12 major sections
- Complete CLI feature documentation
- Step-by-step tutorials and workflows
- Network operations guide
- TensorLogic extensions documentation
- Shell scripting integration examples
- Best practices and optimization tips
- Comprehensive troubleshooting section

#### Example Shell Scripts
- Created 5 production-ready workflow scripts in `examples/scripts/`
- All scripts include error handling and documentation

Scripts:
1. **backup.sh** - Repository backup (config, pins, DAG export to CAR)
2. **restore.sh** - Repository restoration from backup
3. **batch_add.sh** - Batch file upload with CID tracking
4. **monitor.sh** - Real-time monitoring dashboard
5. **sync.sh** - Node-to-node content synchronization

Features:
- Comprehensive error handling
- Progress indicators
- Summary statistics
- Usage documentation in `examples/scripts/README.md`

### Changed
- Major refactoring of main.rs: reduced from 4,825 to 2,079 lines (57% code reduction)
- Extracted all command implementations to modular `src/commands/` files
- Updated README.md with current architecture and refactoring notes
- Gateway command enhanced with TLS support
- TODO.md updated with latest enhancements

### Architecture
- Organized codebase into 15+ modular command files for better maintainability
- Single source of truth for all command logic
- Total codebase: 8,652 lines across all modules

### Quality Assurance
- Zero warnings from cargo build and clippy
- All 103 tests passing (43 unit + 51 integration + 9 doc)
- NO WARNINGS policy maintained

## [0.3.0] - "The Fast & The Wise"

### Added
- Terminal UI (TUI) dashboard with real-time monitoring
  - Overview, Network, Storage, and Help tabs
  - Sparkline graphs for network activity
  - Gauge widgets for resource monitoring
  - Keyboard navigation support
- Plugin system for extensibility
  - Plugin discovery from multiple locations
  - Environment variable support for plugins
  - Metadata querying via --plugin-info flag
  - Commands: `plugin list`, `plugin info`, `plugin run`
- Shell completion generation for bash, zsh, fish, PowerShell, and elvish
- Man page generation via `ipfrs-genman` binary
- Auto-update checking mechanism (check only, install TBD)
- Troubleshooting hints for common errors
- Remote daemon management support
- Shell script integration features
  - Exit codes (0-8) for different error conditions
  - Quiet mode (--quiet/-q) for pipeable output
  - JSON output support (--format json)
  - --no-color flag for logs

### Enhanced
- Interactive REPL shell with:
  - Command history persistence
  - Tab completion for commands
  - Multi-line input support with backslash continuation
  - Built-in aliases and user-defined aliases
- Progress indicators
  - Progress bars for uploads/downloads
  - Spinners for long operations
  - Transfer rate and ETA display
- Output formatting
  - Colored output with terminal detection
  - Table formatting for structured data
  - Human-readable file sizes
  - Compact mode for scripting
- Configuration management
  - Config caching with OnceLock (< 1μs cached load time)
  - Environment variable overrides
  - Remote API support

### Performance
- Config caching: < 1μs for cached loads, < 500μs for uncached
- CLI startup time: < 100ms (measured via benchmarks)
- Command parsing: < 10ms
- Added comprehensive benchmark suite using Criterion

### Testing
- 180 total tests across the project
  - 78 unit tests
  - 51 integration tests
  - 9 doc tests
  - 42 criterion benchmarks
- All tests pass with zero warnings

### Documentation
- Enhanced --help text for all commands with examples
- Migration guide from IPFS/Kubo
- Shell scripting integration guide
- Plugin development guide with example plugins
- Troubleshooting guide integrated into error messages

### Commands
All core IPFS-compatible commands plus IPFRS extensions:
- File operations: init, add, get, cat, ls
- Block operations: block get/put/stat/rm, list
- Network: swarm peers/connect/disconnect/addrs, dht findprovs/provide/findpeer
- Bootstrap: bootstrap list/add/rm
- Repository: repo gc/stat/fsck/version
- Pin management: pin add/rm/ls/verify
- DAG operations: dag get/put/resolve/export/import
- Statistics: stats repo/bw/bitswap, id, info, ping
- Daemon: daemon run/start/stop/status/restart
- Gateway: HTTP gateway server
- TensorLogic extensions:
  - tensor add/get/info/export
  - logic infer/prove/kb-stats/kb-save/kb-load
  - semantic search/index/similar/stats/save/load
  - model add/checkpoint/diff/rollback
  - gradient push/pull/aggregate/history

### Library Interface
Exposed as `ipfrs_cli` library for reusability:
- `ipfrs_cli::commands` - Modular command handlers
- `ipfrs_cli::config` - Configuration management
- `ipfrs_cli::output` - Output formatting utilities
- `ipfrs_cli::progress` - Progress indicators
- `ipfrs_cli::shell` - Interactive REPL
- `ipfrs_cli::plugin` - Plugin system
- `ipfrs_cli::tui` - Terminal UI dashboard
- `ipfrs_cli::utils` - Utility functions

## [0.2.0] - Early Development

### Added
- Basic CLI structure with clap
- Core file operations (add, get, cat)
- Daemon management basics
- Block operations
- Initial configuration system

## [0.1.0] - Initial Release

### Added
- Project structure and initial implementation
- Basic IPFRS integration

[Unreleased]: https://github.com/ipfrs/ipfrs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ipfrs/ipfrs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ipfrs/ipfrs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ipfrs/ipfrs/releases/tag/v0.1.0
