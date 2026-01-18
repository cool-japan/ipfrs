# Contributing to IPFRS

Thank you for your interest in contributing to IPFRS! This document provides guidelines and information to make the contribution process smooth and effective.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Set up your development environment** following the instructions in the README
3. **Create a new branch** for your feature or bugfix
4. **Make your changes**, following our code style and guidelines
5. **Test your changes** thoroughly
6. **Submit a pull request** with a clear description of the changes

## Development Workflow

1. **Check the TODO.md** file for current priorities
2. **Discuss major changes** in an issue before implementing
3. **Follow the test-driven development** approach where possible
4. **Document your code** as you write it
5. **Keep pull requests focused** on a single feature or bugfix

## Coding Standards

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use meaningful variable and function names
- Write comprehensive documentation comments
- Include examples in doc comments for public APIs
- Aim for idiomatic Rust code
- Use `cargo clippy` and fix all warnings
- Format code with `cargo fmt`

## Testing Requirements

- Write unit tests for all new functionality
- Include doc tests for all public APIs
- For numerical algorithms, compare results against reference implementations
- Benchmark performance-critical code

## API Design Principles

- Maintain compatibility with IPFS concepts and patterns
- Use Rust idioms (like traits, generics, and the type system) appropriately
- Follow consistent naming conventions across modules
- Design for both ease of use and performance
- Consider both novice and expert users

## Pull Request Process

1. Ensure all tests pass locally before submitting
2. Update documentation to reflect any changes
3. Add an entry to the CHANGELOG.md file if applicable
4. Pull requests should be linked to an issue where possible
5. Wait for review and address any feedback

## Communication

- **Issues**: Use for bug reports, feature requests, and substantial discussions
- **Pull Requests**: Use for code contributions
- **Discussions**: Use for general questions and ideas

## License

By contributing to IPFRS, you agree that your contributions will be licensed under the project's Apache License 2.0.
