# IPFRS Documentation

This directory contains the mdBook-based documentation for IPFRS.

## Prerequisites

Install mdBook:

```bash
cargo install mdbook
```

## Building the Documentation

Build the documentation:

```bash
cd book
mdbook build
```

The output will be in `book/book/` directory.

## Viewing the Documentation

Serve the documentation locally with live reload:

```bash
cd book
mdbook serve --open
```

This will start a web server at `http://localhost:3000` and open it in your browser.

## Testing

Test all code examples in the documentation:

```bash
cd book
mdbook test
```

## Structure

```
book/
├── book.toml              # mdBook configuration
├── src/                   # Documentation source files
│   ├── SUMMARY.md         # Table of contents
│   ├── introduction.md    # Introduction page
│   ├── getting-started/   # Getting started guides
│   ├── core/              # Core features documentation
│   ├── api/               # API reference
│   ├── bindings/          # Language bindings
│   ├── advanced/          # Advanced topics
│   ├── tutorials/         # Tutorials
│   ├── development/       # Development guides
│   └── reference/         # Reference material
└── theme/                 # Custom theme (optional)
```

## Contributing

To contribute to the documentation:

1. Edit or add markdown files in `src/`
2. Update `src/SUMMARY.md` if adding new pages
3. Test your changes with `mdbook serve`
4. Submit a pull request

## Deployment

### GitHub Pages

To deploy to GitHub Pages:

```bash
# Build the book
mdbook build

# The output is in book/book/
# Deploy the book/ directory to gh-pages branch
```

### Custom Server

Build and copy the output:

```bash
mdbook build
cp -r book/book/ /var/www/ipfrs-docs/
```

## Styling

The documentation uses the default Rust theme. To customize:

1. Create `theme/` directory
2. Generate default theme: `mdbook init --theme`
3. Customize CSS in `theme/css/`
4. Reference custom CSS in `book.toml`

## Search

Full-text search is enabled by default using mdBook's built-in search.

## Maintenance

### Updating Links

Check for broken links:

```bash
mdbook build
# Check the output for broken link warnings
```

### Updating Code Examples

Ensure code examples are tested and up-to-date:

```bash
mdbook test
```

## License

The documentation is licensed under the same license as IPFRS (Apache-2.0).
