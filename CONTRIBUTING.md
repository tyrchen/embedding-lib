# Contributing to embedding-lib

Thank you for your interest in contributing to embedding-lib! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/embedding-lib.git
   cd embedding-lib
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- **Rust** (latest stable version recommended)
- **Git** for version control

### Platform-specific Requirements

#### macOS (for Metal acceleration)
```bash
# Ensure Xcode command line tools are installed
xcode-select --install
```

#### Linux/Windows (for CUDA support)
```bash
# Install CUDA toolkit (version 11.8+ recommended)
# Follow NVIDIA's installation guide for your platform
```

### Building the Project

```bash
# Install dependencies and build
cargo build

# Build with optimizations
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_usage
cargo run --example qwen_example
```

## Development Guidelines

### Code Style

- Follow standard Rust formatting using `rustfmt`
- Use `clippy` for linting
- Write clear, self-documenting code
- Add comprehensive documentation for public APIs

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy -- -D warnings
```

### Documentation

- Document all public APIs with rustdoc comments
- Include examples in documentation where helpful
- Update README.md if adding new features
- Add inline comments for complex logic

### Testing

- Write unit tests for new functionality
- Include integration tests for major features
- Ensure all tests pass before submitting PR
- Test with different models when possible

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### Error Handling

- Use the existing `EmbeddingError` type for errors
- Provide clear, actionable error messages
- Handle edge cases gracefully
- Document error conditions in function docs

### Performance

- Profile performance-critical changes
- Consider memory usage implications
- Test with different batch sizes and concurrency levels
- Benchmark against existing implementations

## Submission Process

### Before Submitting

1. **Test thoroughly**:
   ```bash
   cargo test
   cargo clippy
   cargo fmt --check
   ```

2. **Update documentation** if necessary

3. **Add entry to CHANGELOG.md** for significant changes

4. **Ensure examples still work**:
   ```bash
   cargo run --example basic_usage
   cargo run --example qwen_example
   ```

### Pull Request Guidelines

1. **Create a clear title** describing the change
2. **Write a detailed description** including:
   - What changes were made
   - Why the changes were needed
   - Any breaking changes
   - Performance implications

3. **Reference related issues** using `Fixes #issue-number`

4. **Keep changes focused** - one feature/fix per PR

5. **Include tests** for new functionality

### Pull Request Template

```markdown
## Summary
Brief description of changes

## Changes Made
- List of specific changes
- Use bullet points

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Examples still work
- [ ] Manual testing performed

## Performance Impact
Describe any performance implications

## Breaking Changes
List any breaking changes (if applicable)

## Related Issues
Fixes #issue-number
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Rust version** (`rustc --version`)
- **Operating system** and version
- **Model being used**
- **Minimal reproduction example**
- **Error messages** (full stack trace)
- **Expected vs actual behavior**

### Feature Requests

For feature requests, please include:

- **Clear description** of the feature
- **Use case** and motivation
- **Proposed API** (if applicable)
- **Alternative solutions** considered

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and considerate
- Use clear and professional communication
- Focus on constructive feedback
- Help newcomers and answer questions patiently

### Unacceptable Behavior

- Harassment or discrimination
- Offensive or inappropriate language
- Personal attacks or trolling
- Spam or off-topic discussions

## Getting Help

- **Documentation**: Check the README and API docs first
- **Examples**: Review the examples directory for usage patterns
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas

## Recognition

Contributors will be recognized in the project's acknowledgments. Significant contributors may be invited to become maintainers.

## License

By contributing to embedding-lib, you agree that your contributions will be licensed under the MIT License, the same license that covers the project.

Thank you for contributing to embedding-lib! 🚀
