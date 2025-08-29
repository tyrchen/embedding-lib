# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance text embeddings library built on top of text-embeddings-inference from HuggingFace. It provides a Rust API for generating text embeddings using various transformer models like BAAI/bge, Qwen, and sentence-transformers models.

## Common Development Commands

### Building and Testing
```bash
# Build the project
make build
# or
cargo build

# Run tests (uses nextest)
make test
# or
cargo nextest run --all-features

# Format code
cargo fmt --all

# Lint with clippy
cargo clippy --all-targets --all-features --tests --benches -- -D warnings

# Check dependencies for security/licensing issues
cargo deny check

# Check for typos
typos
```

### Pre-commit Hooks
The project uses pre-commit hooks configured in `.pre-commit-config.yaml`. These run:
- `cargo fmt` - Code formatting
- `cargo deny check` - Dependency checking
- `typos` - Spell checking
- `cargo check` - Compilation checking
- `cargo clippy` - Linting
- `cargo nextest run` - Testing

### Release Process
```bash
# Create a release (tags, updates CHANGELOG.md, pushes)
make release
```

### Git Submodules
```bash
# Update the text-embeddings-inference submodule
make update-submodule
```

## Architecture Overview

### Core Components

1. **TextEmbeddings** (`src/lib.rs:189-463`)
   - Main client struct for generating embeddings
   - Handles model initialization, downloading, and inference
   - Key methods: `new()`, `embed()`, `embed_normalized()`, `health()`

2. **TextEmbeddingsOptions** (`src/lib.rs:74-187`)
   - Configuration struct with builder pattern
   - Controls model loading, batch sizes, concurrency, and pooling methods
   - Supports both HuggingFace Hub models and local model paths

3. **Model Backend Integration**
   - Uses `text-embeddings-backend` and `text-embeddings-core` from HuggingFace
   - Supports multiple backends: Candle (with Metal support on macOS)
   - Handles tokenization, pooling strategies (CLS, Mean, LastToken, Splade)

4. **Error Handling** (`src/lib.rs:42-72`)
   - Custom `EmbeddingError` enum for comprehensive error types
   - Wraps underlying text-embeddings-inference errors

### Model Loading Flow

1. **Model Resolution**: Checks if model_id is a local path or HuggingFace Hub ID
2. **Model Download**: Downloads artifacts from HuggingFace Hub if needed (uses hf-hub crate)
3. **Configuration Loading**: Loads `config.json`, Sentence Transformers configs
4. **Tokenizer Setup**: Initializes tokenizer with special handling for models like Qwen2
5. **Backend Creation**: Instantiates the inference backend with appropriate settings
6. **Warmup**: Pre-allocates resources and validates model health

### Key Dependencies

- **text-embeddings-backend/core**: Core inference engine from HuggingFace (git dependency)
- **tokenizers**: Fast tokenization library
- **hf-hub**: HuggingFace Hub API client for model downloads
- **candle**: ML framework backend (with Metal support on macOS)
- **tokio**: Async runtime

### Testing Strategy

- Unit tests in `src/lib.rs` for configuration builders
- Integration examples in `examples/` directory demonstrating real model usage
- CI/CD via GitHub Actions (`.github/workflows/build.yml`) running on every push/PR

## Working with the Codebase

When modifying the library:

1. **Adding new model support**: Check model architecture handling in `get_backend_model_type()` function
2. **Changing configuration**: Update `TextEmbeddingsOptions` and its builder methods
3. **Performance tuning**: Adjust default batch sizes, concurrency limits in options
4. **Error handling**: Add new error variants to `EmbeddingError` enum as needed

The library is designed to be a simpler, more focused alternative to running the full text-embeddings-inference server, providing direct Rust API access for embedding generation.
