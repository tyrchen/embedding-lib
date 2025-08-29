//! # Text Embeddings Library
//!
//! A high-performance text embeddings library built on top of text-embeddings-inference.
//!
//! ## Example
//!
//! ```rust,no_run
//! use embedding_lib::{TextEmbeddings, TextEmbeddingsOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let options = TextEmbeddingsOptions::new("BAAI/bge-large-en-v1.5".to_string());
//!     let embedder = TextEmbeddings::new(options).await?;
//!
//!     let texts = ["Hello world", "How are you?"];
//!     let embeddings = embedder.embed(&texts).await?;
//!
//!     println!("Generated {} embeddings", embeddings.len());
//!     Ok(())
//! }
//! ```

use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use text_embeddings_backend::{DType, Pool};
use text_embeddings_core::TextEmbeddingsError;
use text_embeddings_core::download::{ST_CONFIG_NAMES, download_artifacts};
use text_embeddings_core::infer::Infer;
use text_embeddings_core::queue::Queue;
use text_embeddings_core::tokenization::{EncodingInput, Tokenization};
use thiserror::Error;
use tokenizers::processors::sequence::Sequence;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{PostProcessorWrapper, Tokenizer};
use tracing::instrument;

/// Errors that can occur when using the text embeddings library
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// Configuration validation error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Model loading or initialization error
    #[error("Model error: {0}")]
    Model(String),

    /// Text embeddings inference error
    #[error("Inference error: {0}")]
    Inference(#[from] TextEmbeddingsError),

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),

    /// I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// General error
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Configuration options for text embeddings
#[derive(Debug)]
pub struct TextEmbeddingsOptions {
    /// Model ID or path (can be HuggingFace model ID or local directory)
    pub model_id: String,
    /// Model revision (git branch/tag/commit)
    pub revision: Option<String>,
    /// Data type for model weights
    pub dtype: Option<DType>,
    /// Pooling method for embeddings
    pub pooling: Option<Pool>,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Maximum batch tokens
    pub max_batch_tokens: usize,
    /// Maximum batch requests
    pub max_batch_requests: Option<usize>,
    /// Number of tokenization workers
    pub tokenization_workers: Option<usize>,
    /// HuggingFace Hub token for private models
    pub hf_token: Option<String>,
    /// HuggingFace Hub cache directory
    pub huggingface_hub_cache: Option<String>,
    /// Dense module path (for some embedding models)
    pub dense_path: Option<String>,
    /// Default prompt for encoding
    pub default_prompt: Option<String>,
    /// Default prompt name for encoding
    pub default_prompt_name: Option<String>,
    /// UDS path
    pub uds_path: Option<String>,
    /// OTLP endpoint
    pub otlp_endpoint: Option<String>,
    /// OTLP service name
    pub otlp_service_name: String,
    /// Max client batch size
    pub max_client_batch_size: usize,
    /// Auto truncate
    pub auto_truncate: bool,
}

impl TextEmbeddingsOptions {
    /// Create new options with the given model ID
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            revision: None,
            dtype: None,
            pooling: None,
            max_concurrent_requests: 512,
            max_batch_tokens: 16384,
            max_batch_requests: None,
            tokenization_workers: None,
            hf_token: None,
            huggingface_hub_cache: None,
            dense_path: Some("2_Dense".to_string()),
            default_prompt: None,
            default_prompt_name: None,
            uds_path: None,
            otlp_endpoint: None,
            otlp_service_name: "embedding-lib".to_string(),
            max_client_batch_size: 32,
            auto_truncate: false,
        }
    }

    /// Set model revision
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = Some(revision);
        self
    }

    /// Set data type
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Set pooling method
    pub fn with_pooling(mut self, pooling: Pool) -> Self {
        self.pooling = Some(pooling);
        self
    }

    /// Set maximum concurrent requests
    pub fn with_max_concurrent_requests(mut self, max_concurrent_requests: usize) -> Self {
        self.max_concurrent_requests = max_concurrent_requests;
        self
    }

    /// Set maximum batch tokens
    pub fn with_max_batch_tokens(mut self, max_batch_tokens: usize) -> Self {
        self.max_batch_tokens = max_batch_tokens;
        self
    }

    /// Set HuggingFace token
    pub fn with_hf_token(mut self, token: String) -> Self {
        self.hf_token = Some(token);
        self
    }

    /// Set default prompt
    pub fn with_default_prompt(mut self, prompt: String) -> Self {
        self.default_prompt = Some(prompt);
        self
    }

    /// Set maximum batch requests
    pub fn with_max_batch_requests(mut self, max_batch_requests: Option<usize>) -> Self {
        self.max_batch_requests = max_batch_requests;
        self
    }
}

/// Main text embeddings client
#[derive(Debug, Clone)]
pub struct TextEmbeddings {
    /// Core inference engine
    infer: Infer,
}

impl TextEmbeddings {
    /// Create a new TextEmbeddings instance
    #[instrument(skip_all)]
    pub async fn new(options: TextEmbeddingsOptions) -> Result<Self, EmbeddingError> {
        let model_id_path = Path::new(&options.model_id);
        let (model_root, api_repo) = if model_id_path.exists() && model_id_path.is_dir() {
            // Using a local model
            (model_id_path.to_path_buf(), None)
        } else {
            let mut builder = ApiBuilder::from_env()
                .with_progress(false)
                .with_token(options.hf_token);

            if let Some(cache_dir) = options.huggingface_hub_cache {
                builder = builder.with_cache_dir(cache_dir.into());
            }

            if let Ok(origin) = std::env::var("HF_HUB_USER_AGENT_ORIGIN") {
                builder = builder.with_user_agent("origin", origin.as_str());
            }

            let api = builder.build().unwrap();
            let api_repo = api.repo(Repo::with_revision(
                options.model_id.clone(),
                RepoType::Model,
                options.revision.clone().unwrap_or("main".to_string()),
            ));

            // Download model from the Hub
            (
                download_artifacts(&api_repo, options.pooling.is_none())
                    .await
                    .context("Could not download model artifacts")?,
                Some(api_repo),
            )
        };

        // Load config
        let config_path = model_root.join("config.json");
        let config = fs::read_to_string(config_path).context("`config.json` not found")?;
        let config: ModelConfig =
            serde_json::from_str(&config).context("Failed to parse `config.json`")?;

        // Set model type from config
        let backend_model_type = get_backend_model_type(&config, &model_root, options.pooling)?;

        // Load tokenizer
        let tokenizer_path = model_root.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect(
            "tokenizer.json not found. text-embeddings-inference only supports fast tokenizers",
        );
        tokenizer.with_padding(None);
        // Qwen2 updates the post processor manually instead of into the tokenizer.json...
        // https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/blob/main/tokenization_qwen.py#L246
        if config.model_type == "qwen2" {
            let template = TemplateProcessing::builder()
                .try_single("$A:0 <|endoftext|>:0")
                .unwrap()
                .try_pair("$A:0 <|endoftext|>:0 $B:1 <|endoftext|>:1")
                .unwrap()
                .special_tokens(vec![("<|endoftext|>", 151643)])
                .build()
                .unwrap();
            match tokenizer.get_post_processor() {
                None => tokenizer.with_post_processor(Some(template)),
                Some(post_processor) => {
                    let post_processor = Sequence::new(vec![
                        post_processor.clone(),
                        PostProcessorWrapper::Template(template),
                    ]);
                    tokenizer.with_post_processor(Some(post_processor))
                }
            };
        }

        // Position IDs offset. Used for Roberta and camembert.
        let position_offset = if &config.model_type == "xlm-roberta"
            || &config.model_type == "camembert"
            || &config.model_type == "roberta"
        {
            config.pad_token_id + 1
        } else {
            0
        };

        // Try to load ST Config
        let mut st_config: Option<STConfig> = None;
        for name in ST_CONFIG_NAMES {
            let config_path = model_root.join(name);
            if let Ok(config) = fs::read_to_string(config_path) {
                st_config = Some(
                    serde_json::from_str(&config).context(format!("Failed to parse `{}`", name))?,
                );
                break;
            }
        }
        let max_input_length = match st_config {
            Some(config) => config.max_seq_length,
            None => {
                tracing::warn!("Could not find a Sentence Transformers config");
                config.max_position_embeddings - position_offset
            }
        };
        tracing::info!("Maximum number of tokens per request: {max_input_length}");

        let tokenization_workers = num_cpus::get();

        // Try to load new ST Config
        let mut new_st_config: Option<NewSTConfig> = None;
        let config_path = model_root.join("config_sentence_transformers.json");
        if let Ok(config) = fs::read_to_string(config_path) {
            new_st_config = Some(
                serde_json::from_str(&config)
                    .context("Failed to parse `config_sentence_transformers.json`")?,
            );
        }
        let prompts = new_st_config.and_then(|c| c.prompts);
        let default_prompt = if let Some(default_prompt_name) = options.default_prompt_name {
            match &prompts {
                None => {
                    return Err(EmbeddingError::Config(format!(
                        "`default-prompt-name` is set to `{default_prompt_name}` but no prompts were found in the Sentence Transformers configuration"
                    )));
                }
                Some(prompts) if !prompts.contains_key(&default_prompt_name) => {
                    return Err(EmbeddingError::Config(format!(
                        "`default-prompt-name` is set to `{default_prompt_name}` but it was not found in the Sentence Transformers prompts. Available prompts: {:?}",
                        prompts.keys()
                    )));
                }
                Some(prompts) => prompts.get(&default_prompt_name).cloned(),
            }
        } else {
            options.default_prompt.clone()
        };

        // Tokenization logic
        let tokenization = Tokenization::new(
            tokenization_workers,
            tokenizer,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
        );

        // Get dtype
        let dtype = options.dtype.unwrap_or_default();

        // Create backend
        tracing::info!("Starting model backend");
        let backend = text_embeddings_backend::Backend::new(
            model_root,
            api_repo,
            dtype,
            backend_model_type,
            options.dense_path.clone(),
            options
                .uds_path
                .unwrap_or("/tmp/text-embeddings-inference-server".to_string()),
            options.otlp_endpoint.clone(),
            options.otlp_service_name.clone(),
        )
        .await
        .context("Could not create backend")?;
        backend
            .health()
            .await
            .context("Model backend is not healthy")?;

        tracing::info!("Warming up model");
        backend
            .warmup(
                max_input_length,
                options.max_batch_tokens,
                options.max_batch_requests,
            )
            .await
            .context("Model backend is not healthy")?;

        let max_batch_requests = backend
            .max_batch_size
            .inspect(|&s| {
                tracing::warn!("Backend does not support a batch size > {s}");
                tracing::warn!("forcing `max_batch_requests={s}`");
            })
            .or(options.max_batch_requests);

        // Queue logic
        let queue = Queue::new(
            backend.padded_model,
            options.max_batch_tokens,
            max_batch_requests,
            options.max_concurrent_requests,
        );

        // Create infer task
        let infer = Infer::new(
            tokenization,
            queue,
            options.max_concurrent_requests,
            backend,
        );

        Ok(Self { infer })
    }

    /// Generate embeddings for the given texts
    #[instrument(skip_all)]
    pub async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            // Acquire a permit to limit concurrent requests
            let permit = self.infer.acquire_permit().await;

            // Generate embedding for this text
            let response = self
                .infer
                .embed_pooled(
                    EncodingInput::Single(text.to_string()),
                    true, // truncate
                    tokenizers::TruncationDirection::Right,
                    None,  // prompt_name
                    false, // normalize
                    None,  // dimensions
                    permit,
                )
                .await?;

            all_embeddings.push(response.results);
        }

        Ok(all_embeddings)
    }

    /// Generate normalized embeddings for the given texts
    #[instrument(skip_all)]
    pub async fn embed_normalized(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let permit = self.infer.acquire_permit().await;

            let response = self
                .infer
                .embed_pooled(
                    EncodingInput::Single(text.to_string()),
                    true, // truncate
                    tokenizers::TruncationDirection::Right,
                    None, // prompt_name
                    true, // normalize
                    None, // dimensions
                    permit,
                )
                .await?;

            all_embeddings.push(response.results);
        }

        Ok(all_embeddings)
    }

    /// Check if the backend is healthy
    pub async fn health(&self) -> bool {
        self.infer.health().await
    }
}

// Helper function to determine backend model type
fn get_backend_model_type(
    config: &ModelConfig,
    model_root: &Path,
    pooling: Option<Pool>,
) -> Result<text_embeddings_backend::ModelType, anyhow::Error> {
    // Check for classifier models
    for arch in &config.architectures {
        if arch.ends_with("Classification") {
            if pooling.is_some() {
                tracing::warn!(
                    "`pooling` arg is set but model is a classifier. Ignoring `pooling` arg."
                );
            }
            return Ok(text_embeddings_backend::ModelType::Classifier);
        }
    }

    // Handle SPLADE pooling
    if matches!(pooling, Some(Pool::Splade)) {
        for arch in &config.architectures {
            if arch.ends_with("MaskedLM") {
                return Ok(text_embeddings_backend::ModelType::Embedding(Pool::Splade));
            }
        }
        return Err(anyhow::anyhow!(
            "Splade pooling is not supported: model is not a ForMaskedLM model"
        ));
    }

    // Determine pooling method for embedding models
    let pool = match pooling {
        Some(pool) => pool,
        None => {
            // Try to load pooling config
            let config_path = model_root.join("1_Pooling/config.json");
            match std::fs::read_to_string(config_path) {
                Ok(config_content) => {
                    let pool_config: PoolConfig =
                        serde_json::from_str(&config_content).map_err(|e| {
                            anyhow::anyhow!("Failed to parse `1_Pooling/config.json`: {}", e)
                        })?;
                    Pool::try_from(pool_config)?
                }
                Err(_) => {
                    if !config.model_type.to_lowercase().contains("bert") {
                        return Err(anyhow::anyhow!(
                            "The `pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model."
                        ));
                    }
                    tracing::warn!(
                        "The `pooling` arg is not set and we could not find a pooling configuration but the model is a BERT variant. Defaulting to `CLS` pooling."
                    );
                    Pool::Cls
                }
            }
        }
    };

    Ok(text_embeddings_backend::ModelType::Embedding(pool))
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub pad_token_id: usize,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
}

#[derive(Debug, Deserialize)]
struct STConfig {
    pub max_seq_length: usize,
}

#[derive(Debug, Deserialize)]
struct NewSTConfig {
    pub prompts: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PoolConfig {
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

impl TryFrom<PoolConfig> for Pool {
    type Error = anyhow::Error;

    fn try_from(config: PoolConfig) -> Result<Self, Self::Error> {
        if config.pooling_mode_cls_token {
            return Ok(Pool::Cls);
        }
        if config.pooling_mode_mean_tokens {
            return Ok(Pool::Mean);
        }
        if config.pooling_mode_lasttoken {
            return Ok(Pool::LastToken);
        }
        Err(anyhow::anyhow!(
            "Pooling config {:?} is not supported",
            config
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_builder() {
        let options = TextEmbeddingsOptions::new("test-model".to_string())
            .with_revision("main".to_string())
            .with_dtype(DType::Float16)
            .with_max_concurrent_requests(256);

        assert_eq!(options.model_id, "test-model");
        assert_eq!(options.revision, Some("main".to_string()));
        assert_eq!(options.dtype, Some(DType::Float16));
        assert_eq!(options.max_concurrent_requests, 256);
    }

    #[tokio::test]
    async fn test_text_embeddings_creation_fails_for_invalid_model() {
        let options = TextEmbeddingsOptions::new("invalid-model-path".to_string());
        let result = TextEmbeddings::new(options).await;
        assert!(result.is_err());
    }
}
