//! Basic usage example for the embedding library
//!
//! This example demonstrates how to use the TextEmbeddings library
//! to generate embeddings for text inputs.

use embedding_lib::{EmbeddingError, TextEmbeddings, TextEmbeddingsOptions};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), EmbeddingError> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "embedding_lib=info,text_embeddings_backend=warn,text_embeddings_core=warn".into()
            }),
        )
        .init();

    println!("Text Embeddings Library - Basic Usage Example");
    println!("==============================================\n");

    // Try to use different models based on availability
    // First check for cached models, then try to download if network is available
    let model_candidates = vec![
        // Check cached models first (these are more likely to work offline)
        ("Qwen/Qwen3-Embedding-0.6B", true), // We know this is often cached
        ("BAAI/bge-small-en-v1.5", true),    // Check if cached
        ("sentence-transformers/all-MiniLM-L6-v2", true), // Check if cached
        // Then try downloading if nothing is cached
        ("BAAI/bge-small-en-v1.5", false), // Popular smaller model
        ("sentence-transformers/all-MiniLM-L6-v2", false), // Very small model
    ];

    let mut embedder = None;

    for (model_id, check_cache_first) in model_candidates {
        println!("Attempting to load model: {}", model_id);

        if check_cache_first {
            // Check if model is actually cached and try to use the local path directly
            let cache_base = format!(
                "{}/.cache/huggingface/hub/models--{}",
                std::env::var("HOME").unwrap_or_default(),
                model_id.replace('/', "--")
            );

            let cache_path = Path::new(&cache_base);
            if !cache_path.exists() {
                println!("  Model not found in cache, skipping...");
                continue;
            }

            // Find the snapshot directory
            let snapshots_dir = cache_path.join("snapshots");
            if snapshots_dir.exists() {
                // Get the latest snapshot
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir)
                    && let Some(snapshot) = entries.filter_map(Result::ok).next()
                {
                    let snapshot_path = snapshot.path();
                    println!("  Found cached model at: {:?}", snapshot_path);

                    // Try to load from the local path directly
                    let options =
                        TextEmbeddingsOptions::new(snapshot_path.to_string_lossy().to_string())
                            .with_max_concurrent_requests(4)
                            .with_max_batch_tokens(128)
                            .with_max_batch_requests(Some(1));

                    match TextEmbeddings::new(options).await {
                        Ok(e) => {
                            println!("  ✓ Loaded from cache successfully!\n");
                            embedder = Some((model_id, e));
                            break;
                        }
                        Err(e) => {
                            eprintln!("  ✗ Failed to load from cache: {}", e);
                        }
                    }
                }
            }
        }

        let options = TextEmbeddingsOptions::new(model_id.to_string())
            .with_max_concurrent_requests(4) // Reduce concurrent requests to avoid hanging
            .with_max_batch_tokens(128) // Smaller batch size for stability
            .with_max_batch_requests(Some(1)); // Limit batch requests

        println!("  Configuration:");
        println!(
            "    Max concurrent requests: {}",
            options.max_concurrent_requests
        );
        println!("    Max batch tokens: {}", options.max_batch_tokens);

        println!("\n  Initializing model...");
        match TextEmbeddings::new(options).await {
            Ok(e) => {
                println!("  ✓ Model loaded successfully!\n");
                embedder = Some((model_id, e));
                break;
            }
            Err(e) => {
                eprintln!("  ✗ Failed to load model: {}\n", e);
                if !check_cache_first {
                    eprintln!(
                        "  Note: This might be due to network issues or the model requiring authentication."
                    );
                    eprintln!(
                        "  You can try setting the HF_TOKEN environment variable if the model is gated.\n"
                    );
                }
            }
        }
    }

    let (model_name, embedder) = embedder
        .ok_or_else(|| EmbeddingError::Model("Could not load any model. Please check your internet connection or ensure you have cached models.".to_string()))?;

    println!("Successfully loaded model: {}", model_name);
    println!("Health check: {}", embedder.health().await);

    // Test embeddings with simple texts
    let texts = [
        "Hello world",
        "How are you today?",
        "This is a test of the embedding system",
        "Machine learning is fascinating",
    ];

    println!("\nGenerating embeddings for {} text samples:", texts.len());
    for text in &texts {
        println!("  - \"{}\"", text);
    }

    let embeddings = embedder.embed(&texts).await?;

    println!(
        "\n✓ Generated {} embeddings successfully!",
        embeddings.len()
    );

    println!("\nEmbedding Details:");
    println!("==================");
    for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
        println!("\n{}. Text: \"{}\"", i + 1, text);
        println!("   Dimensions: {}", embedding.len());

        // Calculate basic statistics
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;

        println!("   Magnitude: {:.4}", magnitude);
        println!("   Mean: {:.4}", mean);

        // Show first 5 values as a sample
        if embedding.len() >= 5 {
            println!(
                "   First 5 values: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]
            );
        }
    }

    // Calculate similarity between first two texts
    if embeddings.len() >= 2 {
        let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
        println!(
            "\nCosine similarity between \"{}\" and \"{}\":",
            texts[0], texts[1]
        );
        println!("  {:.4}", similarity);
    }

    println!("\n✓ Example completed successfully!");

    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}
