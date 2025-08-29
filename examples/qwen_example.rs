//! Example using Qwen embedding models
//!
//! This example demonstrates how to use Qwen models for text embeddings,
//! with proper configuration to avoid hanging issues.

use embedding_lib::{EmbeddingError, TextEmbeddings, TextEmbeddingsOptions};
use text_embeddings_backend::Pool;

#[tokio::main]
async fn main() -> Result<(), EmbeddingError> {
    // Initialize logging with debug level
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "embedding_lib=debug,text_embeddings_backend=info,text_embeddings_core=info".into()
            }),
        )
        .init();

    println!("Qwen Model Text Embeddings Example");
    println!("====================================\n");

    // Use Qwen embedding model optimized for text embeddings
    // Note: You can also try other Qwen models like:
    // - "Qwen/Qwen2-0.5B" (smaller base model)
    // - "Qwen/Qwen2-1.5B" (larger, better quality)
    // - "Alibaba-NLP/gte-Qwen2-1.5B-instruct" (optimized for embeddings)
    let model_id = "Qwen/Qwen3-Embedding-0.6B".to_string();

    println!("Initializing Qwen model: {}", model_id);

    // Configure options optimized for Qwen models
    let options = TextEmbeddingsOptions::new(model_id)
        .with_max_concurrent_requests(4) // Keep low to avoid memory issues
        .with_max_batch_tokens(256) // Smaller batch for stability
        .with_max_batch_requests(Some(2)) // Limit batch size
        .with_pooling(Pool::Mean); // Use mean pooling for better embeddings

    println!("\nConfiguration:");
    println!(
        "  Max concurrent requests: {}",
        options.max_concurrent_requests
    );
    println!("  Max batch tokens: {}", options.max_batch_tokens);
    println!("  Max batch requests: {:?}", options.max_batch_requests);
    println!("  Pooling method: {:?}", options.pooling);

    println!("\nDownloading and initializing model (this may take a while on first run)...");

    // Create embedder instance
    let embedder = match TextEmbeddings::new(options).await {
        Ok(e) => {
            println!("✓ Model initialized successfully!");
            e
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize model: {}", e);
            return Err(e);
        }
    };

    // Check health
    println!("\nChecking model health...");
    if embedder.health().await {
        println!("✓ Model is healthy and ready!");
    } else {
        println!("⚠ Model health check failed, but continuing...");
    }

    // Test with various text samples
    let texts = [
        "What is artificial intelligence?",
        "机器学习是人工智能的一个子领域。", // Chinese text
        "The quick brown fox jumps over the lazy dog.",
        "Qwen is a series of large language models developed by Alibaba Cloud.",
    ];

    println!("\nGenerating embeddings for {} text samples:", texts.len());
    for text in &texts {
        println!("  - \"{}\"", text);
    }

    // Generate embeddings
    let embeddings = embedder.embed(&texts).await?;

    println!("\n✓ Successfully generated {} embeddings", embeddings.len());

    // Display results
    println!("\nEmbedding Results:");
    println!("==================");
    for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
        println!("\n{}. Text: \"{}\"", i + 1, text);
        println!("   Embedding dimensions: {}", embedding.len());

        // Calculate embedding statistics
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let min = embedding.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = embedding.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        println!("   Statistics:");
        println!("     - Magnitude: {:.4}", magnitude);
        println!("     - Mean: {:.4}", mean);
        println!("     - Min: {:.4}", min);
        println!("     - Max: {:.4}", max);

        // Show first 5 values as a sample
        if embedding.len() >= 5 {
            println!(
                "   First 5 values: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]
            );
        }
    }

    // Calculate similarity between embeddings
    println!("\nCosine Similarity Matrix:");
    println!("==========================");
    for i in 0..texts.len() {
        for j in 0..texts.len() {
            let similarity = cosine_similarity(&embeddings[i], &embeddings[j]);
            print!("{:.3}  ", similarity);
        }
        println!();
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
