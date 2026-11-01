use std::{num::NonZeroU32, path::Path};

use anyhow::{Context, Result, ensure};
use llama_cpp_4::{
    context::{
        LlamaContext,
        params::{LlamaContextParams, LlamaPoolingType},
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        AddBos, LlamaModel, Special,
        params::{LlamaLoadMode, LlamaModelParams},
    },
};
use self_cell::self_cell;

// A non-causal embedding must fit in one physical batch; never silently truncate.
const CONTEXT_TOKENS: u32 = 512;

self_cell!(
    struct ModelContext {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
);

pub struct Engine {
    // Context and model must be dropped before the backend.
    cell: ModelContext,
    batch: LlamaBatch,
    dimensions: usize,
    _backend: LlamaBackend,
}

impl Engine {
    pub fn load(path: &Path, threads: u32, dimensions: usize) -> Result<Self> {
        ensure!(
            [128, 256, 512, 768].contains(&dimensions),
            "invalid embedding dimensions"
        );
        let mut backend = LlamaBackend::init()?;
        if std::env::var_os("GEG_DEBUG").is_none() {
            backend.void_logs();
        }
        let model = LlamaModel::load_from_file(
            &backend,
            path,
            &LlamaModelParams::default()
                .with_n_gpu_layers(0)
                .with_load_mode(LlamaLoadMode::Mmap),
        )
        .context("loading GGUF; use a text-only EmbeddingGemma 2 GGUF")?;
        ensure!(
            model.n_embd_out() == 768
                && model.meta_val_str("general.architecture", 128)? == "gemma-embedding2",
            "expected a 768-dimensional EmbeddingGemma model"
        );
        let threads = i32::try_from(threads)?;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS))
            .with_n_batch(CONTEXT_TOKENS)
            .with_n_ubatch(CONTEXT_TOKENS)
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        // The GGUF architecture determines attention masking; do not override it.
        let cell = ModelContext::try_new(model, |model| model.new_context(&backend, params))?;
        Ok(Self {
            cell,
            batch: LlamaBatch::new(CONTEXT_TOKENS as usize, 1),
            dimensions,
            _backend: backend,
        })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.cell.borrow_owner().str_to_token_with_special(
            text,
            AddBos::Always,
            Special::Plaintext,
        )?;
        ensure!(
            tokens.len() <= CONTEXT_TOKENS as usize,
            "input exceeds {CONTEXT_TOKENS} tokens; reduce --chunk-lines or shorten the query"
        );
        self.batch.clear();
        self.batch.add_sequence(&tokens, 0, false)?;
        self.cell.with_dependent_mut(|_, ctx| -> Result<Vec<f32>> {
            ctx.clear_kv_cache();
            ctx.decode(&mut self.batch)?;
            let embedding = ctx.embeddings_seq_ith(0)?;
            // SAFETY: llama-cpp-4 0.9 reports n_embd (512) in the Rust slice,
            // but llama_get_embeddings_seq allocates n_embd_out (768) floats for
            // this architecture. load() verifies gemma-embedding2 and width 768;
            // mean pooling is fixed above, dimensions <= 768, and the context
            // stays alive and unmodified until the data has been copied.
            let mut vector =
                unsafe { std::slice::from_raw_parts(embedding.as_ptr(), self.dimensions) }.to_vec();
            normalize(&mut vector)?;
            Ok(vector)
        })
    }
}

pub fn normalize(vector: &mut [f32]) -> Result<()> {
    ensure!(
        vector.iter().all(|v| v.is_finite()),
        "model returned non-finite embeddings"
    );
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "model returned an invalid embedding norm"
    );
    for v in vector {
        *v /= norm;
    }
    Ok(())
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| a * b)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_cosine() -> Result<()> {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v)?;
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&v, &[0.0, 0.0]), 0.0);
        Ok(())
    }

    #[test]
    fn invalid_embeddings_fail() {
        for mut v in [vec![0.0], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(normalize(&mut v).is_err());
        }
    }
}
