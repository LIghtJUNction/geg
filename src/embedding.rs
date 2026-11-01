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
const BATCH_TOKENS: u32 = 1024;
const BATCH_SEQUENCES: u32 = 8;

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
    pub device: &'static str,
}

impl Engine {
    pub fn load(path: &Path, threads: u32, dimensions: usize, device: &str) -> Result<Self> {
        ensure!(
            [128, 256, 512, 768].contains(&dimensions),
            "invalid embedding dimensions"
        );
        let mut backend = LlamaBackend::init()?;
        if std::env::var_os("GEG_DEBUG").is_none() {
            backend.void_logs();
        }
        let gpu = device != "cpu" && llama_cpp_4::supports_gpu_offload();
        ensure!(
            device != "gpu" || gpu,
            "--device gpu requested but no compatible GPU backend/device is available; check Vulkan/Metal drivers and the build features"
        );
        let selected_device = if gpu { "GPU" } else { "CPU" };
        let model = LlamaModel::load_from_file(
            &backend,
            path,
            &LlamaModelParams::default()
                .with_n_gpu_layers(if gpu { u32::MAX } else { 0 })
                .with_load_mode(LlamaLoadMode::Auto),
        )
        .context("loading GGUF; use a text-only EmbeddingGemma 2 GGUF")?;
        ensure!(
            model.n_embd_out() == 768
                && model.meta_val_str("general.architecture", 128)? == "gemma-embedding2",
            "expected a 768-dimensional EmbeddingGemma model"
        );
        let threads = i32::try_from(threads)?;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS * BATCH_SEQUENCES))
            .with_n_seq_max(BATCH_SEQUENCES)
            .with_n_batch(BATCH_TOKENS)
            .with_n_ubatch(BATCH_TOKENS)
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        // The GGUF architecture determines attention masking; do not override it.
        let cell = ModelContext::try_new(model, |model| model.new_context(&backend, params))?;
        Ok(Self {
            cell,
            batch: LlamaBatch::new(BATCH_TOKENS as usize, BATCH_SEQUENCES as i32),
            dimensions,
            _backend: backend,
            device: selected_device,
        })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_many(&[text])?
            .pop()
            .context("model returned no embedding")
    }

    /// Pack independent sequences into bounded physical batches (no cross-document attention).
    pub fn embed_many(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(texts.len());
        self.batch.clear();
        let mut sequences = 0;
        let mut token_count = 0;
        for text in texts {
            let tokens = self.cell.borrow_owner().str_to_token_with_special(
                text,
                AddBos::Always,
                Special::Plaintext,
            )?;
            ensure!(
                tokens.len() <= CONTEXT_TOKENS as usize,
                "input exceeds {CONTEXT_TOKENS} tokens; reduce --chunk-lines or shorten the query"
            );
            if sequences > 0
                && (sequences == BATCH_SEQUENCES as i32
                    || token_count + tokens.len() > BATCH_TOKENS as usize)
            {
                output.extend(self.encode_batch(sequences)?);
                self.batch.clear();
                sequences = 0;
                token_count = 0;
            }
            self.batch.add_sequence(&tokens, sequences, false)?;
            sequences += 1;
            token_count += tokens.len();
        }
        if sequences > 0 {
            output.extend(self.encode_batch(sequences)?);
        }
        Ok(output)
    }

    fn encode_batch(&mut self, sequences: i32) -> Result<Vec<Vec<f32>>> {
        self.cell
            .with_dependent_mut(|_, ctx| -> Result<Vec<Vec<f32>>> {
                ctx.clear_kv_cache();
                ctx.encode(&mut self.batch)?;
                (0..sequences)
                    .map(|sequence| {
                        let embedding = ctx.embeddings_seq_ith(sequence)?;
                        // SAFETY: llama-cpp-4 0.9 reports n_embd (512) in this slice,
                        // but the verified gemma-embedding2 model allocates n_embd_out
                        // (768) floats per pooled sequence. Copy <=768 floats before
                        // the context is modified or dropped; sequences have unique ids.
                        let mut vector = unsafe {
                            std::slice::from_raw_parts(embedding.as_ptr(), self.dimensions)
                        }
                        .to_vec();
                        normalize(&mut vector)?;
                        Ok(vector)
                    })
                    .collect()
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
