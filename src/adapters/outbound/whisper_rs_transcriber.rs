use std::path::Path;
use std::sync::Arc;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSegment,
};

use crate::domain::model::{TranscribeOptions, TranscriptSegment, WordTimestamp};
use crate::domain::window::AudioWindow;
use crate::ports::progress::{ProgressSink, ProgressUnit};
use crate::ports::transcriber::{Transcriber, TranscriberError};

/// Adaptador de salida: implementa el puerto `Transcriber` usando whisper-rs
/// (bindings de whisper.cpp). El `WhisperContext` se carga una vez y se
/// reutiliza para todas las ventanas de una misma transcripción; cada ventana
/// obtiene su propio `state` (barato) para no arrastrar estado entre ellas.
pub struct WhisperRsTranscriber;

impl WhisperRsTranscriber {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhisperRsTranscriber {
    fn default() -> Self {
        Self::new()
    }
}

/// Agrupa los tokens de un segmento en palabras, usando la misma heurística
/// que whisper.cpp/OpenAI Whisper: un token cuyo texto decodificado empieza
/// con espacio (convención del tokenizer BPE) arranca una palabra nueva; los
/// tokens especiales (SOT/EOT/timestamps/idioma, id >= `token_eot`) se
/// descartan. Sigue siendo tan preciso como el propio whisper.cpp: más fino
/// que el timestamp por segmento, pero no un forced aligner.
fn words_for_segment(
    ctx: &WhisperContext,
    segment: &WhisperSegment<'_>,
    offset_ms: i64,
) -> Vec<WordTimestamp> {
    let eot = ctx.token_eot();
    let mut words = Vec::new();
    let mut current: Option<(String, i64, i64)> = None;

    for i in 0..segment.n_tokens() {
        let Some(token) = segment.get_token(i) else {
            continue;
        };
        if token.token_id() >= eot {
            continue;
        }
        let Ok(text) = token.to_str_lossy() else {
            continue;
        };
        let data = token.token_data();
        if data.t0 < 0 || data.t1 < 0 {
            continue;
        }
        let start_ms = data.t0 * 10 + offset_ms;
        let end_ms = data.t1 * 10 + offset_ms;

        if current.is_none() || text.starts_with(' ') {
            if let Some((word_text, word_start, word_end)) = current.take() {
                words.push(WordTimestamp {
                    start_ms: word_start,
                    end_ms: word_end,
                    text: word_text.trim().to_string(),
                });
            }
            current = Some((text.into_owned(), start_ms, end_ms));
        } else if let Some((word_text, _, word_end)) = current.as_mut() {
            word_text.push_str(&text);
            *word_end = end_ms;
        }
    }

    if let Some((word_text, word_start, word_end)) = current.take() {
        words.push(WordTimestamp {
            start_ms: word_start,
            end_ms: word_end,
            text: word_text.trim().to_string(),
        });
    }

    words
}

impl Transcriber for WhisperRsTranscriber {
    fn transcribe_windows(
        &self,
        model_path: &Path,
        windows: &[AudioWindow],
        options: &TranscribeOptions,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<Vec<Vec<TranscriptSegment>>, TranscriberError> {
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| TranscriberError::ModelLoad("ruta de modelo no es UTF-8".into()))?;

        let ctx = WhisperContext::new_with_params(model_path_str, WhisperContextParameters::default())
            .map_err(|e| TranscriberError::ModelLoad(e.to_string()))?;

        progress.on_start(ProgressUnit::Percent, Some(100));

        let window_count = windows.len().max(1) as u64;
        let mut results = Vec::with_capacity(windows.len());

        for (i, window) in windows.iter().enumerate() {
            let mut state = ctx
                .create_state()
                .map_err(|e| TranscriberError::ModelLoad(e.to_string()))?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(options.threads as i32);
            params.set_translate(options.translate_to_english);
            params.set_language(options.language.as_deref());
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_token_timestamps(options.word_timestamps);

            let progress_cb = progress.clone();
            let window_index = i as u64;
            params.set_progress_callback_safe(move |pct: i32| {
                let pct = pct.clamp(0, 100) as u64;
                let aggregate = (window_index * 100 + pct) / window_count;
                progress_cb.on_progress(aggregate.min(100));
            });

            state
                .full(params, &window.pcm_mono_16k)
                .map_err(|e| TranscriberError::Inference(e.to_string()))?;

            let segments = state
                .as_iter()
                .map(|segment| {
                    let words = options
                        .word_timestamps
                        .then(|| words_for_segment(&ctx, &segment, window.offset_ms));
                    TranscriptSegment {
                        start_ms: segment.start_timestamp() * 10 + window.offset_ms,
                        end_ms: segment.end_timestamp() * 10 + window.offset_ms,
                        text: segment.to_string(),
                        words,
                    }
                })
                .collect();

            results.push(segments);
        }

        progress.on_finish();

        Ok(results)
    }
}
