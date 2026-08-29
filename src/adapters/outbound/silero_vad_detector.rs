use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::domain::model::AudioSamples;
use crate::domain::window::{SpeechSpan, VadConfig};
use crate::ports::vad::{VadError, VoiceActivityDetector};

/// Tamaño de ventana fijo exigido por Silero VAD (v5) a 16kHz: 512 muestras (32ms).
const WINDOW_SAMPLES: usize = 512;
const WINDOW_MS: i64 = (WINDOW_SAMPLES as i64 * 1000) / AudioSamples::SAMPLE_RATE as i64;
/// Estado recurrente del modelo: forma fija (2, 1, 128), se resetea por audio.
const STATE_LEN: usize = 2 * 128;

/// Adaptador de salida: implementa `VoiceActivityDetector` corriendo el
/// modelo ONNX de Silero VAD (v5) vía `ort`. `Session` no es `Sync` de forma
/// segura para llamadas concurrentes sobre el mismo estado recurrente, así
/// que se guarda detrás de un `Mutex` (esta app la usa desde un único hilo,
/// pero el trait exige `Send + Sync`).
pub struct SileroVadDetector {
    session: Mutex<Session>,
}

impl SileroVadDetector {
    pub fn load(model_path: &Path) -> Result<Self, VadError> {
        // Cada inferencia procesa un tensor de 512 floats (32ms de audio) y
        // las llamadas son inherentemente secuenciales (el estado recurrente
        // de un frame alimenta al siguiente): miles de llamadas así de
        // pequeñas. En CPU el paralelismo intra-op no tiene nada que
        // paralelizar y sólo añade overhead de sincronización (de ahí el solo
        // hilo); en GPU medimos algo peor todavía — cada llamada paga
        // lanzamiento de kernel + copia host↔device por 512 floats, ~3s
        // frente a ~180ms en CPU para un audio de 5 min. Por eso este
        // detector se queda en CPU aunque el resto del pipeline use CUDA.
        let session = Session::builder()
            .map_err(|e| VadError::ModelLoad(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| VadError::ModelLoad(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| VadError::ModelLoad(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Corre el modelo sobre una ventana de `WINDOW_SAMPLES` muestras
    /// (rellenada con ceros si es la última y viene incompleta), devolviendo
    /// la probabilidad de voz y actualizando `state` in place.
    fn run_frame(
        session: &mut Session,
        frame: &[f32],
        state: &mut Vec<f32>,
    ) -> Result<f32, VadError> {
        let mut input = frame.to_vec();
        input.resize(WINDOW_SAMPLES, 0.0);

        let input_tensor = Tensor::from_array(([1usize, WINDOW_SAMPLES], input))
            .map_err(|e| VadError::Inference(e.to_string()))?;
        let state_tensor = Tensor::from_array(([2usize, 1usize, 128usize], state.clone()))
            .map_err(|e| VadError::Inference(e.to_string()))?;
        let sr_tensor = Tensor::from_array(([1usize], vec![AudioSamples::SAMPLE_RATE as i64]))
            .map_err(|e| VadError::Inference(e.to_string()))?;

        let outputs = session
            .run(ort::inputs! {
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            })
            .map_err(|e| VadError::Inference(e.to_string()))?;

        let (_, prob) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Inference(e.to_string()))?;
        let probability = *prob.first().ok_or_else(|| {
            VadError::Inference("el modelo VAD no devolvió ninguna probabilidad".into())
        })?;

        let (_, new_state) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Inference(e.to_string()))?;
        state.clear();
        state.extend_from_slice(new_state);

        Ok(probability)
    }
}

impl VoiceActivityDetector for SileroVadDetector {
    fn detect_speech(
        &self,
        _model_path: &Path,
        audio: &AudioSamples,
        config: &VadConfig,
    ) -> Result<Vec<SpeechSpan>, VadError> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| VadError::Inference("el estado del detector VAD quedó envenenado".into()))?;

        let mut state = vec![0.0f32; STATE_LEN];
        let mut raw_spans: Vec<SpeechSpan> = Vec::new();

        for (i, frame) in audio.pcm_mono_16k.chunks(WINDOW_SAMPLES).enumerate() {
            let probability = Self::run_frame(&mut session, frame, &mut state)?;
            if probability < config.threshold {
                continue;
            }

            let start_ms = i as i64 * WINDOW_MS;
            let end_ms = (start_ms + WINDOW_MS).min(audio.duration_ms());

            match raw_spans.last_mut() {
                Some(last) if start_ms - last.end_ms <= 0 => last.end_ms = end_ms,
                _ => raw_spans.push(SpeechSpan { start_ms, end_ms }),
            }
        }

        Ok(merge_and_pad(&raw_spans, config, audio.duration_ms()))
    }
}

/// Fusiona spans separados por menos de `min_silence_ms` y aplica `pad_ms`
/// a cada borde, con clamp a `[0, duration_ms]`.
fn merge_and_pad(spans: &[SpeechSpan], config: &VadConfig, duration_ms: i64) -> Vec<SpeechSpan> {
    let mut merged: Vec<SpeechSpan> = Vec::new();
    for span in spans {
        match merged.last_mut() {
            Some(last) if span.start_ms - last.end_ms < config.min_silence_ms => {
                last.end_ms = span.end_ms;
            }
            _ => merged.push(*span),
        }
    }

    merged
        .into_iter()
        .map(|span| SpeechSpan {
            start_ms: (span.start_ms - config.pad_ms).clamp(0, duration_ms),
            end_ms: (span.end_ms + config.pad_ms).clamp(0, duration_ms),
        })
        .collect()
}
