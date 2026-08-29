use std::path::Path;

use crate::domain::model::AudioSamples;
use crate::domain::window::{SpeechSpan, VadConfig};

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("no se pudo cargar el modelo de detección de voz: {0}")]
    ModelLoad(String),
    #[error("fallo durante la detección de voz: {0}")]
    Inference(String),
}

/// Puerto de salida: detecta los tramos de habla de un audio, en timestamps
/// absolutos respecto a ese audio. La aplicación usa estos tramos para saber
/// qué partes vale la pena mandarle al motor de transcripción, ahorrándose
/// el resto (silencio).
pub trait VoiceActivityDetector: Send + Sync {
    fn detect_speech(
        &self,
        model_path: &Path,
        audio: &AudioSamples,
        config: &VadConfig,
    ) -> Result<Vec<SpeechSpan>, VadError>;
}
