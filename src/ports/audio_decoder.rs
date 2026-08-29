use std::path::Path;

use crate::domain::model::AudioSamples;

#[derive(Debug, thiserror::Error)]
pub enum AudioDecoderError {
    #[error("no se pudo decodificar el audio de entrada: {0}")]
    Decode(String),
}

/// Puerto de salida: convierte cualquier archivo de audio/vídeo soportado en
/// PCM mono 16kHz, el formato que exige whisper.cpp.
pub trait AudioDecoder: Send + Sync {
    fn decode_to_pcm16k_mono(&self, input_path: &Path) -> Result<AudioSamples, AudioDecoderError>;
}
