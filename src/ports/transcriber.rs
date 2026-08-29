use std::path::Path;
use std::sync::Arc;

use crate::domain::model::{TranscribeOptions, Transcript};
use crate::ports::progress::ProgressSink;

#[derive(Debug, thiserror::Error)]
pub enum TranscriberError {
    #[error("fallo durante la transcripción: {0}")]
    Inference(String),
}

/// Puerto de salida: motor que convierte un archivo de audio/vídeo en texto.
/// Decodificación, detección de voz y resolución del modelo quedan a cargo
/// del adaptador concreto — la aplicación sólo entrega la ruta del archivo
/// de entrada tal cual.
pub trait Transcriber: Send + Sync {
    fn transcribe(
        &self,
        input_audio: &Path,
        options: &TranscribeOptions,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<Transcript, TranscriberError>;
}
