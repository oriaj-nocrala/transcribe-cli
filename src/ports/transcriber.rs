use std::path::Path;
use std::sync::Arc;

use crate::domain::model::{TranscribeOptions, TranscriptSegment};
use crate::domain::window::AudioWindow;
use crate::ports::progress::ProgressSink;

#[derive(Debug, thiserror::Error)]
pub enum TranscriberError {
    #[error("no se pudo cargar el modelo whisper: {0}")]
    ModelLoad(String),
    #[error("fallo durante la inferencia: {0}")]
    Inference(String),
}

/// Puerto de salida: motor de inferencia que convierte audio en texto.
/// La aplicación programa contra este trait, no contra whisper-rs directamente.
pub trait Transcriber: Send + Sync {
    /// `windows[i]` produce `resultado[i]`, en el mismo orden (1:1). Los
    /// `TranscriptSegment` devueltos llevan timestamps ABSOLUTOS: el
    /// adaptador debe sumar `window.offset_ms` antes de devolverlos, para que
    /// quien reciba el resultado no tenga que conocer el particionado en
    /// ventanas que hizo posible saltarse los silencios.
    fn transcribe_windows(
        &self,
        model_path: &Path,
        windows: &[AudioWindow],
        options: &TranscribeOptions,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<Vec<Vec<TranscriptSegment>>, TranscriberError>;
}
