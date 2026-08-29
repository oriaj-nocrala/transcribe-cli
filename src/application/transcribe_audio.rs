use std::path::Path;
use std::sync::Arc;

use crate::domain::model::{ModelId, TranscribeOptions, Transcript};
use crate::domain::window::{self, VadConfig};
use crate::ports::audio_decoder::{AudioDecoder, AudioDecoderError};
use crate::ports::model_repository::{ModelRepository, ModelRepositoryError};
use crate::ports::progress::ProgressSink;
use crate::ports::transcriber::{Transcriber, TranscriberError};
use crate::ports::vad::{VadError, VoiceActivityDetector};

#[derive(Debug, thiserror::Error)]
pub enum TranscribeAudioError {
    #[error(transparent)]
    Repository(#[from] ModelRepositoryError),
    #[error(transparent)]
    Decoder(#[from] AudioDecoderError),
    #[error(transparent)]
    Transcriber(#[from] TranscriberError),
    #[error(transparent)]
    Vad(#[from] VadError),
}

pub struct TranscribeAudioUseCase {
    repository: Arc<dyn ModelRepository>,
    decoder: Arc<dyn AudioDecoder>,
    transcriber: Arc<dyn Transcriber>,
    vad: Option<Arc<dyn VoiceActivityDetector>>,
}

impl TranscribeAudioUseCase {
    pub fn new(
        repository: Arc<dyn ModelRepository>,
        decoder: Arc<dyn AudioDecoder>,
        transcriber: Arc<dyn Transcriber>,
        vad: Option<Arc<dyn VoiceActivityDetector>>,
    ) -> Self {
        Self {
            repository,
            decoder,
            transcriber,
            vad,
        }
    }

    pub async fn execute(
        &self,
        model: ModelId,
        input_audio: &Path,
        vad_model_path: Option<&Path>,
        vad_config: &VadConfig,
        options: &TranscribeOptions,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<Transcript, TranscribeAudioError> {
        let model_path = self
            .repository
            .ensure_downloaded(model, progress.clone())
            .await?;
        let audio = self.decoder.decode_to_pcm16k_mono(input_audio)?;

        let windows = match (&self.vad, vad_model_path) {
            (Some(vad), Some(vad_path)) => {
                let spans = vad.detect_speech(vad_path, &audio, vad_config)?;
                window::windows_from_speech_spans(&spans, &audio, vad_config)
            }
            _ => vec![audio.whole_window()],
        };
        if windows.is_empty() {
            return Ok(Transcript::default());
        }

        let per_window = self
            .transcriber
            .transcribe_windows(&model_path, &windows, options, progress)?;
        let segments = per_window.into_iter().flatten().collect();
        Ok(Transcript { segments })
    }
}
