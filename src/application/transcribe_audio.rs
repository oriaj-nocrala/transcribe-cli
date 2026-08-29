use std::path::Path;
use std::sync::Arc;

use crate::domain::model::{TranscribeOptions, Transcript};
use crate::ports::progress::ProgressSink;
use crate::ports::transcriber::{Transcriber, TranscriberError};

pub struct TranscribeAudioUseCase {
    transcriber: Arc<dyn Transcriber>,
}

impl TranscribeAudioUseCase {
    pub fn new(transcriber: Arc<dyn Transcriber>) -> Self {
        Self { transcriber }
    }

    pub fn execute(
        &self,
        input_audio: &Path,
        options: &TranscribeOptions,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<Transcript, TranscriberError> {
        self.transcriber.transcribe(input_audio, options, progress)
    }
}
