use std::path::Path;
use std::process::Command;

use crate::domain::model::AudioSamples;
use crate::ports::audio_decoder::{AudioDecoder, AudioDecoderError};

/// Adaptador de salida: usa el binario `ffmpeg` del sistema para convertir
/// cualquier archivo de audio/vídeo soportado en PCM f32 mono a 16kHz,
/// que es lo que whisper.cpp espera como entrada.
pub struct FfmpegAudioDecoder;

impl FfmpegAudioDecoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FfmpegAudioDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDecoder for FfmpegAudioDecoder {
    fn decode_to_pcm16k_mono(&self, input_path: &Path) -> Result<AudioSamples, AudioDecoderError> {
        if !input_path.is_file() {
            return Err(AudioDecoderError::Decode(format!(
                "el archivo de entrada no existe: {}",
                input_path.display()
            )));
        }

        // ffmpeg -> raw PCM f32le mono 16kHz por stdout, sin tocar el disco.
        let output = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(input_path)
            .args([
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ac", "1",
                "-ar", &AudioSamples::SAMPLE_RATE.to_string(),
                "-",
            ])
            .output()
            .map_err(|e| {
                AudioDecoderError::Decode(format!("no se pudo ejecutar ffmpeg: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AudioDecoderError::Decode(format!(
                "ffmpeg falló (código {:?}): {}",
                output.status.code(),
                stderr.trim()
            )));
        }

        let pcm_mono_16k = output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(AudioSamples { pcm_mono_16k })
    }
}
