use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::domain::model::{TranscribeOptions, Transcript, TranscriptSegment};
use crate::ports::progress::ProgressSink;
use crate::ports::transcriber::{Transcriber, TranscriberError};

/// Adaptador de salida: implementa `Transcriber` invocando el binario
/// `crispasr` (github.com/CrispStrobe/CrispASR) como subproceso. CrispASR
/// resuelve por su cuenta decodificación de audio, VAD (Silero) y descarga
/// del modelo GGUF del backend elegido — este adaptador sólo arma el
/// comando, parsea su salida JSON y mapea al dominio.
pub struct CrispAsrTranscriber {
    binary_path: PathBuf,
}

impl CrispAsrTranscriber {
    /// `binary_path` debe ser la ruta ya resuelta al binario `crispasr`
    /// (ver `ensure_crispasr` en `main.rs`), con sus `.so` bundled en el
    /// mismo directorio.
    pub fn new(binary_path: PathBuf) -> Self {
        Self { binary_path }
    }

    fn lib_dir(&self) -> Option<&Path> {
        self.binary_path.parent()
    }
}

#[derive(Deserialize)]
struct CrispasrOutput {
    transcription: Vec<CrispasrSegment>,
}

#[derive(Deserialize)]
struct CrispasrSegment {
    offsets: CrispasrOffsets,
    text: String,
}

#[derive(Deserialize)]
struct CrispasrOffsets {
    from: i64,
    to: i64,
}

impl Transcriber for CrispAsrTranscriber {
    fn transcribe(
        &self,
        input_audio: &Path,
        options: &TranscribeOptions,
        _progress: Arc<dyn ProgressSink>,
    ) -> Result<Transcript, TranscriberError> {
        // crispasr imprime su propio progreso (descarga de modelo, carga,
        // inferencia) a stderr, heredado directo por esta terminal — no hay
        // nada útil que reportar por `ProgressSink` aquí sin parsear esa
        // salida, y duplicar la barra sería confuso.
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let out_base = std::env::temp_dir().join(format!("transcribe-cli-crispasr-{unique}"));
        let out_json = out_base.with_extension("json");

        let mut cmd = Command::new(&self.binary_path);
        if let Some(lib_dir) = self.lib_dir() {
            cmd.env("LD_LIBRARY_PATH", lib_dir);
        }
        cmd.arg("--backend")
            .arg(&options.backend)
            .arg("-m")
            .arg("auto")
            .arg("-l")
            .arg(options.language.as_deref().unwrap_or("auto"))
            .arg("-t")
            .arg(options.threads.to_string())
            .arg("-np")
            .arg("-oj")
            .arg("-of")
            .arg(&out_base);

        // Acompaña al `-m auto` de arriba: no es una ruta a un fichero, es
        // la preferencia que usa crispasr al resolver el modelo del backend
        // contra su registro (y a descargarlo, si no lo tiene).
        if let Some(quant) = &options.model_quant {
            cmd.arg("--model-quant").arg(quant);
        }

        if options.translate_to_english {
            cmd.arg("-tr");
        }
        if options.vad {
            cmd.arg("--vad").arg("--vad-model").arg("silero");
        }

        cmd.arg(input_audio);

        let status = cmd
            .status()
            .map_err(|e| TranscriberError::Inference(format!("no se pudo ejecutar crispasr: {e}")))?;

        if !status.success() {
            let _ = std::fs::remove_file(&out_json);
            return Err(TranscriberError::Inference(format!(
                "crispasr terminó con error (código {:?})",
                status.code()
            )));
        }

        let raw = std::fs::read_to_string(&out_json).map_err(|e| {
            TranscriberError::Inference(format!(
                "no se pudo leer la salida de crispasr en '{}': {e}",
                out_json.display()
            ))
        })?;
        let _ = std::fs::remove_file(&out_json);

        let parsed: CrispasrOutput = serde_json::from_str(&raw)
            .map_err(|e| TranscriberError::Inference(format!("salida JSON de crispasr inválida: {e}")))?;

        let segments = parsed
            .transcription
            .into_iter()
            .map(|s| TranscriptSegment {
                start_ms: s.offsets.from,
                end_ms: s.offsets.to,
                text: s.text,
            })
            .collect();

        Ok(Transcript { segments })
    }
}
