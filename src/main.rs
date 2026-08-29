mod adapters;
mod application;
mod domain;
mod ports;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;

use adapters::inbound::{Cli, Command, IndicatifProgress, OutputFormat};
use adapters::outbound::{
    FfmpegAudioDecoder, HttpModelRepository, SileroVadDetector, WhisperRsTranscriber,
};
use application::{DownloadModelUseCase, ListModelsUseCase, TranscribeAudioUseCase};
use domain::model::TranscribeOptions;
use domain::window::VadConfig;
use ports::progress::ProgressUnit;
use ports::{AudioDecoder, ModelRepository, Transcriber, VoiceActivityDetector};

/// URL fija del modelo ONNX de Silero VAD (v5), ~2MB: no justifica todavía la
/// abstracción genérica de descarga multi-archivo prevista para cuando llegue
/// Qwen3 (Fase 3 del plan) — es un único archivo, igual que los de whisper.cpp.
const SILERO_VAD_URL: &str =
    "https://huggingface.co/deepghs/silero-vad-onnx/resolve/main/silero_vad.onnx";

fn models_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("no se pudo determinar el directorio home del usuario")?;
    Ok(home.join("models").join("tts"))
}

async fn ensure_silero_vad_model(
    cache_dir: &Path,
    progress: Arc<dyn ports::ProgressSink>,
) -> Result<PathBuf> {
    let path = cache_dir.join("silero_vad.onnx");
    if path.is_file() {
        return Ok(path);
    }

    tokio::fs::create_dir_all(cache_dir).await?;

    let response = reqwest::get(SILERO_VAD_URL)
        .await
        .context("no se pudo descargar el modelo Silero VAD")?
        .error_for_status()
        .context("descarga del modelo Silero VAD falló")?;
    let total = response.content_length();
    progress.on_start(ProgressUnit::Bytes, total);

    let bytes = response.bytes().await?;
    progress.on_progress(bytes.len() as u64);
    tokio::fs::write(&path, &bytes).await?;
    progress.on_finish();

    Ok(path)
}

fn format_timestamp(ms: i64) -> String {
    let total_ms = ms.max(0) as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms / 60_000) % 60;
    let secs = (total_ms / 1000) % 60;
    let millis = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02}.{millis:03}")
}

/// Timestamp corto (sin milisegundos) para la salida Markdown: los
/// milisegundos son ruido para un lector (humano o LLM), no aportan nada que
/// no dé ya el propio orden del texto.
fn format_timestamp_short(ms: i64) -> String {
    let total_secs = ms.max(0) as u64 / 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs / 60) % 60;
    let secs = total_secs % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

/// Markdown pensado para que lo consuma un LLM: encabezado con la fuente y,
/// por cada segmento, su timestamp absoluto (posible gracias al VAD, que
/// conserva la posición real en el audio aunque se salte silencios) seguido
/// del texto en prosa corrida.
fn render_markdown(input: &Path, transcript: &domain::model::Transcript) -> String {
    let title = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".to_string());

    let mut buf = format!("# Transcripción — {title}\n\n");
    for segment in &transcript.segments {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }
        buf.push_str(&format!(
            "**[{}]** {text}\n\n",
            format_timestamp_short(segment.start_ms)
        ));
    }
    buf
}

// Raíz de composición: aquí, y sólo aquí, se conocen los tipos concretos de
// los adaptadores. El resto de la aplicación programa contra los puertos.
#[tokio::main]
async fn main() -> Result<()> {
    // whisper.cpp escribe logs directamente a stderr por defecto; esto los
    // silencia (no configuramos ningún backend `log`/`tracing` detrás).
    whisper_rs::install_logging_hooks();

    let cli = Cli::parse();

    let cache_dir = match &cli.models_dir {
        Some(dir) => dir.clone(),
        None => models_cache_dir()?,
    };
    let repository: Arc<dyn ModelRepository> = Arc::new(HttpModelRepository::new(cache_dir.clone()));
    let decoder: Arc<dyn AudioDecoder> = Arc::new(FfmpegAudioDecoder::new());
    let transcriber: Arc<dyn Transcriber> = Arc::new(WhisperRsTranscriber::new());

    match cli.command {
        Command::Models => {
            let use_case = ListModelsUseCase::new(repository.clone());
            for status in use_case.execute()? {
                let mark = if status.downloaded { "✔" } else { " " };
                println!(
                    "[{mark}] {:<16} ~{} MiB",
                    status.id.to_string(),
                    status.id.approx_size_mib()
                );
            }
        }

        Command::Download { model } => {
            if repository.is_downloaded(model) {
                println!("'{model}' ya está descargado en {}", repository.local_path(model).display());
                return Ok(());
            }
            println!("Descargando '{model}' ({} MiB aprox.)...", model.approx_size_mib());
            let use_case = DownloadModelUseCase::new(repository.clone());
            let progress: Arc<dyn ports::ProgressSink> = Arc::new(IndicatifProgress::new());
            let path = use_case.execute(model, progress).await?;
            println!("Modelo guardado en {}", path.display());
        }

        Command::Transcribe {
            input,
            model,
            language,
            translate,
            threads,
            timestamps,
            word_timestamps,
            format,
            output,
            no_vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_merge_gap_ms,
            vad_pad_ms,
            vad_max_window_secs,
        } => {
            let mut options = TranscribeOptions {
                language,
                translate_to_english: translate,
                word_timestamps,
                ..TranscribeOptions::default()
            };
            if let Some(threads) = threads {
                options.threads = threads;
            }

            if !repository.is_downloaded(model) {
                println!("Modelo '{model}' no está en caché, descargando...");
            }

            let (vad, vad_model_path): (Option<Arc<dyn VoiceActivityDetector>>, Option<PathBuf>) =
                if no_vad {
                    (None, None)
                } else {
                    let progress: Arc<dyn ports::ProgressSink> = Arc::new(IndicatifProgress::new());
                    if !cache_dir.join("silero_vad.onnx").is_file() {
                        println!("Modelo Silero VAD no está en caché, descargando...");
                    }
                    let vad_path = ensure_silero_vad_model(&cache_dir, progress).await?;
                    let detector: Arc<dyn VoiceActivityDetector> =
                        Arc::new(SileroVadDetector::load(&vad_path)?);
                    (Some(detector), Some(vad_path))
                };
            let vad_config = VadConfig {
                threshold: vad_threshold,
                min_silence_ms: vad_min_silence_ms as i64,
                pad_ms: vad_pad_ms as i64,
                max_merge_gap_ms: vad_merge_gap_ms as i64,
                max_window_ms: vad_max_window_secs as i64 * 1000,
            };

            let use_case = TranscribeAudioUseCase::new(repository.clone(), decoder, transcriber, vad);
            let progress: Arc<dyn ports::ProgressSink> = Arc::new(IndicatifProgress::new());
            let transcript = use_case
                .execute(
                    model,
                    &input,
                    vad_model_path.as_deref(),
                    &vad_config,
                    &options,
                    progress,
                )
                .await
                .with_context(|| format!("fallo transcribiendo '{}'", input.display()))?;

            let format = format.unwrap_or_else(|| {
                let output_is_md = output
                    .as_ref()
                    .and_then(|p| p.extension())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));
                if output_is_md {
                    OutputFormat::Md
                } else {
                    OutputFormat::Text
                }
            });

            let rendered = match format {
                OutputFormat::Md => render_markdown(&input, &transcript),
                OutputFormat::Text if timestamps && word_timestamps => {
                    let mut buf = String::new();
                    for segment in &transcript.segments {
                        for word in segment.words.iter().flatten() {
                            buf.push_str(&format!(
                                "[{} --> {}] {}\n",
                                format_timestamp(word.start_ms),
                                format_timestamp(word.end_ms),
                                word.text
                            ));
                        }
                    }
                    buf
                }
                OutputFormat::Text if timestamps => {
                    let mut buf = String::new();
                    for segment in &transcript.segments {
                        buf.push_str(&format!(
                            "[{} --> {}] {}\n",
                            format_timestamp(segment.start_ms),
                            format_timestamp(segment.end_ms),
                            segment.text.trim()
                        ));
                    }
                    buf
                }
                OutputFormat::Text => format!("{}\n", transcript.full_text()),
            };

            print!("{rendered}");

            if let Some(output_path) = output {
                std::fs::write(&output_path, &rendered)
                    .with_context(|| format!("no se pudo escribir '{}'", output_path.display()))?;
                eprintln!("Transcripción guardada en {}", output_path.display());
            }
        }
    }

    Ok(())
}
