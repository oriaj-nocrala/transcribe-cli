mod adapters;
mod application;
mod domain;
mod ports;

use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use futures_util::StreamExt;
use reqwest::Url;
use reqwest::header::{ACCEPT_RANGES, RANGE};
use tokio::io::AsyncWriteExt;

use adapters::inbound::{Cli, Command as CliCommand, IndicatifProgress, OutputFormat};
use adapters::outbound::CrispAsrTranscriber;
use application::TranscribeAudioUseCase;
use domain::model::{Transcript, TranscribeOptions};
use ports::Transcriber;

/// Versión de CrispASR validada en desarrollo — fijar la versión evita que un
/// release nuevo cambie de comportamiento por debajo nuestro sin que nos
/// enteremos.
const CRISPASR_VERSION: &str = "v0.8.30";
/// Único asset soportado por ahora: Linux x86_64 con CUDA 13 (este entorno de
/// desarrollo). Detectar SO/arch/GPU y elegir entre las demás variantes del
/// release (CPU/CUDA/Vulkan/HIP/macOS/Windows) queda para una fase futura.
const CRISPASR_ASSET: &str = "crispasr-linux-x86_64-cuda13.tar.gz";
const CRISPASR_ASSET_DIR: &str = "crispasr-linux-x86_64-cuda13";

fn crispasr_download_url() -> String {
    format!(
        "https://github.com/CrispStrobe/CrispASR/releases/download/{CRISPASR_VERSION}/{CRISPASR_ASSET}"
    )
}

/// Hugging Face y GitHub Releases limitan el throughput por conexión TCP, no
/// por IP: bajar un archivo grande en un único stream se estanca en unos
/// pocos MB/s aunque el enlace dé para mucho más. Repartir el archivo en
/// rangos y pedirlos en paralelo multiplica el throughput real varias veces
/// (medido con este mismo patrón contra HF: ~11x con 8 conexiones).
const PARALLEL_CONNECTIONS: u64 = 8;

fn models_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("no se pudo determinar el directorio home del usuario")?;
    Ok(home.join("models").join("tts"))
}

/// Descarga `url` en `PARALLEL_CONNECTIONS` rangos simultáneos, escribiendo
/// cada uno directamente en su offset del archivo destino.
async fn download_parallel(
    client: &reqwest::Client,
    url: Url,
    total: u64,
    dest: &Path,
    progress: Arc<dyn ports::ProgressSink>,
) -> Result<()> {
    let file = std::fs::File::create(dest)?;
    file.set_len(total)?;
    let file = Arc::new(file);

    let chunk_size = total.div_ceil(PARALLEL_CONNECTIONS);
    let downloaded = Arc::new(AtomicU64::new(0));

    let mut tasks = Vec::new();
    for i in 0..PARALLEL_CONNECTIONS {
        let start = i * chunk_size;
        if start >= total {
            break;
        }
        let end = ((start + chunk_size).min(total)) - 1;

        let client = client.clone();
        let url = url.clone();
        let file = file.clone();
        let downloaded = downloaded.clone();

        tasks.push(tokio::spawn(async move {
            let mut resp = client
                .get(url)
                .header(RANGE, format!("bytes={start}-{end}"))
                .send()
                .await?
                .error_for_status()?;

            let mut offset = start;
            while let Some(chunk) = resp.chunk().await? {
                file.write_at(&chunk, offset)?;
                offset += chunk.len() as u64;
                downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            }
            anyhow::Ok(())
        }));
    }

    let reporter_downloaded = downloaded.clone();
    let reporter_progress = progress.clone();
    let reporter = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            interval.tick().await;
            reporter_progress.on_progress(reporter_downloaded.load(Ordering::Relaxed));
        }
    });

    for task in tasks {
        task.await??;
    }
    reporter.abort();
    progress.on_progress(total);

    Ok(())
}

/// Descarga en un único stream secuencial. Fallback para cuando el servidor
/// no soporta range requests.
async fn download_sequential(
    response: reqwest::Response,
    dest: &Path,
    progress: Arc<dyn ports::ProgressSink>,
) -> Result<()> {
    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress.on_progress(downloaded);
    }
    file.flush().await?;

    Ok(())
}

/// Descarga (si hace falta) y cachea el runtime de CrispASR, devolviendo la
/// ruta al binario `crispasr` ya extraído.
async fn ensure_crispasr(cache_dir: &Path, progress: Arc<dyn ports::ProgressSink>) -> Result<PathBuf> {
    let dest_dir = cache_dir.join("crispasr").join(CRISPASR_VERSION);
    let bin = dest_dir.join(CRISPASR_ASSET_DIR).join("crispasr");
    if bin.is_file() {
        return Ok(bin);
    }

    tokio::fs::create_dir_all(&dest_dir).await?;

    let client = reqwest::Client::new();
    let url = crispasr_download_url();
    let response = client
        .get(&url)
        .send()
        .await
        .context("no se pudo descargar el runtime de CrispASR")?
        .error_for_status()
        .context("descarga del runtime de CrispASR falló")?;

    let total = response.content_length();
    let supports_ranges = response
        .headers()
        .get(ACCEPT_RANGES)
        .is_some_and(|v| v == "bytes");
    let resolved_url = response.url().clone();

    progress.on_start(total);

    let archive_path = dest_dir.join(CRISPASR_ASSET);
    match total {
        Some(total) if supports_ranges => {
            drop(response);
            download_parallel(&client, resolved_url, total, &archive_path, progress.clone()).await?;
        }
        _ => {
            download_sequential(response, &archive_path, progress.clone()).await?;
        }
    }
    progress.on_finish();

    let status = Command::new("tar")
        .arg("xzf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&dest_dir)
        .status()
        .context("no se pudo ejecutar 'tar' para extraer el runtime de CrispASR")?;
    if !status.success() {
        bail!("'tar' terminó con error extrayendo el runtime de CrispASR");
    }
    tokio::fs::remove_file(&archive_path).await.ok();

    if !bin.is_file() {
        bail!(
            "no se encontró el binario 'crispasr' esperado en '{}' tras extraer el runtime",
            bin.display()
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bin)?.permissions();
        perms.set_mode(perms.mode() | 0o111);
        std::fs::set_permissions(&bin, perms)?;
    }

    Ok(bin)
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
/// por cada segmento, su timestamp absoluto seguido del texto en prosa corrida.
fn render_markdown(input: &Path, transcript: &Transcript) -> String {
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

// Raíz de composición: aquí, y sólo aquí, se conoce el tipo concreto del
// adaptador. El resto de la aplicación programa contra los puertos.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let cache_dir = match &cli.models_dir {
        Some(dir) => dir.clone(),
        None => models_cache_dir()?,
    };

    match cli.command {
        CliCommand::Transcribe {
            input,
            backend,
            language,
            translate,
            threads,
            timestamps,
            format,
            output,
            no_vad,
        } => {
            let mut options = TranscribeOptions {
                language,
                translate_to_english: translate,
                backend,
                vad: !no_vad,
                ..TranscribeOptions::default()
            };
            if let Some(threads) = threads {
                options.threads = threads;
            }

            let progress: Arc<dyn ports::ProgressSink> = Arc::new(IndicatifProgress::new());
            let bin_path = ensure_crispasr(&cache_dir, progress.clone()).await?;

            let transcriber: Arc<dyn Transcriber> = Arc::new(CrispAsrTranscriber::new(bin_path));
            let use_case = TranscribeAudioUseCase::new(transcriber);
            let transcript = use_case
                .execute(&input, &options, progress)
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
