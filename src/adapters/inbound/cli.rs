use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Texto plano, con o sin timestamps por segmento (ver `--timestamps`).
    Text,
    /// Markdown legible por humanos y LLMs: un encabezado por segmento con su
    /// timestamp y el texto debajo, en prosa corrida.
    Md,
}

#[derive(Parser)]
#[command(
    name = "transcribe-cli",
    version,
    about = "CLI para transcribir audio con CrispASR (Qwen3-ASR y otros backends), descargando el runtime y los modelos que hagan falta."
)]
pub struct Cli {
    /// Carpeta donde se cachea el runtime de CrispASR.
    /// Por defecto: ~/models/tts (también configurable con $TRANSCRIBE_CLI_MODELS_DIR).
    #[arg(long, global = true, env = "TRANSCRIBE_CLI_MODELS_DIR", value_name = "DIR")]
    pub models_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Transcribe un archivo de audio/vídeo (cualquier formato que CrispASR/ffmpeg entiendan).
    Transcribe {
        /// Archivo de entrada.
        input: PathBuf,

        /// Backend de CrispASR a usar (p.ej. "qwen3-1.7b", "qwen3", "whisper",
        /// "parakeet"). Ver `crispasr --list-backends` para la lista completa.
        #[arg(short, long, default_value = "qwen3-1.7b")]
        backend: String,

        /// Código de idioma del audio (p.ej. "es", "en"). Por defecto, autodetección.
        /// Ojo: algunos backends traducen en vez de transcribir si el idioma
        /// pedido no coincide con el hablado en el audio.
        #[arg(short, long)]
        language: Option<String>,

        /// Traduce el resultado al inglés en vez de transcribir en el idioma original.
        #[arg(short, long)]
        translate: bool,

        /// Número de hilos a usar en la inferencia.
        #[arg(long)]
        threads: Option<u32>,

        /// Incluye timestamps por segmento en la salida de texto plano
        /// (Markdown siempre los incluye, no hace falta esta flag para eso).
        #[arg(long)]
        timestamps: bool,

        /// Formato de salida. Por defecto, texto plano; si no se indica y
        /// `--output` termina en `.md`, se usa Markdown automáticamente.
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,

        /// Guarda la transcripción en este archivo además de imprimirla en stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Desactiva la detección de voz (VAD): transcribe el audio completo,
        /// incluidos los silencios.
        #[arg(long)]
        no_vad: bool,
    },
}
