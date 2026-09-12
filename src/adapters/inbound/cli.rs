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

        /// Cuantización del modelo (p.ej. "q4_k", "q5_k", "q8_0", "f16").
        /// Por defecto, la que resuelva CrispASR: para `qwen3-1.7b` es la
        /// `q4_k`, la más chica y la que más se equivoca en palabras raras y
        /// nombres propios. Si la variante pedida no está en caché, se
        /// descarga sola la primera vez.
        #[arg(long, env = "TRANSCRIBE_CLI_QUANT", value_name = "QUANT")]
        model_quant: Option<String>,

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

        /// Activa la detección de voz (VAD): salta silencios en vez de
        /// transcribir el audio completo. Desactivado por defecto — Silero
        /// clasifica de vez en cuando tramos con voz real como silencio y
        /// esos quedan afuera de la transcripción sin ningún aviso.
        #[arg(long)]
        vad: bool,
    },
}
