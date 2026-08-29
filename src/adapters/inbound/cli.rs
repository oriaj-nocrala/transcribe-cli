use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, Subcommand, ValueEnum};

use crate::domain::model::ModelId;

fn parse_model_id(s: &str) -> Result<ModelId, String> {
    ModelId::from_str(s).map_err(|e| e.to_string())
}

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
    name = "whisper-cli",
    version,
    about = "CLI para transcribir audio con whisper.cpp, descargando los modelos que hagan falta."
)]
pub struct Cli {
    /// Carpeta donde se guardan/buscan los modelos descargados.
    /// Por defecto: ~/models/tts (también configurable con $WHISPER_CLI_MODELS_DIR).
    #[arg(long, global = true, env = "WHISPER_CLI_MODELS_DIR", value_name = "DIR")]
    pub models_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Lista los modelos disponibles y cuáles ya están descargados localmente.
    Models,

    /// Descarga un modelo (si no está ya en caché) sin transcribir nada.
    Download {
        /// tiny | tiny.en | base | base.en | small | small.en | medium | medium.en | large-v3 | large-v3-turbo
        #[arg(value_parser = parse_model_id)]
        model: ModelId,
    },

    /// Transcribe un archivo de audio/vídeo (cualquier formato que ffmpeg entienda).
    Transcribe {
        /// Archivo de entrada.
        input: PathBuf,

        /// Modelo a usar (se descarga automáticamente si falta).
        #[arg(short, long, default_value = "base", value_parser = parse_model_id)]
        model: ModelId,

        /// Código de idioma del audio (p.ej. "es", "en"). Por defecto, autodetección.
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

        /// Pide timestamps por palabra (no sólo por segmento). Más finos que
        /// los de segmento, y suelen reflejar mejor las pausas reales entre
        /// frases (los de segmento a veces salen "pegados": fin de una frase
        /// = inicio de la siguiente). Con `--timestamps` en texto plano,
        /// imprime una línea por palabra en vez de por segmento.
        #[arg(long)]
        word_timestamps: bool,

        /// Formato de salida. Por defecto, texto plano; si no se indica y
        /// `--output` termina en `.md`, se usa Markdown automáticamente.
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,

        /// Guarda la transcripción en este archivo además de imprimirla en stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Desactiva la detección de voz (VAD): transcribe el audio completo,
        /// incluidos los silencios, como antes.
        #[arg(long)]
        no_vad: bool,

        /// Probabilidad mínima (0.0-1.0) para considerar una ventana como voz.
        #[arg(long, default_value_t = 0.5)]
        vad_threshold: f32,

        /// Huecos de silencio más cortos que esto (ms) se fusionan en el mismo tramo de voz.
        #[arg(long, default_value_t = 300)]
        vad_min_silence_ms: u32,

        /// Huecos de silencio entre dos tramos de voz más cortos que esto (ms) se
        /// incluyen en la misma ventana en vez de saltarse (evita pagar el coste
        /// fijo de una inferencia nueva por cada pausa corta entre frases).
        #[arg(long, default_value_t = 2000)]
        vad_merge_gap_ms: u32,

        /// Margen (ms) añadido a cada borde de un tramo de voz para no cortar palabras.
        #[arg(long, default_value_t = 200)]
        vad_pad_ms: u32,

        /// Duración máxima (segundos) de cada ventana de audio enviada al motor.
        #[arg(long, default_value_t = 300)]
        vad_max_window_secs: u32,
    },
}
