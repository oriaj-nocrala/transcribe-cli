//! Tipos del dominio: no dependen de ningún framework, crate de red ni de whisper.cpp.

use std::fmt;
use std::str::FromStr;

use crate::domain::window::AudioWindow;

/// Identifica un modelo ggml de whisper.cpp soportado por la aplicación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV3,
    LargeV3Turbo,
}

impl ModelId {
    pub const ALL: &'static [ModelId] = &[
        ModelId::Tiny,
        ModelId::TinyEn,
        ModelId::Base,
        ModelId::BaseEn,
        ModelId::Small,
        ModelId::SmallEn,
        ModelId::Medium,
        ModelId::MediumEn,
        ModelId::LargeV3,
        ModelId::LargeV3Turbo,
    ];

    /// Fragmento usado tanto en el nombre de archivo (`ggml-{slug}.bin`) como en la CLI.
    pub fn slug(&self) -> &'static str {
        match self {
            ModelId::Tiny => "tiny",
            ModelId::TinyEn => "tiny.en",
            ModelId::Base => "base",
            ModelId::BaseEn => "base.en",
            ModelId::Small => "small",
            ModelId::SmallEn => "small.en",
            ModelId::Medium => "medium",
            ModelId::MediumEn => "medium.en",
            ModelId::LargeV3 => "large-v3",
            ModelId::LargeV3Turbo => "large-v3-turbo",
        }
    }

    pub fn filename(&self) -> String {
        format!("ggml-{}.bin", self.slug())
    }

    /// Tamaño aproximado en MiB, sólo informativo para el usuario.
    pub fn approx_size_mib(&self) -> u32 {
        match self {
            ModelId::Tiny | ModelId::TinyEn => 75,
            ModelId::Base | ModelId::BaseEn => 142,
            ModelId::Small | ModelId::SmallEn => 466,
            ModelId::Medium | ModelId::MediumEn => 1500,
            ModelId::LargeV3 => 2900,
            ModelId::LargeV3Turbo => 1500,
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.slug())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("modelo desconocido: '{0}' (usa `whisper-cli models` para ver los disponibles)")]
pub struct UnknownModelId(pub String);

impl FromStr for ModelId {
    type Err = UnknownModelId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ModelId::ALL
            .iter()
            .copied()
            .find(|m| m.slug() == s)
            .ok_or_else(|| UnknownModelId(s.to_string()))
    }
}

/// Audio PCM mono a 16 kHz, formato exigido por whisper.cpp.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    pub pcm_mono_16k: Vec<f32>,
}

impl AudioSamples {
    pub const SAMPLE_RATE: u32 = 16_000;

    pub fn duration_ms(&self) -> i64 {
        (self.pcm_mono_16k.len() as i64 * 1000) / Self::SAMPLE_RATE as i64
    }

    /// Vista de la totalidad del audio como una única `AudioWindow` sin offset.
    /// Es el camino usado cuando no hay VAD: un solo recorte que cubre todo.
    pub fn whole_window(&self) -> AudioWindow {
        AudioWindow {
            pcm_mono_16k: self.pcm_mono_16k.clone(),
            offset_ms: 0,
        }
    }

    /// Recorta `[start_ms, end_ms)` conservando `start_ms` como offset absoluto.
    pub fn slice_window(&self, start_ms: i64, end_ms: i64) -> AudioWindow {
        let start_sample = (start_ms.max(0) * Self::SAMPLE_RATE as i64 / 1000) as usize;
        let end_sample = (end_ms.max(0) * Self::SAMPLE_RATE as i64 / 1000) as usize;
        let end_sample = end_sample.min(self.pcm_mono_16k.len());
        let start_sample = start_sample.min(end_sample);
        AudioWindow {
            pcm_mono_16k: self.pcm_mono_16k[start_sample..end_sample].to_vec(),
            offset_ms: start_ms.max(0),
        }
    }
}

/// Timestamp de una palabra dentro de un segmento, en ms absolutos.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// Un segmento de texto transcrito con su rango temporal.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// Timestamps por palabra dentro de este segmento, si el motor los dio
    /// (ver `TranscribeOptions::word_timestamps`). Más finos que
    /// `start_ms`/`end_ms` del segmento, aunque siguen siendo tan precisos
    /// como el propio decoder de whisper: no sustituyen a un forced aligner.
    pub words: Option<Vec<WordTimestamp>>,
}

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub segments: Vec<TranscriptSegment>,
}

impl Transcript {
    pub fn full_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Opciones de transcripción independientes del motor concreto.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Código de idioma ISO-639-1 (p.ej. "es", "en"), o `None` para autodetección.
    pub language: Option<String>,
    pub translate_to_english: bool,
    pub threads: u32,
    /// Pide al motor timestamps por palabra además de por segmento (ver
    /// `TranscriptSegment::words`). Whisper.cpp los deriva de sus propios
    /// timestamps por token; siguen siendo aproximados, sólo más finos.
    pub word_timestamps: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            translate_to_english: false,
            // whisper.cpp no escala bien más allá de unos pocos hilos (el
            // cómputo no es tan paralelizable como sugiere el núcleo count);
            // pedir todos los cores disponibles sólo añade overhead de
            // sincronización, y como ahora se hace una llamada por ventana
            // (VAD), ese overhead se paga una vez por ventana. Medido en este
            // proyecto: 24 hilos vs 8 en un audio de 5 min, modelo base,
            // 2 ventanas: ~243s vs ~33s de CPU total, y más rápido en pared.
            threads: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
                .min(8),
            word_timestamps: false,
        }
    }
}
