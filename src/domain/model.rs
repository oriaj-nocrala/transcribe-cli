//! Tipos del dominio: no dependen de ningún framework ni motor de ASR concreto.

/// Un segmento de texto transcrito con su rango temporal, en ms absolutos
/// respecto al audio original.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
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

/// Opciones de transcripción. `backend` es el nombre del motor de CrispASR
/// (p.ej. "qwen3-1.7b", "whisper", "parakeet") — passthrough libre, sin
/// validar contra una lista propia: CrispASR ya valida y da buenos mensajes
/// de error para sus ~100 backends.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Código de idioma ISO-639-1 (p.ej. "es", "en"), o `None` para autodetección.
    pub language: Option<String>,
    pub translate_to_english: bool,
    pub backend: String,
    /// Cuantización preferida del modelo del backend (p.ej. "q4_k", "q5_k",
    /// "q8_0", "f16"), o `None` para dejar que CrispASR resuelva la de su
    /// registro. Un mismo backend está publicado en varias: `qwen3-1.7b` cae
    /// por defecto en la `q4_k`, que es la más chica y también la que más se
    /// equivoca justo en lo que más cuesta caro — palabras raras y nombres
    /// propios. Passthrough libre al `--model-quant` de crispasr, igual que
    /// `backend`: si la variante no existe, el error lo da él.
    pub model_quant: Option<String>,
    /// Si es `false`, transcribe el audio completo sin saltarse silencios.
    pub vad: bool,
    /// Pasa a `-t` de crispasr. Con backends acelerados por GPU (el caso
    /// normal aquí) no cambia el tiempo de transcripción de forma medible
    /// (confirmado: 2, 8 y 24 hilos dan el mismo resultado) — sólo importa
    /// para el VAD (siempre CPU) y para backends sin aceleración GPU.
    pub threads: u32,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            translate_to_english: false,
            backend: "qwen3-1.7b".to_string(),
            model_quant: None,
            vad: true,
            threads: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
                .min(8),
        }
    }
}
