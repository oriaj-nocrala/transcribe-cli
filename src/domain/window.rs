//! Ventanas de audio con offset absoluto y agrupación de spans de habla.
//!
//! `AudioWindow` es la unidad que reciben los motores de transcripción/alineación:
//! un recorte de PCM que conserva su posición temporal respecto al audio original
//! (`offset_ms`), para que los timestamps que produzcan los motores puedan volver
//! a expresarse en la línea de tiempo completa sumando ese offset.

use crate::domain::model::AudioSamples;

/// Recorte de PCM mono 16kHz con su posición absoluta en el audio original.
#[derive(Debug, Clone)]
pub struct AudioWindow {
    pub pcm_mono_16k: Vec<f32>,
    pub offset_ms: i64,
}

/// Tramo de habla detectado por un `VoiceActivityDetector`, en timestamps
/// absolutos respecto al audio original.
#[derive(Debug, Clone, Copy)]
pub struct SpeechSpan {
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Parámetros de detección de voz y agrupación de ventanas.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Probabilidad mínima de voz para considerar una ventana del detector como habla.
    pub threshold: f32,
    /// Huecos de silencio más cortos que esto se fusionan dentro del mismo span
    /// ya en el propio detector (antes de llegar a `windows_from_speech_spans`).
    pub min_silence_ms: i64,
    /// Margen añadido a cada borde de un span para no cortar palabras.
    pub pad_ms: i64,
    /// Huecos de silencio entre dos spans distintos más cortos que esto se
    /// incluyen dentro de la misma ventana en vez de partirla en dos. Sin
    /// esto, cada pausa natural entre frases forzaría una llamada de
    /// inferencia nueva (con su propio coste fijo de arranque), lo que para
    /// audio con muchas pausas cortas puede costar más de lo que ahorra saltarse
    /// silencio. Un silencio más largo que esto sí se salta.
    pub max_merge_gap_ms: i64,
    /// Duración máxima de una `AudioWindow`; un grupo de spans (ya fusionados
    /// por `max_merge_gap_ms`) más largo que esto se trocea en varias ventanas.
    pub max_window_ms: i64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_ms: 300,
            pad_ms: 200,
            max_merge_gap_ms: 2_000,
            max_window_ms: 300_000,
        }
    }
}

/// Agrupa spans de habla en `AudioWindow`, fusionando huecos de silencio
/// cortos (`&lt;= max_merge_gap_ms`) dentro de la misma ventana para amortizar
/// el coste fijo por llamada de inferencia, pero preservando como huecos
/// reales (ventanas separadas, silencio no incluido) cualquier hueco más
/// largo. Un grupo resultante que exceda `max_window_ms` se trocea en cortes
/// fijos de esa duración (p.ej. un monólogo largo sin silencios detectables).
pub fn windows_from_speech_spans(
    spans: &[SpeechSpan],
    audio: &AudioSamples,
    config: &VadConfig,
) -> Vec<AudioWindow> {
    let mut windows = Vec::new();
    let mut group_start: Option<i64> = None;
    let mut group_end: i64 = 0;

    let flush = |start: i64, end: i64, windows: &mut Vec<AudioWindow>| {
        let mut cursor = start;
        while cursor < end {
            let chunk_end = (cursor + config.max_window_ms).min(end);
            windows.push(audio.slice_window(cursor, chunk_end));
            cursor = chunk_end;
        }
    };

    for span in spans {
        let start = span.start_ms.max(0);
        let end = span.end_ms.min(audio.duration_ms());
        if end <= start {
            continue;
        }

        match group_start {
            None => {
                group_start = Some(start);
                group_end = end;
            }
            Some(gs) => {
                let gap = start - group_end;
                if gap <= config.max_merge_gap_ms && end - gs <= config.max_window_ms {
                    group_end = end;
                } else {
                    flush(gs, group_end, &mut windows);
                    group_start = Some(start);
                    group_end = end;
                }
            }
        }
    }

    if let Some(gs) = group_start {
        flush(gs, group_end, &mut windows);
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_of_duration_ms(ms: i64) -> AudioSamples {
        let samples = (ms * AudioSamples::SAMPLE_RATE as i64 / 1000) as usize;
        AudioSamples {
            pcm_mono_16k: vec![0.0; samples],
        }
    }

    fn config(max_merge_gap_ms: i64, max_window_ms: i64) -> VadConfig {
        VadConfig {
            max_merge_gap_ms,
            max_window_ms,
            ..VadConfig::default()
        }
    }

    #[test]
    fn empty_spans_yield_no_windows() {
        let audio = audio_of_duration_ms(10_000);
        let windows = windows_from_speech_spans(&[], &audio, &config(2_000, 5_000));
        assert!(windows.is_empty());
    }

    #[test]
    fn spans_within_merge_gap_share_one_window_including_the_gap() {
        // Dos spans separados por una pausa corta (menor que max_merge_gap_ms):
        // deben compartir una única ventana, silencio de la pausa incluido,
        // para no pagar el coste fijo de una llamada de inferencia extra por
        // cada pausa natural entre frases.
        let audio = audio_of_duration_ms(10_000);
        let spans = [
            SpeechSpan { start_ms: 1_000, end_ms: 2_000 },
            SpeechSpan { start_ms: 2_500, end_ms: 4_000 },
        ];
        let windows = windows_from_speech_spans(&spans, &audio, &config(2_000, 5_000));
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].offset_ms, 1_000);
        assert_eq!(windows[0].pcm_mono_16k.len(), 3_000 * 16);
    }

    #[test]
    fn spans_beyond_merge_gap_stay_in_separate_windows() {
        // El mismo par de spans, pero con max_merge_gap_ms menor que la
        // pausa entre ellos: no deben fusionarse aunque el grupo resultante
        // seguiría cabiendo bajo max_window_ms.
        let audio = audio_of_duration_ms(10_000);
        let spans = [
            SpeechSpan { start_ms: 1_000, end_ms: 2_000 },
            SpeechSpan { start_ms: 2_500, end_ms: 4_000 },
        ];
        let windows = windows_from_speech_spans(&spans, &audio, &config(100, 5_000));
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].offset_ms, 1_000);
        assert_eq!(windows[0].pcm_mono_16k.len(), 1_000 * 16);
        assert_eq!(windows[1].offset_ms, 2_500);
        assert_eq!(windows[1].pcm_mono_16k.len(), 1_500 * 16);
    }

    #[test]
    fn long_silence_gap_is_never_merged_even_under_max_window_ms() {
        // Hueco de silencio real (15s) muy por encima de max_merge_gap_ms,
        // aunque el total quepa cómodo bajo max_window_ms (5 min): no debe
        // colarse en la ventana.
        let audio = audio_of_duration_ms(65_000);
        let spans = [
            SpeechSpan { start_ms: 10_000, end_ms: 30_000 },
            SpeechSpan { start_ms: 45_000, end_ms: 65_000 },
        ];
        let windows = windows_from_speech_spans(&spans, &audio, &config(2_000, 300_000));
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].offset_ms, 10_000);
        assert_eq!(windows[0].pcm_mono_16k.len(), 20_000 * 16);
        assert_eq!(windows[1].offset_ms, 45_000);
        assert_eq!(windows[1].pcm_mono_16k.len(), 20_000 * 16);
    }

    #[test]
    fn spans_far_apart_produce_separate_windows() {
        let audio = audio_of_duration_ms(20_000);
        let spans = [
            SpeechSpan { start_ms: 0, end_ms: 1_000 },
            SpeechSpan { start_ms: 10_000, end_ms: 11_000 },
        ];
        let windows = windows_from_speech_spans(&spans, &audio, &config(2_000, 5_000));
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].offset_ms, 0);
        assert_eq!(windows[1].offset_ms, 10_000);
    }

    #[test]
    fn long_span_without_silence_is_chunked() {
        let audio = audio_of_duration_ms(12_000);
        let spans = [SpeechSpan { start_ms: 0, end_ms: 12_000 }];
        let windows = windows_from_speech_spans(&spans, &audio, &config(2_000, 5_000));
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].offset_ms, 0);
        assert_eq!(windows[0].pcm_mono_16k.len(), 5_000 * 16);
        assert_eq!(windows[1].offset_ms, 5_000);
        assert_eq!(windows[1].pcm_mono_16k.len(), 5_000 * 16);
        assert_eq!(windows[2].offset_ms, 10_000);
        assert_eq!(windows[2].pcm_mono_16k.len(), 2_000 * 16);
    }

    #[test]
    fn merged_group_exceeding_max_window_flushes_early_instead_of_overshooting() {
        // Varios spans cercanos (dentro de max_merge_gap_ms) cuya suma total
        // supera max_window_ms: en vez de colar una ventana más larga que el
        // límite, corta el grupo apenas la siguiente fusión lo excedería.
        let audio = audio_of_duration_ms(12_000);
        let spans = [
            SpeechSpan { start_ms: 0, end_ms: 4_000 },
            SpeechSpan { start_ms: 4_500, end_ms: 8_000 },
            SpeechSpan { start_ms: 8_500, end_ms: 12_000 },
        ];
        let windows = windows_from_speech_spans(&spans, &audio, &config(1_000, 5_000));
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].offset_ms, 0);
        assert_eq!(windows[0].pcm_mono_16k.len(), 4_000 * 16);
        assert_eq!(windows[1].offset_ms, 4_500);
        assert_eq!(windows[1].pcm_mono_16k.len(), 3_500 * 16);
        assert_eq!(windows[2].offset_ms, 8_500);
        assert_eq!(windows[2].pcm_mono_16k.len(), 3_500 * 16);
        for w in &windows {
            assert!(w.pcm_mono_16k.len() as i64 <= 5_000 * 16);
        }
    }
}
