use std::sync::Mutex;

use indicatif::{ProgressBar, ProgressStyle};

use crate::ports::progress::{ProgressSink, ProgressUnit};

const BAR_CHARS: &str = "#>-";

/// Adaptador de entrada/UI: implementa `ProgressSink` mostrando una barra de
/// progreso en terminal con `indicatif`. Se reutiliza tanto para la descarga
/// de modelos (bytes) como para la inferencia (porcentaje), con el mismo
/// estilo visual, para que ambas fases compartan una única UX de progreso.
pub struct IndicatifProgress {
    bar: Mutex<Option<ProgressBar>>,
}

impl IndicatifProgress {
    pub fn new() -> Self {
        Self {
            bar: Mutex::new(None),
        }
    }
}

impl Default for IndicatifProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressSink for IndicatifProgress {
    fn on_start(&self, unit: ProgressUnit, total: Option<u64>) {
        let pb = match (unit, total) {
            (ProgressUnit::Bytes, Some(total)) => {
                let pb = ProgressBar::new(total);
                pb.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                    )
                    .unwrap()
                    .progress_chars(BAR_CHARS),
                );
                pb
            }
            (ProgressUnit::Bytes, None) => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(ProgressStyle::with_template("{spinner:.green} {bytes} descargados").unwrap());
                pb
            }
            (ProgressUnit::Percent, _) => {
                let pb = ProgressBar::new(100);
                pb.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.green} [{bar:40.cyan/blue}] {percent}% ({eta})",
                    )
                    .unwrap()
                    .progress_chars(BAR_CHARS),
                );
                pb
            }
        };
        *self.bar.lock().unwrap() = Some(pb);
    }

    fn on_progress(&self, current: u64) {
        if let Some(pb) = self.bar.lock().unwrap().as_ref() {
            pb.set_position(current);
        }
    }

    fn on_finish(&self) {
        if let Some(pb) = self.bar.lock().unwrap().take() {
            pb.finish_with_message("completado");
        }
    }
}
