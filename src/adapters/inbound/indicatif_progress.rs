use std::sync::Mutex;

use indicatif::{ProgressBar, ProgressStyle};

use crate::ports::progress::ProgressSink;

const BAR_CHARS: &str = "#>-";

/// Adaptador de entrada/UI: implementa `ProgressSink` mostrando una barra de
/// progreso en terminal con `indicatif` para la descarga del runtime de CrispASR.
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
    fn on_start(&self, total: Option<u64>) {
        let pb = match total {
            Some(total) => {
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
            None => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(ProgressStyle::with_template("{spinner:.green} {bytes} descargados").unwrap());
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
