/// Qué representa el total/avance reportado, para que el adaptador de UI pueda
/// elegir una presentación adecuada (p.ej. "3.1/466 MiB" vs "42%").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressUnit {
    Bytes,
    Percent,
}

/// Puerto de salida secundario: notifica avance de una operación larga (descarga, inferencia)
/// sin que el dominio/aplicación sepan si eso termina en una barra de progreso, un log, etc.
/// Se usa tanto para la descarga de modelos como para la propia inferencia, de modo que
/// ambas fases comparten el mismo adaptador de UI.
pub trait ProgressSink: Send + Sync {
    fn on_start(&self, unit: ProgressUnit, total: Option<u64>);
    fn on_progress(&self, current: u64);
    fn on_finish(&self);
}
