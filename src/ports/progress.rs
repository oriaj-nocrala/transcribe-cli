/// Puerto de salida secundario: notifica avance de una descarga larga sin que
/// el resto de la aplicación sepa si eso termina en una barra de progreso,
/// un log, etc.
pub trait ProgressSink: Send + Sync {
    fn on_start(&self, total: Option<u64>);
    fn on_progress(&self, current: u64);
    fn on_finish(&self);
}
