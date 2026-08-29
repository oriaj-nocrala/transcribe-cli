use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::domain::model::ModelId;
use crate::ports::progress::ProgressSink;

#[derive(Debug, thiserror::Error)]
pub enum ModelRepositoryError {
    #[error("fallo de red descargando el modelo: {0}")]
    Network(String),
    #[error("error de E/S accediendo al almacén local de modelos: {0}")]
    Io(#[from] std::io::Error),
}

/// Puerto de salida: cómo obtener y localizar los pesos de un modelo whisper.
/// La aplicación no sabe (ni le importa) si detrás hay HTTP, un caché en disco, S3, etc.
#[async_trait]
pub trait ModelRepository: Send + Sync {
    /// Ruta local donde debería/está el modelo, exista o no todavía.
    fn local_path(&self, id: ModelId) -> PathBuf;

    fn is_downloaded(&self, id: ModelId) -> bool;

    /// Descarga el modelo si hace falta y devuelve la ruta local final.
    async fn ensure_downloaded(
        &self,
        id: ModelId,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<PathBuf, ModelRepositoryError>;
}
