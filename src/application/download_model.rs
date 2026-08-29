use std::path::PathBuf;
use std::sync::Arc;

use crate::domain::model::ModelId;
use crate::ports::model_repository::{ModelRepository, ModelRepositoryError};
use crate::ports::progress::ProgressSink;

pub struct DownloadModelUseCase {
    repository: Arc<dyn ModelRepository>,
}

impl DownloadModelUseCase {
    pub fn new(repository: Arc<dyn ModelRepository>) -> Self {
        Self { repository }
    }

    pub async fn execute(
        &self,
        model: ModelId,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<PathBuf, ModelRepositoryError> {
        self.repository.ensure_downloaded(model, progress).await
    }
}
