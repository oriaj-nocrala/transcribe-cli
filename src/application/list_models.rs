use std::sync::Arc;

use crate::domain::model::ModelId;
use crate::ports::model_repository::{ModelRepository, ModelRepositoryError};

pub struct ModelStatus {
    pub id: ModelId,
    pub downloaded: bool,
}

pub struct ListModelsUseCase {
    repository: Arc<dyn ModelRepository>,
}

impl ListModelsUseCase {
    pub fn new(repository: Arc<dyn ModelRepository>) -> Self {
        Self { repository }
    }

    pub fn execute(&self) -> Result<Vec<ModelStatus>, ModelRepositoryError> {
        Ok(ModelId::ALL
            .iter()
            .map(|&id| ModelStatus {
                id,
                downloaded: self.repository.is_downloaded(id),
            })
            .collect())
    }
}
