use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{ACCEPT_RANGES, RANGE};
use reqwest::Url;
use tokio::io::AsyncWriteExt;

use crate::domain::model::ModelId;
use crate::ports::model_repository::{ModelRepository, ModelRepositoryError};
use crate::ports::progress::{ProgressSink, ProgressUnit};

const HF_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Hugging Face limita el throughput por conexión TCP, no por IP: bajar un
/// modelo en un único stream se estanca en unos pocos MB/s aunque el enlace
/// dé para mucho más. Repartir el archivo en rangos y pedirlos en paralelo
/// multiplica el throughput real varias veces (medido: ~11x con 8 conexiones).
const PARALLEL_CONNECTIONS: u64 = 8;

/// Adaptador de salida: descarga los pesos ggml de whisper.cpp desde Hugging Face
/// y los cachea en disco. Implementa el puerto `ModelRepository`.
pub struct HttpModelRepository {
    cache_dir: PathBuf,
    client: reqwest::Client,
}

impl HttpModelRepository {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            client: reqwest::Client::new(),
        }
    }

    fn url_for(id: ModelId) -> String {
        format!("{HF_BASE_URL}/{}", id.filename())
    }

    fn partial_path(&self, id: ModelId) -> PathBuf {
        self.local_path(id).with_extension("bin.partial")
    }

    /// Descarga `url` en `PARALLEL_CONNECTIONS` rangos simultáneos, escribiendo
    /// cada uno directamente en su offset del archivo destino.
    async fn download_parallel(
        &self,
        url: Url,
        total: u64,
        dest: &std::path::Path,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<(), ModelRepositoryError> {
        let file = std::fs::File::create(dest)?;
        file.set_len(total)?;
        let file = Arc::new(file);

        let chunk_size = total.div_ceil(PARALLEL_CONNECTIONS);
        let downloaded = Arc::new(AtomicU64::new(0));

        let mut tasks = Vec::new();
        for i in 0..PARALLEL_CONNECTIONS {
            let start = i * chunk_size;
            if start >= total {
                break;
            }
            let end = ((start + chunk_size).min(total)) - 1;

            let client = self.client.clone();
            let url = url.clone();
            let file = file.clone();
            let downloaded = downloaded.clone();

            tasks.push(tokio::spawn(async move {
                let mut resp = client
                    .get(url)
                    .header(RANGE, format!("bytes={start}-{end}"))
                    .send()
                    .await
                    .map_err(|e| ModelRepositoryError::Network(e.to_string()))?
                    .error_for_status()
                    .map_err(|e| ModelRepositoryError::Network(e.to_string()))?;

                let mut offset = start;
                while let Some(chunk) = resp
                    .chunk()
                    .await
                    .map_err(|e| ModelRepositoryError::Network(e.to_string()))?
                {
                    file.write_at(&chunk, offset)?;
                    offset += chunk.len() as u64;
                    downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);
                }
                Ok::<(), ModelRepositoryError>(())
            }));
        }

        let reporter_downloaded = downloaded.clone();
        let reporter_progress = progress.clone();
        let reporter = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            loop {
                interval.tick().await;
                reporter_progress.on_progress(reporter_downloaded.load(Ordering::Relaxed));
            }
        });

        for task in tasks {
            task.await
                .map_err(|e| ModelRepositoryError::Network(e.to_string()))??;
        }
        reporter.abort();
        progress.on_progress(total);

        Ok(())
    }

    /// Descarga en un único stream secuencial. Fallback para cuando el
    /// servidor no soporta range requests.
    async fn download_sequential(
        &self,
        response: reqwest::Response,
        dest: &std::path::Path,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<(), ModelRepositoryError> {
        let mut file = tokio::fs::File::create(dest).await?;
        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ModelRepositoryError::Network(e.to_string()))?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            progress.on_progress(downloaded);
        }
        file.flush().await?;

        Ok(())
    }
}

#[async_trait]
impl ModelRepository for HttpModelRepository {
    fn local_path(&self, id: ModelId) -> PathBuf {
        self.cache_dir.join(id.filename())
    }

    fn is_downloaded(&self, id: ModelId) -> bool {
        self.local_path(id).is_file()
    }

    async fn ensure_downloaded(
        &self,
        id: ModelId,
        progress: Arc<dyn ProgressSink>,
    ) -> Result<PathBuf, ModelRepositoryError> {
        let final_path = self.local_path(id);
        if final_path.is_file() {
            return Ok(final_path);
        }

        tokio::fs::create_dir_all(&self.cache_dir).await?;

        let url = Self::url_for(id);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ModelRepositoryError::Network(e.to_string()))?
            .error_for_status()
            .map_err(|e| ModelRepositoryError::Network(e.to_string()))?;

        let total = response.content_length();
        let supports_ranges = response
            .headers()
            .get(ACCEPT_RANGES)
            .is_some_and(|v| v == "bytes");
        let resolved_url = response.url().clone();

        progress.on_start(ProgressUnit::Bytes, total);

        let partial_path = self.partial_path(id);

        match total {
            Some(total) if supports_ranges => {
                drop(response);
                self.download_parallel(resolved_url, total, &partial_path, progress.clone())
                    .await?;
            }
            _ => {
                self.download_sequential(response, &partial_path, progress.clone())
                    .await?;
            }
        }

        tokio::fs::rename(&partial_path, &final_path).await?;
        progress.on_finish();

        Ok(final_path)
    }
}
