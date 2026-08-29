pub mod audio_decoder;
pub mod model_repository;
pub mod progress;
pub mod transcriber;
pub mod vad;

// Facade del hexágono: re-exporta el contrato completo de cada puerto aunque
// el binario actual sólo use una parte; los tipos de error viajan con `?`
// vía `anyhow` y no necesitan nombrarse en el punto de uso.
#[allow(unused_imports)]
pub use audio_decoder::{AudioDecoder, AudioDecoderError};
#[allow(unused_imports)]
pub use model_repository::{ModelRepository, ModelRepositoryError};
#[allow(unused_imports)]
pub use progress::ProgressSink;
#[allow(unused_imports)]
pub use transcriber::{Transcriber, TranscriberError};
#[allow(unused_imports)]
pub use vad::{VadError, VoiceActivityDetector};
