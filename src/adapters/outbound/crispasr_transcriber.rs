use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::domain::model::{TranscribeOptions, Transcript, TranscriptSegment};
use crate::ports::progress::ProgressSink;
use crate::ports::transcriber::{Transcriber, TranscriberError};

/// Adaptador de salida: implementa `Transcriber` invocando el binario
/// `crispasr` (github.com/CrispStrobe/CrispASR) como subproceso. CrispASR
/// resuelve por su cuenta decodificación de audio, VAD (Silero) y descarga
/// del modelo GGUF del backend elegido — este adaptador sólo arma el
/// comando, parsea su salida JSON y mapea al dominio.
pub struct CrispAsrTranscriber {
    binary_path: PathBuf,
}

impl CrispAsrTranscriber {
    /// `binary_path` debe ser la ruta ya resuelta al binario `crispasr`
    /// (ver `ensure_crispasr` en `main.rs`), con sus `.so` bundled en el
    /// mismo directorio.
    pub fn new(binary_path: PathBuf) -> Self {
        Self { binary_path }
    }

    fn lib_dir(&self) -> Option<&Path> {
        self.binary_path.parent()
    }
}

#[derive(Deserialize)]
struct CrispasrOutput {
    transcription: Vec<CrispasrSegment>,
}

#[derive(Deserialize)]
struct CrispasrSegment {
    offsets: CrispasrOffsets,
    text: String,
}

#[derive(Deserialize)]
struct CrispasrOffsets {
    from: i64,
    to: i64,
}

impl Transcriber for CrispAsrTranscriber {
    fn transcribe(
        &self,
        input_audio: &Path,
        options: &TranscribeOptions,
        _progress: Arc<dyn ProgressSink>,
    ) -> Result<Transcript, TranscriberError> {
        // crispasr imprime su propio progreso (descarga de modelo, carga,
        // inferencia) a stderr, heredado directo por esta terminal — no hay
        // nada útil que reportar por `ProgressSink` aquí sin parsear esa
        // salida, y duplicar la barra sería confuso.
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let out_base = std::env::temp_dir().join(format!("transcribe-cli-crispasr-{unique}"));
        let out_json = out_base.with_extension("json");

        let mut cmd = Command::new(&self.binary_path);
        if let Some(lib_dir) = self.lib_dir() {
            cmd.env("LD_LIBRARY_PATH", lib_dir);
        }
        cmd.arg("--backend")
            .arg(&options.backend)
            .arg("-m")
            .arg("auto")
            .arg("-l")
            .arg(options.language.as_deref().unwrap_or("auto"))
            .arg("-t")
            .arg(options.threads.to_string())
            .arg("-np")
            .arg("-oj")
            .arg("-of")
            .arg(&out_base);

        // Acompaña al `-m auto` de arriba: no es una ruta a un fichero, es
        // la preferencia que usa crispasr al resolver el modelo del backend
        // contra su registro (y a descargarlo, si no lo tiene).
        if let Some(quant) = &options.model_quant {
            cmd.arg("--model-quant").arg(quant);
        }

        if options.translate_to_english {
            cmd.arg("-tr");
        }
        if options.vad {
            cmd.arg("--vad").arg("--vad-model").arg("silero");
        }

        cmd.arg(input_audio);

        if let Some(gpu) = gpu_snapshot() {
            eprintln!("GPU antes de empezar: {gpu}");
        }

        // stderr pasa por aquí en vez de heredarse: crispasr avisa de sus
        // degradaciones (VRAM que no alcanzó, ops que cayeron a CPU) en una
        // línea de log y SIGUE, con código 0. Así fue como v0.8.30 tiró 818
        // "cudaMalloc failed" en una corrida que se dio por buena. Cada línea
        // se reenvía tal cual — el progreso se sigue viendo — y además se
        // clasifica.
        cmd.stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| TranscriberError::Inference(format!("no se pudo ejecutar crispasr: {e}")))?;
        let stderr = child.stderr.take().expect("stderr con pipe");
        let monitor = std::thread::spawn(move || {
            let mut health = GpuHealth::default();
            let mut out = std::io::stderr();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = writeln!(out, "{line}");
                health.observe(&line);
            }
            health
        });
        let status = child
            .wait()
            .map_err(|e| TranscriberError::Inference(format!("no se pudo esperar a crispasr: {e}")))?;
        let health = monitor.join().unwrap_or_default();

        // Va ANTES de mirar el código de salida: si crispasr murió por falta
        // de VRAM, eso es lo que hay que decir, no "código 1".
        if let Some(problem) = health.problem() {
            let _ = std::fs::remove_file(&out_json);
            let gpu = gpu_snapshot().map(|g| format!("\nGPU ahora: {g}")).unwrap_or_default();
            return Err(TranscriberError::Gpu(format!("{problem}{gpu}")));
        }

        if !status.success() {
            let _ = std::fs::remove_file(&out_json);
            return Err(TranscriberError::Inference(format!(
                "crispasr terminó con error (código {:?})",
                status.code()
            )));
        }

        let raw = std::fs::read_to_string(&out_json).map_err(|e| {
            TranscriberError::Inference(format!(
                "no se pudo leer la salida de crispasr en '{}': {e}",
                out_json.display()
            ))
        })?;
        let _ = std::fs::remove_file(&out_json);

        let parsed: CrispasrOutput = serde_json::from_str(&raw)
            .map_err(|e| TranscriberError::Inference(format!("salida JSON de crispasr inválida: {e}")))?;

        let segments = parsed
            .transcription
            .into_iter()
            .map(|s| TranscriptSegment {
                start_ms: s.offsets.from,
                end_ms: s.offsets.to,
                text: s.text,
            })
            .collect();

        Ok(Transcript { segments })
    }
}

/// Lo que se sabe de la GPU tras leer el stderr de crispasr. Cualquier
/// degradación invalida la corrida aunque crispasr termine con código 0:
/// la transcripción se usa para fichas clínicas, y un resultado obtenido por
/// un camino distinto al validado (otro encoder, CPU) no se acepta en
/// silencio.
#[derive(Default, Debug)]
struct GpuHealth {
    cuda_device_found: bool,
    /// Primeras líneas distintas que delataron el problema (sin repetir:
    /// crispasr reintenta por tramo y la misma línea sale cientos de veces).
    vram: Vec<String>,
    cpu_fallback: Vec<String>,
    vram_hits: usize,
    cpu_hits: usize,
}

const MAX_EJEMPLOS: usize = 3;

impl GpuHealth {
    fn observe(&mut self, line: &str) {
        let l = line.to_ascii_lowercase();
        if l.contains("ggml_cuda_init: found") && !l.contains("found 0 ") {
            self.cuda_device_found = true;
        }
        if l.contains("cudamalloc failed")
            || l.contains("out of memory")
            || l.contains("vram oom")
            || (l.contains("failed to allocate") && !l.contains("falling back"))
        {
            self.vram_hits += 1;
            push_unique(&mut self.vram, line);
        } else if l.contains("falling back to cpu")
            || l.contains("gpu backend init failed")
            || l.contains("gpu backend unavailable")
            || l.contains("no matching gpu device")
        {
            self.cpu_hits += 1;
            push_unique(&mut self.cpu_fallback, line);
        }
    }

    fn problem(&self) -> Option<String> {
        if self.vram_hits > 0 {
            return Some(format!(
                "VRAM insuficiente: crispasr no pudo reservar memoria en la GPU {} vez/veces. \
                 Cierra lo que esté usando la GPU (navegador, juegos, otro modelo cargado como \
                 llama-server) y vuelve a correr.\n  {}",
                self.vram_hits,
                self.vram.join("\n  ")
            ));
        }
        if self.cpu_hits > 0 {
            return Some(format!(
                "parte del trabajo cayó a CPU ({} vez/veces): el resultado no salió por el camino \
                 validado.\n  {}",
                self.cpu_hits,
                self.cpu_fallback.join("\n  ")
            ));
        }
        if !self.cuda_device_found {
            return Some(
                "crispasr no detectó ninguna GPU CUDA: habría transcrito en CPU. Revisa el driver \
                 de NVIDIA (`nvidia-smi`)."
                    .to_string(),
            );
        }
        None
    }
}

fn push_unique(list: &mut Vec<String>, line: &str) {
    let line = line.trim().to_string();
    if list.len() < MAX_EJEMPLOS && !list.contains(&line) {
        list.push(line);
    }
}

/// Uso de VRAM y procesos de cómputo según `nvidia-smi`, para que el aviso
/// diga quién se está comiendo la GPU. `None` si `nvidia-smi` no está o
/// falla: es diagnóstico, nunca motivo para abortar por sí solo.
fn gpu_snapshot() -> Option<String> {
    let mem = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let mem = String::from_utf8_lossy(&mem.stdout);
    let (used, total) = mem.lines().next()?.split_once(',')?;
    let mut msg = format!("{} de {} MiB en uso", used.trim(), total.trim());

    if let Some(apps) = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let apps: Vec<String> = String::from_utf8_lossy(&apps.stdout)
            .lines()
            .filter_map(|l| {
                // `process_name` trae la línea de comandos entera (Chrome,
                // Discord: kilómetros de flags, con comas adentro): se toman
                // pid y MiB de las puntas y el nombre corto de /proc.
                let (pid, rest) = l.split_once(',')?;
                let (cmdline, mib) = rest.rsplit_once(',')?;
                let pid = pid.trim();
                let name = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                    .map(|c| c.trim().to_string())
                    .unwrap_or_else(|_| {
                        let exe = cmdline.split_whitespace().next().unwrap_or("?");
                        exe.rsplit('/').next().unwrap_or(exe).to_string()
                    });
                Some(format!("{name} (pid {pid}, {} MiB)", mib.trim()))
            })
            .collect();
        if !apps.is_empty() {
            msg.push_str(&format!("; procesos de cómputo: {}", apps.join(", ")));
        }
    }
    Some(msg)
}

#[cfg(test)]
mod tests {
    use super::GpuHealth;

    fn health(lines: &[&str]) -> GpuHealth {
        let mut h = GpuHealth::default();
        for l in lines {
            h.observe(l);
        }
        h
    }

    const CUDA_OK: &str = "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7830 MiB):";

    #[test]
    fn corrida_limpia_no_tiene_problema() {
        let h = health(&[
            CUDA_OK,
            "ggml_cuda_init: CUDA minor version mismatch — compiled 13.0, runtime 13.3. This usually works but may cause subtle issues.",
            "whisper_backend_init_gpu: using CUDA0 backend",
            "ggml_backend_cuda_graph_compute: CUDA graph warmup complete",
        ]);
        assert!(h.problem().is_none(), "{:?}", h.problem());
    }

    #[test]
    fn el_oom_de_v0830_se_detecta_aunque_crispasr_siga() {
        let oom = [
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4481.73 MiB on device 0: cudaMalloc failed: out of memory",
            "alloc_tensor_range: failed to allocate CUDA0 buffer of size 4699439360",
            "crisp_audio: failed to allocate 4.70 GB on 'CUDA0' for the model weights.",
        ];
        let mut lines = vec![CUDA_OK];
        for _ in 0..200 {
            lines.extend(oom);
        }
        let h = health(&lines);
        let p = h.problem().expect("debe avisar");
        assert!(p.starts_with("VRAM insuficiente"), "{p}");
        // Ejemplos acotados y sin repetir, aunque la línea salga 600 veces.
        assert_eq!(h.vram.len(), 3);
    }

    #[test]
    fn caida_a_cpu_se_detecta() {
        let h = health(&[CUDA_OK, "  encoder: graph path failed, falling back to CPU loops"]);
        assert!(h.problem().expect("debe avisar").contains("cayó a CPU"));
    }

    #[test]
    fn sin_gpu_detectada_es_problema() {
        let h = health(&["load_backend: loaded CPU backend from libggml-cpu.so"]);
        assert!(h.problem().expect("debe avisar").contains("ninguna GPU"));
        let h = health(&["ggml_cuda_init: found 0 CUDA devices"]);
        assert!(h.problem().is_some());
    }
}
