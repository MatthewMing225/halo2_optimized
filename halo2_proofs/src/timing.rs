use instant::Instant;
use log::{debug, info};
use std::borrow::Cow;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn perf_writer() -> &'static Mutex<Option<std::fs::File>> {
    static PERF_WRITER: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    PERF_WRITER.get_or_init(|| {
        let file = std::env::var("HALO2_PERF_LOG")
            .ok()
            .and_then(|path| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|err| {
                        log::warn!("failed to open HALO2_PERF_LOG file: {:?}", err);
                        err
                    })
                    .ok()
            });
        Mutex::new(file)
    })
}

fn write_event(label: &str, stage: &str, duration: Duration) {
    if let Ok(mut guard) = perf_writer().lock() {
        if let Some(file) = guard.as_mut() {
            let _ = writeln!(
                file,
                "{},{},{},{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|ts| ts.as_nanos())
                    .unwrap_or_default(),
                label,
                stage,
                duration.as_nanos()
            );
        }
    }
}

/// Helper guard that logs the duration of a scope when dropped.
#[must_use]
#[derive(Debug)]
pub struct PerfGuard {
    label: Cow<'static, str>,
    start: Instant,
    emit_start: bool,
}

impl PerfGuard {
    /// Creates a new guard and immediately logs the start of the scope.
    pub fn new(label: impl Into<Cow<'static, str>>) -> Self {
        let label = label.into();
        debug!("perf.start: {}", &label);
        write_event(&label, "start", Duration::ZERO);
        Self {
            label,
            start: Instant::now(),
            emit_start: true,
        }
    }

    /// Creates a guard without emitting an initial start event.
    pub fn new_silent(label: impl Into<Cow<'static, str>>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
            emit_start: false,
        }
    }

    /// Explicitly record the elapsed time without dropping the guard.
    pub fn checkpoint(&mut self, message: &str) {
        let elapsed = self.start.elapsed();
        info!("perf.checkpoint: {} {} {:?}", self.label, message, elapsed);
        write_event(
            &format!("{}:{}", self.label, message),
            "checkpoint",
            elapsed,
        );
    }
}

impl Drop for PerfGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if self.emit_start {
            info!("perf.end: {} took {:?}", self.label, elapsed);
            write_event(&self.label, "end", elapsed);
        } else {
            info!("perf: {} took {:?}", self.label, elapsed);
            write_event(&self.label, "span", elapsed);
        }
    }
}
