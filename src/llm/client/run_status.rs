use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use super::streaming::MULTI_PROGRESS;
use super::usage_tracker;

/// Live global status row pinned at the bottom of the indicatif feed.
///
/// Shows: bar, stage, call count, est tokens, cost, tool/retry/trunc counters,
/// elapsed. Created once; per-call bars use `insert_before(&GLOBAL_BAR, …)`
/// so this row stays at the bottom.
///
/// When `--cost-and-usage` is on and captures exist, cost/tokens reflect real
/// reported values. Otherwise we show `~`-prefixed estimates from chars/4
/// accumulated at the agent_executor funnel.
pub struct RunStatus {
    bar: ProgressBar,
    items_done: AtomicU64,
    total_hint: AtomicU64,
    llm_calls: AtomicU64,
    tool_calls: AtomicU64,
    retries: AtomicU64,
    truncations: AtomicU64,
    est_input_tokens: AtomicU64,
    est_output_tokens: AtomicU64,
    stage: Mutex<String>,
    has_real_data: AtomicBool,
}

static RUN_STATUS: LazyLock<RunStatus> = LazyLock::new(|| {
    let bar = MULTI_PROGRESS.add(ProgressBar::new(0));
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar:.cyan/blue} {pos}/{len} · {msg} · {elapsed_precise}")
            .expect("valid progress template"),
    );
    bar.enable_steady_tick(Duration::from_millis(500));
    bar.set_message("initialising…");
    RunStatus {
        bar,
        items_done: AtomicU64::new(0),
        total_hint: AtomicU64::new(0),
        llm_calls: AtomicU64::new(0),
        tool_calls: AtomicU64::new(0),
        retries: AtomicU64::new(0),
        truncations: AtomicU64::new(0),
        est_input_tokens: AtomicU64::new(0),
        est_output_tokens: AtomicU64::new(0),
        stage: Mutex::new(String::new()),
        has_real_data: AtomicBool::new(false),
    }
});

pub fn global() -> &'static RunStatus {
    &RUN_STATUS
}

/// Return the underlying `ProgressBar` so callers can `insert_before` their
/// own bars above the global row.
pub fn global_bar() -> ProgressBar {
    RUN_STATUS.bar.clone()
}

impl RunStatus {
    /// Mark a work item done (cache hit or real call). Advances pos.
    pub fn item_resolved(&self) {
        self.items_done.fetch_add(1, Ordering::Relaxed);
        self.update_message();
    }

    /// An LLM call completed. Advances pos and accumulates estimated tokens
    /// from chars (chars/4). pass `(0, 0)` for cache hits that contributed
    /// no token activity.
    pub fn call_completed(&self, input_chars: usize, output_chars: usize) {
        self.llm_calls.fetch_add(1, Ordering::Relaxed);
        self.items_done.fetch_add(1, Ordering::Relaxed);
        self.est_input_tokens
            .fetch_add((input_chars / 4).max(1) as u64, Ordering::Relaxed);
        self.est_output_tokens
            .fetch_add((output_chars / 4).max(1) as u64, Ordering::Relaxed);
        self.update_message();
    }

    pub fn set_stage(&self, stage: &str) {
        *self.stage.lock().unwrap() = stage.to_string();
        self.update_message();
    }

    /// Grow the progress-bar total hint by `n`.
    pub fn add_total(&self, n: usize) {
        self.total_hint.fetch_add(n as u64, Ordering::Relaxed);
    }

    /// Mark the bar's length at its current total, called at stable points
    /// (e.g., at fan-out launch).
    pub fn commit_total(&self) {
        let t = self.total_hint.load(Ordering::Relaxed);
        self.bar.set_length(t);
    }

    pub fn inc_retry(&self) {
        self.retries.fetch_add(1, Ordering::Relaxed);
        self.update_message();
    }

    pub fn inc_truncation(&self) {
        self.truncations.fetch_add(1, Ordering::Relaxed);
        self.update_message();
    }

    pub fn inc_tool_calls(&self, n: u64) {
        if n > 0 {
            self.tool_calls.fetch_add(n, Ordering::Relaxed);
            self.update_message();
        }
    }

    /// Signal that real cost/usage data is available (called on first record).
    pub fn mark_has_real_data(&self) {
        self.has_real_data.store(true, Ordering::Relaxed);
    }

    /// Finish the bar with a final summary.
    pub fn finish(&self) {
        self.update_message();
        let d = self.items_done.load(Ordering::Relaxed);
        let c = self.llm_calls.load(Ordering::Relaxed);
        let t = self.total_hint.load(Ordering::Relaxed);
        let stage = self.stage.lock().unwrap().clone();
        let msg = if stage.is_empty() {
            format!("✓ done · {d}/{t} items · {c} calls")
        } else {
            format!("✓ {stage} done · {d}/{t} items · {c} calls")
        };
        self.bar.finish_with_message(msg);
    }

    fn update_message(&self) {
        let items = self.items_done.load(Ordering::Relaxed);
        let total = self.total_hint.load(Ordering::Relaxed);
        let calls = self.llm_calls.load(Ordering::Relaxed);
        let stage = self.stage.lock().unwrap().clone();

        let est_in = self.est_input_tokens.load(Ordering::Relaxed);
        let est_out = self.est_output_tokens.load(Ordering::Relaxed);
        let tools = self.tool_calls.load(Ordering::Relaxed);
        let retries = self.retries.load(Ordering::Relaxed);
        let truncs = self.truncations.load(Ordering::Relaxed);

        // Sync bar position/length.
        self.bar.set_position(items);
        self.bar.set_length(total);

        // Build the message tail.
        let mut parts: Vec<String> = Vec::new();

        if !stage.is_empty() {
            parts.push(stage);
        }

        parts.push(format!("{calls} call"));

        // Tokens: when real data exists show reported; otherwise ~ estimates.
        if self.has_real_data.load(Ordering::Relaxed) {
            if let Some(snap) = usage_tracker::snapshot_totals() {
                parts.push(format!("↑{} ↓{} tok", snap.0, snap.1));
                if snap.3 > 0.0 {
                    parts.push(format!("${:.4}", snap.3));
                }
            } else {
                parts.push(format!("~{} ~{} tok", Self::format_num(est_in), Self::format_num(est_out)));
            }
        } else {
            parts.push(format!(
                "~{} ~{} tok",
                Self::format_num(est_in),
                Self::format_num(est_out)
            ));
        }

        if tools > 0 {
            parts.push(format!("tools {tools}"));
        }
        if retries > 0 {
            parts.push(format!("retries {retries}"));
        }
        if truncs > 0 {
            parts.push(format!("trunc {truncs}"));
        }

        self.bar.set_message(parts.join(" · "));
    }

    fn format_num(n: u64) -> String {
        if n >= 1_000_000 {
            format!("{:.1}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}K", n as f64 / 1_000.0)
        } else {
            n.to_string()
        }
    }
}