//! Cost & usage tracking data model and global accumulator.
//!
//! When `llm.cost_and_usage` is enabled, every provider call records a
//! [`UsageRecord`] into the process-wide [`UsageTracker`]. Records merge
//! rig-reported token usage with LiteLLM-reported cost/metadata captured
//! from the raw HTTP response (see [`super::usage_capture`]).

use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::sync::{LazyLock, Mutex};

/// Source of a record's cost figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    /// Cost reported by LiteLLM (response body `usage.cost` or
    /// `x-litellm-response-cost` header).
    #[default]
    LiteLlm,
    /// Cost estimated locally from the pricing table.
    Estimated,
    /// No cost data available.
    None,
}

/// Per-call usage record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UsageRecord {
    pub execution_id: Option<String>,
    pub timestamp: DateTime<chrono::Utc>,
    pub duration_ms: u64,
    pub model: String,
    pub base_url: String,
    pub provider: String,
    pub agent_tag: Option<String>,
    pub cache_scope: Option<String>,
    pub stream: bool,
    pub success: bool,
    pub error: Option<String>,
    // Request params
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub prompt_chars: usize,
    pub response_chars: usize,
    // Token usage
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub tool_use_prompt_tokens: u64,
    pub reasoning_tokens: u64,
    // Cost
    pub cost_usd: Option<f64>,
    pub cost_source: CostSource,
    pub cost_input_usd: Option<f64>,
    pub cost_output_usd: Option<f64>,
    pub cost_cache_read_usd: Option<f64>,
    pub cost_cache_creation_usd: Option<f64>,
    pub cost_reasoning_usd: Option<f64>,
    pub cost_tool_usage_usd: Option<f64>,
    // LiteLLM metadata (x-litellm-* headers)
    pub litellm_call_id: Option<String>,
    pub litellm_model_id: Option<String>,
    pub litellm_model_name: Option<String>,
    pub litellm_cache_key: Option<String>,
    pub litellm_model_api_base: Option<String>,
    pub litellm_version: Option<String>,
    pub litellm_key_spend: Option<f64>,
    pub litellm_response_duration_ms: Option<u64>,
    pub litellm_overhead_duration_ms: Option<u64>,
    pub litellm_callback_duration_ms: Option<u64>,
    // Completion metadata
    pub finish_reason: Option<String>,
    pub tool_calls_count: Option<u32>,
    pub retry_count: u32,
}

impl Default for UsageRecord {
    fn default() -> Self {
        Self {
            execution_id: None,
            timestamp: chrono::Utc::now(),
            duration_ms: 0,
            model: String::new(),
            base_url: String::new(),
            provider: String::new(),
            agent_tag: None,
            cache_scope: None,
            stream: false,
            success: true,
            error: None,
            temperature: None,
            max_tokens: None,
            prompt_chars: 0,
            response_chars: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
            cost_usd: None,
            cost_source: CostSource::None,
            cost_input_usd: None,
            cost_output_usd: None,
            cost_cache_read_usd: None,
            cost_cache_creation_usd: None,
            cost_reasoning_usd: None,
            cost_tool_usage_usd: None,
            litellm_call_id: None,
            litellm_model_id: None,
            litellm_model_name: None,
            litellm_cache_key: None,
            litellm_model_api_base: None,
            litellm_version: None,
            litellm_key_spend: None,
            litellm_response_duration_ms: None,
            litellm_overhead_duration_ms: None,
            litellm_callback_duration_ms: None,
            finish_reason: None,
            tool_calls_count: None,
            retry_count: 0,
        }
    }
}

impl UsageRecord {
    /// Estimate cost from the configured pricing table when no provider cost
    /// was reported.
    pub fn estimate_cost(
        &mut self,
        pricing: Option<&crate::config::ModelPricing>,
    ) {
        if self.cost_usd.is_some() || self.cost_source == CostSource::LiteLlm {
            return;
        }
        let Some(p) = pricing else {
            return;
        };
        let input_rate = p.input_per_1k;
        let cache_read_rate = if p.cache_read_per_1k > 0.0 {
            p.cache_read_per_1k
        } else {
            input_rate
        };
        let cache_create_rate = if p.cache_creation_per_1k > 0.0 {
            p.cache_creation_per_1k
        } else {
            input_rate
        };
        let standard_input =
            (self.input_tokens as f64 - self.cache_creation_input_tokens as f64
                - self.cached_input_tokens as f64)
                .max(0.0);
        let cost = standard_input / 1000.0 * input_rate
            + self.output_tokens as f64 / 1000.0 * p.output_per_1k
            + self.cache_creation_input_tokens as f64 / 1000.0 * cache_create_rate
            + self.cached_input_tokens as f64 / 1000.0 * cache_read_rate;
        self.cost_usd = Some(cost);
        self.cost_source = CostSource::Estimated;
        self.cost_input_usd = Some(
            (standard_input / 1000.0 * input_rate)
                + self.cache_creation_input_tokens as f64 / 1000.0 * cache_create_rate
                + self.cached_input_tokens as f64 / 1000.0 * cache_read_rate,
        );
        self.cost_output_usd = Some(self.output_tokens as f64 / 1000.0 * p.output_per_1k);
    }
}

/// Per-model aggregate for the execution report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelUsageSummary {
    pub model: String,
    pub calls: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub cost_usd: f64,
    pub avg_duration_ms: f64,
}

/// Aggregated per-execution usage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionUsageReport {
    pub execution_id: String,
    pub started_at: DateTime<chrono::Utc>,
    pub finished_at: DateTime<chrono::Utc>,
    pub total_duration_ms: u64,
    pub project_name: String,
    pub base_url: String,
    pub provider: String,
    pub total_calls: usize,
    pub failed_calls: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_tokens: u64,
    pub total_cached_input_tokens: u64,
    pub total_cache_creation_input_tokens: u64,
    pub total_reasoning_tokens: u64,
    pub total_cost_usd: f64,
    pub litellm_reported_calls: usize,
    pub estimated_calls: usize,
    pub models: Vec<ModelUsageSummary>,
    pub calls: Vec<UsageRecord>,
}

/// Process-wide usage accumulator.
#[derive(Default)]
pub struct UsageTracker {
    inner: Mutex<TrackerState>,
}

#[derive(Default)]
struct TrackerState {
    execution_id: String,
    started_at: Option<DateTime<chrono::Utc>>,
    records: Vec<UsageRecord>,
    project_name: String,
    base_url: String,
    provider: String,
}

static USAGE_TRACKER: LazyLock<UsageTracker> = LazyLock::new(UsageTracker::default);

impl UsageTracker {
    /// Global tracker singleton.
    pub fn global() -> &'static UsageTracker {
        &USAGE_TRACKER
    }

    /// Begin a new execution, discarding any previous records.
    pub fn begin_execution(
        &self,
        execution_id: String,
        project_name: String,
        base_url: String,
        provider: String,
    ) {
        let mut st = self.inner.lock().unwrap();
        *st = TrackerState {
            execution_id,
            started_at: Some(chrono::Utc::now()),
            project_name,
            base_url,
            provider,
            records: Vec::new(),
        };
    }

    /// Record one call.
    pub fn record(&self, record: UsageRecord) {
        self.inner.lock().unwrap().records.push(record);
    }

    /// The execution id currently in progress, if any.
    pub fn current_execution_id(&self) -> Option<String> {
        let id = self.inner.lock().unwrap().execution_id.clone();
        (!id.is_empty()).then_some(id)
    }

    /// Build the aggregated execution report, marking the finish time.
    pub fn build_report(&self) -> Option<ExecutionUsageReport> {
        let st = self.inner.lock().unwrap();
        let started_at = st.started_at?;
        let finished_at = chrono::Utc::now();
        let mut models: std::collections::HashMap<String, ModelUsageSummary> =
            std::collections::HashMap::new();
        let mut totals = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        let mut total_cost = 0.0f64;
        let mut failed = 0usize;
        let mut litellm_calls = 0usize;
        let mut estimated_calls = 0usize;
        for r in &st.records {
            let m = models.entry(r.model.clone()).or_default();
            m.model = r.model.clone();
            m.calls += 1;
            m.input_tokens += r.input_tokens;
            m.output_tokens += r.output_tokens;
            m.total_tokens += r.total_tokens;
            m.cached_input_tokens += r.cached_input_tokens;
            m.cache_creation_input_tokens += r.cache_creation_input_tokens;
            m.reasoning_tokens += r.reasoning_tokens;
            m.cost_usd += r.cost_usd.unwrap_or(0.0);
            m.avg_duration_ms += r.duration_ms as f64;
            totals.0 += r.input_tokens;
            totals.1 += r.output_tokens;
            totals.2 += r.total_tokens;
            totals.3 += r.cached_input_tokens;
            totals.4 += r.cache_creation_input_tokens;
            totals.5 += r.reasoning_tokens;
            total_cost += r.cost_usd.unwrap_or(0.0);
            if !r.success {
                failed += 1;
            }
            match r.cost_source {
                CostSource::LiteLlm => litellm_calls += 1,
                CostSource::Estimated => estimated_calls += 1,
                CostSource::None => {}
            }
        }
        let mut models: Vec<ModelUsageSummary> = models.into_values().collect();
        for m in &mut models {
            if m.calls > 0 {
                m.avg_duration_ms /= m.calls as f64;
            }
        }
        models.sort_by(|a, b| b.cost_usd.partial_cmp(&a.cost_usd).unwrap_or(std::cmp::Ordering::Equal));
        Some(ExecutionUsageReport {
            execution_id: st.execution_id.clone(),
            started_at,
            finished_at,
            total_duration_ms: (finished_at - started_at).num_milliseconds() as u64,
            project_name: st.project_name.clone(),
            base_url: st.base_url.clone(),
            provider: st.provider.clone(),
            total_calls: st.records.len(),
            failed_calls: failed,
            total_input_tokens: totals.0,
            total_output_tokens: totals.1,
            total_tokens: totals.2,
            total_cached_input_tokens: totals.3,
            total_cache_creation_input_tokens: totals.4,
            total_reasoning_tokens: totals.5,
            total_cost_usd: total_cost,
            litellm_reported_calls: litellm_calls,
            estimated_calls,
            models,
            calls: st.records.clone(),
        })
    }
}

/// Render the markdown cost & usage report for one execution.
pub fn render_markdown_report(report: &ExecutionUsageReport) -> String {
    let mut s = String::new();
    s.push_str("# Cost & Usage Report\n\n");
    s.push_str(&format!(
        "- **Execution**: `{}`\n- **Started**: {} (UTC)\n- **Finished**: {}\n- **Wall duration**: {:.2}s\n",
        report.execution_id,
        report.started_at.format("%Y-%m-%d %H:%M:%S UTC"),
        report.finished_at.format("%Y-%m-%d %H:%M:%S UTC"),
        report.total_duration_ms as f64 / 1000.0,
    ));
    s.push_str(&format!(
        "- **Project**: {}\n- **Provider**: {}\n- **Base URL**: `{}`\n\n",
        if report.project_name.is_empty() { "-" } else { &report.project_name },
        report.provider,
        report.base_url,
    ));

    s.push_str("## Summary\n\n");
    s.push_str("| Metric | Value |\n|---|---|\n");
    s.push_str(&format!("| Total calls | {} ({} failed) |\n", report.calls.len(), report.failed_calls));
    s.push_str(&format!("| Total cost | ${:.4} |\n", report.total_cost_usd));
    s.push_str(&format!("| Input tokens | {} |\n", report.total_input_tokens));
    s.push_str(&format!("| Output tokens | {} |\n", report.total_output_tokens));
    s.push_str(&format!("| Cached input tokens (read) | {} |\n", report.total_cached_input_tokens));
    s.push_str(&format!("| Cache creation tokens | {} |\n", report.total_cache_creation_input_tokens));
    s.push_str(&format!("| Reasoning tokens | {} |\n", report.total_reasoning_tokens));
    s.push_str(&format!("| Total tokens | {} |\n", report.total_tokens));
    s.push_str(&format!(
        "| Cost source | LiteLLM: {} calls, Estimated: {} calls |\n\n",
        report.litellm_reported_calls, report.estimated_calls,
    ));

    if !report.models.is_empty() {
        s.push_str("## Per-Model Breakdown\n\n");
        s.push_str("| Model | Calls | In tok | Out tok | Cached | Cache write | Reasoning | Cost USD | Avg ms |\n");
        s.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for m in &report.models {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | ${:.4} | {:.0} |\n",
                m.model, m.calls, m.input_tokens, m.output_tokens,
                m.cached_input_tokens, m.cache_creation_input_tokens,
                m.reasoning_tokens, m.cost_usd, m.avg_duration_ms,
            ));
        }
        s.push('\n');
    }

    if !report.calls.is_empty() {
        s.push_str("## Per-Call Detail\n\n");
        s.push_str("| Time | Agent | Model | In | Out | Cached | Reasoning | Cost USD | Source | ms | Stream | OK | Finish |\n");
        s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for c in &report.calls {
            let source = match c.cost_source {
                CostSource::LiteLlm => "litellm",
                CostSource::Estimated => "est",
                CostSource::None => "-",
            };
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                c.timestamp.format("%H:%M:%S%.3f"),
                c.agent_tag.as_deref().unwrap_or("-"),
                c.model,
                c.input_tokens,
                c.output_tokens,
                c.cached_input_tokens,
                c.reasoning_tokens,
                c.cost_usd.map(|v| format!("${:.4}", v)).unwrap_or_else(|| "-".into()),
                source,
                c.duration_ms,
                c.stream,
                if c.success { "✓" } else { "✗" },
                c.finish_reason.as_deref().unwrap_or("-"),
            ));
        }
        s.push('\n');

        let litellm_meta: Vec<&UsageRecord> = report
            .calls
            .iter()
            .filter(|c| c.litellm_call_id.is_some() || c.litellm_model_id.is_some())
            .collect();
        if !litellm_meta.is_empty() {
            s.push_str("## LiteLLM Metadata\n\n");
            s.push_str("| Call ID | Model ID | Model Name | API Base | Resp ms | Overhead ms | Key Spend |\n");
            s.push_str("|---|---|---|---|---|---|---|\n");
            for c in &litellm_meta {
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} |\n",
                    c.litellm_call_id.as_deref().unwrap_or("-"),
                    c.litellm_model_id.as_deref().unwrap_or("-"),
                    c.litellm_model_name.as_deref().unwrap_or("-"),
                    c.litellm_model_api_base.as_deref().unwrap_or("-"),
                    opt_u64(&c.litellm_response_duration_ms),
                    opt_u64(&c.litellm_overhead_duration_ms),
                    c.litellm_key_spend.map(|v| format!("${:.4}", v)).unwrap_or_else(|| "-".into()),
                ));
            }
            s.push('\n');
        }
    }

    s
}

fn opt_u64(v: &Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "-".into())
}

/// Persist the execution report to `<internal>/cost_usage/<id>.json` and
/// append per-call lines to `<internal>/cost_usage/calls.jsonl`.
pub fn persist(
    internal_path: &std::path::Path,
    report: &ExecutionUsageReport,
) -> anyhow::Result<()> {
    let dir = internal_path.join("cost_usage");
    std::fs::create_dir_all(&dir)?;
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(dir.join(format!("{}.json", report.execution_id)), json)?;
    let mut jsonl = String::new();
    for c in &report.calls {
        jsonl.push_str(&serde_json::to_string(c)?);
        jsonl.push('\n');
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("calls.jsonl"))?;
    f.write_all(jsonl.as_bytes())?;
    Ok(())
}

/// Drain HTTP captures for the current task and, when `cost_and_usage` is
/// enabled, build and record a [`UsageRecord`] for each. Always drains (to
/// keep the capture sink bounded) but only records when enabled.
#[allow(clippy::too_many_arguments)]
pub fn record_from_captures(
    config: &crate::config::LLMConfig,
    model: &str,
    provider: &str,
    agent_tag: Option<String>,
    duration_ms: u64,
    _stream: bool,
    prompt_chars: usize,
    response_chars: usize,
    success: bool,
    error: Option<String>,
) -> usize {
    let captures = crate::llm::client::usage_capture::take_captures_for_current_task();
    if !config.cost_and_usage {
        return 0;
    }
    let n_captures = captures.len();
    if n_captures == 0 {
        return 0;
    }
    let execution_id = UsageTracker::global().current_execution_id();
    // One funnel call may cover several provider calls (ReAct turns,
    // retries). Wall time and prompt/response sizes are funnel-scoped, so
    // they are attributed to the first capture only, and the wall time is
    // additionally spread evenly to keep per-model averages honest.
    let even_duration_ms = duration_ms / n_captures as u64;
    let mut n = 0;
    for (idx, cap) in captures.into_iter().enumerate() {
        let mut rec = crate::llm::client::usage_capture::extract_usage_record(&cap);
        if rec.model.is_empty() {
            rec.model = model.to_string();
        }
        rec.provider = provider.to_string();
        rec.execution_id = execution_id.clone();
        rec.agent_tag = agent_tag.clone();
        rec.prompt_chars = if idx == 0 { prompt_chars } else { 0 };
        rec.response_chars = if idx == 0 { response_chars } else { 0 };
        rec.duration_ms = if idx == 0 {
            duration_ms.saturating_sub(even_duration_ms * (n_captures as u64 - 1))
        } else {
            even_duration_ms
        };
        if !success {
            rec.success = false;
            rec.error = error.clone();
        }
        let pricing = config.pricing_table.get(&rec.model);
        rec.estimate_cost(pricing);
        eprintln!(
            "💰 {} · {}↑ {}↓ tok · {} · {}ms",
            if rec.model.is_empty() { "model" } else { &rec.model },
            rec.input_tokens,
            rec.output_tokens,
            rec.cost_usd
                .map(|c| format!("${:.4}", c))
                .unwrap_or_else(|| "cost n/a".into()),
            rec.duration_ms,
        );
        UsageTracker::global().record(rec);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_cost_uses_pricing_table() {
        let mut r = UsageRecord {
            input_tokens: 2000,
            output_tokens: 1000,
            cached_input_tokens: 500,
            ..Default::default()
        };
        r.estimate_cost(Some(&crate::config::ModelPricing {
            input_per_1k: 0.001,
            output_per_1k: 0.002,
            cache_read_per_1k: 0.0001,
            cache_creation_per_1k: 0.0,
        }));
        assert_eq!(r.cost_source, CostSource::Estimated);
        // standard input = 2000 - 500 = 1500 => 1.5k * 0.001 = 0.0015
        // output 1k * 0.002 = 0.002 ; cache read 0.5k * 0.0001 = 0.00005
        let cost = r.cost_usd.unwrap();
        assert!((cost - 0.00355).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn report_aggregates() {
        let t = UsageTracker::default();
        t.begin_execution("test".into(), "p".into(), "http://x".into(), "openai".into());
        t.record(UsageRecord {
            model: "m".into(),
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            cost_usd: Some(0.01),
            cost_source: CostSource::LiteLlm,
            ..Default::default()
        });
        let rep = t.build_report().unwrap();
        assert_eq!(rep.calls.len(), 1);
        assert_eq!(rep.failed_calls, 0);
        assert_eq!(rep.total_input_tokens, 10);
        assert!((rep.total_cost_usd - 0.01).abs() < 1e-9);
        assert_eq!(rep.models[0].calls, 1);
        let md = render_markdown_report(&rep);
        assert!(md.contains("Cost & Usage Report"));
        assert!(md.contains("Total cost | $0.0100"));
    }
}
