//! Streaming inference helpers.
//!
//! Drains rig streaming responses into a complete string — deltas are
//! accumulated, JSON is only ever parsed from the finished text — while
//! reporting live progress on stderr via `indicatif`. Also parses
//! OpenAI-compatible SSE chunks for the reqwest HTTP fallback path.

use anyhow::{Context, Result};
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rig_core::agent::{MultiTurnStreamItem, StreamingResult};
use rig_core::streaming::{StreamedAssistantContent, StreamedUserContent};
use serde_json::Value;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::LazyLock;
use std::time::Duration;

/// Shared `MultiProgress` manager so concurrent provider calls each get their
/// own stderr line without overwriting one another.
static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Rough character-to-token ratio used for live tok/s estimates.
const CHARS_PER_TOKEN: usize = 4;

/// Number of physical lines kept in the rolling echo window.
const ECHO_WINDOW_LINES: usize = 16;

/// Fallback terminal width when stderr is not a TTY.
const ECHO_FALLBACK_WIDTH: usize = 100;

/// Which of the streamed channels a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EchoKind {
    Think,
    Output,
    Tool,
}

impl EchoKind {
    fn prefix(self) -> &'static str {
        match self {
            EchoKind::Think => "think│ ",
            EchoKind::Output => "out  │ ",
            EchoKind::Tool => "tool │ ",
        }
    }
}

/// Wrap `line` into chunks of at most `budget` chars, so a long streamed line
/// occupies multiple terminal lines. Multibyte-safe: splits on char boundaries.
fn wrap_line(line: &str, budget: usize) -> Vec<String> {
    if budget == 0 || line.is_empty() {
        return vec![line.to_string()];
    }
    let mut chunks = Vec::new();
    let mut rem = line;
    while !rem.is_empty() {
        let mut it = rem.chars();
        let chunk: String = it.by_ref().take(budget).collect();
        chunks.push(chunk);
        rem = it.as_str();
    }
    chunks
}

/// Rolling tail window of the most recent streamed lines.
///
/// Pure rendering state: deltas are appended, physical lines are flushed on
/// newlines and on channel changes, and only the newest `cap` lines survive.
/// Partial lines are kept as the "current" line so the window can be redrawn
/// on every delta.
struct RollingTail {
    lines: VecDeque<(EchoKind, String)>,
    current: Option<(EchoKind, String)>,
    cap: usize,
}

impl RollingTail {
    fn new(cap: usize) -> Self {
        Self {
            lines: VecDeque::new(),
            current: None,
            cap,
        }
    }

    fn push(&mut self, kind: EchoKind, text: &str) {
        match self.current.as_mut() {
            Some((current_kind, buf)) if *current_kind == kind => buf.push_str(text),
            _ => {
                self.flush_current();
                self.current = Some((kind, text.to_string()));
            }
        }

        // Flush every complete physical line; keep the trailing partial one.
        while let Some((line_kind, mut buf)) = self.current.take() {
            match buf.find('\n') {
                Some(idx) => {
                    let rest = buf.split_off(idx + 1);
                    buf.truncate(idx);
                    self.lines.push_back((line_kind, buf));
                    self.current = (!rest.is_empty()).then_some((line_kind, rest));
                }
                None => {
                    self.current = Some((line_kind, buf));
                    break;
                }
            }
        }

        self.trim();
    }

    /// Record a self-contained marker line (tool call / run / result).
    fn mark(&mut self, kind: EchoKind, text: &str) {
        self.flush_current();
        self.lines.push_back((kind, text.to_string()));
        self.trim();
    }

    fn flush_current(&mut self) {
        if let Some((kind, buf)) = self.current.take().filter(|(_, buf)| !buf.is_empty()) {
            self.lines.push_back((kind, buf));
        }
    }

    fn trim(&mut self) {
        while self.lines.len() > self.cap {
            self.lines.pop_front();
        }
    }

    /// Render the window as a newline-joined block. A long entry whose physical
    /// line exceeds `width` is wrapped across multiple terminal lines.
    fn render(&self, width: usize) -> String {
        let mut out = Vec::with_capacity(self.lines.len() + 1);
        for (kind, line) in self.lines.iter().chain(self.current.iter()) {
            let prefix = kind.prefix();
            let budget = width.saturating_sub(prefix.chars().count()).max(1);
            for chunk in wrap_line(line, budget) {
                out.push(format!("{prefix}{chunk}"));
            }
        }
        out.join("\n")
    }
}

/// Rolling echo pane drawn above the spinner: one `indicatif` bar whose
/// message is the multi-line tail window.
struct EchoPane {
    bar: ProgressBar,
    tail: RollingTail,
}

fn echo_width() -> usize {
    let cols = console::Term::stderr().size().1 as usize;
    if cols == 0 { ECHO_FALLBACK_WIDTH } else { cols }
}

/// Live stderr progress reporter for a single request.
///
/// Renders a spinner plus a status message on every delta, giving real-time
/// tok/s and internal state (tool calls, reasoning, etc.). When `echo` is on,
/// a rolling tail window of the streamed thinking and output is drawn above
/// the spinner. The bar is cleared when the request finishes or fails.
pub struct StreamProgress {
    bar: ProgressBar,
    echo: Option<EchoPane>,
    label: String,
}

impl StreamProgress {
    /// Create a new progress reporter.
    ///
    /// `label` is usually the model name; `status` is the initial human-readable
    /// state (e.g. "streaming" or "calling"). `echo` enables the rolling
    /// thinking/output window; leave it off for non-streaming calls that never
    /// produce deltas.
    pub fn new(label: &str, status: &str, echo: bool) -> Self {
        // The echo pane is added first so `MultiProgress` draws it above the
        // spinner line.
        let echo = echo.then(|| {
            let bar = MULTI_PROGRESS.add(ProgressBar::new_spinner());
            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{wide_msg}")
                    .expect("valid progress template"),
            );
            EchoPane {
                bar,
                tail: RollingTail::new(ECHO_WINDOW_LINES),
            }
        });
        let bar = MULTI_PROGRESS.add(ProgressBar::new_spinner());
        bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg} · {pos} tok · {per_sec} · {elapsed_precise}")
                .expect("valid progress template"),
        );
        bar.set_message(format!("{label} · {status}"));
        bar.enable_steady_tick(Duration::from_millis(120));
        Self {
            bar,
            echo,
            label: label.to_string(),
        }
    }

    /// Record more streamed characters; updates the bar immediately.
    /// `indicatif` throttles actual terminal draws internally, so we can call
    /// this on every delta without overwhelming stderr.
    pub fn advance(&self, chars: usize) {
        self.bar.inc((chars / CHARS_PER_TOKEN).max(1) as u64);
    }

    /// Update the status portion of the progress line (e.g. "tool: file_reader").
    pub fn set_status(&self, status: &str) {
        self.bar.set_message(format!("{} · {}", self.label, status));
    }

    /// Echo a reasoning (thinking) delta into the rolling window.
    pub fn push_think(&mut self, text: &str) {
        self.echo_push(EchoKind::Think, text);
    }

    /// Echo an assistant output delta into the rolling window.
    pub fn push_output(&mut self, text: &str) {
        self.echo_push(EchoKind::Output, text);
    }

    /// Record a self-contained tool marker line in the rolling window.
    pub fn mark_tool(&mut self, text: &str) {
        if let Some(pane) = self.echo.as_mut() {
            pane.tail.mark(EchoKind::Tool, text);
            redraw(&pane.bar, &pane.tail);
        }
    }

    fn echo_push(&mut self, kind: EchoKind, text: &str) {
        if let Some(pane) = self.echo.as_mut() {
            pane.tail.push(kind, text);
            redraw(&pane.bar, &pane.tail);
        }
    }

    /// Clear the rendered lines so they don't collide with subsequent output.
    pub fn finish(&self) {
        self.bar.finish_and_clear();
        if let Some(pane) = self.echo.as_ref() {
            pane.bar.finish_and_clear();
        }
    }
}

fn redraw(bar: &ProgressBar, tail: &RollingTail) {
    bar.set_message(tail.render(echo_width()));
}

/// Run an async operation with a live spinner that shows `label · status`.
///
/// The spinner is cleared as soon as the future resolves, regardless of success
/// or failure. This gives users feedback during non-streaming provider calls
/// that would otherwise appear frozen.
pub async fn with_spinner<T, F>(label: &str, status: &str, fut: F) -> T
where
    F: Future<Output = T>,
{
    let progress = StreamProgress::new(label, status, false);
    let result = fut.await;
    progress.finish();
    result
}

/// Drain a rig streaming prompt into its complete assistant text.
///
/// Aggregates assistant text deltas as they arrive and prefers the provider's
/// final aggregated response when it carries text. Nothing is parsed until the
/// stream completes: callers feed the returned string to their existing
/// parse-and-validate logic, which never sees partial JSON.
///
/// `label` (typically the model name) enables the stderr progress line; pass
/// `None` to disable it. `echo` additionally draws a rolling tail window of the
/// streamed thinking and output; it has no effect without a label and never
/// changes the returned text.
pub async fn drain_to_string<R>(
    mut stream: StreamingResult<R>,
    label: Option<&str>,
    echo: bool,
) -> Result<String> {
    let mut progress = label.map(|l| StreamProgress::new(l, "streaming", echo));
    let mut acc = String::new();
    let mut final_text: Option<String> = None;
    // Providers that emit both `ReasoningDelta` and a closing `Reasoning`
    // block would otherwise print the same thinking twice. The flag is scoped
    // to one provider call: it resets at each `CompletionCall` so a later turn
    // that emits only a `Reasoning` block still echoes.
    let mut saw_reasoning_delta = false;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                if let Some(p) = progress.as_mut() {
                    p.advance(text.text.chars().count());
                    p.push_output(&text.text);
                }
                acc.push_str(&text.text);
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            )) => {
                // Reasoning is not part of the answer but keeps the line alive
                // during long thinking phases.
                saw_reasoning_delta = true;
                if let Some(p) = progress.as_mut() {
                    p.advance(reasoning.chars().count());
                    p.set_status("thinking");
                    p.push_think(&reasoning);
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                if let Some(p) = progress.as_mut() {
                    let display = reasoning.display_text();
                    p.advance(display.chars().count());
                    p.set_status("thinking");
                    if !saw_reasoning_delta {
                        p.push_think(&display);
                    }
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            })) => {
                if let Some(p) = progress.as_mut() {
                    p.set_status(&format!("tool: {}", tool_call.function.name));
                    p.mark_tool(&format!("» tool: {}", tool_call.function.name));
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCallDelta { .. },
            )) => {
                // Tool-call argument deltas: keep the spinner alive but don't
                // count them as answer tokens.
                if let Some(p) = progress.as_ref() {
                    p.set_status("tool call");
                }
            }
            Ok(MultiTurnStreamItem::ToolExecutionStart { tool_call, .. }) => {
                if let Some(p) = progress.as_mut() {
                    p.set_status(&format!("running {}", tool_call.function.name));
                    p.mark_tool(&format!("» run: {}", tool_call.function.name));
                }
            }
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                if let Some(p) = progress.as_mut() {
                    p.set_status("tool result");
                    p.mark_tool("» result");
                }
            }
            Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                saw_reasoning_delta = false;
                if let Some(p) = progress.as_ref() {
                    p.set_status("provider call completed");
                }
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_text = Some(res.output);
            }
            // Unknown / Final assistant content / other provider-native items.
            Ok(_) => {}
            Err(err) => {
                if let Some(p) = progress.as_ref() {
                    p.finish();
                }
                // Keep the inner rig error in the anyhow chain so existing
                // HTTP-fallback triggers (ApiResponse/untagged enum/JsonError
                // substring checks) still match through `format!("{:#}")`.
                return Err(err).context("streaming inference request failed");
            }
        }
    }

    if let Some(p) = progress.as_ref() {
        p.finish();
    }

    // Prefer the provider-aggregated final response when it carries text;
    // fall back to the accumulated deltas otherwise.
    Ok(final_text.filter(|text| !text.is_empty()).unwrap_or(acc))
}

/// One parsed delta from an OpenAI-compatible chat-completion SSE chunk.
///
/// `Reasoning` covers the thinking channels used by OpenRouter/DeepSeek-style
/// endpoints; it must never be accumulated into the answer text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseDelta {
    Content(String),
    Reasoning(String),
}

/// Extract a delta from one OpenAI-compatible chat-completion SSE `data:`
/// payload.
///
/// Returns `None` for `[DONE]`, role-only first chunks, heartbeats, payloads
/// without a recognized delta text field, and unparseable payloads. Non-empty
/// content wins over reasoning when a chunk carries both; an empty `content`
/// string is ignored so a reasoning field on the same chunk still surfaces.
pub fn openai_sse_delta(data: &str) -> Option<SseDelta> {
    let trimmed = data.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;
    let delta = value.get("choices")?.as_array()?.first()?.get("delta")?;
    if let Some(text) = delta
        .get("content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return Some(SseDelta::Content(text.to_string()));
    }
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = delta.get(key).and_then(Value::as_str) {
            return Some(SseDelta::Reasoning(text.to_string()));
        }
    }
    None
}

/// Consume an OpenAI-compatible SSE response body into its complete content,
/// reporting live progress under `label`, and store the raw SSE exchange
/// into the cost/usage capture sink for the current task (used by the raw
/// HTTP fallback paths that bypass the capturing transport).
///
/// The caller is responsible for the request body (`"stream": true`) and for
/// checking the HTTP status before calling this.
pub async fn collect_openai_sse_with_capture(
    response: reqwest::Response,
    label: &str,
    request_url: String,
    request_model: String,
    echo: bool,
) -> Result<String> {
    collect_openai_sse_impl(response, label, Some((request_url, request_model)), echo).await
}

async fn collect_openai_sse_impl(
    response: reqwest::Response,
    label: &str,
    capture: Option<(String, String)>,
    echo: bool,
) -> Result<String> {
    use eventsource_stream::Eventsource;

    let (status, headers) = if capture.is_some() {
        (
            Some(response.status().as_u16()),
            Some(
                response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect::<Vec<_>>(),
            ),
        )
    } else {
        (None, None)
    };

    let mut progress = StreamProgress::new(label, "streaming", echo);
    let mut text = String::new();
    let mut raw = capture.is_some().then(String::new);
    let mut events = response.bytes_stream().eventsource();

    while let Some(event) = events.next().await {
        let event = event.map_err(|e| anyhow::anyhow!("SSE stream error: {}", e))?;
        if let Some(raw) = raw.as_mut() {
            raw.push_str("data: ");
            raw.push_str(&event.data);
            raw.push('\n');
        }
        if event.data.trim() == "[DONE]" {
            break;
        }
        match openai_sse_delta(&event.data) {
            Some(SseDelta::Content(delta)) => {
                progress.advance(delta.chars().count());
                progress.push_output(&delta);
                text.push_str(&delta);
            }
            Some(SseDelta::Reasoning(delta)) => {
                // Thinking never becomes part of the answer text.
                progress.advance(delta.chars().count());
                progress.push_think(&delta);
            }
            None => {}
        }
    }
    progress.finish();

    if let (Some((url, model)), Some(status), Some(headers), Some(body)) =
        (capture, status, headers, raw)
    {
        crate::llm::client::usage_capture::capture_response(
            crate::llm::client::usage_capture::CapturedResponse {
                request_url: url,
                request_model: Some(model),
                status,
                headers,
                body,
                stream: true,
            },
        );
    }

    if text.is_empty() {
        anyhow::bail!(
            "streaming response contained no content — if the endpoint does not support \
             \"stream\": true (it may have returned a plain JSON body), set stream = false \
             in the profile's [llm] section"
        );
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::Usage;

    #[test]
    fn sse_delta_extracts_content() {
        let data = r#"{"choices":[{"index":0,"delta":{"content":"hel"}}]}"#;
        assert_eq!(
            openai_sse_delta(data),
            Some(SseDelta::Content("hel".to_string()))
        );
    }

    #[test]
    fn sse_delta_ignores_empty_content_and_falls_through_to_reasoning() {
        let data =
            r#"{"choices":[{"index":0,"delta":{"content":"","reasoning_content":"think"}}]}"#;
        assert_eq!(
            openai_sse_delta(data),
            Some(SseDelta::Reasoning("think".to_string()))
        );
        let data = r#"{"choices":[{"index":0,"delta":{"content":""}}]}"#;
        assert_eq!(openai_sse_delta(data), None);
    }

    #[test]
    fn sse_delta_extracts_reasoning_content() {
        let data = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}"#;
        assert_eq!(
            openai_sse_delta(data),
            Some(SseDelta::Reasoning("think".to_string()))
        );
    }

    #[test]
    fn sse_delta_extracts_short_reasoning_field() {
        let data = r#"{"choices":[{"index":0,"delta":{"reasoning":"hmm"}}]}"#;
        assert_eq!(
            openai_sse_delta(data),
            Some(SseDelta::Reasoning("hmm".to_string()))
        );
    }

    #[test]
    fn sse_delta_prefers_content_over_reasoning() {
        let data = r#"{"choices":[{"index":0,"delta":{"content":"a","reasoning_content":"b"}}]}"#;
        assert_eq!(
            openai_sse_delta(data),
            Some(SseDelta::Content("a".to_string()))
        );
    }

    #[test]
    fn sse_delta_ignores_role_only_chunk() {
        let data = r#"{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        assert_eq!(openai_sse_delta(data), None);
    }

    #[test]
    fn sse_delta_ignores_done_heartbeat_and_garbage() {
        assert_eq!(openai_sse_delta("[DONE]"), None);
        assert_eq!(openai_sse_delta("  [DONE]\n"), None);
        assert_eq!(openai_sse_delta(""), None);
        assert_eq!(openai_sse_delta(": keep-alive comment"), None);
        assert_eq!(openai_sse_delta("not json at all"), None);
        assert_eq!(openai_sse_delta(r#"{"usage":{}}"#), None);
    }

    #[test]
    fn progress_advances_by_estimated_tokens() {
        let progress = StreamProgress::new("test-model", "streaming", false);
        progress.advance(800); // ~200 tokens
        assert_eq!(progress.bar.position(), 200);
        progress.finish();
    }

    #[test]
    fn progress_status_can_be_updated() {
        let progress = StreamProgress::new("test-model", "streaming", false);
        progress.set_status("tool: file_reader");
        // We can't easily read the rendered message, but we can verify the bar
        // is still alive and the method does not panic.
        assert!(!progress.bar.is_finished());
        progress.finish();
    }

    #[test]
    fn rolling_tail_keeps_last_lines() {
        let mut tail = RollingTail::new(2);
        tail.push(EchoKind::Output, "one\ntwo\nthree\n");
        let rendered = tail.render(200);
        assert_eq!(rendered, "out  │ two\nout  │ three");
    }

    #[test]
    fn rolling_tail_flushes_on_kind_change() {
        let mut tail = RollingTail::new(4);
        tail.push(EchoKind::Think, "pondering");
        tail.push(EchoKind::Output, "answer");
        assert_eq!(tail.render(200), "think│ pondering\nout  │ answer");
    }

    #[test]
    fn rolling_tail_marks_are_standalone_lines() {
        let mut tail = RollingTail::new(4);
        tail.push(EchoKind::Output, "partial");
        tail.mark(EchoKind::Tool, "tool: x");
        tail.push(EchoKind::Output, "more");
        assert_eq!(
            tail.render(200),
            "out  │ partial\ntool │ tool: x\nout  │ more"
        );
    }

    #[test]
    fn rolling_tail_wraps_long_lines() {
        let mut tail = RollingTail::new(1);
        tail.push(EchoKind::Output, "abcdefghij");
        assert_eq!(tail.render(13), "out  │ abcdef\nout  │ ghij");
    }

    fn text_item(text: &str) -> MultiTurnStreamItem<()> {
        MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::text(text))
    }

    fn final_item(text: &str) -> MultiTurnStreamItem<()> {
        MultiTurnStreamItem::final_response(
            rig_core::OneOrMany::one(rig_core::completion::AssistantContent::text(text)),
            Usage::new(),
        )
    }

    #[tokio::test]
    async fn drain_accumulates_text_deltas_without_progress() {
        let items = vec![
            Ok(text_item("{")),
            Ok(text_item("\"a\"")),
            Ok(final_item("{\"a\":1}")),
        ];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let out = drain_to_string(stream, None, false).await.unwrap();
        // Final response wins when non-empty (provider-aggregated text).
        assert_eq!(out, "{\"a\":1}");
    }

    #[tokio::test]
    async fn drain_falls_back_to_accumulated_deltas_without_final_response() {
        let items = vec![Ok(text_item("part1")), Ok(text_item("part2"))];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let out = drain_to_string(stream, None, false).await.unwrap();
        assert_eq!(out, "part1part2");
    }

    #[tokio::test]
    async fn drain_prefers_final_response_but_ignores_empty_one() {
        let items = vec![Ok(text_item("delta-text")), Ok(final_item(""))];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let out = drain_to_string(stream, None, false).await.unwrap();
        assert_eq!(out, "delta-text");
    }

    #[tokio::test]
    async fn drain_with_echo_returns_same_text() {
        let items = vec![Ok(text_item("out1")), Ok(text_item("out2"))];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let out = drain_to_string(stream, Some("test-model"), true)
            .await
            .unwrap();
        assert_eq!(out, "out1out2");
    }

    #[tokio::test]
    async fn drain_propagates_stream_errors_with_fallback_triggers_intact() {
        let items: Vec<Result<MultiTurnStreamItem<()>, rig_core::agent::StreamingError>> =
            vec![Err(rig_core::agent::StreamingError::Completion(
                rig_core::completion::CompletionError::ProviderError(
                    "ApiResponse mismatch".to_string(),
                ),
            ))];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let err = drain_to_string(stream, None, false).await.unwrap_err();
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("ApiResponse"),
            "fallback trigger must survive formatting: {rendered}"
        );
    }
}
