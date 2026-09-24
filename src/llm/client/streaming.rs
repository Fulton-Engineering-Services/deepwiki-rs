//! Streaming inference helpers.
//!
//! Drains rig streaming responses into a complete string — deltas are
//! accumulated, JSON is only ever parsed from the finished text — while
//! reporting throttled progress on stderr. Also parses OpenAI-compatible SSE
//! chunks for the reqwest HTTP fallback path.

use anyhow::{Context, Result};
use futures::StreamExt;
use rig_core::agent::{MultiTurnStreamItem, StreamingResult};
use rig_core::streaming::StreamedAssistantContent;
use serde_json::Value;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Minimum interval between stderr progress renders.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

/// Number of streaming requests currently rendering progress. With more than
/// one live stream (e.g. `max_parallels > 1`), in-place carriage-return
/// updates would overwrite each other, so concurrent streams fall back to
/// plain periodic lines.
static ACTIVE_STREAMS: AtomicUsize = AtomicUsize::new(0);

/// Throttled stderr progress reporter for a single streaming request.
///
/// Renders at most one line per [`PROGRESS_INTERVAL`]: an in-place
/// carriage-return update on a TTY when this is the only live stream (plain
/// periodic lines otherwise), and clears the line when the request completes
/// or fails.
pub struct StreamProgress {
    label: String,
    start: Instant,
    last_render: Instant,
    chars: usize,
    out_is_tty: bool,
    rendered: bool,
}

impl StreamProgress {
    pub fn new(label: &str) -> Self {
        ACTIVE_STREAMS.fetch_add(1, Ordering::SeqCst);
        Self {
            label: label.to_string(),
            start: Instant::now(),
            last_render: Instant::now(),
            chars: 0,
            out_is_tty: std::io::stderr().is_terminal(),
            rendered: false,
        }
    }

    /// Record more streamed characters; render if the throttle interval elapsed.
    pub fn advance(&mut self, chars: usize) {
        self.chars += chars;
        let now = Instant::now();
        if now.duration_since(self.last_render) < PROGRESS_INTERVAL {
            return;
        }
        self.last_render = now;
        let line = self.format_line(now);
        if self.out_is_tty && ACTIVE_STREAMS.load(Ordering::SeqCst) == 1 {
            eprint!("\r\x1b[2K{line}");
            let _ = std::io::stderr().flush();
            self.rendered = true;
        } else {
            eprintln!("{line}");
        }
    }

    /// Clear the rendered line (TTY only) so it doesn't collide with
    /// subsequent status output.
    pub fn finish(&mut self) {
        if self.rendered {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
        self.rendered = false;
    }

    fn format_line(&self, now: Instant) -> String {
        let elapsed = now.duration_since(self.start).as_secs_f64().max(0.001);
        // Rough token estimate: ~4 characters per token.
        let est_tokens = self.chars / 4;
        format!(
            "⏳ {} · ~{} tok · {:.1} tok/s · {:.1}s",
            self.label,
            est_tokens,
            est_tokens as f64 / elapsed,
            elapsed
        )
    }
}

impl Drop for StreamProgress {
    fn drop(&mut self) {
        ACTIVE_STREAMS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Drain a rig streaming prompt into its complete assistant text.
///
/// Aggregates assistant text deltas as they arrive and prefers the provider's
/// final aggregated response when it carries text. Nothing is parsed until the
/// stream completes: callers feed the returned string to their existing
/// parse-and-validate logic, which never sees partial JSON.
///
/// `label` (typically the model name) enables the stderr progress line; pass
/// `None` to disable it.
pub async fn drain_to_string<R>(
    mut stream: StreamingResult<R>,
    label: Option<&str>,
) -> Result<String> {
    let mut progress = label.map(StreamProgress::new);
    let mut acc = String::new();
    let mut final_text: Option<String> = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                if let Some(p) = progress.as_mut() {
                    p.advance(text.text.chars().count());
                }
                acc.push_str(&text.text);
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            )) => {
                // Reasoning is not part of the answer but keeps the line alive
                // during long thinking phases.
                if let Some(p) = progress.as_mut() {
                    p.advance(reasoning.chars().count());
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                if let Some(p) = progress.as_mut() {
                    p.advance(reasoning.display_text().chars().count());
                }
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_text = Some(res.output);
            }
            // Tool calls/results and final model payloads: no progress text.
            Ok(_) => {}
            Err(err) => {
                if let Some(p) = progress.as_mut() {
                    p.finish();
                }
                // Keep the inner rig error in the anyhow chain so existing
                // HTTP-fallback triggers (ApiResponse/untagged enum/JsonError
                // substring checks) still match through `format!("{:#}")`.
                return Err(err).context("streaming inference request failed");
            }
        }
    }

    if let Some(p) = progress.as_mut() {
        p.finish();
    }

    // Prefer the provider-aggregated final response when it carries text;
    // fall back to the accumulated deltas otherwise.
    Ok(final_text.filter(|text| !text.is_empty()).unwrap_or(acc))
}

/// Extract a content delta from one OpenAI-compatible chat-completion SSE
/// `data:` payload.
///
/// Returns `None` for `[DONE]`, role-only first chunks, heartbeats, payloads
/// without a `delta.content` field, and unparseable payloads.
pub fn openai_sse_delta(data: &str) -> Option<String> {
    let trimmed = data.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

/// Consume an OpenAI-compatible SSE response body into its complete content,
/// reporting throttled progress under `label`.
///
/// The caller is responsible for the request body (`"stream": true`) and for
/// checking the HTTP status before calling this.
pub async fn collect_openai_sse(response: reqwest::Response, label: &str) -> Result<String> {
    use eventsource_stream::Eventsource;

    let mut progress = StreamProgress::new(label);
    let mut text = String::new();
    let mut events = response.bytes_stream().eventsource();

    while let Some(event) = events.next().await {
        let event = event.map_err(|e| anyhow::anyhow!("SSE stream error: {}", e))?;
        if event.data.trim() == "[DONE]" {
            break;
        }
        if let Some(delta) = openai_sse_delta(&event.data) {
            progress.advance(delta.chars().count());
            text.push_str(&delta);
        }
    }
    progress.finish();

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
        assert_eq!(openai_sse_delta(data).as_deref(), Some("hel"));
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
    fn progress_line_reports_model_tokens_and_rate() {
        let mut progress = StreamProgress::new("test-model");
        progress.start = Instant::now() - Duration::from_secs(2);
        progress.chars = 800; // 200 est. tokens over 2s => 100 tok/s
        let line = progress.format_line(progress.start + Duration::from_secs(2));
        assert!(line.contains("test-model"), "line: {line}");
        assert!(line.contains("~200 tok"), "line: {line}");
        assert!(line.contains("100.0 tok/s"), "line: {line}");
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
        let out = drain_to_string(stream, None).await.unwrap();
        // Final response wins when non-empty (provider-aggregated text).
        assert_eq!(out, "{\"a\":1}");
    }

    #[tokio::test]
    async fn drain_falls_back_to_accumulated_deltas_without_final_response() {
        let items = vec![Ok(text_item("part1")), Ok(text_item("part2"))];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let out = drain_to_string(stream, None).await.unwrap();
        assert_eq!(out, "part1part2");
    }

    #[tokio::test]
    async fn drain_prefers_final_response_but_ignores_empty_one() {
        let items = vec![Ok(text_item("delta-text")), Ok(final_item(""))];
        let stream: StreamingResult<()> = Box::pin(futures::stream::iter(items));
        let out = drain_to_string(stream, None).await.unwrap();
        assert_eq!(out, "delta-text");
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
        let err = drain_to_string(stream, None).await.unwrap_err();
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("ApiResponse"),
            "fallback trigger must survive formatting: {rendered}"
        );
    }
}
