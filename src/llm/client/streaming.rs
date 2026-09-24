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
use std::future::Future;
use std::sync::LazyLock;
use std::time::Duration;

/// Shared `MultiProgress` manager so concurrent provider calls each get their
/// own stderr line without overwriting one another.
static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Rough character-to-token ratio used for live tok/s estimates.
const CHARS_PER_TOKEN: usize = 4;

/// Live stderr progress reporter for a single request.
///
/// Renders a spinner plus a status message on every delta, giving real-time
/// tok/s and internal state (tool calls, reasoning, etc.). The bar is cleared
/// when the request finishes or fails.
pub struct StreamProgress {
    bar: ProgressBar,
    label: String,
}

impl StreamProgress {
    /// Create a new progress reporter.
    ///
    /// `label` is usually the model name; `status` is the initial human-readable
    /// state (e.g. "streaming" or "calling").
    pub fn new(label: &str, status: &str) -> Self {
        let bar = MULTI_PROGRESS.add(ProgressBar::new_spinner());
        bar.set_style(
            ProgressStyle::default_spinner()
                .template(
                    "{spinner:.green} {msg} · {pos} tok · {per_sec} · {elapsed_precise}",
                )
                .expect("valid progress template"),
        );
        bar.set_message(format!("{label} · {status}"));
        bar.enable_steady_tick(Duration::from_millis(120));
        Self {
            bar,
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

    /// Clear the rendered line so it doesn't collide with subsequent output.
    pub fn finish(&self) {
        self.bar.finish_and_clear();
    }
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
    let progress = StreamProgress::new(label, status);
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
/// `None` to disable it.
pub async fn drain_to_string<R>(
    mut stream: StreamingResult<R>,
    label: Option<&str>,
) -> Result<String> {
    let progress = label.map(|l| StreamProgress::new(l, "streaming"));
    let mut acc = String::new();
    let mut final_text: Option<String> = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                if let Some(p) = progress.as_ref() {
                    p.advance(text.text.chars().count());
                }
                acc.push_str(&text.text);
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            )) => {
                // Reasoning is not part of the answer but keeps the line alive
                // during long thinking phases.
                if let Some(p) = progress.as_ref() {
                    p.advance(reasoning.chars().count());
                    p.set_status("thinking");
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                if let Some(p) = progress.as_ref() {
                    p.advance(reasoning.display_text().chars().count());
                    p.set_status("thinking");
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            })) => {
                if let Some(p) = progress.as_ref() {
                    p.set_status(&format!("tool: {}", tool_call.function.name));
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
                if let Some(p) = progress.as_ref() {
                    p.set_status(&format!("running {}", tool_call.function.name));
                }
            }
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                if let Some(p) = progress.as_ref() {
                    p.set_status("tool result");
                }
            }
            Ok(MultiTurnStreamItem::CompletionCall(_)) => {
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
) -> Result<String> {
    collect_openai_sse_impl(response, label, Some((request_url, request_model))).await
}

async fn collect_openai_sse_impl(
    response: reqwest::Response,
    label: &str,
    capture: Option<(String, String)>,
) -> Result<String> {
    use eventsource_stream::Eventsource;

    let (status, headers) = if capture.is_some() {
        (
            Some(response.status().as_u16()),
            Some(
                response
                    .headers()
                    .iter()
                    .map(|(k, v)| {
                        (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())
                    })
                    .collect::<Vec<_>>(),
            ),
        )
    } else {
        (None, None)
    };

    let progress = StreamProgress::new(label, "streaming");
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
        if let Some(delta) = openai_sse_delta(&event.data) {
            progress.advance(delta.chars().count());
            text.push_str(&delta);
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
    fn progress_advances_by_estimated_tokens() {
        let progress = StreamProgress::new("test-model", "streaming");
        progress.advance(800); // ~200 tokens
        assert_eq!(progress.bar.position(), 200);
        progress.finish();
    }

    #[test]
    fn progress_status_can_be_updated() {
        let progress = StreamProgress::new("test-model", "streaming");
        progress.set_status("tool: file_reader");
        // We can't easily read the rendered message, but we can verify the bar
        // is still alive and the method does not panic.
        assert!(!progress.bar.is_finished());
        progress.finish();
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
