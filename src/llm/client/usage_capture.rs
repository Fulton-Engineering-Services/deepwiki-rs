//! HTTP response capture layer for cost & usage tracking.
//!
//! Wraps the provider HTTP client (`rig_core::http_client::HttpClientExt`)
//! so the raw response body and headers are snapshotted BEFORE rig parses
//! them into typed structs (which drop `usage.cost` and `x-litellm-*`
//! headers). Captures are keyed by the tokio task that issued the call; the
//! call site drains them via [`take_captures_for_current_task`] and turns
//! them into [`UsageRecord`]s (see [`extract_usage_record`]).

use bytes::Bytes;
use futures::Stream;
use rig_core::http_client::sse::BoxedStream;
use rig_core::http_client::{
    self, HttpClientExt, LazyBody, Request, Response, StreamingResponse,
};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{LazyLock, Mutex};

/// One snapshotted HTTP exchange (request meta + full response).
#[derive(Debug, Clone)]
pub struct CapturedResponse {
    pub request_url: String,
    pub request_model: Option<String>,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    /// Full response body (non-streaming) or accumulated SSE text (streaming).
    pub body: String,
    pub stream: bool,
}

impl CapturedResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Origin part of the request URL, e.g. `https://gateway:4000`.
    pub fn origin(&self) -> String {
        if let Some((scheme, rest)) = self.request_url.split_once("://") {
            let authority = rest.split('/').next().unwrap_or(rest);
            return format!("{}://{}", scheme, authority);
        }
        self.request_url.clone()
    }
}

#[derive(Default)]
struct CaptureStore {
    by_task: Mutex<HashMap<tokio::task::Id, Vec<CapturedResponse>>>,
}

static CAPTURE_STORE: LazyLock<CaptureStore> = LazyLock::new(CaptureStore::default);

fn current_task_key() -> Option<tokio::task::Id> {
    // `task::id()` panics when polled outside a spawned task (e.g. the main
    // future under `#[tokio::main]`); `try_id` degrades to no-capture there.
    tokio::task::try_id()
}

fn store(captured: CapturedResponse) {
    if let Some(key) = current_task_key() {
        let mut map = CAPTURE_STORE.by_task.lock().unwrap();
        map.entry(key).or_default().push(captured);
    }
}

/// Drain the captures recorded by the current tokio task (in issue order).
pub fn take_captures_for_current_task() -> Vec<CapturedResponse> {
    let Some(key) = current_task_key() else {
        return Vec::new();
    };
    let mut map = CAPTURE_STORE.by_task.lock().unwrap();
    map.remove(&key).unwrap_or_default()
}

/// Record a capture for the current task from a code path that makes its own
/// HTTP call (e.g. the raw OpenAI-compatible fallback) rather than going
/// through [`CapturingHttpClient`].
pub fn capture_response(cap: CapturedResponse) {
    store(cap);
}

/// Wrapper client that tees every response through the capture sink while
/// delegating transport to the inner client.
#[derive(Clone)]
pub struct CapturingHttpClient<H = http_client::ReqwestClient> {
    inner: H,
}

impl<H: Default> Default for CapturingHttpClient<H> {
    fn default() -> Self {
        Self { inner: H::default() }
    }
}

impl<H: std::fmt::Debug> std::fmt::Debug for CapturingHttpClient<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturingHttpClient")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<H> CapturingHttpClient<H> {
    pub fn new(inner: H) -> Self {
        Self { inner }
    }
}

/// Capturing client over rig's default transport — the concrete `H` used by
/// every OpenAI-family provider client in this crate.
pub type CapturingClient = CapturingHttpClient<rig_core::http_client::ReqwestClient>;

/// OpenAI chat-completions model wired to the capturing transport.
pub type OpenAIModel =
    rig_core::providers::openai::completion::CompletionModel<CapturingClient>;

fn collect_headers(headers: &rig_core::http_client::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect()
}

fn extract_model(body: &Bytes) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str().map(str::to_string)))
}

impl<H> HttpClientExt for CapturingHttpClient<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let inner = self.inner.clone();
        let (parts, body) = req.into_parts();
        let body_bytes: Bytes = body.into();
        let model = extract_model(&body_bytes);
        let url = parts.uri.to_string();
        let req2 = Request::from_parts(parts, body_bytes);
        async move {
            let resp: Response<LazyBody<Bytes>> = inner.send::<Bytes, Bytes>(req2).await?;
            let status = resp.status();
            let header_map = resp.headers().clone();
            let headers = collect_headers(&header_map);
            let body_fut = resp.into_body();

            let lazy: LazyBody<U> = Box::pin(async move {
                let bytes = body_fut.await?;
                store(CapturedResponse {
                    request_url: url,
                    request_model: model,
                    status: status.as_u16(),
                    headers,
                    body: String::from_utf8_lossy(&bytes).to_string(),
                    stream: false,
                });
                Ok(U::from(bytes))
            });

            let mut builder = Response::builder().status(status);
            if let Some(hs) = builder.headers_mut() {
                *hs = header_map;
            }
            builder.body(lazy).map_err(http_client::Error::Protocol)
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<http_client::MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        // Multipart is not used for chat completions; delegate without capture.
        let inner = self.inner.clone();
        async move { inner.send_multipart(req).await }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let inner = self.inner.clone();
        let (parts, body) = req.into_parts();
        let body_bytes: Bytes = body.into();
        let model = extract_model(&body_bytes);
        let url = parts.uri.to_string();
        let req2 = Request::from_parts(parts, body_bytes);
        async move {
            let resp: StreamingResponse = inner.send_streaming(req2).await?;
            let status = resp.status();
            let header_map = resp.headers().clone();
            let headers = collect_headers(&header_map);

            let tee = TeeStream {
                inner: resp.into_body(),
                url,
                model,
                status: status.as_u16(),
                headers,
                body: String::new(),
            };

            let boxed: BoxedStream = Box::pin(tee);
            let mut builder = Response::builder().status(status);
            if let Some(hs) = builder.headers_mut() {
                *hs = header_map;
            }
            builder.body(boxed).map_err(http_client::Error::Protocol)
        }
    }
}

/// Streaming tee: forwards chunks untouched to rig while accumulating the
/// raw SSE text; the accumulated body is stored into the capture sink when
/// the stream ends.
struct TeeStream {
    inner: BoxedStream,
    url: String,
    model: Option<String>,
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Stream for TeeStream {
    type Item = http_client::Result<Bytes>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                self.body.push_str(&String::from_utf8_lossy(&chunk));
                std::task::Poll::Ready(Some(Ok(chunk)))
            }
            std::task::Poll::Ready(None) => {
                store(CapturedResponse {
                    request_url: self.url.clone(),
                    request_model: self.model.clone(),
                    status: self.status,
                    headers: self.headers.clone(),
                    body: std::mem::take(&mut self.body),
                    stream: true,
                });
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Ready(Some(Err(e))) => std::task::Poll::Ready(Some(Err(e))),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

/// Build a [`UsageRecord`] from one captured HTTP exchange, merging the
/// response body's `usage` object with LiteLLM's `x-litellm-*` headers.
pub fn extract_usage_record(cap: &CapturedResponse) -> crate::llm::client::usage_tracker::UsageRecord {
    use crate::llm::client::usage_tracker::{CostSource, UsageRecord};

    let mut rec = UsageRecord {
        timestamp: chrono::Utc::now(),
        model: cap
            .request_model
            .clone()
            .unwrap_or_default(),
        base_url: cap.origin(),
        stream: cap.stream,
        success: (200..300).contains(&cap.status),
        ..Default::default()
    };

    // Locate the JSON object carrying a `usage` map: the whole body for
    // non-streaming responses, or the terminal SSE data chunk for streams.
    if let Some((usage, finish_reason)) = find_usage_object(cap) {
        rec.finish_reason = finish_reason;
        rec.input_tokens = u64_field(&usage, "prompt_tokens")
            .or_else(|| u64_field(&usage, "input_tokens"))
            .unwrap_or(0);
        rec.output_tokens = u64_field(&usage, "completion_tokens")
            .or_else(|| u64_field(&usage, "output_tokens"))
            .unwrap_or(0);
        rec.total_tokens = u64_field(&usage, "total_tokens").unwrap_or(rec.input_tokens + rec.output_tokens);
        rec.cached_input_tokens = usage
            .get("prompt_tokens_details")
            .and_then(|d| u64_field(d, "cached_tokens"))
            .or_else(|| u64_field(&usage, "cache_read_input_tokens"))
            .unwrap_or(0);
        rec.cache_creation_input_tokens = u64_field(&usage, "cache_creation_input_tokens").unwrap_or(0);
        rec.reasoning_tokens = usage
            .get("completion_tokens_details")
            .and_then(|d| u64_field(d, "reasoning_tokens"))
            .or_else(|| u64_field(&usage, "reasoning_tokens"))
            .unwrap_or(0);
        rec.tool_use_prompt_tokens = u64_field(&usage, "tool_use_prompt_tokens").unwrap_or(0);
        if let Some(cost) = f64_field(&usage, "cost") {
            rec.cost_usd = Some(cost);
            rec.cost_source = CostSource::LiteLlm;
        }
    }

    apply_litellm_headers(&mut rec, cap);
    rec
}

fn find_usage_object(cap: &CapturedResponse) -> Option<(serde_json::Value, Option<String>)> {
    if !cap.stream {
        let v: serde_json::Value = serde_json::from_str(&cap.body).ok()?;
        let usage = v.get("usage").filter(|u| !u.is_null())?.clone();
        let finish = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason"))
            .and_then(|r| r.as_str())
            .map(str::to_string);
        return Some((usage, finish));
    }

    // Streaming: scan `data:` lines; the terminal usage chunk carries the
    // usage object (LiteLLM injects one with choices=[] before [DONE]).
    let mut found: Option<serde_json::Value> = None;
    let mut finish_reason: Option<String> = None;
    for line in cap.body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            found = Some(u.clone());
        }
        if let Some(reason) = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason"))
            .and_then(|r| r.as_str())
        {
            finish_reason = Some(reason.to_string());
        }
    }
    found.map(|u| (u, finish_reason))
}

fn apply_litellm_headers(rec: &mut crate::llm::client::usage_tracker::UsageRecord, cap: &CapturedResponse) {
    let h = |name: &str| cap.header(name).map(str::to_string);
    rec.litellm_call_id = h("x-litellm-call-id");
    rec.litellm_model_id = h("x-litellm-model-id");
    rec.litellm_model_name = h("x-litellm-model-name");
    rec.litellm_cache_key = h("x-litellm-cache-key");
    rec.litellm_model_api_base = h("x-litellm-model-api-base");
    rec.litellm_version = h("x-litellm-version");
    rec.litellm_key_spend = h("x-litellm-key-spend").and_then(|v| v.parse().ok());
    rec.litellm_response_duration_ms = h("x-litellm-response-duration-ms").and_then(|v| v.parse().ok());
    rec.litellm_overhead_duration_ms = h("x-litellm-overhead-duration-ms").and_then(|v| v.parse().ok());
    rec.litellm_callback_duration_ms = h("x-litellm-callback-duration-ms").and_then(|v| v.parse().ok());

    if rec.cost_usd.is_none()
        && let Some(cost) = h("x-litellm-response-cost").and_then(|v| v.parse::<f64>().ok())
    {
        rec.cost_usd = Some(cost);
        rec.cost_source = crate::llm::client::usage_tracker::CostSource::LiteLlm;
    }
    rec.cost_input_usd = h("x-litellm-response-cost-input").and_then(|v| v.parse().ok());
    rec.cost_output_usd = h("x-litellm-response-cost-output").and_then(|v| v.parse().ok());
    rec.cost_cache_read_usd = h("x-litellm-response-cost-cache-read").and_then(|v| v.parse().ok());
    rec.cost_cache_creation_usd = h("x-litellm-response-cost-cache-creation").and_then(|v| v.parse().ok());
    rec.cost_reasoning_usd = h("x-litellm-response-cost-reasoning").and_then(|v| v.parse().ok());
    rec.cost_tool_usage_usd = h("x-litellm-response-cost-tool-usage").and_then(|v| v.parse().ok());
}

fn u64_field(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

fn f64_field(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| {
        x.as_f64().or_else(|| x.as_i64().map(|i| i as f64))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_non_streaming_usage_and_headers() {
        let body = r#"{"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":{"reasoning_tokens":5},"cost":0.0123}}"#;
        let cap = CapturedResponse {
            request_url: "https://gw.example:4000/v1/chat/completions".into(),
            request_model: Some("m".into()),
            status: 200,
            headers: vec![
                ("x-litellm-call-id".into(), "abc".into()),
                ("x-litellm-response-cost-input".into(), "0.0001".into()),
            ],
            body: body.into(),
            stream: false,
        };
        let rec = extract_usage_record(&cap);
        assert_eq!(rec.input_tokens, 100);
        assert_eq!(rec.output_tokens, 50);
        assert_eq!(rec.cached_input_tokens, 20);
        assert_eq!(rec.reasoning_tokens, 5);
        assert_eq!(rec.cost_usd, Some(0.0123));
        assert_eq!(rec.litellm_call_id.as_deref(), Some("abc"));
        assert_eq!(rec.cost_input_usd, Some(0.0001));
        assert_eq!(rec.base_url, "https://gw.example:4000");
    }

    #[test]
    fn extracts_streaming_terminal_usage() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10,\"cost\":0.0005}}\n\ndata:";
        let cap = CapturedResponse {
            request_url: "http://gw/v1/chat/completions".into(),
            request_model: Some("m".into()),
            status: 200,
            headers: vec![("x-litellm-response-cost".into(), "0.0006".into())],
            body: body.into(),
            stream: true,
        };
        let rec = extract_usage_record(&cap);
        assert_eq!(rec.input_tokens, 7);
        assert_eq!(rec.output_tokens, 3);
        assert_eq!(rec.cost_usd, Some(0.0005));
        assert!(rec.stream);
    }

    #[tokio::test]
    async fn send_preserves_response_headers() {
        let client = CapturingHttpClient::new(MockInner);
        let resp = client
            .send::<Bytes, Bytes>(Request::new(Bytes::new()))
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        assert_eq!(
            resp.headers().get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
    }

    #[tokio::test]
    async fn streaming_preserves_content_type_for_sse_check() {
        // Run inside a spawned task so the capture sink can key by task id.
        let (status, content_type, body_text, captures) =
            tokio::spawn(async move {
                let client = CapturingHttpClient::new(MockInner);
                let resp =
                    client.send_streaming(Request::new(Bytes::new())).await.unwrap();
                let status = resp.status().as_u16();
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();

                // Chunks pass through the tee unchanged.
                use futures::StreamExt;
                let mut body = resp.into_body();
                let mut acc: Vec<u8> = Vec::new();
                while let Some(chunk) = body.next().await {
                    acc.extend_from_slice(&chunk.unwrap());
                }
                let text = String::from_utf8_lossy(&acc).to_string();
                let captures = take_captures_for_current_task();
                (status, content_type, text, captures)
            })
            .await
            .unwrap();

        assert_eq!(status, 200);
        // rig's GenericEventSource rejects responses whose content-type is
        // not text/event-stream; the rebuilt response must keep it.
        assert_eq!(content_type, "text/event-stream");
        assert!(body_text.contains("\"content\":\"hi\""));

        // The tee stored the raw exchange for the current task.
        assert_eq!(captures.len(), 1);
        assert!(captures[0].stream);
        assert_eq!(captures[0].body, body_text);
        assert_eq!(
            captures[0].header("content-type"),
            Some("text/event-stream")
        );
    }

    /// Minimal inner transport that answers with an SSE-shaped response so
    /// the capturing wrapper's header preservation can be verified.
    #[derive(Clone, Default)]
    struct MockInner;

    impl HttpClientExt for MockInner {
        fn send<T, U>(
            &self,
            req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            T: Into<Bytes> + WasmCompatSend,
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            let (_, body) = req.into_parts();
            let _body_bytes: Bytes = body.into();
            async move {
                let mut b = Response::builder().status(200);
                if let Some(hs) = b.headers_mut() {
                    hs.insert("content-type", "application/json".parse().unwrap());
                }
                let lazy: LazyBody<U> = Box::pin(async {
                    Ok(U::from(Bytes::from_static(b"{}")))
                });
                b.body(lazy).map_err(http_client::Error::Protocol)
            }
        }

        fn send_multipart<U>(
            &self,
            req: Request<http_client::MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            let _ = req;
            async move {
                let mut b = Response::builder().status(200);
                let lazy: LazyBody<U> = Box::pin(async {
                    Ok(U::from(Bytes::from_static(b"{}")))
                });
                b.body(lazy).map_err(http_client::Error::Protocol)
            }
        }

        fn send_streaming<T>(
            &self,
            req: Request<T>,
        ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
        where
            T: Into<Bytes> + WasmCompatSend,
        {
            let (_, body) = req.into_parts();
            let _body_bytes: Bytes = body.into();
            async move {
                let chunks: Vec<http_client::Result<Bytes>> = vec![
                    Ok(Bytes::from_static(
                        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    )),
                    Ok(Bytes::from_static(b"data:\n")),
                ];
                let boxed: BoxedStream = Box::pin(futures::stream::iter(chunks));
                let mut b = Response::builder().status(200);
                if let Some(hs) = b.headers_mut() {
                    hs.insert("content-type", "text/event-stream".parse().unwrap());
                }
                b.body(boxed).map_err(http_client::Error::Protocol)
            }
        }
    }
}
