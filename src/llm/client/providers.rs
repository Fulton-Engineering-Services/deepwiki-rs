//! LLM Provider support module

use anyhow::Result;
use rig_core::{
    agent::Agent,
    client::CompletionClient,
    completion::{CompletionModel, GetTokenUsage, Prompt},
    extractor::Extractor,
    providers::gemini::completion::gemini_api_types::{AdditionalParameters, GenerationConfig},
    streaming::StreamingPrompt,
    wasm_compat::WasmCompatSend,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    config::{LLMConfig, LLMProvider},
    llm::tools::time::AgentToolTime,
};

use super::ollama_extractor::OllamaExtractorWrapper;
use super::openai_compatible_extractor::OpenAICompatibleExtractorWrapper;
use super::streaming::{collect_openai_sse, drain_to_string, with_spinner};
use super::usage_capture::{
    take_captures_for_current_task, CapturingClient, OpenAIModel,
};
use super::usage_tracker;

/// Unified Provider client enum
#[derive(Clone)]
pub enum ProviderClient {
    OpenAI(rig_core::providers::openai::CompletionsClient<CapturingClient>),
    Moonshot(rig_core::providers::moonshot::Client),
    DeepSeek(rig_core::providers::deepseek::Client),
    Mistral(rig_core::providers::mistral::Client),
    OpenRouter(rig_core::providers::openrouter::Client),
    Anthropic(rig_core::providers::anthropic::Client),
    Gemini(rig_core::providers::gemini::Client),
    Ollama(rig_core::providers::ollama::Client),
}

impl ProviderClient {
    /// Create corresponding provider client based on configuration
    pub fn new(config: &LLMConfig) -> Result<Self> {
        match config.provider {
            LLMProvider::OpenAI => {
                let http = CapturingClient::new(rig_core::http_client::ReqwestClient::default());
                let client = if config.api_base_url != "https://api.openai.com/v1" {
                    rig_core::providers::openai::Client::builder()
                        .api_key(&config.api_key)
                        .base_url(&config.api_base_url)
                        .http_client(http)
                        .build()?
                        .completions_api()
                } else {
                    rig_core::providers::openai::Client::builder()
                        .api_key(&config.api_key)
                        .http_client(http)
                        .build()?
                        .completions_api()
                };
                Ok(ProviderClient::OpenAI(client))
            }
            LLMProvider::Moonshot => {
                let client = rig_core::providers::moonshot::Client::builder()
                    .api_key(&config.api_key)
                    .base_url(&config.api_base_url)
                    .build()?;
                Ok(ProviderClient::Moonshot(client))
            }
            LLMProvider::DeepSeek => {
                let client = rig_core::providers::deepseek::Client::builder()
                    .api_key(&config.api_key)
                    .base_url(&config.api_base_url)
                    .build()?;
                Ok(ProviderClient::DeepSeek(client))
            }
            LLMProvider::Mistral => {
                let client = rig_core::providers::mistral::Client::new(&config.api_key)?;
                Ok(ProviderClient::Mistral(client))
            }
            LLMProvider::OpenRouter => {
                let client = rig_core::providers::openrouter::Client::new(&config.api_key)?;
                Ok(ProviderClient::OpenRouter(client))
            }
            LLMProvider::Anthropic => {
                // Only override base_url if it looks like an Anthropic endpoint.
                // This prevents accidentally using a non-Anthropic URL (e.g., modelscope)
                // when api_base_url is set to a global default.
                let normalized_url = config.api_base_url.to_lowercase().trim_end_matches('/').to_string();
                let use_custom_url = normalized_url != "https://api.anthropic.com"
                    && normalized_url.contains("anthropic");
                let client = if use_custom_url {
                    rig_core::providers::anthropic::Client::builder()
                        .api_key(&config.api_key)
                        .base_url(&config.api_base_url)
                        .build()?
                } else {
                    rig_core::providers::anthropic::Client::new(&config.api_key)?
                };
                Ok(ProviderClient::Anthropic(client))
            }
            LLMProvider::Gemini => {
                let client = rig_core::providers::gemini::Client::new(&config.api_key)?;
                Ok(ProviderClient::Gemini(client))
            }
            LLMProvider::Ollama => {
                let client = rig_core::providers::ollama::Client::builder()
                    .api_key(rig_core::client::Nothing)
                    .base_url(&config.api_base_url)
                    .build()?;
                Ok(ProviderClient::Ollama(client))
            }
        }
    }

    /// Create Agent
    pub fn create_agent(
        &self,
        model: &str,
        system_prompt: &str,
        config: &LLMConfig,
    ) -> ProviderAgent {
        match self {
            ProviderClient::OpenAI(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into());

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::OpenAI {
                    agent,
                    base_url: config.api_base_url.clone(),
                    model: model.to_string(),
                    api_key: config.api_key.clone(),
                    system_prompt: system_prompt.to_string(),
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                    stream: config.stream_enabled(),
                }
            }
            ProviderClient::Moonshot(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::Moonshot(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::DeepSeek(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::DeepSeek(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Mistral(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::Mistral(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::OpenRouter(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::OpenRouter(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Anthropic(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into());

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::Anthropic(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Gemini(client) => {
                let gen_cfg = GenerationConfig::default();
                let cfg = AdditionalParameters::default().with_config(gen_cfg);

                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into());

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .additional_params(serde_json::to_value(cfg).unwrap())
                    .build();
                ProviderAgent::Gemini(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Ollama(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into());

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();
                ProviderAgent::Ollama(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
        }
    }

    /// Create Agent with tools
    pub fn create_agent_with_tools(
        &self,
        model: &str,
        system_prompt: &str,
        config: &LLMConfig,
        file_explorer: &crate::llm::tools::file_explorer::AgentToolFileExplorer,
        file_reader: &crate::llm::tools::file_reader::AgentToolFileReader,
    ) -> ProviderAgent {
        let tool_time = AgentToolTime::new();

        match self {
            ProviderClient::OpenAI(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::OpenAI {
                    agent,
                    base_url: config.api_base_url.clone(),
                    model: model.to_string(),
                    api_key: config.api_key.clone(),
                    system_prompt: system_prompt.to_string(),
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                    stream: config.stream_enabled(),
                }
            }
            ProviderClient::Moonshot(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::Moonshot(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::DeepSeek(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::DeepSeek(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Mistral(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::Mistral(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::OpenRouter(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::OpenRouter(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Anthropic(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::Anthropic(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Gemini(client) => {
                let gen_cfg = GenerationConfig::default();
                let cfg = AdditionalParameters::default().with_config(gen_cfg);

                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .additional_params(serde_json::to_value(cfg).unwrap())
                    .build();
                ProviderAgent::Gemini(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
            ProviderClient::Ollama(client) => {
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .default_max_turns(config.max_turns);

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder
                    .tool(file_explorer.clone())
                    .tool(file_reader.clone())
                    .tool(tool_time)
                    .build();
                ProviderAgent::Ollama(AgentHandle {
                    agent,
                    model: model.to_string(),
                    stream: config.stream_enabled(),
                })
            }
        }
    }

    /// Create Extractor
    pub fn create_extractor<T>(
        &self,
        model: &str,
        system_prompt: &str,
        config: &LLMConfig,
    ) -> ProviderExtractor<T>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
    {
        match self {
            ProviderClient::OpenAI(client) => {
                // Create agent for OpenAI-compatible provider
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into());

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();

                // Wrap with OpenAICompatibleExtractorWrapper for HTTP fallback
                let wrapper = OpenAICompatibleExtractorWrapper::new(
                    agent,
                    config.retry_attempts,
                    config.api_base_url.clone(),
                    model.to_string(),
                    config.api_key.clone(),
                    config.stream_enabled(),
                    config.max_tokens,
                    config.temperature,
                );

                ProviderExtractor::OpenAI(wrapper)
            }
            ProviderClient::Moonshot(client) => {
                let extractor = client
                    .extractor::<T>(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .build();
                ProviderExtractor::Moonshot(extractor)
            }
            ProviderClient::DeepSeek(client) => {
                let extractor = client
                    .extractor::<T>(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .build();
                ProviderExtractor::DeepSeek(extractor)
            }
            ProviderClient::Mistral(client) => {
                let extractor = client
                    .extractor::<T>(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .build();
                ProviderExtractor::Mistral(extractor)
            }
            ProviderClient::OpenRouter(client) => {
                let extractor = client
                    .extractor::<T>(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .build();
                ProviderExtractor::OpenRouter(extractor)
            }
            ProviderClient::Anthropic(client) => {
                let extractor = client
                    .extractor::<T>(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .build();
                ProviderExtractor::Anthropic(extractor)
            }
            ProviderClient::Gemini(client) => {
                let gen_cfg = GenerationConfig::default();
                let cfg = AdditionalParameters::default().with_config(gen_cfg);

                let extractor = client
                    .extractor::<T>(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into())
                    .additional_params(serde_json::to_value(cfg).unwrap())
                    .build();
                ProviderExtractor::Gemini(extractor)
            }
            ProviderClient::Ollama(client) => {
                // Create standard agent for Ollama
                let mut builder = client
                    .agent(model)
                    .preamble(system_prompt)
                    .max_tokens(config.max_tokens.into());

                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }

                let agent = builder.build();

                // Wrap with OllamaExtractorWrapper to handle structured output
                // Pass base_url and model for HTTP fallback when rig fails
                let wrapper = OllamaExtractorWrapper::with_config(
                    agent,
                    config.retry_attempts,
                    config.api_base_url.clone(),
                    model.to_string(),
                    config.stream_enabled(),
                );

                ProviderExtractor::Ollama(wrapper)
            }
        }
    }
}

/// Unified Agent enum
pub enum ProviderAgent {
    OpenAI {
        agent: Agent<OpenAIModel>,
        base_url: String,
        model: String,
        api_key: String,
        /// Kept so the raw HTTP fallback path can replay the same preamble
        /// (the rig agent itself cannot expose its configured preamble).
        system_prompt: String,
        max_tokens: u32,
        temperature: Option<f64>,
        stream: bool,
    },
    Mistral(AgentHandle<Agent<rig_core::providers::mistral::CompletionModel>>),
    OpenRouter(AgentHandle<Agent<rig_core::providers::openrouter::CompletionModel>>),
    Anthropic(AgentHandle<Agent<rig_core::providers::anthropic::completion::CompletionModel>>),
    Gemini(AgentHandle<Agent<rig_core::providers::gemini::completion::CompletionModel>>),
    Moonshot(AgentHandle<Agent<rig_core::providers::moonshot::CompletionModel>>),
    DeepSeek(AgentHandle<Agent<rig_core::providers::deepseek::CompletionModel>>),
    Ollama(AgentHandle<Agent<rig_core::providers::ollama::CompletionModel>>),
}

/// Agent payload plus the transport settings shared by the tuple-style
/// ProviderAgent variants: the model label (for progress reporting) and the
/// per-provider streaming flag resolved from `LLMConfig::stream_enabled`.
pub struct AgentHandle<A> {
    pub agent: A,
    pub model: String,
    pub stream: bool,
}

impl ProviderAgent {
    /// Execute prompt with HTTP fallback for OpenAI-compatible providers
    pub async fn prompt(&self, prompt: &str, concurrency: usize) -> Result<String> {
        let concurrency = concurrency.max(1);
        // Discard any stale captures from an uncovered caller so the sink
        // cannot grow unbounded; the real drain happens at the funnel.
        let _ = take_captures_for_current_task();
        let _ = usage_tracker::UsageTracker::global();
        match self {
            ProviderAgent::OpenAI { agent, base_url, model, api_key, system_prompt, max_tokens, temperature, stream } => {
                // Try rig agent first (streaming when enabled) with concurrency
                let rig_result = if *stream {
                    let stream_items = agent
                        .stream_prompt(prompt)
                        .tool_concurrency(concurrency)
                        .await;
                    drain_to_string(stream_items, Some(model)).await
                } else {
                    with_spinner(model, "calling", async {
                        agent.prompt(prompt).tool_concurrency(concurrency).await
                    })
                    .await
                    .map_err(anyhow::Error::from)
                };
                match rig_result {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        let error_msg = format!("{:?}", e);
                        // Check if it's an API response parsing error
                        if error_msg.contains("ApiResponse")
                            || error_msg.contains("untagged enum")
                            || error_msg.contains("JsonError")
                        {
                            // Fallback to direct HTTP call (SSE when streaming)
                            Self::prompt_via_http(
                                base_url,
                                model,
                                api_key,
                                system_prompt,
                                *max_tokens,
                                *temperature,
                                prompt,
                                *stream,
                            )
                            .await
                        } else {
                            Err(e)
                        }
                    }
                }
            }
            ProviderAgent::Moonshot(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
            ProviderAgent::DeepSeek(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
            ProviderAgent::Mistral(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
            ProviderAgent::OpenRouter(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
            ProviderAgent::Anthropic(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
            ProviderAgent::Gemini(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
            ProviderAgent::Ollama(handle) => {
                Self::prompt_single(&handle.agent, &handle.model, handle.stream, prompt, concurrency).await
            }
        }
    }

    /// Prompt a single agent without an HTTP fallback: stream (with tool
    /// concurrency preserved) when enabled, otherwise use the classic
    /// non-streaming request path.
    async fn prompt_single<M>(
        agent: &Agent<M>,
        model: &str,
        stream: bool,
        prompt: &str,
        concurrency: usize,
    ) -> Result<String>
    where
        M: CompletionModel + 'static,
        M::StreamingResponse: GetTokenUsage + WasmCompatSend,
    {
        if stream {
            let stream_items = agent
                .stream_prompt(prompt)
                .tool_concurrency(concurrency)
                .await;
            drain_to_string(stream_items, Some(model)).await
        } else {
            with_spinner(model, "calling", async {
                agent.prompt(prompt).tool_concurrency(concurrency).await
            })
            .await
            .map_err(|e| e.into())
        }
    }

    /// Direct HTTP call to OpenAI-compatible API
    ///
    /// Mirrors the rig agent configuration: the system prompt is sent as the
    /// `system` message and max_tokens / temperature come from config instead
    /// of hardcoded values (previously 4096 / 0.7, which silently truncated
    /// long documents and dropped the agent's role/output-format preamble).
    ///
    /// With `stream = true` the request uses SSE and collects the deltas into
    /// the same complete string the JSON path would have returned.
    #[allow(clippy::too_many_arguments)]
    async fn prompt_via_http(
        base_url: &str,
        model: &str,
        api_key: &str,
        system_prompt: &str,
        max_tokens: u32,
        temperature: Option<f64>,
        prompt: &str,
        stream: bool,
    ) -> Result<String> {
        // Streaming: bound per-read activity instead of a total request
        // timeout, which would otherwise kill long-running SSE responses.
        let client = if stream {
            reqwest::Client::builder()
                .read_timeout(std::time::Duration::from_secs(120))
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?
        } else {
            reqwest::Client::new()
        };

        let mut messages = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": prompt
        }));

        let mut request_body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        });
        if let Some(temp) = temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if stream {
            request_body["stream"] = serde_json::json!(true);
        }

        let request = client
            .post(format!("{}/chat/completions", base_url.trim_end_matches('/')))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body);
        let request = if stream {
            request
        } else {
            request.timeout(std::time::Duration::from_secs(120))
        };
        let response = request
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI-compatible API HTTP error {}: {}", status, body);
        }

        if stream {
            return collect_openai_sse(response, model).await;
        }

        with_spinner(model, "reading response", async {
            let status = response.status();
            let headers: Vec<(String, String)> = response
                .headers()
                .iter()
                .map(|(k, v)| {
                    (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())
                })
                .collect();
            let body_text = response
                .text()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to read HTTP response: {}", e))?;
            crate::llm::client::usage_capture::capture_response(
                crate::llm::client::usage_capture::CapturedResponse {
                    request_url: format!(
                        "{}/chat/completions",
                        base_url.trim_end_matches('/')
                    ),
                    request_model: Some(model.to_string()),
                    status: status.as_u16(),
                    headers,
                    body: body_text.clone(),
                    stream: false,
                },
            );
            let json: serde_json::Value = serde_json::from_str(&body_text)
                .map_err(|e| anyhow::anyhow!("Failed to parse HTTP response: {}", e))?;

            let content = json
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .ok_or_else(|| anyhow::anyhow!("Invalid OpenAI API response format"))?;

            Ok(content.to_string())
        })
        .await
    }
}

/// Unified Extractor enum
pub enum ProviderExtractor<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    OpenAI(OpenAICompatibleExtractorWrapper<T>),
    Mistral(Extractor<rig_core::providers::mistral::CompletionModel, T>),
    OpenRouter(Extractor<rig_core::providers::openrouter::CompletionModel, T>),
    Anthropic(Extractor<rig_core::providers::anthropic::completion::CompletionModel, T>),
    Gemini(Extractor<rig_core::providers::gemini::completion::CompletionModel, T>),
    Moonshot(Extractor<rig_core::providers::moonshot::CompletionModel, T>),
    DeepSeek(Extractor<rig_core::providers::deepseek::CompletionModel, T>),
    Ollama(OllamaExtractorWrapper<T>),
}

impl<T> ProviderExtractor<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    /// Execute extraction
    pub async fn extract(&self, prompt: &str) -> Result<T> {
        match self {
            ProviderExtractor::OpenAI(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::Moonshot(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::DeepSeek(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::Mistral(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::OpenRouter(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::Anthropic(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::Gemini(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
            ProviderExtractor::Ollama(extractor) => {
                extractor.extract(prompt).await.map_err(|e| e.into())
            }
        }
    }
}
