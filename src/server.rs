use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{sse::Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::SyncSender,
    Arc,
};
use std::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;

use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::metrics::Metrics;
use crate::scheduler::{self, GenerationParams, InferenceRequest, StreamEvent};

// ─── OpenAI-compatible types ─────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::One(stop) => vec![stop],
            StopSequences::Many(stops) => stops,
        }
    }
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
}

fn default_max_tokens() -> usize { 1024 }
fn default_temperature() -> f32 { 1.0 }
fn default_min_p() -> f32 { 0.05 }
fn default_repetition_penalty() -> f32 { 1.0 }

#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

pub struct ApiError {
    status: StatusCode,
    message: String,
    code: String,
}

impl ApiError {
    fn bad_request(code: &str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            code: code.to_string(),
        }
    }

    fn too_many_requests(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
            code: "queue_full".to_string(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            code: "internal_error".to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = ErrorResponse {
            error: ErrorDetail {
                message: self.message,
                error_type: "invalid_request_error".to_string(),
                code: self.code,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

// ─── Server state ────────────────────────────────────────────────────────────

pub struct AppState {
    pub request_tx: SyncSender<InferenceRequest>,
    pub metrics: Arc<Metrics>,
    pub tokenizer: tokenizers::Tokenizer,
    pub max_context_len: usize,
}

// ─── Chat template ───────────────────────────────────────────────────────────

const BUILT_IN_STOP_SEQUENCES: &[&str] = &[
    "<end_of_turn>",
    "<eos>",
    "<start_of_turn>",
    "</start_of_turn>",
];

fn apply_chat_template(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", msg.role, msg.content));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

fn find_stop_position(text: &str, request_stop: Option<&[String]>) -> Option<usize> {
    let mut earliest = BUILT_IN_STOP_SEQUENCES.iter().filter_map(|stop| text.find(stop)).min();

    if let Some(request_stop) = request_stop {
        let request_earliest = request_stop.iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| text.find(stop))
            .min();
        earliest = match (earliest, request_earliest) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }

    earliest
}

fn trim_stop_sequences(text: &mut String, request_stop: Option<&[String]>) -> bool {
    if let Some(pos) = find_stop_position(text, request_stop) {
        text.truncate(pos);
        return true;
    }
    false
}

fn stop_prefix_holdback_len(text: &str, request_stop: Option<&[String]>) -> usize {
    let mut max_len = 0;

    for (start, _) in text.char_indices() {
        let suffix = &text[start..];
        if BUILT_IN_STOP_SEQUENCES.iter().any(|stop| stop.starts_with(suffix))
            || request_stop
                .map(|stops| stops.iter().any(|stop| !stop.is_empty() && stop.starts_with(suffix)))
                .unwrap_or(false)
        {
            max_len = max_len.max(text.len() - start);
        }
    }

    max_len
}

fn trim_stream_safe_text(text: &mut String, request_stop: Option<&[String]>) -> bool {
    if trim_stop_sequences(text, request_stop) {
        return true;
    }

    let holdback_len = stop_prefix_holdback_len(text, request_stop);
    if holdback_len > 0 {
        text.truncate(text.len() - holdback_len);
    }

    false
}

// ─── Handlers ────────────────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "gemma-4-e4b-q4".to_string(),
            object: "model".to_string(),
            owned_by: "local".to_string(),
        }],
    })
}

fn enqueue_request(
    state: &AppState,
    input_ids: Vec<usize>,
    params: GenerationParams,
) -> Result<(tokio::sync::mpsc::Receiver<StreamEvent>, Arc<AtomicBool>), ApiError> {
    let (response_tx, response_rx) = tokio::sync::mpsc::channel(64);
    let cancel = Arc::new(AtomicBool::new(false));
    let prompt_tokens = input_ids.len();
    let request = InferenceRequest {
        id: format!("req-{}", uuid::Uuid::new_v4()),
        input_ids,
        params,
        response_tx,
        cancel: cancel.clone(),
        created_at: Instant::now(),
    };

    state.metrics.record_enqueue(prompt_tokens);
    if state.request_tx.try_send(request).is_err() {
        state.metrics.record_queue_full();
        return Err(ApiError::too_many_requests("scheduler queue is full"));
    }

    Ok((response_rx, cancel))
}

fn generation_params_from_request(req: &ChatCompletionRequest) -> Result<GenerationParams, ApiError> {
    validate_request(req)?;

    Ok(GenerationParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        min_p: req.min_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        eos_token_ids: vec![1, 106],
        request_timeout: Duration::from_secs(60),
    })
}

fn validate_request(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("empty_messages", "messages must not be empty"));
    }

    if req.max_tokens == 0 {
        return Err(ApiError::bad_request("invalid_max_tokens", "max_tokens must be greater than 0"));
    }

    if !req.temperature.is_finite() || req.temperature < 0.0 || req.temperature > 5.0 {
        return Err(ApiError::bad_request("invalid_temperature", "temperature must be between 0 and 5"));
    }

    if !req.min_p.is_finite() || req.min_p < 0.0 || req.min_p > 1.0 {
        return Err(ApiError::bad_request("invalid_min_p", "min_p must be between 0 and 1"));
    }

    if !req.repetition_penalty.is_finite() || req.repetition_penalty <= 0.0 || req.repetition_penalty > 10.0 {
        return Err(ApiError::bad_request(
            "invalid_repetition_penalty",
            "repetition_penalty must be greater than 0 and at most 10",
        ));
    }

    if !req.frequency_penalty.is_finite() || req.frequency_penalty < -2.0 || req.frequency_penalty > 2.0 {
        return Err(ApiError::bad_request(
            "invalid_frequency_penalty",
            "frequency_penalty must be between -2 and 2",
        ));
    }

    Ok(())
}

fn encode_prompt(state: &AppState, messages: &[Message]) -> Result<Vec<usize>, ApiError> {
    let prompt = apply_chat_template(messages);
    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|err| ApiError::bad_request("tokenizer_error", format!("failed to tokenize prompt: {}", err)))?;
    Ok(encoding.get_ids().iter().map(|&t| t as usize).collect())
}

fn validate_context_len(
    prompt_tokens: usize,
    max_tokens: usize,
    max_context_len: usize,
) -> Result<(), ApiError> {
    if prompt_tokens >= max_context_len {
        return Err(ApiError::bad_request(
            "context_length_exceeded",
            format!(
                "prompt has {} tokens, which exceeds the model context limit of {}",
                prompt_tokens, max_context_len
            ),
        ));
    }

    if prompt_tokens + max_tokens > max_context_len {
        return Err(ApiError::bad_request(
            "context_length_exceeded",
            format!(
                "prompt tokens ({}) plus max_tokens ({}) exceeds the model context limit of {}",
                prompt_tokens, max_tokens, max_context_len
            ),
        ));
    }

    Ok(())
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render_prometheus(),
    )
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    if req.stream {
        Ok(chat_completions_stream(state, req).await?.into_response())
    } else {
        Ok(chat_completions_sync(state, req).await?.into_response())
    }
}

use axum::response::IntoResponse;

async fn chat_completions_sync(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let generation_params = generation_params_from_request(&req)?;
    let input_ids = encode_prompt(&state, &req.messages)?;
    let prompt_tokens = input_ids.len();
    validate_context_len(prompt_tokens, generation_params.max_tokens, state.max_context_len)?;
    let request_stop = req.stop.map(StopSequences::into_vec);

    let (mut response_rx, cancel) =
        enqueue_request(&state, input_ids, generation_params)?;
    let mut output_tokens = Vec::new();
    let mut finish_reason = "stop".to_string();

    while let Some(event) = response_rx.recv().await {
        match event {
            StreamEvent::Token { token_id } => {
                output_tokens.push(token_id as u32);

                let decoded_so_far = state.tokenizer.decode(&output_tokens, false).unwrap_or_default();
                if find_stop_position(&decoded_so_far, request_stop.as_deref()).is_some() {
                    cancel.store(true, Ordering::Relaxed);
                    break;
                }
            }
            StreamEvent::Done { finish_reason: reason } => {
                finish_reason = reason;
                break;
            }
            StreamEvent::Error { message } => return Err(ApiError::internal(message)),
        }
    }

    let mut text = state.tokenizer.decode(&output_tokens, true).unwrap_or_default();
    trim_stop_sequences(&mut text, request_stop.as_deref());
    // Strip thinking/reasoning content (Gemma4 thinking mode)
    if text.starts_with("thought\n") {
        // Find the end of thinking block - look for double newline or end
        if let Some(end_pos) = text.find("\n...end_of_turn") {
            text = text[end_pos..].trim_start_matches("\n...end_of_turn").to_string();
        } else if let Some(end_pos) = text.rfind("\n\n") {
            // Take only the last paragraph as the actual response
            text = text[end_pos..].trim().to_string();
        }
    }
    let completion_tokens = output_tokens.len();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: "gemma-4-e4b-q4".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message { role: "assistant".to_string(), content: text },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response))
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let generation_params = generation_params_from_request(&req)?;
    let input_ids = encode_prompt(&state, &req.messages)?;
    validate_context_len(input_ids.len(), generation_params.max_tokens, state.max_context_len)?;
    let request_stop = req.stop.map(StopSequences::into_vec);
    let request_result = enqueue_request(&state, input_ids, generation_params)?;

    tokio::spawn(async move {
        // Send role delta first
        let role_chunk = ChatCompletionChunk {
            id: chat_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: "gemma-4-e4b-q4".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant".to_string()), content: None },
                finish_reason: None,
            }],
        };
        let _ = tx.send(Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap()))).await;

        let mut output_tokens = Vec::new();
        let mut emitted_text = String::new();

        let mut finish_reason = "stop".to_string();

        let (mut response_rx, cancel) = request_result;
        while let Some(event) = response_rx.recv().await {
            match event {
                StreamEvent::Token { token_id } => {
                    output_tokens.push(token_id as u32);
                    let mut visible_text = state.tokenizer.decode(&output_tokens, false).unwrap_or_default();
                    let stopped = trim_stream_safe_text(&mut visible_text, request_stop.as_deref());
                    let tok_str = if visible_text.starts_with(&emitted_text) {
                        visible_text[emitted_text.len()..].to_string()
                    } else {
                        String::new()
                    };
                    emitted_text = visible_text;

                    if !tok_str.is_empty() {
                        let chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: "gemma-4-e4b-q4".to_string(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta { role: None, content: Some(tok_str) },
                                finish_reason: None,
                            }],
                        };

                        if tx.send(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))).await.is_err() {
                            cancel.store(true, Ordering::Relaxed);
                            break;
                        }
                    }

                    if stopped {
                        cancel.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                StreamEvent::Done { finish_reason: reason } => {
                    finish_reason = reason;
                    break;
                }
                StreamEvent::Error { message } => {
                    finish_reason = format!("error: {}", message);
                    break;
                }
            }
        }

        // Send final chunk with finish_reason
        let done_chunk = ChatCompletionChunk {
            id: chat_id,
            object: "chat.completion.chunk".to_string(),
            created,
            model: "gemma-4-e4b-q4".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: None },
                finish_reason: Some(finish_reason),
            }],
        };
        let _ = tx.send(Ok(Event::default().data(serde_json::to_string(&done_chunk).unwrap()))).await;
        let _ = tx.send(Ok(Event::default().data("[DONE]".to_string()))).await;
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

pub async fn run_server(model: Gemma4GpuModel, tokenizer: tokenizers::Tokenizer, port: u16) {
    let max_context_len = model.kv_capacity as usize;
    let metrics = Arc::new(Metrics::new());
    let request_tx = scheduler::spawn_scheduler(model, 32, 4, metrics.clone());
    let state = Arc::new(AppState {
        request_tx,
        metrics,
        tokenizer,
        max_context_len,
    });

    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("🚀 Gemma4 E4B server listening on http://{}", addr);
    println!("   Compatible with OpenAI API: /v1/chat/completions");
    println!("   Models: /v1/models");
    println!("   Health: /health");
    println!("   Metrics: /metrics");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
