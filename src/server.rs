use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::sampling;

// ─── OpenAI-compatible types ─────────────────────────────────────────────────

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
    pub stop: Option<Vec<String>>,
}

fn default_max_tokens() -> usize { 1024 }
fn default_temperature() -> f32 { 1.0 }

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

// ─── Server state ────────────────────────────────────────────────────────────

pub struct AppState {
    pub model: Mutex<Gemma4GpuModel>,
    pub tokenizer: tokenizers::Tokenizer,
}

// ─── Chat template ───────────────────────────────────────────────────────────

fn apply_chat_template(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", msg.role, msg.content));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
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

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, StatusCode> {
    if req.stream {
        Ok(chat_completions_stream(state, req).await.into_response())
    } else {
        Ok(chat_completions_sync(state, req).await.into_response())
    }
}

use axum::response::IntoResponse;

async fn chat_completions_sync(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> impl IntoResponse {
    let prompt = apply_chat_template(&req.messages);
    let encoding = state.tokenizer.encode(prompt.as_str(), true).unwrap();
    let input_ids: Vec<usize> = encoding.get_ids().iter().map(|&t| t as usize).collect();
    let prompt_tokens = input_ids.len();

    let mut model = state.model.lock().unwrap();

    // Reset KV cache for new request
    model.kv_seq_len = 0;
    model.total_tokens = 0;

    // Prefill
    let mut logits = model.forward_prefill(&input_ids);

    // Decode
    let mut output_tokens = Vec::new();
    let eos_tokens: &[usize] = &[1, 106];

    for _ in 0..req.max_tokens {
        let next_token = sampling::sample(&logits, req.temperature, 0.05);

        if eos_tokens.contains(&next_token) {
            break;
        }
        output_tokens.push(next_token as u32);

        // Check if decoded text contains end-of-turn marker
        let decoded_so_far = state.tokenizer.decode(&output_tokens, false).unwrap_or_default();
        if decoded_so_far.contains("<end_of_turn>") || decoded_so_far.contains("<eos>") {
            // Trim the stop string from output
            break;
        }

        logits = model.forward_single_token(next_token);
    }

    drop(model);

    let mut text = state.tokenizer.decode(&output_tokens, true).unwrap_or_default();
    // Remove any trailing stop sequences
    if let Some(pos) = text.find("<end_of_turn>") {
        text.truncate(pos);
    }
    if let Some(pos) = text.find("<eos>") {
        text.truncate(pos);
    }
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
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Json(response)
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    tokio::task::spawn_blocking(move || {
        let prompt = apply_chat_template(&req.messages);
        let encoding = state.tokenizer.encode(prompt.as_str(), true).unwrap();
        let input_ids: Vec<usize> = encoding.get_ids().iter().map(|&t| t as usize).collect();

        let mut model = state.model.lock().unwrap();

        // Reset KV cache
        model.kv_seq_len = 0;
        model.total_tokens = 0;

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
        let _ = tx.blocking_send(Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap())));

        // Prefill
        let mut logits = model.forward_prefill(&input_ids);

        let eos_tokens: &[usize] = &[1, 106];

        for _ in 0..req.max_tokens {
            let next_token = sampling::sample(&logits, req.temperature, 0.05);

            if eos_tokens.contains(&next_token) {
                break;
            }

            let tok_str = state.tokenizer.decode(&[next_token as u32], false).unwrap_or_default();

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

            if tx.blocking_send(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))).is_err() {
                break; // Client disconnected
            }

            logits = model.forward_single_token(next_token);
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
                finish_reason: Some("stop".to_string()),
            }],
        };
        let _ = tx.blocking_send(Ok(Event::default().data(serde_json::to_string(&done_chunk).unwrap())));
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]".to_string())));
    });

    Sse::new(ReceiverStream::new(rx))
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

pub async fn run_server(model: Gemma4GpuModel, tokenizer: tokenizers::Tokenizer, port: u16) {
    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
    });

    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("🚀 Gemma4 E4B server listening on http://{}", addr);
    println!("   Compatible with OpenAI API: /v1/chat/completions");
    println!("   Models: /v1/models");
    println!("   Health: /health");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
