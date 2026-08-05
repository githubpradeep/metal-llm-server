//! Thin serial OpenAI-compatible serve path for DeepSeek-V4-Flash.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::stream::StreamExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokenizers::Tokenizer;

use super::{
    encode_chat_user_ids, Dsv4GpuModel, DSV4_DEFAULT_SYSTEM, DSV4_EOS_ID,
};

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<Dsv4GpuModel>>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
}

fn default_max() -> usize {
    128
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

fn encode_messages(tok: &Tokenizer, messages: &[ChatMessage], nothink: bool) -> Vec<usize> {
    let mut system = None;
    let mut user_parts = Vec::new();
    for m in messages {
        let c = m.content.as_deref().unwrap_or("");
        match m.role.as_str() {
            "system" => system = Some(c.to_string()),
            "user" => user_parts.push(c.to_string()),
            "assistant" => {
                if let Some(last) = user_parts.last_mut() {
                    last.push_str("\n\nAssistant: ");
                    last.push_str(c);
                }
            }
            _ => {}
        }
    }
    let user = user_parts.join("\n");
    let sys = system.as_deref().unwrap_or(DSV4_DEFAULT_SYSTEM);
    encode_chat_user_ids(tok, &user, nothink, Some(sys))
}

async fn health() -> impl IntoResponse {
    Json(json!({"status":"ok","arch":"deepseek4"}))
}

async fn list_models(State(st): State<AppState>) -> impl IntoResponse {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: st.model_id.clone(),
            object: "model",
            owned_by: "llama-sinks",
        }],
    })
}

fn sample_next(logits: &[f32], temperature: f32) -> usize {
    if temperature <= 1e-5 {
        return Dsv4GpuModel::sample_greedy(logits);
    }
    let inv_t = 1.0 / temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, &l) in logits.iter().enumerate() {
        let p = ((l - max_l) * inv_t).exp();
        probs[i] = p;
        sum += p;
    }
    let mut r: f32 = rand::thread_rng().gen::<f32>() * sum;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

async fn chat_completions(State(st): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let ids = encode_messages(&st.tokenizer, &req.messages, true);
    let max_new = req.max_tokens.max(1).min(2048);
    let temperature = req.temperature.unwrap_or(0.0);
    let model_id = st.model_id.clone();

    if req.stream {
        let (tx, rx) = mpsc::channel::<String>(16);
        let model = st.model.clone();
        let tok = st.tokenizer.clone();
        tokio::task::spawn_blocking(move || {
            let mut m = model.lock().unwrap();
            m.reset();
            let mut logits = m.forward_prefill(&ids);
            let _ = tx.blocking_send(format!(
                "{}",
                json!({
                    "id": "dsv4-1",
                    "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]
                })
            ));
            for _ in 0..max_new {
                let next = sample_next(&logits, temperature);
                if next as u32 == DSV4_EOS_ID {
                    break;
                }
                let piece = tok.decode(&[next as u32], true).unwrap_or_default();
                let _ = tx.blocking_send(format!(
                    "{}",
                    json!({
                        "id": "dsv4-1",
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [{"index":0,"delta":{"content": piece},"finish_reason":null}]
                    })
                ));
                logits = m.forward_token_logits(next);
            }
            let _ = tx.blocking_send(format!(
                "{}",
                json!({
                    "id": "dsv4-1",
                    "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [{"index":0,"delta":{},"finish_reason":"stop"}]
                })
            ));
            let _ = tx.blocking_send("[DONE]".to_string());
        });
        let stream = ReceiverStream::new(rx).map(|msg| {
            Ok::<Event, Infallible>(Event::default().data(msg))
        });
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let model = st.model.clone();
    let tok = st.tokenizer.clone();
    let text = tokio::task::spawn_blocking(move || {
        let mut m = model.lock().unwrap();
        m.reset();
        let mut logits = m.forward_prefill(&ids);
        let mut out_ids = Vec::new();
        for _ in 0..max_new {
            let next = sample_next(&logits, temperature);
            if next as u32 == DSV4_EOS_ID {
                break;
            }
            out_ids.push(next as u32);
            logits = m.forward_token_logits(next);
        }
        tok.decode(&out_ids, true).unwrap_or_default()
    })
    .await
    .unwrap_or_default();

    Json(json!({
        "id": "dsv4-1",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    }))
    .into_response()
}

/// Run a serial DeepSeek OpenAI-compatible server (single request at a time).
pub async fn run_dsv4_server(model: Dsv4GpuModel, tokenizer: Tokenizer, port: u16) {
    let model_id = "deepseek-v4-flash".to_string();
    let st = AppState {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        model_id,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(st);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!("DeepSeek-V4 Flash server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}
