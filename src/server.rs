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
    atomic::{AtomicU8, Ordering},
    mpsc::SyncSender,
    Arc,
};
use std::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;

use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::metrics::Metrics;
use crate::scheduler::{
    self, GenerationParams, InferenceRequest, StreamEvent, CANCEL_CLIENT, CANCEL_NONE, CANCEL_STOP,
};

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
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
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
    // Accepted for OpenAI compatibility (not currently used by the sampler).
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    // Accepted for OpenAI compatibility; tool selection is left to the model.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize, Clone)]
pub struct Tool {
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,
    pub function: FunctionDef,
}

#[derive(Deserialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string, per the OpenAI spec.
    pub arguments: String,
}

#[derive(Serialize, Clone)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: FunctionCallDelta,
}

#[derive(Serialize)]
struct AssistantMessageOut {
    role: String,
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

fn strip_native_tool_calls(text: &mut String) {
    while let Some(start) = text.find(NATIVE_TOOL_CALL_PREFIX) {
        let suffix_start = text[start..].find(NATIVE_TOOL_CALL_SUFFIX);
        match suffix_start {
            Some(end) => {
                text.replace_range(start..start + end + NATIVE_TOOL_CALL_SUFFIX.len(), "");
            }
            None => break,
        }
    }
    *text = text.trim().to_string();
}

fn default_max_tokens() -> usize {
    1024
}

fn effective_max_tokens(req: &ChatCompletionRequest) -> usize {
    req.max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or_else(default_max_tokens)
}

fn response_model(req: &ChatCompletionRequest) -> &str {
    req.model.as_deref().unwrap_or("gemma-4-e4b-q4")
}

fn assistant_message_out(
    text: String,
    tool_calls: Vec<ToolCall>,
    split_mode: ChannelSplitMode,
) -> AssistantMessageOut {
    let (reasoning, mut content) = if !tool_calls.is_empty() {
        split_tool_generation_output(&text)
    } else {
        let (reasoning, content) = split_reasoning_and_content_with_mode(&text, split_mode);
        finalize_reasoning_content_split(reasoning, content, &text, split_mode)
    };
    strip_native_tool_calls(&mut content);
    let mut reasoning_content = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning.clone())
    };

    if !tool_calls.is_empty() {
        AssistantMessageOut {
            role: "assistant".to_string(),
            content: Some(serde_json::Value::Null),
            reasoning_content,
            tool_calls: Some(tool_calls),
        }
    } else {
        AssistantMessageOut {
            role: "assistant".to_string(),
            content: Some(serde_json::Value::String(content)),
            reasoning_content,
            tool_calls: None,
        }
    }
}

fn strip_thinking_content(text: &mut String) {
    let (_, content) = split_reasoning_and_content(text);
    *text = content;
}

#[derive(Clone, Copy, Default)]
struct ChannelSplitMode {
    /// Plain text with no channel markup is reasoning (tool-awaiting / primed-thought turn).
    plain_text_as_reasoning: bool,
}

fn has_channel_markup(text: &str) -> bool {
    text.contains(CHANNEL_START) || text.contains(CHANNEL_END)
}

fn parse_named_channel_body(body: &str) -> (String, String) {
    if let Some(newline) = body.find('\n') {
        (
            body[..newline].trim().to_string(),
            body[newline + 1..].trim().to_string(),
        )
    } else {
        (body.trim().to_string(), String::new())
    }
}

fn join_visible_parts(parts: &[String]) -> String {
    parts
        .iter()
        .map(String::as_str)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split only when `<|channel>final` is present. Standalone `<channel|>` closes a
/// thought segment but does not start the visible answer; text after an explicit
/// `<|channel>thought` block is treated as content.
fn split_channel_markup(text: &str) -> (String, String) {
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();
    let mut rest = text;
    let mut after_explicit_thought = false;

    loop {
        if let Some(start) = rest.find(CHANNEL_START) {
            if start > 0 {
                let segment = rest[..start].trim();
                if !segment.is_empty() {
                    if after_explicit_thought {
                        content_parts.push(segment.to_string());
                        after_explicit_thought = false;
                    } else {
                        reasoning_parts.push(segment.to_string());
                    }
                }
            }
            rest = &rest[start + CHANNEL_START.len()..];
            if let Some(end) = rest.find(CHANNEL_END) {
                let body = &rest[..end];
                rest = &rest[end + CHANNEL_END.len()..];
                let (name, body_text) = parse_named_channel_body(body);
                match name.as_str() {
                    "final" => {
                        after_explicit_thought = false;
                        if !body_text.is_empty() {
                            content_parts.push(body_text);
                        }
                    }
                    "thought" => {
                        after_explicit_thought = true;
                        if !body_text.is_empty() {
                            reasoning_parts.push(body_text);
                        }
                    }
                    _ => {
                        after_explicit_thought = false;
                        if !body_text.is_empty() {
                            reasoning_parts.push(body_text);
                        }
                    }
                }
                continue;
            }

            let (name, body_text) = parse_named_channel_body(rest);
            match name.as_str() {
                "final" => {
                    if !body_text.is_empty() {
                        content_parts.push(body_text);
                    }
                }
                "thought" => {
                    if !body_text.is_empty() {
                        reasoning_parts.push(body_text);
                    }
                }
                _ => {
                    if !body_text.is_empty() {
                        reasoning_parts.push(body_text);
                    }
                }
            }
            break;
        }

        if let Some(end) = rest.find(CHANNEL_END) {
            let segment = rest[..end].trim();
            if !segment.is_empty() {
                reasoning_parts.push(segment.to_string());
            }
            rest = &rest[end + CHANNEL_END.len()..];
            after_explicit_thought = false;
            continue;
        }

        let segment = rest.trim();
        if !segment.is_empty() {
            if after_explicit_thought {
                content_parts.push(segment.to_string());
            } else if content_parts.is_empty() {
                reasoning_parts.push(segment.to_string());
            } else {
                content_parts.push(segment.to_string());
            }
        }
        break;
    }

    let mut reasoning = join_visible_parts(&reasoning_parts);
    let mut content = join_visible_parts(&content_parts);
    trim_stop_sequences(&mut reasoning, None);
    trim_stop_sequences(&mut content, None);
    (reasoning, content)
}

fn find_implicit_answer_start(text: &str) -> Option<usize> {
    const MARKERS: &[&str] = &[
        "\n\nThis is ",
        "\n\nHere is ",
        "\n\nThe film ",
        "\n\n The film ",
        "\n\nThe movie ",
        "\n\n The movie ",
        "\n\nOnce Upon ",
        "coherent summary.\n\n",
        "concise summary.\n\n",
        ".\nThis is ",
        ".\nHere is ",
        ".\nThe film ",
        ".\nThe movie ",
        "\nThis is ",
        "\nHere is ",
        "\nThe film ",
        "\nThe movie ",
        ".This is ",
        ".Here is ",
    ];
    let marker_start = MARKERS
        .iter()
        .filter_map(|marker| {
            text.rfind(marker).map(|idx| {
                if let Some(after_newline) = marker.rfind('\n') {
                    idx + after_newline + 1
                } else {
                    idx + 1
                }
            })
        })
        .max();
    match (marker_start, find_paragraph_answer_start(text)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// After a blank line, treat non-planning paragraphs as the deliverable answer.
fn find_paragraph_answer_start(text: &str) -> Option<usize> {
    const PLANNING_PREFIXES: &[&str] = &[
        "User ",
        "The user ",
        "I need ",
        "I will ",
        "I should ",
        "Let me ",
        "**Analysis",
        "Analysis",
        "Okay",
        "Ok,",
        "First,",
        "Step ",
    ];

    let mut best: Option<usize> = None;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            let mut para_start = i + 2;
            while para_start < bytes.len() && bytes[para_start] == b' ' {
                para_start += 1;
            }
            if para_start < bytes.len() {
                let rest = text[para_start..].trim_start();
                if !rest.is_empty() {
                    let is_planning = PLANNING_PREFIXES
                        .iter()
                        .any(|prefix| rest.starts_with(prefix));
                    if !is_planning && rest.len() >= 15 {
                        let actual_start = para_start + (text[para_start..].len() - rest.len());
                        best = Some(best.map_or(actual_start, |b| b.max(actual_start)));
                    }
                }
            }
            i = para_start.max(i + 2);
        } else {
            i += 1;
        }
    }
    best
}

fn finalize_reasoning_content_split(
    reasoning: String,
    content: String,
    raw: &str,
    mode: ChannelSplitMode,
) -> (String, String) {
    if !content.is_empty() {
        return (reasoning, content);
    }
    if !has_channel_markup(raw) && !mode.plain_text_as_reasoning {
        return (String::new(), reasoning);
    }
    apply_implicit_answer_split(reasoning, content)
}

fn apply_implicit_answer_split(reasoning: String, content: String) -> (String, String) {
    if !content.is_empty() {
        return (reasoning, content);
    }
    if let Some(idx) = find_implicit_answer_start(&reasoning) {
        let answer = reasoning[idx..].trim().to_string();
        let kept = reasoning[..idx].trim().to_string();
        if !answer.is_empty() {
            return (kept, answer);
        }
    }
    (reasoning, content)
}

fn split_reasoning_and_content_with_mode(
    text: &str,
    mode: ChannelSplitMode,
) -> (String, String) {
    if !has_channel_markup(text) {
        let mut plain = text.to_string();
        trim_stop_sequences(&mut plain, None);
        if mode.plain_text_as_reasoning {
            return apply_implicit_answer_split(plain, String::new());
        }
        return (String::new(), plain);
    }
    let (reasoning, content) = split_channel_markup(text);
    apply_implicit_answer_split(reasoning, content)
}

fn split_reasoning_and_content(text: &str) -> (String, String) {
    split_reasoning_and_content_with_mode(text, ChannelSplitMode::default())
}

/// When tools are enabled, model output before a tool call is reasoning (even if the
/// prompt already closed an empty `<|channel>thought` block).
fn split_tool_generation_output(text: &str) -> (String, String) {
    if let Some(idx) = text.find(NATIVE_TOOL_CALL_TRIGGER) {
        let mut reasoning = text[..idx].to_string();
        trim_stop_sequences(&mut reasoning, None);
        return (reasoning, String::new());
    }
    if looks_like_tool_code_output(text) {
        if let Some(idx) = text.to_ascii_lowercase().find("tool_code") {
            let reasoning = text[..idx].trim().to_string();
            return (reasoning, String::new());
        }
        return (String::new(), String::new());
    }
    if text.contains(CHANNEL_START) || text.contains(CHANNEL_END) {
        return split_reasoning_and_content_with_mode(
            text,
            ChannelSplitMode {
                plain_text_as_reasoning: true,
            },
        );
    }
    let mut reasoning = text.to_string();
    trim_stop_sequences(&mut reasoning, None);
    (reasoning, String::new())
}

/// During streaming on a tool-awaiting turn, use channel tags for reasoning/content
/// when the model emits them. Fall back to tool-style split only for plain pre-tool
/// preamble or while a native tool call is being streamed.
fn use_tool_generation_stream_split(
    visible_text: &str,
    in_tool_call: bool,
    tool_generation_mode: bool,
) -> bool {
    if in_tool_call {
        return true;
    }
    if !tool_generation_mode {
        return false;
    }
    if visible_text.contains(CHANNEL_START) || visible_text.contains(CHANNEL_END) {
        return false;
    }
    true
}

fn compute_stream_deltas(
    visible_text: &str,
    emitted_reasoning_len: usize,
    emitted_content_len: usize,
    for_tools: bool,
    split_mode: ChannelSplitMode,
) -> (String, String, usize, usize) {
    let (reasoning, content) = if for_tools {
        split_tool_generation_output(visible_text)
    } else {
        split_reasoning_and_content_with_mode(visible_text, split_mode)
    };
    if for_tools {
        let new_reasoning = if reasoning.len() > emitted_reasoning_len {
            reasoning[emitted_reasoning_len..].to_string()
        } else {
            String::new()
        };
        let new_content = if content.len() > emitted_content_len {
            content[emitted_content_len..].to_string()
        } else {
            String::new()
        };
        (new_reasoning, new_content, reasoning.len(), content.len())
    } else {
        let new_reasoning = if reasoning.len() > emitted_reasoning_len {
            reasoning[emitted_reasoning_len..].to_string()
        } else {
            String::new()
        };
        let new_content = if content.len() > emitted_content_len {
            content[emitted_content_len..].to_string()
        } else {
            String::new()
        };
        // When content shrinks (reclassified from content to reasoning
        // because the model just emitted a standalone `<channel|>`
        // delimiter), suppress the duplicate re-emission — the text was
        // already sent as content in earlier deltas.
        let new_reasoning = if content.len() < emitted_content_len {
            String::new()
        } else {
            new_reasoning
        };
        (new_reasoning, new_content, reasoning.len(), content.len())
    }
}

fn gemma4_string(value: &str) -> String {
    format!("<|\"|>{}<|\"|>", value)
}

fn json_value_to_gemma4(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => gemma4_string(s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(items) => {
            let inner = items
                .iter()
                .map(json_value_to_gemma4)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        serde_json::Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(k, v)| format!("{k}:{}", json_value_to_gemma4(v)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

fn json_schema_to_gemma4_params(schema: &serde_json::Value) -> String {
    let Some(obj) = schema.as_object() else {
        return json_value_to_gemma4(schema);
    };

    let mut members = Vec::new();
    if let Some(properties) = obj.get("properties").and_then(|v| v.as_object()) {
        let props: Vec<String> = properties
            .iter()
            .map(|(key, value)| gemma4_schema_property(key, value))
            .collect();
        members.push(format!("properties:{{{}}}", props.join(",")));
    }
    if let Some(required) = obj.get("required").and_then(|v| v.as_array()) {
        let req: Vec<String> = required
            .iter()
            .filter_map(|value| value.as_str().map(gemma4_string))
            .collect();
        members.push(format!("required:[{}]", req.join(",")));
    }
    if let Some(typ) = obj.get("type").and_then(|v| v.as_str()) {
        members.push(format!("type:{}", gemma4_string(&typ.to_uppercase())));
    }

    if members.is_empty() {
        "{}".to_string()
    } else {
        format!("{{{}}}", members.join(","))
    }
}

fn gemma4_schema_property(name: &str, schema: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    if let Some(description) = schema.get("description").and_then(|v| v.as_str()) {
        parts.push(format!("description:{}", gemma4_string(description)));
    }
    if let Some(typ) = schema.get("type").and_then(|v| v.as_str()) {
        parts.push(format!("type:{}", gemma4_string(&typ.to_uppercase())));
    }
    format!("{name}:{{{}}}", parts.join(","))
}

fn looks_like_tool_code_output(text: &str) -> bool {
    let lower = text.trim_start().to_ascii_lowercase();
    lower.starts_with("tool_code") || "tool_code".starts_with(&lower)
}

fn parse_tool_code_fallback(text: &str) -> Option<ToolCall> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    if !first.eq_ignore_ascii_case("tool_code") {
        return None;
    }
    let name = lines.next()?.to_string();
    let command = lines.collect::<Vec<_>>().join(" ");
    let arguments = if name == "bash" {
        let cmd = match command.as_str() {
            "" => "ls -F".to_string(),
            "ls" => "ls -F".to_string(),
            other => other.to_string(),
        };
        serde_json::json!({ "command": cmd }).to_string()
    } else {
        "{}".to_string()
    };
    Some(ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: FunctionCall { name, arguments },
    })
}

fn render_gemma4_tool_declarations(tools: &[Tool]) -> String {
    let mut s = String::new();
    for tool in tools {
        let f = &tool.function;
        let description = gemma4_string(f.description.as_deref().unwrap_or(""));
        let parameters = f
            .parameters
            .as_ref()
            .map(json_schema_to_gemma4_params)
            .unwrap_or_else(|| "{}".to_string());
        s.push_str(GEMMA4_TOOL_START);
        s.push_str("declaration:");
        s.push_str(&f.name);
        s.push_str("{description:");
        s.push_str(&description);
        s.push_str(",parameters:");
        s.push_str(&parameters);
        s.push('}');
        s.push_str(GEMMA4_TOOL_END);
    }
    s
}

fn render_assistant_tool_call_native(tc: &ToolCall) -> String {
    let args = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
        .unwrap_or_else(|_| serde_json::json!({}));
    format!(
        "{NATIVE_TOOL_CALL_PREFIX}{}{}<tool_call|>",
        tc.function.name,
        json_value_to_gemma4(&args)
    )
}

fn render_tool_response_native(name: &str, content: &str) -> String {
    format!(
        "{NATIVE_TOOL_RESPONSE_PREFIX}{name}{{value:{}}}{NATIVE_TOOL_RESPONSE_SUFFIX}",
        gemma4_string(content)
    )
}

/// Best-effort JSON args string from a partial or complete gemma4 dict (`{key:<|"|>val...`).
fn gemma4_dict_to_json_args_incremental(dict: &str) -> String {
    const S: &str = "<|\"|>";
    let trimmed = dict.trim();
    let has_close = trimmed.ends_with('}');
    let body = trimmed
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim_end_matches(',');

    let mut json = String::from("{");
    let mut rest = body;
    let mut first = true;

    while !rest.is_empty() {
        let Some(colon) = rest.find(':') else {
            break;
        };
        let key = rest[..colon].trim();
        if key.is_empty() {
            break;
        }
        rest = rest[colon + 1..].trim_start();

        if !first {
            json.push(',');
        }
        first = false;
        json.push('"');
        json.push_str(key);
        json.push_str("\":");

        if rest.starts_with(S) {
            let after = &rest[S.len()..];
            if let Some(end) = after.find(S) {
                json.push('"');
                for ch in after[..end].chars() {
                    match ch {
                        '"' => json.push_str("\\\""),
                        '\\' => json.push_str("\\\\"),
                        c => json.push(c),
                    }
                }
                json.push('"');
                rest = after[end + S.len()..].trim_start();
                if rest.starts_with(',') {
                    rest = rest[1..].trim_start();
                }
            } else {
                json.push('"');
                for ch in after.chars() {
                    match ch {
                        '"' => json.push_str("\\\""),
                        '\\' => json.push_str("\\\\"),
                        c => json.push(c),
                    }
                }
                break;
            }
        } else if rest.starts_with("true") {
            json.push_str("true");
            rest = rest[4..].trim_start();
            if rest.starts_with(',') {
                rest = rest[1..].trim_start();
            }
        } else if rest.starts_with("false") {
            json.push_str("false");
            rest = rest[5..].trim_start();
            if rest.starts_with(',') {
                rest = rest[1..].trim_start();
            }
        } else if rest.starts_with("null") {
            json.push_str("null");
            rest = rest[4..].trim_start();
            if rest.starts_with(',') {
                rest = rest[1..].trim_start();
            }
        } else {
            let end = rest
                .find(',')
                .unwrap_or(rest.len());
            let num = rest[..end].trim();
            if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+')
            {
                json.push_str(num);
                rest = rest[end..].trim_start();
                if rest.starts_with(',') {
                    rest = rest[1..].trim_start();
                }
            } else {
                break;
            }
        }
    }

    if has_close {
        json.push('}');
    }
    json
}

fn extract_active_tool_call(text: &str) -> Option<(String, String)> {
    let idx = text.rfind(NATIVE_TOOL_CALL_TRIGGER)?;
    let after_trigger = &text[idx + NATIVE_TOOL_CALL_TRIGGER.len()..];
    let after = after_trigger
        .strip_prefix("call:")
        .or_else(|| after_trigger.strip_prefix("call"))?;
    let brace = after.find('{')?;
    let name = after[..brace].trim();
    if name.is_empty() {
        return None;
    }
    let end = after.find(NATIVE_TOOL_CALL_SUFFIX).unwrap_or(after.len());
    let dict = after[brace..end].to_string();
    Some((name.to_string(), dict))
}

fn allowed_tool_names(tools: &[Tool]) -> Vec<String> {
    tools.iter().map(|t| t.function.name.clone()).collect()
}

fn is_allowed_tool_name(name: &str, allowed: &[String]) -> bool {
    allowed.iter().any(|n| n == name)
}

fn is_complete_json_object(arguments: &str) -> bool {
    let trimmed = arguments.trim();
    trimmed.starts_with('{') && trimmed.ends_with('}')
}

fn bash_args_missing_command(arguments: &str) -> bool {
    let trimmed = arguments.trim();
    if trimmed.is_empty() || trimmed == "{" {
        return false;
    }
    if trimmed == "{}" {
        return true;
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(serde_json::Value::Object(map)) => map
            .get("command")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().is_empty())
            .unwrap_or(true),
        Ok(_) => true,
        Err(_) => trimmed.ends_with('}') && !trimmed.contains("command:"),
    }
}

fn fill_default_tool_args(name: &str, arguments: &str) -> (String, String) {
    if name == "bash"
        && is_complete_json_object(arguments)
        && bash_args_missing_command(arguments)
    {
        return (
            name.to_string(),
            r#"{"command":"ls -F"}"#.to_string(),
        );
    }
    (name.to_string(), arguments.to_string())
}

fn tool_call_arguments_header(json_args: &str) -> String {
    if is_complete_json_object(json_args) {
        json_args.to_string()
    } else if json_args.starts_with('{') {
        "{".to_string()
    } else {
        String::new()
    }
}

fn tool_call_arguments_delta(emitted: &str, target: &str) -> Option<String> {
    if target.len() <= emitted.len() {
        return None;
    }
    if !target.starts_with(emitted) {
        return None;
    }
    let delta = &target[emitted.len()..];
    let accumulated = format!("{}{}", emitted, delta);
    if accumulated == "{}" {
        return None;
    }
    Some(delta.to_string())
}

/// Map hallucinated tool names (e.g. `list_files`) to declared tools (`bash`).
fn normalize_tool_call(name: &str, arguments: &str, allowed: &[String]) -> (String, String) {
    if allowed.is_empty() || is_allowed_tool_name(name, allowed) {
        return fill_default_tool_args(name, arguments);
    }

    let lower = name.to_lowercase().replace('-', "_");
    let args_empty = arguments.trim().is_empty() || arguments.trim() == "{}";

    if (lower.contains("list") || lower == "ls" || lower == "dir")
        && allowed.iter().any(|n| n == "bash")
    {
        let args = if args_empty {
            r#"{"command":"ls -F"}"#.to_string()
        } else {
            arguments.to_string()
        };
        return fill_default_tool_args("bash", &args);
    }

    if lower.contains("read") && allowed.iter().any(|n| n == "read") {
        return fill_default_tool_args("read", arguments);
    }

    if lower.contains("write") && allowed.iter().any(|n| n == "write") {
        return fill_default_tool_args("write", arguments);
    }

    if lower.contains("edit") && allowed.iter().any(|n| n == "edit") {
        return fill_default_tool_args("edit", arguments);
    }

    if (lower.contains("bash")
        || lower.contains("shell")
        || lower.contains("exec")
        || lower.contains("run"))
        && allowed.iter().any(|n| n == "bash")
    {
        return fill_default_tool_args("bash", arguments);
    }

    if let Some(best) = allowed.iter().find(|n| {
        let nl = n.to_lowercase();
        lower.starts_with(&nl) || nl.starts_with(&lower)
    }) {
        return fill_default_tool_args(best, arguments);
    }

    fill_default_tool_args(name, arguments)
}

fn normalize_tool_calls(calls: Vec<ToolCall>, allowed: &[String]) -> Vec<ToolCall> {
    if allowed.is_empty() {
        return calls;
    }
    calls
        .into_iter()
        .map(|mut tc| {
            let (name, args) =
                normalize_tool_call(&tc.function.name, &tc.function.arguments, allowed);
            tc.function.name = name;
            tc.function.arguments = args;
            tc
        })
        .collect()
}

struct IncrementalToolCallEmitter {
    call_id: String,
    index: usize,
    json_args_emitted: String,
    header_sent: bool,
}

impl IncrementalToolCallEmitter {
    fn new(index: usize) -> Self {
        Self {
            call_id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            index,
            json_args_emitted: String::new(),
            header_sent: false,
        }
    }

    fn update(
        &mut self,
        decoded_text: &str,
        id: &str,
        created: i64,
        model: &str,
        allowed_tools: &[String],
    ) -> Vec<String> {
        let mut chunks = Vec::new();
        let Some((raw_name, dict)) = extract_active_tool_call(decoded_text) else {
            return chunks;
        };

        let json_args = gemma4_dict_to_json_args_incremental(&dict);
        let (name, json_args) = normalize_tool_call(&raw_name, &json_args, allowed_tools);

        if !self.header_sent {
            if !dict.contains('{') && json_args.is_empty() {
                return chunks;
            }
            self.header_sent = true;
            let first_args = tool_call_arguments_header(&json_args);
            chunks.push(stream_chunk_json(
                id,
                created,
                model,
                serde_json::json!({
                    "tool_calls": [{
                        "index": self.index,
                        "id": self.call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": first_args
                        }
                    }]
                }),
                None,
            ));
            self.json_args_emitted = first_args.clone();
            if let Some(delta) = tool_call_arguments_delta(&self.json_args_emitted, &json_args) {
                chunks.push(stream_chunk_json(
                    id,
                    created,
                    model,
                    serde_json::json!({
                        "tool_calls": [{
                            "index": self.index,
                            "function": {
                                "arguments": delta
                            }
                        }]
                    }),
                    None,
                ));
                self.json_args_emitted = json_args.clone();
            }
            return chunks;
        }

        if let Some(delta) = tool_call_arguments_delta(&self.json_args_emitted, &json_args) {
            chunks.push(stream_chunk_json(
                id,
                created,
                model,
                serde_json::json!({
                    "tool_calls": [{
                        "index": self.index,
                        "function": {
                            "arguments": delta
                        }
                    }]
                }),
                None,
            ));
            self.json_args_emitted = json_args;
        }

        chunks
    }

    fn finalize(
        &mut self,
        decoded_text: &str,
        id: &str,
        created: i64,
        model: &str,
        allowed_tools: &[String],
    ) -> Vec<String> {
        let mut chunks = self.update(decoded_text, id, created, model, allowed_tools);
        let parsed = parse_tool_calls(
            decoded_text,
            if allowed_tools.is_empty() {
                None
            } else {
                Some(allowed_tools)
            },
            true,
        );
        let Some(tc) = parsed.first() else {
            return chunks;
        };
        if let Some(delta) = tool_call_arguments_delta(&self.json_args_emitted, &tc.function.arguments) {
            chunks.push(stream_chunk_json(
                id,
                created,
                model,
                serde_json::json!({
                    "tool_calls": [{
                        "index": self.index,
                        "function": {
                            "arguments": delta
                        }
                    }]
                }),
                None,
            ));
            self.json_args_emitted = tc.function.arguments.clone();
        }
        chunks
    }
}

fn gemma4_dict_to_json_object(dict: &str) -> Option<serde_json::Value> {
    const GEMMA4_STR: &str = "<|\"|>";
    let mut out = String::new();
    let mut rest = dict.trim();
    while let Some(start) = rest.find(GEMMA4_STR) {
        out.push_str(&rest[..start]);
        let after = &rest[start + GEMMA4_STR.len()..];
        let end = after.find(GEMMA4_STR)?;
        out.push('"');
        out.push_str(&after[..end]);
        out.push('"');
        rest = &after[end + GEMMA4_STR.len()..];
    }
    out.push_str(rest);

    let body = out.trim().trim_start_matches('{').trim_end_matches('}');
    if body.is_empty() {
        return Some(serde_json::json!({}));
    }

    let mut map = serde_json::Map::new();
    for pair in body.split(',') {
        let (key, value) = pair.split_once(':')?;
        let key = key.trim().trim_matches('"');
        let value = value.trim();
        let parsed = if value.starts_with('"') {
            serde_json::from_str(value).ok()?
        } else if value == "true" || value == "false" {
            serde_json::Value::Bool(value == "true")
        } else if value == "null" {
            serde_json::Value::Null
        } else if let Ok(n) = value.parse::<f64>() {
            serde_json::Number::from_f64(n)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(value.to_string()))
        } else {
            serde_json::Value::String(value.to_string())
        };
        map.insert(key.to_string(), parsed);
    }
    Some(serde_json::Value::Object(map))
}

fn decode_output_text(
    tokenizer: &tokenizers::Tokenizer,
    output_tokens: &[u32],
    request_stop: Option<&[String]>,
) -> String {
    let mut text = tokenizer
        .decode(output_tokens, true)
        .unwrap_or_default();
    trim_stop_sequences(&mut text, request_stop);
    strip_thinking_content(&mut text);
    text
}

fn stream_chunk_json(
    id: &str,
    created: i64,
    model: &str,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
    .to_string()
}

struct StreamTimingsSnapshot {
    prompt_ms: f64,
    predicted_ms: f64,
    prompt_tokens: usize,
    completion_tokens: usize,
}

fn stream_usage_chunk_json(
    id: &str,
    created: i64,
    model: &str,
    timings: &StreamTimingsSnapshot,
) -> String {
    let prompt_per_token_ms = if timings.prompt_tokens > 0 {
        timings.prompt_ms / timings.prompt_tokens as f64
    } else {
        0.0
    };
    let predicted_per_token_ms = if timings.completion_tokens > 0 {
        timings.predicted_ms / timings.completion_tokens as f64
    } else {
        0.0
    };
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [],
        "usage": {
            "prompt_tokens": timings.prompt_tokens,
            "completion_tokens": timings.completion_tokens,
            "total_tokens": timings.prompt_tokens + timings.completion_tokens,
            "prompt_tokens_details": { "cached_tokens": 0 },
        },
        "timings": {
            "cache_n": 0,
            "prompt_n": timings.prompt_tokens,
            "prompt_ms": timings.prompt_ms,
            "prompt_per_token_ms": prompt_per_token_ms,
            "prompt_per_second": if timings.prompt_ms > 0.0 {
                timings.prompt_tokens as f64 / (timings.prompt_ms / 1000.0)
            } else { 0.0 },
            "predicted_n": timings.completion_tokens,
            "predicted_ms": timings.predicted_ms,
            "predicted_per_token_ms": predicted_per_token_ms,
            "predicted_per_second": if timings.predicted_ms > 0.0 {
                timings.completion_tokens as f64 / (timings.predicted_ms / 1000.0)
            } else { 0.0 },
        }
    })
    .to_string()
}

fn should_include_stream_usage(stream_options: Option<&StreamOptions>) -> bool {
    stream_options.map(|opts| opts.include_usage).unwrap_or(true)
}

fn stream_tool_call_chunks(
    id: &str,
    created: i64,
    model: &str,
    tool_calls: &[ToolCall],
) -> Vec<String> {
    let mut chunks = Vec::new();
    for (index, tc) in tool_calls.iter().enumerate() {
        chunks.push(stream_chunk_json(
            id,
            created,
            model,
            serde_json::json!({
                "tool_calls": [{
                    "index": index,
                    "id": tc.id,
                    "type": tc.call_type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": "",
                    }
                }]
            }),
            None,
        ));
        chunks.push(stream_chunk_json(
            id,
            created,
            model,
            serde_json::json!({
                "tool_calls": [{
                    "index": index,
                    "function": {
                        "arguments": tc.function.arguments,
                    }
                }]
            }),
            None,
        ));
    }
    chunks
}

fn tool_choice_is_required(tool_choice: Option<&serde_json::Value>) -> bool {
    matches!(
        tool_choice,
        Some(serde_json::Value::String(value)) if value == "required"
    )
}

fn tool_choice_is_none(tool_choice: Option<&serde_json::Value>) -> bool {
    matches!(
        tool_choice,
        Some(serde_json::Value::String(value)) if value == "none"
    )
}

fn should_require_tool_call(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> bool {
    if tool_choice_is_none(tool_choice) {
        return false;
    }
    if tool_choice_is_required(tool_choice) {
        return true;
    }
    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    if !has_tools {
        return false;
    }
    matches!(messages.last().map(|m| m.role.as_str()), Some("user"))
}

/// True when the conversation still needs a tool call before an answer turn
/// (user asked, no tool result yet). False once a tool message follows the request,
/// even if the client appended a trailing empty user continuation message.
fn awaits_tool_call(messages: &[Message]) -> bool {
    let mut seen_tool = false;
    for msg in messages.iter().rev() {
        match msg.role.as_str() {
            "tool" => seen_tool = true,
            "user" => {
                if msg.content.as_ref().is_some_and(|c| c.trim().is_empty()) {
                    continue;
                }
                return !seen_tool;
            }
            "assistant" if msg.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) => {
                return !seen_tool;
            }
            _ => {}
        }
    }
    false
}

fn should_force_tool_call(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> bool {
    should_require_tool_call(messages, tools, tool_choice) && awaits_tool_call(messages)
}

fn last_user_message_content(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
}

fn user_wants_directory_listing(messages: &[Message]) -> bool {
    last_user_message_content(messages)
        .map(|content| {
            let lower = content.to_ascii_lowercase();
            lower.contains("list file")
                || lower.contains("list files")
                || lower.contains("list dir")
                || lower.contains("list directory")
                || lower.trim() == "ls"
        })
        .unwrap_or(false)
}

fn make_bash_tool_call(command: &str) -> ToolCall {
    ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "bash".to_string(),
            arguments: serde_json::json!({ "command": command }).to_string(),
        },
    }
}

fn make_read_tool_call(path: &str) -> ToolCall {
    ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read".to_string(),
            arguments: serde_json::json!({ "path": path }).to_string(),
        },
    }
}

/// Pi-style file reference: `@"path/with spaces.md"` or `@test.md`.
fn extract_at_file_path(content: &str) -> Option<String> {
    if let Some(start) = content.find("@\"") {
        let rest = &content[start + 2..];
        let end = rest.find('"')?;
        let path = rest[..end].trim();
        if !path.is_empty() {
            return Some(path.to_string());
        }
    }

    let at = content.find('@')?;
    let rest = &content[at + 1..];
    if rest.starts_with('"') {
        return None;
    }
    let end = rest
        .char_indices()
        .find(|(_, c)| c.is_whitespace())
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    let path = rest[..end].trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':'));
    if path.is_empty() {
        return None;
    }
    Some(path.to_string())
}

fn infer_read_path_from_user(messages: &[Message]) -> Option<String> {
    let content = last_user_message_content(messages)?;
    extract_at_file_path(content)
}

fn min_decode_tokens_for_request(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> usize {
    // Prefill can spuriously peak on <eos>/<turn|>/\n. Require a couple of
    // non-EOS tokens on answer turns — but keep this small so short replies
    // (e.g. "Hello") are not forced into filler monologue.
    if should_require_tool_call(messages, tools, tool_choice) {
        0
    } else {
        2
    }
}

fn user_message_implies_read(messages: &[Message]) -> bool {
    last_user_message_content(messages).is_some_and(|content| {
        if extract_at_file_path(content).is_some() {
            return true;
        }
        let lower = content.to_ascii_lowercase();
        lower.contains("summar") || lower.contains("read file") || lower.contains("read the file")
    })
}

fn parse_plain_read_invocation(text: &str) -> Option<ToolCall> {
    for line in text.lines() {
        let line = line.trim();
        let Some(path) = line
            .strip_prefix("read ")
            .or_else(|| line.strip_prefix("read_file "))
        else {
            continue;
        };
        let path = path.trim();
        if path.is_empty() || path.starts_with('{') || path.contains("<|tool_call>") {
            continue;
        }
        return Some(make_read_tool_call(path));
    }
    None
}

fn infer_tool_call_from_context(
    messages: &[Message],
    tools: Option<&[Tool]>,
    text: &str,
) -> Option<ToolCall> {
    let names: Vec<String> = tools?.iter().map(|t| t.function.name.clone()).collect();
    if names.is_empty() {
        return None;
    }

    if names.iter().any(|n| n == "bash") && user_wants_directory_listing(messages) {
        return Some(make_bash_tool_call("ls -F"));
    }

    let lower = text.to_ascii_lowercase();
    if names.iter().any(|n| n == "bash")
        && user_wants_directory_listing(messages)
        && (lower.contains("bash") || lower.contains("`ls`") || lower.contains(" ls "))
    {
        return Some(make_bash_tool_call("ls -F"));
    }

    if names.iter().any(|n| n == "read") && user_message_implies_read(messages) {
        if let Some(path) = infer_read_path_from_user(messages) {
            return Some(make_read_tool_call(&path));
        }
    }

    None
}

fn resolve_tool_calls(
    text: &str,
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> Vec<ToolCall> {
    let allowed_names = tools.map(allowed_tool_names);
    let allowed_ref = allowed_names.as_deref();
    let infer_plain_read = awaits_tool_call(messages);
    let calls = parse_tool_calls(text, allowed_ref, infer_plain_read);
    if !calls.is_empty() {
        return calls;
    }
    if should_force_tool_call(messages, tools, tool_choice) {
        if let Some(call) = infer_tool_call_from_context(messages, tools, text) {
            return finish_parse_tool_calls(vec![call], allowed_ref);
        }
    }
    calls
}

/// Tool calls we can derive from the request alone (pi `@"path"`, `list files`, etc.).
fn infer_tool_calls_without_generation(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> Vec<ToolCall> {
    resolve_tool_calls("", messages, tools, tool_choice)
}
fn default_temperature() -> f32 {
    1.0
}
fn default_min_p() -> f32 {
    0.05
}
fn default_repetition_penalty() -> f32 {
    1.0
}

#[derive(Deserialize, Serialize, Clone, Default)]
pub struct Message {
    pub role: String,
    #[serde(
        default,
        deserialize_with = "deserialize_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present on `role: "tool"` messages, linking back to the assistant call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Function name for `role: "tool"` messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI allows `content` to be either a plain string, `null`, or an array of
/// content parts (e.g. `[{ "type": "text", "text": "hi" }]`). Accept all forms
/// and flatten array content down to a single string by concatenating text
/// parts.
fn deserialize_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ContentField {
        Text(String),
        Parts(Vec<ContentPart>),
    }

    #[derive(Deserialize)]
    struct ContentPart {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        text: Option<String>,
    }

    let field = Option::<ContentField>::deserialize(deserializer)?;
    Ok(match field {
        None => None,
        Some(ContentField::Text(s)) => Some(s),
        Some(ContentField::Parts(parts)) => {
            let mut out = String::new();
            for part in parts {
                if part.kind == "text" {
                    if let Some(text) = part.text {
                        out.push_str(&text);
                    }
                }
            }
            Some(out)
        }
    })
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
    pub message: AssistantMessageOut,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
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
    pub runtime_config: ServerRuntimeConfig,
}

impl AppState {
    fn request_timeout(&self) -> Duration {
        self.runtime_config.request_timeout
    }

    fn render_metrics(&self) -> String {
        let mut output = self.metrics.render_prometheus();
        output.push_str(&self.runtime_config.render_prometheus());
        output
    }
}

#[derive(Clone)]
pub struct ServerRuntimeConfig {
    pub queue_depth: usize,
    pub kv_pool_slots: usize,
    pub request_timeout: Duration,
    pub max_prefill_tokens_per_tick: Option<usize>,
}

impl ServerRuntimeConfig {
    fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            queue_depth: parse_usize(&mut lookup, "LLAMA_QUEUE_DEPTH")
                .unwrap_or(32)
                .max(1),
            kv_pool_slots: parse_usize(&mut lookup, "LLAMA_KV_POOL_SLOTS")
                .unwrap_or(4)
                .max(1),
            request_timeout: Duration::from_secs(
                parse_u64(&mut lookup, "LLAMA_REQUEST_TIMEOUT_SECS")
                    .unwrap_or(300)
                    .max(1),
            ),
            max_prefill_tokens_per_tick: parse_usize(&mut lookup, "LLAMA_PREFILL_TOKENS_PER_TICK")
                .filter(|tokens| *tokens > 0),
        }
    }

    fn render_prometheus(&self) -> String {
        let prefill_tokens_per_tick = self.max_prefill_tokens_per_tick.unwrap_or(0);

        format!(
            concat!(
                "# HELP llama_config_queue_depth Configured scheduler queue depth.\n",
                "# TYPE llama_config_queue_depth gauge\n",
                "llama_config_queue_depth {}\n",
                "# HELP llama_config_kv_pool_slots Configured KV cache pool slots.\n",
                "# TYPE llama_config_kv_pool_slots gauge\n",
                "llama_config_kv_pool_slots {}\n",
                "# HELP llama_config_request_timeout_secs Configured request timeout in seconds.\n",
                "# TYPE llama_config_request_timeout_secs gauge\n",
                "llama_config_request_timeout_secs {}\n",
                "# HELP llama_config_prefill_tokens_per_tick Configured prefill tokens per scheduler tick; 0 means model default.\n",
                "# TYPE llama_config_prefill_tokens_per_tick gauge\n",
                "llama_config_prefill_tokens_per_tick {}\n",
            ),
            self.queue_depth,
            self.kv_pool_slots,
            self.request_timeout.as_secs(),
            prefill_tokens_per_tick,
        )
    }
}

fn parse_usize(lookup: &mut impl FnMut(&str) -> Option<String>, name: &str) -> Option<usize> {
    lookup(name)?.parse().ok()
}

fn parse_u64(lookup: &mut impl FnMut(&str) -> Option<String>, name: &str) -> Option<u64> {
    lookup(name)?.parse().ok()
}

// ─── Chat template ───────────────────────────────────────────────────────────

// Gemma 4 chat control tokens (see GGUF: <|turn>=105, <turn|>=106).
const TURN_START: &str = "<|turn>";
const TURN_END: &str = "<turn|>";
const CHANNEL_START: &str = "<|channel>";
const CHANNEL_END: &str = "<channel|>";
const GEMMA4_TOOL_START: &str = "<|tool>";
const GEMMA4_TOOL_END: &str = "<tool|>";
const NATIVE_TOOL_CALL_PREFIX: &str = "<|tool_call>call:";
const NATIVE_TOOL_CALL_TRIGGER: &str = "<|tool_call>";
const NATIVE_TOOL_CALL_SUFFIX: &str = "<tool_call|>";
const NATIVE_TOOL_RESPONSE_PREFIX: &str = "<|tool_response>response:";
const NATIVE_TOOL_RESPONSE_SUFFIX: &str = "<tool_response|>";
const SYSTEM_FINGERPRINT: &str = "local-7f3a2c1b";
/// Empty thought channel used by Gemma 4 12B/26B/31B when thinking is off.
/// E2B/E4B must NOT use this — official template ends at `<|turn>model\n` only.
#[allow(dead_code)]
const EMPTY_THOUGHT_PREFIX: &str = "<|channel>thought\n<channel|>";

fn generation_priming_suffix(
    _messages: &[Message],
    _tools: Option<&[Tool]>,
    _tool_choice: Option<&serde_json::Value>,
) -> &'static str {
    // Gemma 4 E2B/E4B (our target): thinking-off generation prompt is just
    // `<|turn>model\n` with no empty thought stub. Priming
    // `<|channel>thought\n<channel|>` is for 12B/26B/31B only and makes E2B
    // emit meta-narration ("The user wants…") instead of the answer.
    ""
}

const BUILT_IN_OUTPUT_TRIM_SEQUENCES: &[&str] = &[
    TURN_END,
    TURN_START,
    NATIVE_TOOL_CALL_SUFFIX,
    "<end_of_turn>",
    "<eos>",
    "<start_of_turn>",
    "</start_of_turn>",
];

/// Strips channel markup from text so the visible content flows correctly in
/// SSE streaming. Handles two cases:
///   1. Paired `<|channel>...<channel|>` blocks (model's internal deliberation).
///   2. Standalone `<channel|>` tags the model emits as a terminator.
fn strip_channel_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        // First handle any paired block.
        match rest.find(CHANNEL_START) {
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_start = &rest[start + CHANNEL_START.len()..];
                match after_start.find(CHANNEL_END) {
                    Some(end) => {
                        rest = &after_start[end + CHANNEL_END.len()..];
                        continue;
                    }
                    None => {
                        // Unclosed channel opening — discard it and everything after.
                        break;
                    }
                }
            }
            None => {}
        }
        // No (more) opening tag. Strip any standalone closing tag.
        match rest.find(CHANNEL_END) {
            Some(pos) => {
                out.push_str(&rest[..pos]);
                rest = &rest[pos + CHANNEL_END.len()..];
                // Loop to handle another closing tag that may follow.
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

// Only these end the underlying sampler loop. Channel / tool markup tokens are
// normal model output and must not cancel generation mid-turn.
const BUILT_IN_GENERATION_STOP_SEQUENCES: &[&str] = &[TURN_END, TURN_START];

const TURN_STOP_PREFIXES: &[&str] = &["<", "<|", "<|t", "<|tu", "<|tur", "<|turn"];
const OTHER_OUTPUT_TRIM_PREFIXES: &[&str] = &[
    "<e",
    "<en",
    "<end",
    "<end_",
    "<end_o",
    "<end_of",
    "<end_of_",
    "<end_of_t",
    "<end_of_tu",
    "<end_of_turn",
];
const TOOL_CALL_STOP_PREFIXES: &[&str] = &[
    "<",
    "<t",
    "<to",
    "<too",
    "<tool",
    "<tool_",
    "<tool_c",
    "<tool_ca",
    "<tool_cal",
    "<tool_call",
    "<tool_call|",
    "<tool_call|>",
];

fn render_tool_instructions(tools: &[Tool], tool_choice: Option<&serde_json::Value>) -> String {
    let mut s = String::new();
    s.push_str(
        "You have access to the following tools. When you need to use a tool, \
respond with one or more fenced code blocks tagged `tool_call`, each containing \
a single JSON object with \"name\" and \"arguments\" keys. You may also emit a bare \
JSON object with \"name\" and \"arguments\". Emit nothing else when calling a tool. \
Only call tools from this list.\n\n",
    );
    s.push_str("Available tools:\n");
    for tool in tools {
        let f = &tool.function;
        s.push_str("- ");
        s.push_str(&f.name);
        if let Some(desc) = &f.description {
            s.push_str(": ");
            s.push_str(desc);
        }
        s.push('\n');
        if let Some(params) = &f.parameters {
            s.push_str("  parameters (JSON schema): ");
            s.push_str(&params.to_string());
            s.push('\n');
        }
    }
    if let Some(first) = tools.first() {
        s.push_str(&format!(
            "\nExample:\n```tool_call\n{{\"name\": \"{}\", \"arguments\": {{}}}}\n```\n",
            first.function.name
        ));
    }
    s.push_str(
        "\nTo call a tool, output exactly:\n```tool_call\n{\"name\": \"tool_name\", \
\"arguments\": {\"arg\": \"value\"}}\n```\n",
    );
    if tool_choice_is_required(tool_choice) {
        s.push_str("\nYou must call a tool to answer this request.\n");
    }
    s
}

fn apply_chat_template(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();

    let mut system_parts = Vec::new();
    for msg in messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content {
                if !content.is_empty() {
                    system_parts.push(content.clone());
                }
            }
        }
    }

    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    let include_tool_declarations = has_tools && awaits_tool_call(messages);
    if include_tool_declarations || !system_parts.is_empty() {
        prompt.push_str(TURN_START);
        prompt.push_str("system\n");
        prompt.push_str("<|think|>\n");
        if !system_parts.is_empty() {
            prompt.push_str(&system_parts.join("\n\n"));
            if include_tool_declarations {
                prompt.push('\n');
            }
        }
        if let Some(tools) = tools {
            if include_tool_declarations && !tools.is_empty() {
                prompt.push_str(&render_gemma4_tool_declarations(tools));
            }
        }
        if should_force_tool_call(messages, tools, tool_choice) {
            prompt.push_str("\nYou must call a tool to answer this request.");
        }
        prompt.push_str(TURN_END);
        prompt.push('\n');
    }

    let mut ends_in_open_model_turn = false;
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.role == "system" {
            i += 1;
            continue;
        }

        if msg.role == "tool" {
            let name = msg.name.as_deref().unwrap_or("tool");
            prompt.push_str(&render_tool_response_native(
                name,
                msg.content.as_deref().unwrap_or(""),
            ));
            i += 1;
            continue;
        }

        if msg.role == "user" {
            ends_in_open_model_turn = false;
        }

        let mapped = if msg.role == "assistant" {
            "model"
        } else {
            "user"
        };
        prompt.push_str(TURN_START);
        prompt.push_str(mapped);
        prompt.push('\n');
        if let Some(content) = &msg.content {
            prompt.push_str(content);
        }
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                prompt.push_str(&render_assistant_tool_call_native(tc));
            }
        }
        i += 1;

        if msg.role == "assistant" && msg.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) {
            ends_in_open_model_turn = true;
            while i < messages.len() && messages[i].role == "tool" {
                let tm = &messages[i];
                let name = tm.name.as_deref().unwrap_or("tool");
                prompt.push_str(&render_tool_response_native(
                    name,
                    tm.content.as_deref().unwrap_or(""),
                ));
                i += 1;
            }
            while i < messages.len()
                && messages[i].role == "assistant"
                && messages[i]
                    .tool_calls
                    .as_ref()
                    .map_or(true, |t| t.is_empty())
            {
                if let Some(content) = &messages[i].content {
                    prompt.push_str(content);
                }
                ends_in_open_model_turn = false;
                i += 1;
            }
        }

        if !ends_in_open_model_turn {
            prompt.push_str(TURN_END);
            prompt.push('\n');
        }
    }

    if ends_in_open_model_turn {
        prompt.push_str(generation_priming_suffix(messages, tools, tool_choice));
    } else {
        prompt.push_str(TURN_START);
        prompt.push_str("model\n");
        prompt.push_str(generation_priming_suffix(messages, tools, tool_choice));
    }

    prompt
}

/// Parse tool calls emitted by the model. Supports the prompted
/// ```` ```tool_call ```` fenced-block format as well as
/// `<tool_call>...</tool_call>` tags, with one JSON object per block.
fn parse_tool_calls(
    text: &str,
    allowed_names: Option<&[String]>,
    infer_plain_read: bool,
) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    fn push_from_json(calls: &mut Vec<ToolCall>, json_str: &str) {
        let trimmed = json_str.trim();
        if trimmed.is_empty() {
            return;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(call) = tool_call_from_value(&val) {
                calls.push(call);
            }
        }
    }

    // Gemma4 native: <|tool_call>call:read{path:<|"|>a<|"|>}<tool_call|>
    let mut rest = text;
    while let Some(start) = rest.find(NATIVE_TOOL_CALL_TRIGGER) {
        let after_trigger = &rest[start + NATIVE_TOOL_CALL_TRIGGER.len()..];
        let after = after_trigger
            .strip_prefix("call:")
            .or_else(|| after_trigger.strip_prefix("call"))
            .unwrap_or(after_trigger);
        let body = if let Some(end) = after.find(NATIVE_TOOL_CALL_SUFFIX) {
            &after[..end]
        } else {
            after.trim()
        };
        if let Some(brace) = body.find('{') {
            let name = body[..brace].trim();
            let dict = &body[brace..];
            if let Some(args_val) = gemma4_dict_to_json_object(dict) {
                push_unique_call(
                    &mut calls,
                    ToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4()),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: serde_json::to_string(&args_val).unwrap_or_default(),
                        },
                    },
                );
            }
        }
        let advance = if let Some(end) = after.find(NATIVE_TOOL_CALL_SUFFIX) {
            start + NATIVE_TOOL_CALL_TRIGGER.len() + end + NATIVE_TOOL_CALL_SUFFIX.len()
        } else {
            rest.len()
        };
        if advance <= start {
            break;
        }
        rest = &rest[advance..];
    }
    if !calls.is_empty() {
        return finish_parse_tool_calls(calls, allowed_names.as_deref());
    }

    // Fenced ```tool_call blocks.
    let mut rest = text;
    while let Some(start) = rest.find("```tool_call") {
        let after = &rest[start + "```tool_call".len()..];
        if let Some(end) = after.find("```") {
            push_from_json(&mut calls, &after[..end]);
            rest = &after[end + 3..];
        } else {
            break;
        }
    }

    // <tool_call> ... </tool_call> tags.
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        if let Some(end) = after.find("</tool_call>") {
            push_from_json(&mut calls, &after[..end]);
            rest = &after[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    if calls.is_empty() {
        let mut rest = text;
        while let Some(start) = rest.find("```json") {
            let after = &rest[start + "```json".len()..];
            let json_body = after.strip_prefix('\n').unwrap_or(after);
            if let Some(end) = json_body.find("```") {
                push_from_json(&mut calls, &json_body[..end]);
                rest = &json_body[end + 3..];
            } else {
                break;
            }
        }

        for val in iter_json_objects(text) {
            if let Some(call) = tool_call_from_value(&val) {
                push_unique_call(&mut calls, call);
            }
        }
    }

    if calls.is_empty() && infer_plain_read {
        if let Some(call) = parse_plain_read_invocation(text) {
            calls.push(call);
        }
    }

    if calls.is_empty() {
        if let Some(call) = parse_tool_code_fallback(text) {
            calls.push(call);
        }
    }

    finish_parse_tool_calls(calls, allowed_names.as_deref())
}

fn finish_parse_tool_calls(calls: Vec<ToolCall>, allowed_names: Option<&[String]>) -> Vec<ToolCall> {
    match allowed_names {
        Some(names) if !names.is_empty() => normalize_tool_calls(calls, names),
        _ => calls,
    }
}

fn push_unique_call(calls: &mut Vec<ToolCall>, call: ToolCall) {
    if !calls
        .iter()
        .any(|existing| {
            existing.function.name == call.function.name
                && existing.function.arguments == call.function.arguments
        })
    {
        calls.push(call);
    }
}

fn iter_json_objects(text: &str) -> Vec<serde_json::Value> {
    let mut objects = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if text.as_bytes().get(i) == Some(&b'{') {
            if let Some((end, json_str)) = extract_balanced_json(&text[i..]) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if val.is_object() {
                        objects.push(val);
                    }
                }
                i += end;
                continue;
            }
        }
        i += 1;
    }
    objects
}

fn extract_balanced_json(s: &str) -> Option<(usize, &str)> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if b == b'\\' {
                escape = true;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((i + 1, &s[..=i]));
                }
            }
            _ => {}
        }
    }
    None
}

fn tool_call_from_value(val: &serde_json::Value) -> Option<ToolCall> {
    if let Some(items) = val.get("tool_calls").and_then(|v| v.as_array()) {
        return items.iter().find_map(tool_call_from_value);
    }

    if val.get("type").and_then(|t| t.as_str()) == Some("function") {
        if let Some(func) = val.get("function") {
            let name = func.get("name")?.as_str()?.to_string();
            let arguments = match func.get("arguments") {
                Some(args) if args.is_string() => args.as_str().unwrap_or("").to_string(),
                Some(args) => serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()),
                None => "{}".to_string(),
            };
            return Some(ToolCall {
                id: val
                    .get("id")
                    .and_then(|id| id.as_str())
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
                call_type: "function".to_string(),
                function: FunctionCall { name, arguments },
            });
        }
    }

    let name = val.get("name")?.as_str()?.to_string();
    let arguments = match val.get("arguments").or_else(|| val.get("parameters")) {
        Some(args) if args.is_string() => args.as_str().unwrap_or("").to_string(),
        Some(args) => serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: FunctionCall { name, arguments },
    })
}

fn find_stop_position(text: &str, stops: &[&str], request_stop: Option<&[String]>) -> Option<usize> {
    let mut earliest = stops.iter().filter_map(|stop| text.find(stop)).min();

    if let Some(request_stop) = request_stop {
        let request_earliest = request_stop
            .iter()
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

fn find_generation_stop_position(text: &str, request_stop: Option<&[String]>) -> Option<usize> {
    let mut earliest = find_stop_position(text, BUILT_IN_GENERATION_STOP_SEQUENCES, None);

    // A completed native tool call ends the assistant turn.
    if let Some(tool_end) = text.find(NATIVE_TOOL_CALL_SUFFIX) {
        earliest = Some(match earliest {
            Some(pos) => pos.min(tool_end),
            None => tool_end,
        });
    }

    if let Some(request_stop) = request_stop {
        let request_earliest = request_stop
            .iter()
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
    if let Some(pos) = find_stop_position(text, BUILT_IN_OUTPUT_TRIM_SEQUENCES, request_stop) {
        text.truncate(pos);
        return true;
    }
    false
}

fn stop_prefix_holdback_len(
    text: &str,
    in_tool_call: bool,
    request_stop: Option<&[String]>,
) -> usize {
    let prefixes = if in_tool_call {
        TOOL_CALL_STOP_PREFIXES
    } else {
        TURN_STOP_PREFIXES
    };

    let mut max_len = 0;
    for (start, _) in text.char_indices() {
        let suffix = &text[start..];
        if prefixes.iter().any(|prefix| *prefix == suffix)
            || (!in_tool_call
                && OTHER_OUTPUT_TRIM_PREFIXES
                    .iter()
                    .any(|prefix| *prefix == suffix))
        {
            max_len = max_len.max(text.len() - start);
        }
        if request_stop.is_some_and(|stops| {
            stops
                .iter()
                .any(|stop| !stop.is_empty() && stop.starts_with(suffix))
        }) {
            max_len = max_len.max(text.len() - start);
        }
    }
    max_len
}

fn trim_stream_safe_text(
    text: &mut String,
    request_stop: Option<&[String]>,
    in_tool_call: bool,
) -> bool {
    if trim_stop_sequences(text, request_stop) {
        return true;
    }

    let holdback_len = stop_prefix_holdback_len(text, in_tool_call, request_stop);
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
) -> Result<(tokio::sync::mpsc::Receiver<StreamEvent>, Arc<AtomicU8>), ApiError> {
    let (response_tx, response_rx) = tokio::sync::mpsc::channel(64);
    let cancel = Arc::new(AtomicU8::new(CANCEL_NONE));
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

fn generation_params_from_request(
    req: &ChatCompletionRequest,
    request_timeout: Duration,
) -> Result<GenerationParams, ApiError> {
    validate_request(req)?;

    Ok(GenerationParams {
        max_tokens: effective_max_tokens(req),
        temperature: req.temperature,
        min_p: req.min_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        eos_token_ids: vec![1, 106],
        min_decode_tokens: min_decode_tokens_for_request(
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        ),
        request_timeout,
    })
}

fn validate_request(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(
            "empty_messages",
            "messages must not be empty",
        ));
    }

    if effective_max_tokens(req) == 0 {
        return Err(ApiError::bad_request(
            "invalid_max_tokens",
            "max_tokens must be greater than 0",
        ));
    }

    if !req.temperature.is_finite() || req.temperature < 0.0 || req.temperature > 5.0 {
        return Err(ApiError::bad_request(
            "invalid_temperature",
            "temperature must be between 0 and 5",
        ));
    }

    if !req.min_p.is_finite() || req.min_p < 0.0 || req.min_p > 1.0 {
        return Err(ApiError::bad_request(
            "invalid_min_p",
            "min_p must be between 0 and 1",
        ));
    }

    if !req.repetition_penalty.is_finite()
        || req.repetition_penalty <= 0.0
        || req.repetition_penalty > 10.0
    {
        return Err(ApiError::bad_request(
            "invalid_repetition_penalty",
            "repetition_penalty must be greater than 0 and at most 10",
        ));
    }

    if !req.frequency_penalty.is_finite()
        || req.frequency_penalty < -2.0
        || req.frequency_penalty > 2.0
    {
        return Err(ApiError::bad_request(
            "invalid_frequency_penalty",
            "frequency_penalty must be between -2 and 2",
        ));
    }

    Ok(())
}

const TOOL_CONTENT_TRUNCATION_NOTICE: &str =
    "\n\n[... file content truncated to fit context window ...]";

fn truncate_content_to_budget(content: &mut String, target_len: usize) {
    if content.len() <= target_len {
        return;
    }
    let notice = TOOL_CONTENT_TRUNCATION_NOTICE;
    let budget = target_len.saturating_sub(notice.len() + 32);
    if budget < 512 {
        content.truncate(target_len.saturating_sub(notice.len()));
        if !content.contains("truncated to fit context window") {
            content.push_str(notice);
        }
        return;
    }
    let head = budget * 70 / 100;
    let tail = budget - head;
    let head_end = content
        .char_indices()
        .nth(head)
        .map(|(i, _)| i)
        .unwrap_or(content.len());
    let tail_start = content
        .char_indices()
        .nth(content.chars().count().saturating_sub(tail))
        .map(|(i, _)| i)
        .unwrap_or(head_end);
    let new_content = format!(
        "{}\n\n[... middle of file omitted for context limit ...]\n\n{}{}",
        &content[..head_end],
        &content[tail_start..],
        notice
    );
    *content = new_content;
}

fn count_prompt_tokens(
    tokenizer: &tokenizers::Tokenizer,
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> Result<usize, ApiError> {
    let prompt = apply_chat_template(messages, tools, tool_choice);
    let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|err| {
        ApiError::bad_request(
            "tokenizer_error",
            format!("failed to tokenize prompt: {}", err),
        )
    })?;
    Ok(encoding.get_ids().len())
}

/// Shrink `role: tool` payloads so the rendered prompt fits the KV budget.
fn fit_messages_to_context(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
    tokenizer: &tokenizers::Tokenizer,
    max_prompt_tokens: usize,
) -> Result<Vec<Message>, ApiError> {
    let mut fitted = messages.to_vec();
    if count_prompt_tokens(tokenizer, &fitted, tools, tool_choice)? <= max_prompt_tokens {
        return Ok(fitted);
    }

    for _ in 0..48 {
        let current = count_prompt_tokens(tokenizer, &fitted, tools, tool_choice)?;
        if current <= max_prompt_tokens {
            return Ok(fitted);
        }

        let Some((index, _)) = fitted
            .iter()
            .enumerate()
            .filter_map(|(index, msg)| {
                let content = msg.content.as_ref()?;
                if content.len() < 512 {
                    return None;
                }
                if msg.role == "tool" {
                    Some((index, content.len()))
                } else {
                    None
                }
            })
            .max_by_key(|(_, len)| *len)
        else {
            break;
        };

        let shrink_ratio = max_prompt_tokens as f64 / current as f64;
        let content = fitted[index].content.as_mut().expect("tool content");
        let min_chars = 512;
        if content.len() <= min_chars {
            break;
        }
        let target_len = ((content.len() as f64) * shrink_ratio * 0.92) as usize;
        let target_len = target_len.clamp(min_chars, content.len().saturating_sub(1));
        truncate_content_to_budget(content, target_len);
    }

    let final_count = count_prompt_tokens(tokenizer, &fitted, tools, tool_choice)?;
    if final_count > max_prompt_tokens {
        return Err(ApiError::bad_request(
            "context_length_exceeded",
            format!(
                "prompt has {} tokens after truncating tool results; context limit is {}",
                final_count, max_prompt_tokens
            ),
        ));
    }
    if fitted.iter().any(|msg| {
        msg.content.as_deref().is_some_and(|c| {
            c.contains("truncated to fit context window")
        })
    }) {
        eprintln!(
            "   Prompt fit: truncated tool content to {} tokens (budget {})",
            final_count, max_prompt_tokens
        );
    }
    Ok(fitted)
}

/// How many prompt tokens to allow when fitting messages.
///
/// Clients (e.g. agents) often send `max_tokens` ≈ full context. Reserving that
/// entire amount for completion would leave a 1-token prompt budget. Cap the
/// reserved completion share at half the window so long tool results still fit;
/// completion is clamped to the remainder after the prompt is known.
fn prompt_token_budget(max_context_len: usize, requested_max_tokens: usize) -> usize {
    let max_reserve = (max_context_len / 2).max(1).min(max_context_len.saturating_sub(1));
    let reserved = requested_max_tokens.min(max_reserve);
    max_context_len.saturating_sub(reserved).max(1)
}

fn clamp_max_tokens_to_context(
    prompt_tokens: usize,
    requested_max_tokens: usize,
    max_context_len: usize,
) -> Result<usize, ApiError> {
    if prompt_tokens >= max_context_len {
        return Err(ApiError::bad_request(
            "context_length_exceeded",
            format!(
                "prompt has {} tokens, which exceeds the model context limit of {}",
                prompt_tokens, max_context_len
            ),
        ));
    }
    let remaining = max_context_len - prompt_tokens;
    Ok(requested_max_tokens.min(remaining).max(1))
}

fn encode_prompt(
    state: &AppState,
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
    max_tokens: usize,
) -> Result<Vec<usize>, ApiError> {
    let max_prompt_tokens = prompt_token_budget(state.max_context_len, max_tokens);
    let fitted = fit_messages_to_context(
        messages,
        tools,
        tool_choice,
        &state.tokenizer,
        max_prompt_tokens,
    )?;
    let prompt = apply_chat_template(&fitted, tools, tool_choice);
    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|err| {
            ApiError::bad_request(
                "tokenizer_error",
                format!("failed to tokenize prompt: {}", err),
            )
        })?;
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
        state.render_metrics(),
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
    let mut generation_params = generation_params_from_request(&req, state.request_timeout())?;
    let input_ids = encode_prompt(
        &state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        generation_params.max_tokens,
    )?;
    let prompt_tokens = input_ids.len();
    generation_params.max_tokens = clamp_max_tokens_to_context(
        prompt_tokens,
        generation_params.max_tokens,
        state.max_context_len,
    )?;
    validate_context_len(
        prompt_tokens,
        generation_params.max_tokens,
        state.max_context_len,
    )?;
    let has_tools = req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    let model_name = response_model(&req).to_string();
    let request_stop = req.stop.map(StopSequences::into_vec);

    let inferred_tool_calls = if has_tools {
        infer_tool_calls_without_generation(
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        )
    } else {
        Vec::new()
    };
    if !inferred_tool_calls.is_empty() {
        let message = assistant_message_out(
            String::new(),
            inferred_tool_calls.clone(),
            ChannelSplitMode::default(),
        );
        return Ok(Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model_name,
            choices: vec![Choice {
                index: 0,
                message,
                finish_reason: "tool_calls".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: 0,
                total_tokens: prompt_tokens,
            },
        }));
    }

    let (mut response_rx, cancel) = enqueue_request(&state, input_ids, generation_params)?;
    let mut output_tokens = Vec::new();
    let mut finish_reason = "stop".to_string();

    while let Some(event) = response_rx.recv().await {
        match event {
            StreamEvent::Token { token_id } => {
                output_tokens.push(token_id as u32);
                let decoded_text = state
                    .tokenizer
                    .decode(&output_tokens.iter().map(|&t| t).collect::<Vec<u32>>(), false)
                    .unwrap_or_default();
                if find_generation_stop_position(&decoded_text, request_stop.as_deref()).is_some() {
                    cancel.store(CANCEL_STOP, Ordering::Relaxed);
                    break;
                }
            }
            StreamEvent::Done {
                finish_reason: reason,
            } => {
                finish_reason = reason;
                break;
            }
            StreamEvent::Error { message } => return Err(ApiError::internal(message)),
        }
    }

    let text = {
        let mut raw = state
            .tokenizer
            .decode(&output_tokens.iter().map(|&t| t).collect::<Vec<u32>>(), true)
            .unwrap_or_default();
        trim_stop_sequences(&mut raw, request_stop.as_deref());
        raw
    };
    let completion_tokens = output_tokens.len();

    let tool_calls = if has_tools {
        resolve_tool_calls(
            &text,
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        )
    } else {
        Vec::new()
    };

    if !tool_calls.is_empty() {
        finish_reason = "tool_calls".to_string();
    }

    let split_mode = ChannelSplitMode {
        plain_text_as_reasoning: has_tools && awaits_tool_call(&req.messages),
    };
    let message = assistant_message_out(text, tool_calls, split_mode);

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message,
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
    let mut generation_params = generation_params_from_request(&req, state.request_timeout())?;
    let input_ids = encode_prompt(
        &state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        generation_params.max_tokens,
    )?;
    let prompt_tokens = input_ids.len();
    generation_params.max_tokens = clamp_max_tokens_to_context(
        prompt_tokens,
        generation_params.max_tokens,
        state.max_context_len,
    )?;
    validate_context_len(
        prompt_tokens,
        generation_params.max_tokens,
        state.max_context_len,
    )?;
    let has_tools = req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    let model_name = response_model(&req).to_string();
    let request_stop = req.stop.map(StopSequences::into_vec);
    let include_usage = should_include_stream_usage(req.stream_options.as_ref());
    let allowed_tool_names = req
        .tools
        .as_ref()
        .map(|tools| allowed_tool_names(tools.as_slice()))
        .unwrap_or_default();
    let messages_for_resolve = req.messages.clone();
    let tools_for_resolve = req.tools.clone();
    let tool_choice_for_resolve = req.tool_choice.clone();

    let inferred_tool_calls = if has_tools {
        infer_tool_calls_without_generation(
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        )
    } else {
        Vec::new()
    };
    if !inferred_tool_calls.is_empty() {
        tokio::spawn(async move {
            let role_chunk_data = stream_chunk_json(
                &chat_id,
                created,
                &model_name,
                serde_json::json!({"role": "assistant", "content": null}),
                None,
            );
            let _ = tx
                .send(Ok(Event::default().data(role_chunk_data)))
                .await;
            for chunk_data in
                stream_tool_call_chunks(&chat_id, created, &model_name, &inferred_tool_calls)
            {
                let _ = tx
                    .send(Ok(Event::default().data(chunk_data)))
                    .await;
            }
            let done_chunk_data = stream_chunk_json(
                &chat_id,
                created,
                &model_name,
                serde_json::json!({}),
                Some("tool_calls"),
            );
            let _ = tx
                .send(Ok(Event::default().data(done_chunk_data)))
                .await;
            if include_usage {
                let timings = StreamTimingsSnapshot {
                    prompt_ms: 0.0,
                    predicted_ms: 0.0,
                    prompt_tokens,
                    completion_tokens: 0,
                };
                let usage_chunk_data =
                    stream_usage_chunk_json(&chat_id, created, &model_name, &timings);
                let _ = tx
                    .send(Ok(Event::default().data(usage_chunk_data)))
                    .await;
            }
            let _ = tx
                .send(Ok(Event::default().data("[DONE]".to_string())))
                .await;
        });
        return Ok(sse_stream(rx));
    }

    let request_result = enqueue_request(&state, input_ids, generation_params)?;

    tokio::spawn(async move {
        let stream_started = Instant::now();
        let tool_generation_mode = has_tools
            && should_force_tool_call(
                &messages_for_resolve,
                tools_for_resolve.as_deref(),
                tool_choice_for_resolve.as_ref(),
            );
        let split_mode = ChannelSplitMode {
            plain_text_as_reasoning: has_tools && awaits_tool_call(&messages_for_resolve),
        };
        let role_chunk_data = stream_chunk_json(
            &chat_id,
            created,
            &model_name,
            serde_json::json!({"role": "assistant", "content": null}),
            None,
        );
        let _ = tx
            .send(Ok(Event::default().data(role_chunk_data)))
            .await;

        let mut output_tokens = Vec::new();
        let mut decoded_text = String::new();
        let mut emitted_reasoning_len = 0usize;
        let mut emitted_content_len = 0usize;
        let mut first_token_at: Option<Instant> = None;
        let mut saw_tool_call_marker = false;
        let mut in_tool_call = false;
        let mut tool_emitter: Option<IncrementalToolCallEmitter> = None;
        let mut early_tool_calls: Option<Vec<ToolCall>> = None;

        let mut finish_reason = "stop".to_string();

        let (mut response_rx, cancel) = request_result;
        while let Some(event) = response_rx.recv().await {
            match event {
                StreamEvent::Token { token_id } => {
                    output_tokens.push(token_id as u32);
                    if first_token_at.is_none() {
                        first_token_at = Some(Instant::now());
                    }

                    let tok_str = state
                        .tokenizer
                        .decode(&[token_id as u32], false)
                        .unwrap_or_default();
                    decoded_text.push_str(&tok_str);

                    let mut visible_text = decoded_text.clone();
                    if visible_text.contains(NATIVE_TOOL_CALL_TRIGGER) {
                        in_tool_call = true;
                        saw_tool_call_marker = true;
                    }
                    let mut stopped = trim_stream_safe_text(
                        &mut visible_text,
                        request_stop.as_deref(),
                        in_tool_call,
                    );

                    let use_tool_split = use_tool_generation_stream_split(
                        &visible_text,
                        in_tool_call,
                        tool_generation_mode,
                    );
                    let (new_reasoning, new_content, new_er, new_ec) = {
                        let (nr, nc, er, ec) = compute_stream_deltas(
                            &visible_text,
                            emitted_reasoning_len,
                            emitted_content_len,
                            use_tool_split,
                            split_mode,
                        );
                        emitted_reasoning_len = er;
                        emitted_content_len = ec;
                        (nr, nc, er, ec)
                    };

                    if !new_reasoning.is_empty() {
                        let chunk_data = stream_chunk_json(
                            &chat_id,
                            created,
                            &model_name,
                            serde_json::json!({"reasoning_content": new_reasoning}),
                            None,
                        );
                        if tx
                            .send(Ok(Event::default().data(chunk_data)))
                            .await
                            .is_err()
                        {
                            cancel.store(CANCEL_CLIENT, Ordering::Relaxed);
                            break;
                        }
                    }

                    if !new_content.is_empty() && !in_tool_call {
                        let chunk_data = stream_chunk_json(
                            &chat_id,
                            created,
                            &model_name,
                            serde_json::json!({"content": new_content}),
                            None,
                        );
                        if tx
                            .send(Ok(Event::default().data(chunk_data)))
                            .await
                            .is_err()
                        {
                            cancel.store(CANCEL_CLIENT, Ordering::Relaxed);
                            break;
                        }
                    }

                    if has_tools && in_tool_call {
                        if tool_emitter.is_none() {
                            tool_emitter = Some(IncrementalToolCallEmitter::new(0));
                        }
                        if let Some(emitter) = &mut tool_emitter {
                            for chunk_data in emitter.update(
                                &decoded_text,
                                &chat_id,
                                created,
                                &model_name,
                                &allowed_tool_names,
                            ) {
                                if tx
                                    .send(Ok(Event::default().data(chunk_data)))
                                    .await
                                    .is_err()
                                {
                                    cancel.store(CANCEL_CLIENT, Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                    }

                    if find_generation_stop_position(&decoded_text, request_stop.as_deref()).is_some() {
                        cancel.store(CANCEL_STOP, Ordering::Relaxed);
                        break;
                    }

                    if tool_generation_mode
                        && !saw_tool_call_marker
                        && tool_emitter.as_ref().is_none_or(|e| !e.header_sent)
                    {
                        let resolved = resolve_tool_calls(
                            &decoded_text,
                            &messages_for_resolve,
                            tools_for_resolve.as_deref(),
                            tool_choice_for_resolve.as_ref(),
                        );
                        if !resolved.is_empty() {
                            early_tool_calls = Some(resolved);
                            cancel.store(CANCEL_STOP, Ordering::Relaxed);
                            break;
                        }
                    }

                    if stopped {
                        cancel.store(CANCEL_STOP, Ordering::Relaxed);
                        break;
                    }
                }
                StreamEvent::Done {
                    finish_reason: reason,
                } => {
                    finish_reason = reason;
                    break;
                }
                StreamEvent::Error { message } => {
                    finish_reason = format!("error: {}", message);
                    break;
                }
            }
        }

        // Promote any trailing answer text that only became visible at stream end.
        if !in_tool_call && !decoded_text.is_empty() {
            let mut flush_text = decoded_text.clone();
            trim_stop_sequences(&mut flush_text, request_stop.as_deref());
            if !flush_text.is_empty() {
                let use_tool_split = use_tool_generation_stream_split(
                    &flush_text,
                    in_tool_call,
                    tool_generation_mode,
                );
                let (mut reasoning, mut content) = if use_tool_split {
                    split_tool_generation_output(&flush_text)
                } else {
                    split_reasoning_and_content_with_mode(&flush_text, split_mode)
                };
                (reasoning, content) =
                    finalize_reasoning_content_split(reasoning, content, &flush_text, split_mode);

                let new_reasoning = if reasoning.len() > emitted_reasoning_len {
                    reasoning[emitted_reasoning_len..].to_string()
                } else {
                    String::new()
                };
                let new_content = if content.len() > emitted_content_len {
                    content[emitted_content_len..].to_string()
                } else {
                    String::new()
                };

                if !new_reasoning.is_empty() {
                    let chunk_data = stream_chunk_json(
                        &chat_id,
                        created,
                        &model_name,
                        serde_json::json!({"reasoning_content": new_reasoning}),
                        None,
                    );
                    let _ = tx.send(Ok(Event::default().data(chunk_data))).await;
                }
                if !new_content.is_empty() {
                    let chunk_data = stream_chunk_json(
                        &chat_id,
                        created,
                        &model_name,
                        serde_json::json!({"content": new_content}),
                        None,
                    );
                    let _ = tx.send(Ok(Event::default().data(chunk_data))).await;
                }
            }
        }

        let raw_text = {
            let mut full = state
                .tokenizer
                .decode(&output_tokens, true)
                .unwrap_or_default();
            trim_stop_sequences(&mut full, request_stop.as_deref());
            full
        };

        if has_tools {
            if let Some(emitter) = &mut tool_emitter {
                for chunk_data in emitter.finalize(
                    &raw_text,
                    &chat_id,
                    created,
                    &model_name,
                    &allowed_tool_names,
                ) {
                    let _ = tx
                        .send(Ok(Event::default().data(chunk_data)))
                        .await;
                }
            }

            let streamed_tool_call = tool_emitter
                .as_ref()
                .is_some_and(|e| e.header_sent);

            if streamed_tool_call {
                finish_reason = "tool_calls".to_string();
            } else if let Some(tool_calls) = early_tool_calls {
                finish_reason = "tool_calls".to_string();
                for chunk_data in
                    stream_tool_call_chunks(&chat_id, created, &model_name, &tool_calls)
                {
                    let _ = tx
                        .send(Ok(Event::default().data(chunk_data)))
                        .await;
                }
            } else {
                let tool_calls = resolve_tool_calls(
                    &raw_text,
                    &messages_for_resolve,
                    tools_for_resolve.as_deref(),
                    tool_choice_for_resolve.as_ref(),
                );
                if !tool_calls.is_empty() {
                    finish_reason = "tool_calls".to_string();
                    for chunk_data in
                        stream_tool_call_chunks(&chat_id, created, &model_name, &tool_calls)
                    {
                        let _ = tx
                            .send(Ok(Event::default().data(chunk_data)))
                            .await;
                    }
                }
            }
        }

        let done_chunk_data = stream_chunk_json(
            &chat_id,
            created,
            &model_name,
            serde_json::json!({}),
            Some(&finish_reason),
        );
        let _ = tx
            .send(Ok(Event::default().data(done_chunk_data)))
            .await;

        if include_usage {
            let stream_end = Instant::now();
            let prompt_ms = first_token_at
                .map(|t| t.duration_since(stream_started).as_secs_f64() * 1000.0)
                .unwrap_or_else(|| stream_end.duration_since(stream_started).as_secs_f64() * 1000.0);
            let predicted_ms = first_token_at
                .map(|t| stream_end.duration_since(t).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let timings = StreamTimingsSnapshot {
                prompt_ms,
                predicted_ms,
                prompt_tokens,
                completion_tokens: output_tokens.len(),
            };
            let usage_chunk_data = stream_usage_chunk_json(
                &chat_id,
                created,
                &model_name,
                &timings,
            );
            let _ = tx
                .send(Ok(Event::default().data(usage_chunk_data)))
                .await;
        }

        let _ = tx
            .send(Ok(Event::default().data("[DONE]".to_string())))
            .await;
    });

    Ok(sse_stream(rx))
}

fn sse_stream(
    rx: tokio::sync::mpsc::Receiver<Result<Event, std::convert::Infallible>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    Sse::new(ReceiverStream::new(rx)).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

pub async fn run_server(model: Gemma4GpuModel, tokenizer: tokenizers::Tokenizer, port: u16) {
    let max_context_len = model.kv_capacity as usize;
    let runtime_config = ServerRuntimeConfig::from_env();
    let metrics = Arc::new(Metrics::new());
    let scheduler_config = scheduler::SchedulerConfig {
        max_prefill_tokens_per_tick: runtime_config.max_prefill_tokens_per_tick,
    };
    let request_tx = scheduler::spawn_scheduler_with_config(
        model,
        runtime_config.queue_depth,
        runtime_config.kv_pool_slots,
        metrics.clone(),
        scheduler_config,
    );
    let state = Arc::new(AppState {
        request_tx,
        metrics,
        tokenizer,
        max_context_len,
        runtime_config: runtime_config.clone(),
    });

    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("🚀 Gemma4 E4B server listening on http://{}", addr);
    println!("   Compatible with OpenAI API: /v1/chat/completions");
    println!("   Models: /v1/models");
    println!("   Health: /health");
    println!("   Metrics: /metrics");
    println!(
        "   Runtime: queue_depth={}, kv_pool_slots={}, request_timeout_secs={}, prefill_tokens_per_tick={}",
        runtime_config.queue_depth,
        runtime_config.kv_pool_slots,
        runtime_config.request_timeout.as_secs(),
        runtime_config
            .max_prefill_tokens_per_tick
            .map(|tokens| tokens.to_string())
            .unwrap_or_else(|| "model_default".to_string()),
    );
    println!(
        "   Context: {} tokens max (override with LLAMA_CTX_SIZE, up to 200000)",
        max_context_len
    );

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: Some("gemma-4-e4b-q4".to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("hello".to_string()),
                ..Default::default()
            }],
            max_tokens: Some(16),
            max_completion_tokens: None,
            temperature: 0.7,
            stream: false,
            stop: None,
            min_p: 0.05,
            top_k: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            top_p: None,
            tools: None,
            tool_choice: None,
            stream_options: None,
        }
    }

    fn should_include_stream_usage_defaults_true() {
        assert!(should_include_stream_usage(None));
        assert!(should_include_stream_usage(Some(&StreamOptions {
            include_usage: true,
        })));
        assert!(!should_include_stream_usage(Some(&StreamOptions {
            include_usage: false,
        })));
    }

    #[test]
    fn trim_stop_sequences_prefers_earliest_builtin_or_request_stop() {
        let request_stop = vec!["CUSTOM_STOP".to_string()];
        let mut text = "hello CUSTOM_STOP ignored <turn|>".to_string();

        assert!(trim_stop_sequences(&mut text, Some(&request_stop)));
        assert_eq!(text, "hello ");

        let mut built_in_first = "hello <turn|> CUSTOM_STOP".to_string();
        assert!(trim_stop_sequences(
            &mut built_in_first,
            Some(&request_stop)
        ));
        assert_eq!(built_in_first, "hello ");
    }

    #[test]
    fn stream_trim_holds_back_partial_stop_prefixes() {
        let request_stop = vec!["CUSTOM_STOP".to_string()];
        let mut text = "hello <end".to_string();

        assert!(!trim_stream_safe_text(&mut text, Some(&request_stop), false));
        assert_eq!(text, "hello ");

        let mut custom = "hello CUSTOM_".to_string();
        assert!(!trim_stream_safe_text(&mut custom, Some(&request_stop), false));
        assert_eq!(custom, "hello ");
    }

    #[test]
    fn generation_does_not_stop_at_channel_end() {
        let text = "<|channel>thought\nhello<channel|><|tool_call>call:bash{command:<|\"|>ls<|\"|>}";
        assert!(find_generation_stop_position(text, None).is_none());
    }

    #[test]
    fn generation_stops_after_complete_native_tool_call() {
        let text = "<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>";
        assert_eq!(
            find_generation_stop_position(text, None),
            Some(text.find(NATIVE_TOOL_CALL_SUFFIX).unwrap())
        );
    }

    #[test]
    fn parse_tool_calls_handles_fenced_and_tagged_blocks() {
        let fenced = "Sure!\n```tool_call\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.txt\"}}\n```";
        let calls = parse_tool_calls(fenced, None, true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].call_type, "function");
        assert_eq!(calls[0].function.arguments, "{\"path\":\"a.txt\"}");

        let tagged = "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}</tool_call>";
        let calls = parse_tool_calls(tagged, None, true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");

        let multi = "```tool_call\n{\"name\": \"a\", \"arguments\": {}}\n```\n```tool_call\n{\"name\": \"b\", \"arguments\": {}}\n```";
        assert_eq!(parse_tool_calls(multi, None, true).len(), 2);

        assert!(parse_tool_calls("just a normal answer", None, true).is_empty());

        let bare = r#"{"name": "read_file", "arguments": {"path": "package.json"}}"#;
        let calls = parse_tool_calls(bare, None, true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[test]
    fn effective_max_tokens_prefers_max_completion_tokens() {
        let mut req = valid_request();
        req.max_tokens = Some(100);
        req.max_completion_tokens = Some(256);
        assert_eq!(effective_max_tokens(&req), 256);

        req.max_completion_tokens = None;
        assert_eq!(effective_max_tokens(&req), 100);

        req.max_tokens = None;
        assert_eq!(effective_max_tokens(&req), 1024);
    }

    #[test]
    fn prompt_token_budget_does_not_collapse_when_max_tokens_equals_context() {
        // Agent clients often set max_tokens ≈ context_len.
        assert_eq!(prompt_token_budget(16384, 16384), 8192);
        assert_eq!(prompt_token_budget(16384, 1), 16383);
        assert_eq!(
            clamp_max_tokens_to_context(5420, 16384, 16384).ok(),
            Some(10964)
        );
    }

    #[test]
    fn assistant_message_out_uses_null_content_for_tool_calls() {
        let message = assistant_message_out(
            String::new(),
            vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read".to_string(),
                    arguments: "{}".to_string(),
                },
            }],
            ChannelSplitMode::default(),
        );
        let json = serde_json::to_value(message).unwrap();
        assert!(json.get("content").unwrap().is_null());
        assert_eq!(json.get("tool_calls").unwrap().as_array().unwrap().len(), 1);
    }

    #[test]
    fn apply_chat_template_uses_peg_gemma4_system_turn() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some("be helpful".to_string()),
                ..Default::default()
            },
            Message {
                role: "user".to_string(),
                content: Some("hi".to_string()),
                ..Default::default()
            },
        ];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: Some("Read a file".to_string()),
                parameters: None,
            },
        }];

        let prompt = apply_chat_template(&messages, Some(&tools), None);
        assert!(prompt.contains("<|turn>system\n"));
        assert!(prompt.contains("<|think|>"));
        assert!(prompt.contains("be helpful"));
        assert!(prompt.contains("<|tool>declaration:read"));
        assert!(prompt.contains("<|turn>user\nhi<turn|>"));
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn apply_chat_template_ends_at_model_turn_for_plain_chat() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("Say hello".to_string()),
            ..Default::default()
        }];
        let prompt = apply_chat_template(&messages, None, None);
        assert!(prompt.contains("<|turn>user\nSay hello<turn|>"));
        // E2B/E4B thinking-off: no empty thought channel stub.
        assert!(prompt.ends_with("<|turn>model\n"));
        assert!(!prompt.contains(EMPTY_THOUGHT_PREFIX));
    }

    #[test]
    fn split_reasoning_puts_plain_chat_in_content_without_tools() {
        let text = "Hello! Here is a short summary.";
        let (reasoning, content) = split_reasoning_and_content_with_mode(
            text,
            ChannelSplitMode {
                plain_text_as_reasoning: false,
            },
        );
        assert!(reasoning.is_empty());
        assert_eq!(content, text);
    }

    #[test]
    fn apply_chat_template_primes_empty_thought_for_tools_only_request() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("list files".to_string()),
            ..Default::default()
        }];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "bash".to_string(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": { "type": "string" }
                    }
                })),
            },
        }];

        let prompt = apply_chat_template(&messages, Some(&tools), None);
        assert!(prompt.contains("<|turn>system\n"));
        assert!(prompt.contains("<|think|>"));
        assert!(prompt.contains("<|tool>declaration:bash"));
        assert!(prompt.contains("<|turn>user\nlist files<turn|>"));
        assert!(prompt.ends_with("<|turn>model\n"));
        assert!(!prompt.contains(EMPTY_THOUGHT_PREFIX));
    }

    #[test]
    fn extract_at_file_path_handles_quoted_and_unquoted_pi_references() {
        assert_eq!(
            extract_at_file_path(r#"summarize @"Sports Highlights.md""#),
            Some("Sports Highlights.md".to_string())
        );
        assert_eq!(
            extract_at_file_path("summarize @test.md"),
            Some("test.md".to_string())
        );
        assert_eq!(
            extract_at_file_path("read @src/foo.rs please"),
            Some("src/foo.rs".to_string())
        );
    }

    #[test]
    fn infer_tool_calls_without_generation_handles_pi_summarize_at_file() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(
                "summarize @\"Sports Highlights — Catch Up to Live Highlights Design.md\""
                    .to_string(),
            ),
            ..Default::default()
        }];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let calls =
            infer_tool_calls_without_generation(&messages, Some(&tools), None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"Sports Highlights — Catch Up to Live Highlights Design.md"}"#
        );
    }

    #[test]
    fn infer_tool_calls_without_generation_handles_unquoted_at_file() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("summarize @test.md".to_string()),
            ..Default::default()
        }];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let calls =
            infer_tool_calls_without_generation(&messages, Some(&tools), None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"test.md"}"#);
    }

    #[test]
    fn resolve_tool_calls_infers_read_for_pi_at_file_summarize() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(
                "summarize @\"Sports Highlights — Catch Up to Live Highlights Design.md\""
                    .to_string(),
            ),
            ..Default::default()
        }];
        let tools = vec![
            Tool {
                tool_type: Some("function".to_string()),
                function: FunctionDef {
                    name: "read".to_string(),
                    description: None,
                    parameters: None,
                },
            },
            Tool {
                tool_type: Some("function".to_string()),
                function: FunctionDef {
                    name: "bash".to_string(),
                    description: None,
                    parameters: None,
                },
            },
        ];
        let prose = "The user wants me to summarize the file. I need to use the read tool first.";
        let calls = resolve_tool_calls(prose, &messages, Some(&tools), None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"Sports Highlights — Catch Up to Live Highlights Design.md"}"#
        );
    }

    #[test]
    fn parse_plain_read_invocation_handles_read_line() {
        let text = "I will read the file.\nread Sports Highlights.md\n";
        let calls = parse_tool_calls(text, Some(&["read".to_string()]), true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"Sports Highlights.md"}"#
        );
    }

    #[test]
    fn resolve_tool_calls_infers_bash_ls_for_list_files_request() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("list files".to_string()),
            ..Default::default()
        }];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "bash".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let prose = "I need to list the files. I will use the `bash` tool to execute `ls`.";
        let calls = resolve_tool_calls(prose, &messages, Some(&tools), None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn split_reasoning_and_content_treats_final_channel_as_content() {
        let text = "<|channel>final\nSummary of the document.<channel|>";
        let (reasoning, content) = split_reasoning_and_content(text);
        assert!(reasoning.is_empty());
        assert_eq!(content, "Summary of the document.");
    }

    #[test]
    fn generation_priming_suffix_is_empty_for_e2b_post_tool_answer() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("summarize @test.md".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: r#"{"path":"test.md"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("file body".to_string()),
                name: Some("read".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
        ];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: None,
            },
        }];
        assert_eq!(
            generation_priming_suffix(&messages, Some(&tools), None),
            ""
        );
    }

    #[test]
    fn awaits_tool_call_false_after_read_tool_result() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("summarize @test.md".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: r#"{"path":"test.md"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("# Title\n\nBody.".to_string()),
                name: Some("read".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
        ];
        assert!(!awaits_tool_call(&messages));
        assert!(!should_force_tool_call(&messages, Some(&[Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: None,
            },
        }]), None));
    }

    #[test]
    fn awaits_tool_call_false_with_trailing_empty_user_after_tool() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("summarize @test.md".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: r#"{"path":"test.md"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("file body".to_string()),
                name: Some("read".to_string()),
                ..Default::default()
            },
            Message {
                role: "user".to_string(),
                content: Some(String::new()),
                ..Default::default()
            },
        ];
        assert!(!awaits_tool_call(&messages));
    }

    #[test]
    fn resolve_tool_calls_does_not_reinfer_read_on_answer_turn() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("summarize @test.md".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: r#"{"path":"test.md"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("# Sports Highlights\n\nCatch-up to live.".to_string()),
                name: Some("read".to_string()),
                ..Default::default()
            },
        ];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let calls = resolve_tool_calls(
            "read Sports Highlights — Catch Up to Live Highlights Design.md",
            &messages,
            Some(&tools),
            None,
        );
        assert!(calls.is_empty());
    }

    #[test]
    fn apply_chat_template_omits_tool_declarations_on_post_tool_summarize() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("summarize @test.md".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: r#"{"path":"test.md"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("# Title\n\nBody.".to_string()),
                name: Some("read".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
        ];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: Some("Read a file".to_string()),
                parameters: None,
            },
        }];
        let prompt = apply_chat_template(&messages, Some(&tools), None);
        assert!(!prompt.contains("declaration:read"));
        assert!(prompt.contains("# Title"));
        // Post-tool answer: remain in open model turn with no empty-thought stub.
        assert!(prompt.contains("<|tool_response>response:read{"));
        assert!(!prompt.contains("<|channel>thought\n<channel|>"));
        assert!(!prompt.ends_with("<turn|>\n"));
    }

    #[test]
    fn compute_stream_deltas_streams_thought_as_reasoning_on_answer_turn() {
        let text = "<|channel>thought\nHere is the summary.<channel|>";
        let (new_reasoning, new_content, _, _) =
            compute_stream_deltas(text, 0, 0, false, ChannelSplitMode::default());
        assert_eq!(new_reasoning, "Here is the summary.");
        assert!(new_content.is_empty());
    }

    #[test]
    fn compute_stream_deltas_keeps_post_channel_text_in_reasoning_until_final() {
        let text = "Plan.<channel|>Still planning.";
        let mode = ChannelSplitMode {
            plain_text_as_reasoning: true,
        };
        let (new_reasoning, new_content, _, _) =
            compute_stream_deltas(text, 0, 0, false, mode);
        assert_eq!(new_reasoning, "Plan.\nStill planning.");
        assert!(new_content.is_empty());
    }

    #[test]
    fn split_reasoning_keeps_analysis_in_reasoning_after_standalone_channel_close() {
        let text = "The user wants a summary.<channel|>The user provided details.\n\n**Analysis:**\n* bullet";
        let (reasoning, content) = split_reasoning_and_content_with_mode(
            text,
            ChannelSplitMode {
                plain_text_as_reasoning: true,
            },
        );
        assert!(content.is_empty());
        assert!(reasoning.contains("Analysis"));
        assert!(reasoning.contains("The user provided"));
    }

    #[test]
    fn split_reasoning_promotes_implicit_final_answer_after_channel_markup() {
        let text = "Plan.<channel|>More plan.I will synthesize this into a concise summary.This is a description of the film.";
        let (reasoning, content) = split_reasoning_and_content_with_mode(
            text,
            ChannelSplitMode {
                plain_text_as_reasoning: true,
            },
        );
        assert!(reasoning.contains("Plan."));
        assert!(reasoning.contains("synthesize"));
        assert_eq!(content, "This is a description of the film.");
    }

    #[test]
    fn compute_stream_deltas_streams_final_channel_as_content_on_answer_turn() {
        let text = "<|channel>thought\nPlanning the summary.<channel|><|channel>final\nHere is the summary.<channel|>";
        let (new_reasoning, new_content, _, _) =
            compute_stream_deltas(text, 0, 0, false, ChannelSplitMode::default());
        assert_eq!(new_reasoning, "Planning the summary.");
        assert_eq!(new_content, "Here is the summary.");
    }

    #[test]
    fn compute_stream_deltas_streams_content_after_tool_result_turn() {
        let text = "Here is a concise summary of the document.";
        let (new_reasoning, new_content, _, _) =
            compute_stream_deltas(text, 0, 0, false, ChannelSplitMode::default());
        assert!(new_reasoning.is_empty());
        assert_eq!(new_content, text);
    }

    #[test]
    fn split_reasoning_promotes_plain_text_answer_without_channel_markup() {
        let text = "Plan.\n\nI will synthesize this into a concise summary.\nThis is a description of the film.";
        let (reasoning, content) = split_reasoning_and_content_with_mode(
            text,
            ChannelSplitMode {
                plain_text_as_reasoning: true,
            },
        );
        assert!(reasoning.contains("synthesize"));
        assert_eq!(content, "This is a description of the film.");
    }

    #[test]
    fn split_reasoning_promotes_movie_summary_after_planning_paragraph() {
        let text = concat!(
            "User wants me to summarize the provided text about the movie \"Once Upon a Time\".\n",
            "The user has provided a descriptive text about the film.\n",
            "I need to synthesize this information into a coherent summary.\n\n",
            "The movie \"Once Upon a Time\" is described as a film set in 1969 Los Angeles.",
        );
        let (reasoning, content) = split_reasoning_and_content_with_mode(
            text,
            ChannelSplitMode {
                plain_text_as_reasoning: true,
            },
        );
        assert!(reasoning.contains("synthesize"));
        assert!(content.starts_with("The movie \"Once Upon a Time\""));
    }

    #[test]
    fn split_tool_generation_output_treats_pre_tool_text_as_reasoning() {
        let text = "I will list files.<|tool_call>call:bash{command:<|\"|>ls -F<|\"|>}";
        let (reasoning, content) = split_tool_generation_output(text);
        assert_eq!(reasoning, "I will list files.");
        assert!(content.is_empty());
    }

    #[test]
    fn use_tool_generation_stream_split_prefers_channel_tags_on_tool_turn() {
        assert!(!use_tool_generation_stream_split(
            "<|channel>final\nHere is the summary.",
            false,
            true,
        ));
        assert!(!use_tool_generation_stream_split(
            "planning<channel|>answer",
            false,
            true,
        ));
        assert!(use_tool_generation_stream_split(
            "plain preamble before tool call",
            false,
            true,
        ));
        assert!(use_tool_generation_stream_split(
            "streaming tool args",
            true,
            true,
        ));
        assert!(!use_tool_generation_stream_split(
            "plain answer text",
            false,
            false,
        ));
    }

    #[test]
    fn compute_stream_deltas_streams_content_on_tool_turn_with_final_channel() {
        let text = "<|channel>final\nHere is the summary.";
        let (new_reasoning, new_content, _, _) =
            compute_stream_deltas(text, 0, 0, false, ChannelSplitMode::default());
        assert!(new_reasoning.is_empty());
        assert_eq!(new_content, "Here is the summary.");
    }

    #[test]
    fn parse_tool_calls_handles_tool_code_fallback_format() {
        let allowed = vec!["bash".to_string()];
        let text = "tool_code\nbash\nls\n";
        let calls = parse_tool_calls(text, Some(&allowed), true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn gemma4_tool_parameters_match_jinja_shape() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": { "type": "string" }
            }
        });
        let rendered = json_schema_to_gemma4_params(&schema);
        assert!(rendered.contains("properties:{command:{type:<|\"|>STRING<|\"|>}"));
        assert!(rendered.contains("required:[<|\"|>command<|\"|>]"));
        assert!(rendered.contains("type:<|\"|>OBJECT<|\"|>"));
    }

    #[test]
    fn gemma4_dict_to_json_args_incremental_handles_partial_and_complete() {
        let partial = "{command:<|\"|>ls -F";
        assert_eq!(
            gemma4_dict_to_json_args_incremental(partial),
            r#"{"command":"ls -F"#
        );

        let complete = r#"{command:<|"|>ls -F<|"|>}"#;
        assert_eq!(
            gemma4_dict_to_json_args_incremental(complete),
            r#"{"command":"ls -F"}"#
        );
    }

    #[test]
    fn tool_call_arguments_delta_skips_bare_empty_object() {
        assert_eq!(
            tool_call_arguments_delta("{", r#"{"command":"ls -F"}"#).as_deref(),
            Some(r#""command":"ls -F"}"#)
        );
        assert!(tool_call_arguments_delta("{", "{}").is_none());
    }

    #[test]
    fn incremental_tool_call_emitter_finalize_after_stop_suffix() {
        let mut emitter = IncrementalToolCallEmitter::new(0);
        let allowed = vec![
            "read".to_string(),
            "bash".to_string(),
            "edit".to_string(),
            "write".to_string(),
        ];
        // Simulate streaming `bash{` then generation stopping on `<tool_call|>`.
        let partial = r#"<|channel>thought
think<channel|><|tool_call>call:bash{"#;
        let early = emitter.update(partial, "id", 1, "model", &allowed);
        let full = r#"<|channel>thought
think<channel|><|tool_call>call:bash{}<tool_call|>"#;
        let late = emitter.finalize(full, "id", 1, "model", &allowed);
        let combined_args: String = early
            .iter()
            .chain(late.iter())
            .filter_map(|c| {
                let v: serde_json::Value = serde_json::from_str(c).ok()?;
                v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .map(|s| s.to_string())
            })
            .collect();
        assert_eq!(combined_args, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn incremental_tool_call_emitter_streams_argument_deltas() {
        let mut emitter = IncrementalToolCallEmitter::new(0);
        let text = r#"<|channel>thought
think<channel|><|tool_call>call:bash{command:<|"|>ls -F<|"|>}"#;
        let allowed = vec![
            "read".to_string(),
            "bash".to_string(),
            "edit".to_string(),
            "write".to_string(),
        ];
        let chunks = emitter.update(text, "id", 1, "model", &allowed);
        assert!(emitter.header_sent);
        assert!(!chunks.is_empty());
        let combined_args: String = chunks
            .iter()
            .filter_map(|c| {
                let v: serde_json::Value = serde_json::from_str(c).ok()?;
                v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .map(|s| s.to_string())
            })
            .collect();
        assert_eq!(combined_args, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn apply_chat_template_renders_tool_results_and_assistant_calls() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("read a.txt".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: "{\"path\":\"a.txt\"}".to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("file contents".to_string()),
                name: Some("read".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
        ];

        let prompt = apply_chat_template(&messages, None, None);
        assert!(prompt.contains("<|turn>model\n"));
        assert!(prompt.contains("<|tool_call>call:read{"));
        assert!(prompt.contains("<|tool_response>response:read{value:"));
        assert!(prompt.contains("file contents"));
        assert!(!prompt.contains("Tool result for"));
    }

    #[test]
    fn apply_chat_template_primes_generation_after_tool_result_in_open_turn() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("summarize @test.md".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: r#"{"path":"test.md"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("# Title\n\nLong file body.".to_string()),
                name: Some("read".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
        ];
        let tools = vec![Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let prompt = apply_chat_template(&messages, Some(&tools), None);
        assert!(prompt.contains("<|tool_response>response:read{value:"));
        // Stay in the open model turn; E2B/E4B do not prime an empty thought channel.
        assert!(prompt.contains("<|tool_response>response:read{"));
        assert!(!prompt.contains("<|channel>thought\n<channel|>"));
        assert!(!prompt.ends_with("<turn|>\n"));
    }

    #[test]
    fn apply_chat_template_merges_tool_response_into_model_turn() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("list files".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "bash".to_string(),
                        arguments: r#"{"command":"ls -F"}"#.to_string(),
                    },
                }]),
                ..Default::default()
            },
            Message {
                role: "tool".to_string(),
                content: Some("a.md\nb.csv".to_string()),
                name: Some("bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            Message {
                role: "assistant".to_string(),
                content: Some("I have listed the files.".to_string()),
                ..Default::default()
            },
            Message {
                role: "user".to_string(),
                content: Some("list files".to_string()),
                ..Default::default()
            },
        ];

        let prompt = apply_chat_template(&messages, None, None);
        assert!(prompt.contains(
            "<|tool_call>call:bash{command:<|\"|>ls -F<|\"|>}<tool_call|><|tool_response>response:bash{value:<|\"|>a.md\nb.csv<|\"|>}<tool_response|>I have listed the files.<turn|>"
        ));
    }

    #[test]
    fn parse_tool_calls_handles_gemma4_native_format() {
        let native = r#"<|tool_call>call:read{path:<|"|>package.json<|"|>}<tool_call|>"#;
        let calls = parse_tool_calls(native, None, true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"package.json"}"#);

        let without_suffix =
            r#"<|tool_call>call:bash{command:<|"|>ls -F<|"|>}"#;
        let calls = parse_tool_calls(without_suffix, None, true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn parse_tool_calls_remaps_hallucinated_list_files_to_bash() {
        let allowed = vec![
            "read".to_string(),
            "bash".to_string(),
            "edit".to_string(),
            "write".to_string(),
        ];
        let native = r#"<|tool_call>call:list_files{}<tool_call|>"#;
        let calls = parse_tool_calls(native, Some(&allowed), true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn parse_tool_calls_fills_empty_bash_args_with_ls() {
        let allowed = vec![
            "read".to_string(),
            "bash".to_string(),
            "edit".to_string(),
            "write".to_string(),
        ];
        let native = r#"<|tool_call>call:bash{}<tool_call|>"#;
        let calls = parse_tool_calls(native, Some(&allowed), true);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn incremental_tool_call_emitter_fills_empty_bash_args() {
        let mut emitter = IncrementalToolCallEmitter::new(0);
        let text = r#"<|channel>thought
think<channel|><|tool_call>call:bash{}<tool_call|>"#;
        let allowed = vec![
            "read".to_string(),
            "bash".to_string(),
            "edit".to_string(),
            "write".to_string(),
        ];
        let chunks = emitter.update(text, "id", 1, "model", &allowed);
        let combined_args: String = chunks
            .iter()
            .filter_map(|c| {
                let v: serde_json::Value = serde_json::from_str(c).ok()?;
                v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .map(|s| s.to_string())
            })
            .collect();
        assert_eq!(combined_args, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn split_reasoning_and_content_splits_channel_tags() {
        let text = "<|channel>thought\nThe user said hi.<channel|>Hello!";
        let (reasoning, content) = split_reasoning_and_content(text);
        assert_eq!(reasoning, "The user said hi.");
        assert_eq!(content, "Hello!");
    }

    #[test]
    fn deserialize_message_accepts_array_and_null_content() {
        let array: Message = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(array.content.as_deref(), Some("hi"));

        let null_content: Message = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}]}"#,
        )
        .unwrap();
        assert_eq!(null_content.content, None);
        assert_eq!(null_content.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn validate_request_rejects_common_bad_inputs() {
        let mut req = valid_request();
        req.messages.clear();
        let err = validate_request(&req).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "empty_messages");

        let mut req = valid_request();
        req.max_tokens = Some(0);
        assert_eq!(
            validate_request(&req).unwrap_err().code,
            "invalid_max_tokens"
        );

        let mut req = valid_request();
        req.temperature = -0.1;
        assert_eq!(
            validate_request(&req).unwrap_err().code,
            "invalid_temperature"
        );

        let mut req = valid_request();
        req.min_p = 1.1;
        assert_eq!(validate_request(&req).unwrap_err().code, "invalid_min_p");

        let mut req = valid_request();
        req.repetition_penalty = 0.0;
        assert_eq!(
            validate_request(&req).unwrap_err().code,
            "invalid_repetition_penalty"
        );

        let mut req = valid_request();
        req.frequency_penalty = 3.0;
        assert_eq!(
            validate_request(&req).unwrap_err().code,
            "invalid_frequency_penalty"
        );
    }

    #[test]
    fn validate_context_len_rejects_over_limit_requests() {
        assert!(validate_context_len(8, 8, 16).is_ok());
        assert_eq!(
            validate_context_len(16, 1, 16).unwrap_err().code,
            "context_length_exceeded"
        );
        assert_eq!(
            validate_context_len(15, 2, 16).unwrap_err().code,
            "context_length_exceeded"
        );
    }

    #[test]
    fn runtime_config_defaults_are_production_safe() {
        let config = ServerRuntimeConfig::from_lookup(|_| None);

        assert_eq!(config.queue_depth, 32);
        assert_eq!(config.kv_pool_slots, 4);
        assert_eq!(config.request_timeout, Duration::from_secs(300));
        assert_eq!(config.max_prefill_tokens_per_tick, None);
    }

    #[test]
    fn runtime_config_applies_env_overrides_and_clamps_zeroes() {
        let config = ServerRuntimeConfig::from_lookup(|name| match name {
            "LLAMA_QUEUE_DEPTH" => Some("0".to_string()),
            "LLAMA_KV_POOL_SLOTS" => Some("8".to_string()),
            "LLAMA_REQUEST_TIMEOUT_SECS" => Some("0".to_string()),
            "LLAMA_PREFILL_TOKENS_PER_TICK" => Some("32".to_string()),
            _ => None,
        });

        assert_eq!(config.queue_depth, 1);
        assert_eq!(config.kv_pool_slots, 8);
        assert_eq!(config.request_timeout, Duration::from_secs(1));
        assert_eq!(config.max_prefill_tokens_per_tick, Some(32));
    }

    #[test]
    fn runtime_config_metrics_expose_static_scheduler_knobs() {
        let config = ServerRuntimeConfig {
            queue_depth: 16,
            kv_pool_slots: 2,
            request_timeout: Duration::from_secs(30),
            max_prefill_tokens_per_tick: Some(64),
        };
        let metrics = config.render_prometheus();

        assert!(metrics.contains("llama_config_queue_depth 16"));
        assert!(metrics.contains("llama_config_kv_pool_slots 2"));
        assert!(metrics.contains("llama_config_request_timeout_secs 30"));
        assert!(metrics.contains("llama_config_prefill_tokens_per_tick 64"));
    }
}
