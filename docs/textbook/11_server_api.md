# 11 — The HTTP Layer: From JSON to Tokens and Back

This chapter is about everything that happens *outside* the GPU: turning an
OpenAI-shaped chat request into token ids, handing it to the scheduler,
turning a token stream back into SSE deltas, and doing all of that without
letting an async runtime touch the model.

`src/server.rs` is big (~4700 lines) but the structure is small. Learn the
five phases and you can navigate it.

---

## Part A — The surface

```3465:3473:src/server.rs
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}
```

Five routes, one of which does the work. `/v1/models` is duplicated at
`/models` because clients disagree about the prefix. There is deliberately
**no** completion endpoint and **no** cancel endpoint — cancellation is
inferred from the stream (Part E).

`AppState` is shared immutably via `Arc`: the tokenizer, the request sender,
metrics, `max_context_len`, and runtime config. Note what is *not* in it: the
model. The model lives on the scheduler thread and is unreachable from any
handler. That is enforced by construction, not by convention, and it is why no
handler can accidentally block on the GPU.

`run_server_with_mtp` picks the engine at startup:

```3479:3489:src/server.rs
/// Serve mode with optional MTP (speculative draft/verify) decoding.
///
/// When `mtp_assistant` is `Some`, the server uses the serial MTP scheduler:
/// requests are prefilled and decoded one at a time via draft/verify (the batched
/// multi-slot scheduler is bypassed, since MTP is inherently single-sequence).
pub async fn run_server_with_mtp(
    model: Gemma4GpuModel,
    tokenizer: tokenizers::Tokenizer,
    port: u16,
    mtp_assistant: Option<Gemma4MtpAssistant>,
) { ... }
```

Both schedulers consume the same `InferenceRequest` and emit the same
`StreamEvent`s, so the HTTP layer does not know which one it is talking to.
That interface is the reason MTP could be developed without touching the API
code at all — worth copying as a design habit.

---

## Part B — Phase 1: request → prompt string

`encode_prompt` does three things in order, and the order matters:

```2803:2829:src/server.rs
fn encode_prompt(state: &AppState, messages: &[Message], tools: Option<&[Tool]>,
                 tool_choice: Option<&serde_json::Value>, max_tokens: usize)
    -> Result<Vec<usize>, ApiError>
{
    let max_prompt_tokens = prompt_token_budget(state.max_context_len, max_tokens);
    let fitted = fit_messages_to_context(messages, tools, tool_choice,
                                        &state.tokenizer, max_prompt_tokens)?;
    let prompt = apply_chat_template(&fitted, tools, tool_choice);
    let encoding = state.tokenizer.encode(prompt.as_str(), true)
        .map_err(|err| ApiError::bad_request("tokenizer_error", ...))?;
    Ok(encoding.get_ids().iter().map(|&t| t as usize).collect())
}
```

1. **Budget** — how many prompt tokens may we spend, given `max_tokens` must
   also fit in the context window.
2. **Fit** — drop or trim history until the prompt fits that budget. This is
   where long conversations get truncated, and it happens *before* templating
   so the template is applied to what will actually be sent.
3. **Template + tokenize.**

### B.1 The chat template is model-specific and load-bearing

`apply_chat_template` (~1977) renders Gemma4's turn structure. The pieces that
surprise people coming from Llama-style templates:

- **Turn markers** — `<|turn>` / `<turn|>` style control tokens delimit
  role turns. The exact ids appear in the scheduler's first-token blocklist
  (Ch 12 Part D.1) precisely because the model can emit them where they would
  produce an empty first chunk.
- **Channels** — Gemma4 separates *thought* from *final answer* with
  `<|channel>` / `<channel|>` delimiters. This is how the server can expose
  reasoning separately from content.
- **Tool declarations** — tools are rendered into the prompt in a Gemma4
  specific syntax, with its own string quoting (`gemma4_string`,
  `json_value_to_gemma4`: strings are wrapped in `<|"|>…<|"|>`, objects become
  `k:v` pairs). JSON goes in, a Gemma4-flavored literal comes out.
- **Generation priming** — the template ends by opening the model's turn, so
  the first sampled token continues an assistant message rather than starting
  a new turn.

The template has a dozen behavioral unit tests (`apply_chat_template_*` in
the test module). Read their names as a spec: system-turn handling, ending at
the model turn for plain chat, priming an empty thought for tools-only
requests, omitting tool declarations when summarizing after a tool result,
merging tool responses into the model turn. **Template edits without a test
are how a server silently starts producing subtly worse answers.**

`state.tokenizer.encode(prompt, true)` — the `true` adds special tokens
(BOS). The tokenizer itself came from GGUF metadata (Ch 03 Part B), so the ids
in the template, the ids in the blocklist, and the ids in `eos_token_ids` all
originate from the same file.

---

## Part C — Phase 2: parameters and validation

```2573:2594:src/server.rs
fn generation_params_from_request(req: &ChatCompletionRequest, request_timeout: Duration)
    -> Result<GenerationParams, ApiError>
{
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
            &req.messages, req.tools.as_deref(), req.tool_choice.as_ref()),
        request_timeout,
    })
}
```

`GenerationParams` is the whole contract between HTTP and engine. Two fields
are not in the OpenAI API at all:

- **`eos_token_ids: vec![1, 106]`** — `<eos>` and the end-of-turn marker.
  Turn-end must terminate a response even though it is not the model's EOS.
- **`min_decode_tokens`** — computed *from the request shape*. A tools-only
  request or a post-tool summarization has different degenerate-empty-answer
  risk than plain chat, so the floor differs. This feeds the re-sampling loop
  in Ch 12 Part D.1.

### C.1 Context arithmetic happens twice, on purpose

```2785:2801:src/server.rs
fn clamp_max_tokens_to_context(prompt_tokens: usize, requested_max_tokens: usize,
                               max_context_len: usize) -> Result<usize, ApiError> {
    if prompt_tokens >= max_context_len {
        return Err(ApiError::bad_request("context_length_exceeded", ...));
    }
    let remaining = max_context_len - prompt_tokens;
    Ok(requested_max_tokens.min(remaining).max(1))
}
```

then

```2846:2854:src/server.rs
if prompt_tokens + max_tokens > max_context_len {
    return Err(ApiError::bad_request("context_length_exceeded",
        format!("prompt tokens ({}) plus max_tokens ({}) exceeds the model context limit of {}",
            prompt_tokens, max_tokens, max_context_len)));
}
```

First clamp (be generous: shrink `max_tokens` to what fits), then validate
(be strict: reject if it still does not). Both exist because a request can be
unsatisfiable in two different ways, and the error messages differ.

The tests pin the boundaries exactly:

```4679:4687:src/server.rs
assert!(validate_context_len(8, 8, 16).is_ok());
assert_eq!(validate_context_len(16, 1, 16).unwrap_err().code, "context_length_exceeded");
assert_eq!(validate_context_len(15, 2, 16).unwrap_err().code, "context_length_exceeded");
```

`8+8=16` fits exactly. `15+2=17` does not. Off-by-one here means a request
that overruns the KV slot mid-generation, which fails much less politely.

---

## Part D — Phase 3: enqueue and the queue-full boundary

```2547:2571:src/server.rs
fn enqueue_request(state: &AppState, input_ids: Vec<usize>, params: GenerationParams)
    -> Result<(tokio::sync::mpsc::Receiver<StreamEvent>, Arc<AtomicU8>), ApiError>
{
    let (response_tx, response_rx) = tokio::sync::mpsc::channel(64);
    let cancel = Arc::new(AtomicU8::new(CANCEL_NONE));
    let prompt_tokens = input_ids.len();
    let request = InferenceRequest {
        id: format!("req-{}", uuid::Uuid::new_v4()),
        input_ids, params, response_tx,
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
```

Everything important about backpressure is in these 20 lines:

- **`try_send`, never `send`.** A full queue returns **429** immediately. No
  unbounded latency, no unbounded memory. Default `queue_depth = 32`
  (`LLAMA_QUEUE_DEPTH`).
- **Two separate limits.** The queue (32) is admission to the *scheduler*; KV
  pool slots (4, `LLAMA_KV_POOL_SLOTS`) are admission to the *GPU*. A request
  can pass the first and be rejected by the second (Ch 12 Part B.1) — those
  are different errors with different meanings.
- **Response channel capacity 64.** If the client cannot keep up, the channel
  fills, `blocking_send` fails on the scheduler thread, and the request is
  cancelled. That is the flow-control path for slow consumers.
- **The handler gets back exactly two things:** the receiver and the cancel
  flag. That pair *is* the request handle.

---

## Part E — Phase 4: streaming out

`chat_completions_stream` sets up an SSE channel and spawns a task:

```3023:3023:src/server.rs
let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);
```

The spawned task loop, in essence:

```text
for each StreamEvent from the scheduler:
  Token{id} → push id, decode the full token list to text,
              split into (reasoning, content),
              diff against what was already emitted,
              send only the new suffixes as SSE deltas
  Done{r}   → send finish_reason, optional usage chunk, "[DONE]"
  Error{m}  → send an error chunk
```

### E.1 Why the whole text is re-decoded each token

Because detokenization is **not** a per-token map. Byte-level BPE tokens can
combine into a single character, and Gemma4's channel delimiters can only be
recognized in the decoded string. So the task keeps the token list, decodes
it, and computes deltas by length:

```526:572:src/server.rs
fn compute_stream_deltas(visible_text: &str, emitted_reasoning_len: usize,
                         emitted_content_len: usize, for_tools: bool,
                         split_mode: ChannelSplitMode) -> (String, String, usize, usize)
{
    let (reasoning, content) = if for_tools {
        split_tool_generation_output(visible_text)
    } else {
        split_reasoning_and_content_with_mode(visible_text, split_mode)
    };
    ...
    // When content shrinks (reclassified from content to reasoning
    // because the model just emitted a standalone `<channel|>`
    // delimiter), suppress the duplicate re-emission — the text was
    // already sent as content in earlier deltas.
    let new_reasoning = if content.len() < emitted_content_len { String::new() } else { new_reasoning };
    (new_reasoning, new_content, reasoning.len(), content.len())
}
```

That comment describes a genuinely tricky bug class, and it is worth
understanding because it generalizes to any incremental parser over a growing
string:

1. Model emits text; the splitter classifies it as **content**; you stream it.
2. Model then emits `<channel|>`, which retroactively makes that text part of
   the **reasoning** section.
3. Now `content` is *shorter* than what you already emitted, and `reasoning`
   is longer.
4. Naively diffing would re-emit the same words as reasoning — the user sees
   duplicated text.

The fix: when content shrinks, suppress the reasoning delta. **A monotonic
"emitted length" cursor is only valid over a monotonic string.** Channel
delimiters break monotonicity, so the code detects the shrink explicitly.

The `compute_stream_deltas_*` tests cover each case: thought streamed as
reasoning, post-channel text kept in reasoning until final, final channel
streamed as content, content after a tool-result turn, tool turn with a final
channel.

### E.2 Cancellation, written from here

Every send site does the same thing on failure:

```3206:3213:src/server.rs
if tx.send(Ok(Event::default().data(chunk_data))).await.is_err() {
    cancel.store(CANCEL_CLIENT, Ordering::Relaxed);
    break;
}
```

and stop-sequence detection does:

```3258:3261:src/server.rs
if find_generation_stop_position(&decoded_text, request_stop.as_deref()).is_some() {
    cancel.store(CANCEL_STOP, Ordering::Relaxed);
    break;
}
```

This is the answer to "why an atomic flag rather than just dropping the
channel": **stop sequences live in decoded text, not in token ids.** A stop
string can straddle token boundaries, so only this task — which owns the
detokenizer — can recognize it, and it needs a way to say "stop, and call it a
normal stop." Same mechanism for early-resolved tool calls
(`CANCEL_STOP` + `early_tool_calls`).

### E.3 The zero-forward path

```3059:3115:src/server.rs
let inferred_tool_calls = if has_tools {
    infer_tool_calls_without_generation(&req.messages, req.tools.as_deref(), req.tool_choice.as_ref())
} else { Vec::new() };
if !inferred_tool_calls.is_empty() {
    tokio::spawn(async move { /* role chunk, tool_call chunks, finish, usage, [DONE] */ });
    return Ok(sse_stream(rx));
}
```

If the tool call is fully determined by the request (e.g. a required
single-function `tool_choice`), the server synthesizes the response and
**never enqueues anything**. Zero GPU work, correct OpenAI-shaped SSE
including a usage chunk with `completion_tokens: 0`.

Worth noticing as a serving pattern: the cheapest inference is the inference
you can prove you do not need.

### E.4 Sync vs stream

`chat_completions_sync` consumes the *same* `StreamEvent` stream but
accumulates instead of emitting, then returns one JSON body. It also watches
for stop sequences and sets `CANCEL_STOP` the same way (~2951). One engine
protocol, two presentations — do not add engine features that only one of them
can express.

---

### E.5 What the client actually receives

Everything above is easier to reason about once you have seen the bytes. A
streaming completion is a sequence of `text/event-stream` frames — each one the
literal characters `data: `, a JSON object, and a blank line:

```text
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,
       "model":"gemma-4","choices":[{"index":0,"delta":{"role":"assistant"},
       "finish_reason":null}]}

data: {"…","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"…","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: {"…","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

Four properties of this format drive the code in Parts E.1–E.4:

**Deltas are differences, so the server must remember what it sent.** The frame
carries `" capital"`, not `"The capital"`. But the scheduler hands back
*cumulative decoded text* (it re-detokenizes because token boundaries and UTF-8
character boundaries do not align). Hence `compute_stream_deltas` and the
`emitted_reasoning_len` bookkeeping: the difference between "what the model has
produced" and "what this connection has already been told" is state that lives in
the HTTP task, not the engine.

**A partial UTF-8 sequence must not be emitted.** Detokenizing token-by-token can
produce half of a multi-byte character. If you slice cumulative text at a byte
offset that lands mid-character, the JSON encoder either panics or emits a
replacement char that never gets fixed. Slicing on character boundaries is
therefore not politeness, it is correctness — and it is why the delta logic works
on decoded text rather than tokens.

**Frames are one-way and unacknowledged.** The only signal that the client left
is a failed send, which is exactly what Part E.4's `CANCEL_CLIENT` store reacts
to. There is no heartbeat requirement in the protocol, but note the consequence:
a client that disconnects during a long *prefill* is not detected until the first
frame is attempted.

**The terminator is a literal, not JSON.** `data: [DONE]` is a convention from
the OpenAI API, not a valid chunk object. Clients that `json.loads` every frame
without special-casing it will throw on the last one — worth knowing when you are
debugging an integration and the text arrived fine but the client reports an
error.

For non-streaming requests, the same generation loop runs and the same deltas are
computed; they are simply concatenated and returned as one `chat.completion`
object with a `message` instead of a `delta`. Keeping one code path with two
serializations is why stop-sequence and tool-call behavior cannot drift between
the two modes.

---

## Part F — Runtime configuration

```4692:4697:src/server.rs
let config = ServerRuntimeConfig::from_lookup(|_| None);
assert_eq!(config.queue_depth, 32);
assert_eq!(config.kv_pool_slots, 4);
assert_eq!(config.request_timeout, Duration::from_secs(300));
assert_eq!(config.max_prefill_tokens_per_tick, None);
```

| Env | Default | Effect |
|---|---|---|
| `LLAMA_QUEUE_DEPTH` | 32 | Channel capacity; overflow → 429 |
| `LLAMA_KV_POOL_SLOTS` | 4 | Concurrent sequences; ×KV bytes (Ch 13) |
| `LLAMA_REQUEST_TIMEOUT_SECS` | 300 | Checked at admission and every tick |
| `LLAMA_PREFILL_TOKENS_PER_TICK` | unset | Prefill budget per tick (Ch 12 E.1) |

`from_lookup` takes a closure instead of reading the environment directly,
which is why these defaults are unit-testable at all — including the
`clamps_zeroes` test that ensures `LLAMA_QUEUE_DEPTH=0` cannot produce a
zero-capacity channel. Dependency injection for env vars costs one closure and
buys real tests.

---

## Part G — Exercises

1. Trace a `{"stream": true}` request with tools and a required tool_choice
   that is *not* fully determined. Which functions run in which order, and
   where does the first GPU work happen?

2. `prompt_tokens = 8190`, `max_context_len = 8192`, `max_tokens = 100`.
   What does `clamp_max_tokens_to_context` return? Does
   `validate_context_len` then pass?

3. Why is the prompt fitted to a budget *before* templating rather than after?

4. The model emits `A B C <channel|> D`. Assume the splitter classifies
   `A B C` as content until the delimiter arrives. Write the sequence of
   `(new_reasoning, new_content)` pairs `compute_stream_deltas` produces, and
   show the shrink-suppression firing.

5. `queue_depth = 32` and `kv_pool_slots = 4`. Describe the states of 40
   simultaneous requests and the exact error each group receives.

6. Response channel capacity is 64. What happens if a client reads one token
   per second while the model generates 40/s? Which component notices first?

7. Why can no HTTP handler touch the model? Point to the specific reason in
   `AppState` and say what would break if you added a `&Gemma4GpuModel` field.

---

## Checklist

- [ ] I can name all five routes and the one that does work.
- [ ] I know the three steps of `encode_prompt` and why that order.
- [ ] I can list the two `GenerationParams` fields that are not OpenAI API.
- [ ] I know why context length is both clamped and validated.
- [ ] I can explain `try_send` → 429 and the two independent admission limits.
- [ ] I can explain why the full text is re-decoded every token.
- [ ] I can explain the content-shrink suppression bug and its general form.
- [ ] I know why cancellation is an atomic and who writes each value.

**Next:** [12_scheduler_batching.md](12_scheduler_batching.md) — the other
side of the channel.
