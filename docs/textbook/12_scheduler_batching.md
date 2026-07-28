# 12 — The Scheduler: Continuous Batching in One Thread

The scheduler is the smallest interesting file in the repo and the one most
likely to bite you. It is ~900 lines of single-threaded Rust that owns the
GPU, and every design choice in it follows from one constraint:

> **There is exactly one GPU, so there is exactly one thread that touches it.**

Everything else — continuous batching, fairness, cancellation, timeouts — is
built on top of that with plain `Vec` manipulation and no locks.

This chapter walks the real control flow. Open `src/scheduler.rs` and follow
along; every code block below is quoted from it.

---

## Part A — The shape of the thing

### A.1 Two channels, two worlds

```text
 Tokio async world              std::sync boundary          Scheduler thread
 ─────────────────              ──────────────────          ────────────────
 HTTP handler (server.rs)
   builds InferenceRequest ──►  std::sync::mpsc::SyncSender
                                        │
                                        ▼
                                   request_rx.recv()  ──►  Scheduler::run
                                                                │
   SSE stream  ◄── tokio::mpsc::Receiver ◄── response_tx.blocking_send(StreamEvent)
```

Note the asymmetry, it is deliberate:

- **Inbound** is `std::sync::mpsc` — the scheduler thread is not async, so it
  wants a blocking `recv()`.
- **Outbound** is `tokio::sync::mpsc` — the HTTP handler *is* async, so the
  scheduler uses `blocking_send` to push into a Tokio channel.

```32:45:src/scheduler.rs
pub struct InferenceRequest {
    pub id: String,
    pub input_ids: Vec<usize>,
    pub params: GenerationParams,
    pub response_tx: mpsc::Sender<StreamEvent>,   // tokio sender
    pub cancel: Arc<AtomicU8>,
    pub created_at: Instant,
}

pub enum StreamEvent {
    Token { token_id: usize },
    Done { finish_reason: String },
    Error { message: String },
}
```

`StreamEvent` has exactly three variants. Once you know that, the whole
protocol between engine and HTTP layer fits in your head: a stream of tokens
terminated by exactly one `Done` or one `Error`.

### A.2 The state machine per request

```rust
enum ActivePhase { Prefilling, Decoding }

struct ActiveRequest {
    request: InferenceRequest,
    slot: usize,                 // KV pool slot — the scarce resource
    phase: ActivePhase,
    prefill_cursor: usize,       // how many prompt tokens are in the KV cache
    logits: Vec<f32>,            // output of the last forward, input to sampling
    generated_tokens: Vec<usize>,// for repetition/frequency penalties
    completion_tokens: usize,
    // ... timing fields for metrics
}
```

`prefill_cursor` is the single source of truth for "how far into the prompt
are we." `prefill_cursor == input_ids.len()` is the condition to flip to
`Decoding`.

---

## Part B — The main loop, line by line

```78:102:src/scheduler.rs
pub fn run(mut self, request_rx: Receiver<InferenceRequest>) {
    let mut active = Vec::new();
    let mut receiver_open = true;

    while receiver_open || !active.is_empty() {
        if active.is_empty() && receiver_open {
            match request_rx.recv() {                    // (1) blocking
                Ok(request) => self.admit_request(request, &mut active),
                Err(_) => { receiver_open = false; continue; }
            }
        }

        if receiver_open && self.engine.available_slots() > 0 {
            receiver_open = self.try_admit_available(&request_rx, &mut active);  // (2)
        }

        if !active.is_empty() {
            self.decode_active_round(&mut active);        // (3)
            self.prefill_active_round(&mut active);       // (4)
        }
    }
}
```

Four behaviors, each worth a sentence:

**(1) Idle blocks, busy never blocks.** When `active` is empty the thread
parks in `recv()` — zero CPU burn while idle. The instant there is work, it
never blocks again, because blocking would stall tokens for every active
request.

**(2) Opportunistic admission.** `try_admit_available` drains the queue
non-blockingly while slots exist:

```154:159:src/scheduler.rs
while self.engine.available_slots() > 0 {
    match request_rx.try_recv() {
        Ok(request) => self.admit_request(request, active),
        Err(TryRecvError::Empty) => return true,
        Err(TryRecvError::Disconnected) => return false,
    }
}
```

The return value *is* `receiver_open`. Notice the shutdown semantics in the
`while` condition of `run`: once the sender is dropped, `receiver_open` goes
false, but the loop keeps running until `active` drains. Requests already
admitted always get finished.

**(3) then (4) — decode before prefill.** This ordering is the core latency
decision. Decoding requests are already streaming to a user who is watching
tokens appear; prefilling requests are waiting on a spinner. Decode first
means an incoming 4k-token prompt cannot stall everyone else's stream by a
full prefill.

This is *continuous batching*: within one tick, some requests are prefilling
and some are decoding, and they progress independently. Contrast with static
batching, where a batch is formed, run to completion, and only then does the
next batch start (head-of-line blocking, awful tail latency).

### B.1 Admission, and the one failure that is loud

```107:113:src/scheduler.rs
let Some(slot) = self.engine.allocate_slot() else {
    let _ = request.response_tx.blocking_send(StreamEvent::Error {
        message: "KV cache pool is full".to_string(),
    });
    self.finish_request(&request, RequestFinish::new("error_kv_pool_full", 0));
    return;
};
```

There is no queueing beyond the channel: if no slot is free, the request is
rejected immediately with an error. Backpressure is explicit rather than
unbounded latency. `admit_request` then re-checks timeout and cancellation
*after* allocating (and releases the slot if either fires) — a request can sit
in the channel long enough to expire before it is ever admitted.

---

## Part C — `decode_active_round`: batch, forward, reap

Three stages, always in this order.

### C.1 Stage 1 — prepare (CPU only)

```171:193:src/scheduler.rs
while index < round_len && index < active.len() {
    if active[index].phase != ActivePhase::Decoding { index += 1; continue; }

    match prepare_decode_token(&mut active[index]) {
        DecodePreparation::Forward(next_token) => {
            decode_batch.push(PreparedDecode {
                active_index: index,
                input: DecodeInput { slot: active[index].slot, token_id: next_token },
            });
        }
        DecodePreparation::Finish(finish) => {
            finished.push(FinishedRequest::done(index, finish));
        }
    }
    index += 1;
}
```

`round_len` is captured **before** the loop. Combined with the
`index < active.len()` guard this makes the iteration robust even though the
vector can shrink later in the tick. A subtle but important habit: never
iterate a `Vec` by index across code that may remove elements without
bounding both ends.

This stage does the sampling for the *previous* forward's logits — see
Part D. Nothing touches the GPU yet.

### C.2 Stage 2 — one batched forward

```206:225:src/scheduler.rs
for (prepared, output) in decode_batch.into_iter().zip(self.engine.decode_batch(&inputs)) {
    match output {
        Ok(forward) => {
            let active_request = &mut active[prepared.active_index];
            active_request.logits = forward.logits;
            active_request.decode_compute_latency += forward.latency;
            self.metrics.record_decode_compute(forward.latency);
        }
        Err(message) => { finished.push(FinishedRequest::error(...)); }
    }
}
```

One call: `engine.decode_batch(&inputs)`, N rows in, N logit vectors out.
Results are matched back by **position** in the zip — the engine must return
outputs in input order, and a per-row error does not sink the batch.

Weights are read once for the whole batch, so batch-2 decode costs far less
than 2× batch-1 (see Ch 13 and the MTP verify numbers in `AGENTS.md`).

### C.3 Stage 3 — reap, in reverse

```227:246:src/scheduler.rs
finished.sort_by_key(|finish| finish.active_index);
finished.dedup_by_key(|finish| finish.active_index);
for finish in finished.into_iter().rev() {
    let active_request = active.swap_remove(finish.active_index);
    // ... send Error/Done, record metrics ...
    let _ = self.engine.release_slot(active_request.slot);
}
```

Read this pattern until it is automatic, because it appears twice (decode and
prefill) and it encodes three separate hazards:

1. **`sort` + `dedup`** — a request can be added to `finished` by both the
   prepare stage and the forward stage in the same tick. Removing the same
   index twice would evict an innocent request.
2. **`.rev()`** — remove from the back forward so earlier indices stay valid.
3. **`swap_remove`** is O(1) but *reorders* `active` by moving the last
   element into the hole. That is fine here because nothing depends on
   `active` order — except `next_prefill_index`, which is why the prefill
   round-robin cursor is only an approximation of fairness, not a guarantee.

And always: `release_slot`. A leaked slot permanently shrinks capacity, and
the symptom is "server gets slower and then rejects everything," far from
the cause.

---

## Part D — `prepare_decode_token`: where the token is actually chosen

This function is the whole sampling + stopping policy. It reads
`active.logits` (produced by the *previous* tick's forward) and returns
either the next token or a finish.

Checks happen in a specific order:

```text
1. timeout?                      → Done{"timeout"}
2. cancelled?                    → Done{reason}
3. completion_tokens >= max?     → Done{"stop"}
4. sample from logits
5. re-sample loop (EOS / first-token blocklist)
6. sampled EOS?                  → Done{"stop"}
7. count it, push to generated_tokens
8. send StreamEvent::Token       → send failure means client vanished
9. cancelled again?              → Done{reason}
10. hit max after increment?     → Done{"stop"}
11. Forward(next_token)
```

Steps 2 and 9 both check cancellation — before and after emitting. That is
not redundancy for its own sake: the token send is the point where a
disconnected client becomes observable, and a `/cancel` may have landed
during it.

### D.1 The re-sampling guard

```631:659:src/scheduler.rs
let mut guard = 0;
loop {
    let block_eos = active.completion_tokens < active.request.params.min_decode_tokens
        && active.request.params.eos_token_ids.contains(&next_token);
    let block_first =
        active.completion_tokens == 0 && FIRST_TOKEN_BLOCKLIST.contains(&next_token);
    if (!block_eos && !block_first) || guard >= 64 { break; }

    let mut masked_logits = active.logits.clone();
    if block_first {
        for &blocked in FIRST_TOKEN_BLOCKLIST { masked_logits[blocked] = f32::NEG_INFINITY; }
    }
    for eos in &active.request.params.eos_token_ids { masked_logits[*eos] = f32::NEG_INFINITY; }
    next_token = sampling::sample_with_params(&masked_logits, &sampling_params, &active.generated_tokens);
    guard += 1;
}
```

Two policies, both fixing real Gemma4 behaviors:

- **`min_decode_tokens`** — the model sometimes emits `<eos>` immediately.
  Masking EOS to `−∞` and re-sampling forces at least N tokens of content.
- **`FIRST_TOKEN_BLOCKLIST = [1, 100, 101, 105, 106, 107]`** — `<eos>`,
  channel markers, turn markers, and a bare newline. These are legal
  mid-stream but produce an empty or malformed first chunk if they lead the
  answer. The comment in the source names each id; keep it in sync if the
  tokenizer changes.

`guard >= 64` is the escape hatch. With temperature 0 and a degenerate
distribution, masking may not change the argmax outcome in a useful way;
without the guard this is an infinite loop that hangs the **entire server**,
because this is the one thread. Whenever you add a re-sample rule to a
single-threaded scheduler, add its bound in the same commit.

---

## Part E — `prefill_active_round` and fair water-filling

### E.1 Why chunk prompts at all

A 32k prompt cannot be one forward — activation scratch and attention cost
scale with chunk length, and a multi-second GPU submission blocks every
other request's decode. So prompts are split into chunks of at most
`engine.max_prefill_chunk_tokens()`.

Per tick there is also a **global** budget:

```349:356:src/scheduler.rs
fn max_prefill_tokens_per_tick(&self) -> usize {
    let max_chunk_tokens = self.engine.max_prefill_chunk_tokens();
    self.config.max_prefill_tokens_per_tick
        .unwrap_or(max_chunk_tokens)
        .min(max_chunk_tokens)
        .max(1)
}
```

Default: one chunk's worth of tokens across *all* prefilling requests per
tick. That caps how long one tick can take, which caps decode jitter.

### E.2 The allocator

```512:545:src/scheduler.rs
while remaining_budget > 0 {
    let eligible_count = candidates.iter().zip(&allocations)
        .filter(|((_, remaining), allocated)| **allocated < *remaining)
        .count();
    if eligible_count == 0 { break; }

    let quantum = (remaining_budget / eligible_count).max(1);
    let mut made_progress = false;
    for ((_, remaining), allocated) in candidates.iter().zip(&mut allocations) {
        if *allocated >= *remaining { continue; }
        let take = (*remaining - *allocated).min(quantum).min(remaining_budget);
        if take == 0 { continue; }
        *allocated += take;
        remaining_budget -= take;
        made_progress = true;
        if remaining_budget == 0 { break; }
    }
    if !made_progress { break; }
}
```

This is **water-filling**: give everyone an equal share; whoever cannot use
their share returns it to the pool; repeat. It is the same algorithm as
max-min fair bandwidth allocation.

Worked trace — budget 512, three prefilling requests needing 100, 1000, 50
(each already clamped to `max_chunk_tokens`):

```text
round 1: eligible = 3, quantum = 512/3 = 170
         A: min(100−0, 170, 512)  = 100 → A=100, budget=412
         B: min(1000−0, 170, 412) = 170 → B=170, budget=242
         C: min(50−0, 170, 242)   =  50 → C=50,  budget=192
round 2: eligible = 1 (only B), quantum = 192
         B: min(1000−170, 192, 192) = 192 → B=362, budget=0
result: A=100 (done), B=362 (continues next tick), C=50 (done)
```

Total exactly 512. Small requests finish this tick instead of being starved
behind the big one, and the big one still gets the leftovers rather than the
budget being wasted.

Two termination conditions matter:

- `quantum = (remaining/eligible).max(1)` — without `.max(1)`, a budget
  smaller than the eligible count yields quantum 0 and nobody progresses.
- `made_progress` — belt-and-braces against any future edit that could make
  every `take` zero. Again: infinite loop here = dead server.

`start_index` rotates via `self.next_prefill_index`, so the *iteration order*
of candidates rotates across ticks. With `swap_remove` shuffling `active`,
treat this as best-effort anti-starvation, not strict round-robin. The unit
tests at the bottom of the file pin the intended behavior:

```text
prefill_plan_spreads_total_tokens_per_tick
prefill_plan_redistributes_short_chunks_until_budget_is_spent
prefill_plan_round_robins_from_previous_cursor
prefill_plan_gives_single_prefill_request_full_budget
```

Read those tests — they are the cheapest available spec of the algorithm.

### E.3 Chunk preparation and `want_logits`

```585:591:src/scheduler.rs
let chunk_start = active.prefill_cursor;
let chunk_end = (chunk_start + chunk_size).min(active.request.input_ids.len());
PrefillPreparation::Forward(PrefillInput {
    slot: active.slot,
    token_ids: active.request.input_ids[chunk_start..chunk_end].to_vec(),
    want_logits: chunk_end >= active.request.input_ids.len(),
})
```

`want_logits` is only true for the **final** chunk. Intermediate chunks exist
to populate the KV cache; their logits are garbage-to-be-discarded, so the
engine skips the `lm_head` matvec — a ~440 MB weight read against a 262k
vocab. On a 4-chunk prompt that removes three of them.

### E.4 Post-forward bookkeeping

```302:315:src/scheduler.rs
Ok(forward) => {
    let active_request = &mut active[prepared.active_index];
    active_request.logits = forward.logits;
    active_request.prefill_cursor += prepared.token_count;
    active_request.prefill_chunks_done += 1;
    active_request.prefill_latency += forward.latency;
    self.metrics.record_prefill_chunk(prepared.token_count, forward.latency);

    if active_request.prefill_cursor >= active_request.request.input_ids.len() {
        active_request.phase = ActivePhase::Decoding;
        active_request.decode_started_at = Some(Instant::now());
        self.metrics.record_prefill_to_decode();
    }
}
```

The phase flip is exactly one comparison. Because prefill runs *after*
decode in the tick, a request that finishes prefilling here starts decoding
on the **next** tick — that one-tick delay is the TTFT floor and it is
intentional.

---

## Part F — Cancellation: one atomic, three meanings

```698:704:src/scheduler.rs
pub fn cancellation_finish_reason(request: &InferenceRequest) -> Option<&'static str> {
    match request.cancel.load(Ordering::Relaxed) {
        CANCEL_STOP => Some("stop"),
        CANCEL_CLIENT => Some("cancelled"),
        _ => None,
    }
}
```

`Arc<AtomicU8>`: the HTTP side writes, the scheduler reads. No mutex, no
channel, no allocation. `Ordering::Relaxed` is sufficient — the only thing
being communicated is the u8 itself; there is no dependent data whose
visibility must be ordered with it.

Who writes it, and why (all in `server.rs`, inside the streaming/sync task):

| Trigger | Value | Reason | Set at |
|---|---|---|---|
| A `stop` sequence appears in decoded text | `CANCEL_STOP` | `"stop"` | `find_generation_stop_position(...)` is `Some` |
| Tool call resolved early in tool-generation mode | `CANCEL_STOP` | `"stop"` | after `resolve_tool_calls` returns non-empty |
| SSE send fails (client vanished) | `CANCEL_CLIENT` | `"cancelled"` | `tx.send(...).is_err()` |
| `created_at.elapsed() >= request_timeout` | — | `"timeout"` | checked by the scheduler itself |

Note there is no cancel endpoint: `create_router` exposes only `/health`,
`/metrics`, `/v1/models`, `/models`, and `/v1/chat/completions`. Cancellation
is always a consequence of something the HTTP task observes in the token
stream, or of the client going away.

Why an atomic instead of just closing the channel: **stop sequences are
detected on decoded text, not on token ids.** A stop string can span several
tokens, so only the HTTP side (which owns the incremental detokenizer) can
recognize it. It needs a way to tell the scheduler "stop now, and call it a
normal stop" — that is `CANCEL_STOP`.

There is also an implicit path: `response_tx.blocking_send` returning `Err`
means the receiver is gone, and `prepare_decode_token` step 8 treats that as
`"cancelled"` without waiting for the atomic.

---

## Part G — What this design buys and what it costs

**Buys:**

- No locks anywhere on the hot path. No lock ordering to reason about, no
  contention, no priority inversion.
- Deterministic tick structure — easy to profile and to reason about
  latency bounds.
- Batching falls out naturally: whatever is `Decoding` this tick is the
  batch.

**Costs:**

- **Any hang is total.** An unbounded loop, a blocking call, or a GPU wait
  that never returns takes down every request. Hence the `guard >= 64` and
  `made_progress` bounds.
- CPU-side sampling for all rows is serial within the tick. At large batch
  and 262k vocab, that becomes measurable.
- No priority classes and no queue: a full pool rejects rather than waits.

---

## Part H — Exercises

1. `active` has 3 decoding and 2 prefilling requests, budget 512, chunk max
   512, remaining prompts 700 and 40. Write out the full tick: what does
   `decode_batch` receive, what does `plan_prefill_round` allocate?

2. Remove `.rev()` from the reap loop. Construct the exact sequence of two
   finished indices that then evicts a live request. Which one survives?

3. Remove `dedup_by_key`. Describe the double-remove scenario concretely
   (which stage adds the duplicate index, and when).

4. Set `min_decode_tokens = 5` and delete the `guard >= 64` check. Describe
   the failure and why it affects other requests too.

5. Why is `want_logits` false for intermediate chunks safe? What is the one
   thing the engine must still do for those chunks?

6. Trace water-filling for budget 100 with four requests needing
   10/10/10/1000. How many rounds? Final allocation?

7. The prefill cursor rotates, but `swap_remove` reorders `active`.
   Construct a case where a request is skipped for several consecutive ticks.

---

## Checklist

- [ ] I can draw the two-channel architecture and say why the types differ.
- [ ] I can explain why decode runs before prefill in every tick.
- [ ] I can write the three stages of `decode_active_round` from memory.
- [ ] I can justify `sort`, `dedup`, `.rev()`, and `swap_remove` individually.
- [ ] I can hand-run water-filling on a new example and get the exact split.
- [ ] I know all three cancellation paths and their reason strings.
- [ ] I can name every unbounded-loop guard in the file and why it exists.

**Next:** [13_kv_pool.md](13_kv_pool.md) — the slot allocator and
`BatchEngine` that this chapter calls into.
