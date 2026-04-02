"""
Multi-AI Collaboration Hub
Connects Claude, ChatGPT, and Perplexity so they can talk to each other.
"""

import os
import json
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import anthropic
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI(title="Multi-AI Collaboration Hub")
templates = Jinja2Templates(directory="templates")

# ── API clients ─────────────────────────────────────────────────────────────

def get_anthropic_client():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key == "your_anthropic_api_key_here":
        return None
    return anthropic.AsyncAnthropic(api_key=key)

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your_openai_api_key_here":
        return None
    return AsyncOpenAI(api_key=key)

def get_perplexity_client():
    key = os.getenv("PERPLEXITY_API_KEY", "")
    if not key or key == "your_perplexity_api_key_here":
        return None
    return AsyncOpenAI(api_key=key, base_url="https://api.perplexity.ai")


# ── Individual AI callers ────────────────────────────────────────────────────

async def stream_claude(messages: list[dict], system: str = "") -> AsyncGenerator[str, None]:
    client = get_anthropic_client()
    if not client:
        yield "⚠️ Claude API key not configured."
        return
    try:
        kwargs = {
            "model": "claude-opus-4-6",
            "max_tokens": 2048,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n⚠️ Claude error: {e}"


async def stream_chatgpt(messages: list[dict], system: str = "") -> AsyncGenerator[str, None]:
    client = get_openai_client()
    if not client:
        yield "⚠️ ChatGPT API key not configured."
        return
    try:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=oai_messages,
            stream=True,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n⚠️ ChatGPT error: {e}"


async def stream_perplexity(messages: list[dict], system: str = "") -> AsyncGenerator[str, None]:
    client = get_perplexity_client()
    if not client:
        yield "⚠️ Perplexity API key not configured."
        return
    try:
        pplx_messages = []
        if system:
            pplx_messages.append({"role": "system", "content": system})
        pplx_messages.extend(messages)
        stream = await client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=pplx_messages,
            stream=True,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n⚠️ Perplexity error: {e}"


# ── Non-streaming helpers (collect full response) ────────────────────────────

async def call_claude(messages: list[dict], system: str = "") -> str:
    parts = []
    async for chunk in stream_claude(messages, system):
        parts.append(chunk)
    return "".join(parts)


async def call_chatgpt(messages: list[dict], system: str = "") -> str:
    parts = []
    async for chunk in stream_chatgpt(messages, system):
        parts.append(chunk)
    return "".join(parts)


async def call_perplexity(messages: list[dict], system: str = "") -> str:
    parts = []
    async for chunk in stream_perplexity(messages, system):
        parts.append(chunk)
    return "".join(parts)


# ── SSE helpers ──────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Mode: Parallel ───────────────────────────────────────────────────────────
# All three AIs answer the same question simultaneously.

async def mode_parallel(prompt: str, enabled: dict) -> AsyncGenerator[str, None]:
    yield sse("status", {"msg": "All AIs answering simultaneously..."})

    user_msg = [{"role": "user", "content": prompt}]

    # Launch all enabled AIs concurrently
    tasks = {}
    if enabled.get("claude"):
        tasks["claude"] = asyncio.create_task(collect_with_events("claude", stream_claude(user_msg)))
    if enabled.get("chatgpt"):
        tasks["chatgpt"] = asyncio.create_task(collect_with_events("chatgpt", stream_chatgpt(user_msg)))
    if enabled.get("perplexity"):
        tasks["perplexity"] = asyncio.create_task(collect_with_events("perplexity", stream_perplexity(user_msg)))

    # Stream token-by-token from a shared queue
    queue: asyncio.Queue = asyncio.Queue()

    async def feed(ai_name: str, gen: AsyncGenerator):
        async for token in gen:
            await queue.put(sse("token", {"ai": ai_name, "text": token}))
        await queue.put(sse("done_ai", {"ai": ai_name}))

    feeders = []
    if enabled.get("claude"):
        feeders.append(asyncio.create_task(feed("claude", stream_claude(user_msg))))
    if enabled.get("chatgpt"):
        feeders.append(asyncio.create_task(feed("chatgpt", stream_chatgpt(user_msg))))
    if enabled.get("perplexity"):
        feeders.append(asyncio.create_task(feed("perplexity", stream_perplexity(user_msg))))

    finished = 0
    total = len(feeders)
    while finished < total:
        item = await queue.get()
        yield item
        if '"done_ai"' in item or "done_ai" in item:
            finished += 1

    yield sse("complete", {"msg": "All AIs have answered."})


async def collect_with_events(name: str, gen: AsyncGenerator) -> str:
    parts = []
    async for chunk in gen:
        parts.append(chunk)
    return "".join(parts)


# ── Mode: Discussion ─────────────────────────────────────────────────────────
# AIs take turns. Each sees what the previous AI said and builds on it.

AI_NAMES = {"claude": "Claude", "chatgpt": "ChatGPT", "perplexity": "Perplexity"}

async def mode_discussion(prompt: str, enabled: dict, rounds: int = 1) -> AsyncGenerator[str, None]:
    active = [ai for ai in ["claude", "chatgpt", "perplexity"] if enabled.get(ai)]
    if not active:
        yield sse("error", {"msg": "No AIs enabled."})
        return

    yield sse("status", {"msg": f"Starting discussion with {len(active)} AIs for {rounds} round(s)..."})

    # Conversation history shared across all AIs (as a neutral transcript)
    transcript: list[str] = []
    user_msg = [{"role": "user", "content": prompt}]

    for round_num in range(1, rounds + 1):
        yield sse("round", {"round": round_num})
        for ai in active:
            yield sse("status", {"msg": f"Round {round_num}: {AI_NAMES[ai]} is responding..."})

            # Build context: original question + transcript so far
            if transcript:
                context = (
                    f"The user asked: {prompt}\n\n"
                    f"Here is the discussion so far:\n"
                    + "\n\n".join(transcript)
                    + f"\n\nNow it is your turn ({AI_NAMES[ai]}). "
                    "Read what the other AIs said, add your own perspective, "
                    "agree or disagree with specific points, and advance the discussion."
                )
                messages = [{"role": "user", "content": context}]
            else:
                messages = user_msg

            response_parts: list[str] = []
            stream_fn = {"claude": stream_claude, "chatgpt": stream_chatgpt, "perplexity": stream_perplexity}[ai]

            async for token in stream_fn(messages):
                yield sse("token", {"ai": ai, "text": token})
                response_parts.append(token)

            full_response = "".join(response_parts)
            transcript.append(f"**{AI_NAMES[ai]}**: {full_response}")
            yield sse("done_ai", {"ai": ai})

    yield sse("complete", {"msg": "Discussion complete.", "transcript": "\n\n".join(transcript)})


# ── Mode: Group Work ─────────────────────────────────────────────────────────
# Claude acts as coordinator, breaks the task into sub-tasks, assigns them,
# then synthesizes all results into a final answer.

async def mode_group_work(prompt: str, enabled: dict) -> AsyncGenerator[str, None]:
    active = [ai for ai in ["claude", "chatgpt", "perplexity"] if enabled.get(ai)]
    if not active:
        yield sse("error", {"msg": "No AIs enabled."})
        return

    if "claude" not in active:
        yield sse("status", {"msg": "Note: Claude is recommended as coordinator. Using first available AI."})
        coordinator = active[0]
    else:
        coordinator = "claude"

    workers = [ai for ai in active if ai != coordinator]

    yield sse("status", {"msg": f"Coordinator ({AI_NAMES[coordinator]}) is breaking down the task..."})

    # Step 1: Coordinator breaks down the task
    worker_names = ", ".join(AI_NAMES[w] for w in workers) if workers else "itself"
    breakdown_prompt = (
        f"You are coordinating a team of AI assistants to complete this task:\n\n{prompt}\n\n"
        f"Your team members are: {worker_names}.\n"
        f"Break this task into {max(len(workers), 1)} clear sub-tasks — one per team member. "
        "Output ONLY a numbered list like:\n"
        "1. [Sub-task for first AI]\n"
        "2. [Sub-task for second AI]\n"
        "Be specific and actionable. Each sub-task should be self-contained."
    )

    breakdown_parts: list[str] = []
    stream_fn = {"claude": stream_claude, "chatgpt": stream_chatgpt, "perplexity": stream_perplexity}[coordinator]
    async for token in stream_fn([{"role": "user", "content": breakdown_prompt}]):
        yield sse("token", {"ai": coordinator, "text": token, "phase": "breakdown"})
        breakdown_parts.append(token)

    breakdown = "".join(breakdown_parts)
    yield sse("done_ai", {"ai": coordinator, "phase": "breakdown"})
    yield sse("status", {"msg": "Sub-tasks assigned. Workers are executing..."})

    # Step 2: Parse sub-tasks and assign to workers
    lines = [l.strip() for l in breakdown.strip().split("\n") if l.strip()]
    subtasks = []
    for line in lines:
        for prefix in ["1.", "2.", "3.", "4.", "5."]:
            if line.startswith(prefix):
                subtasks.append(line[len(prefix):].strip())
                break

    if not subtasks:
        subtasks = [breakdown]  # fallback: give the whole thing to each worker

    # Assign sub-tasks to workers (cycle if more subtasks than workers)
    results: dict[str, str] = {}

    async def run_worker(ai: str, subtask: str):
        parts: list[str] = []
        wfn = {"claude": stream_claude, "chatgpt": stream_chatgpt, "perplexity": stream_perplexity}[ai]
        async for token in wfn([{"role": "user", "content": subtask}]):
            await _worker_queue.put(sse("token", {"ai": ai, "text": token, "phase": "work"}))
            parts.append(token)
        results[ai] = "".join(parts)
        await _worker_queue.put(sse("done_ai", {"ai": ai, "phase": "work"}))

    _worker_queue: asyncio.Queue = asyncio.Queue()

    worker_tasks = []
    for i, subtask in enumerate(subtasks):
        if workers:
            ai = workers[i % len(workers)]
        else:
            ai = coordinator
        worker_tasks.append(asyncio.create_task(run_worker(ai, subtask)))

    finished = 0
    total = len(worker_tasks)
    while finished < total:
        item = await _worker_queue.get()
        yield item
        if "done_ai" in item and '"work"' in item:
            finished += 1

    await asyncio.gather(*worker_tasks)

    # Step 3: Coordinator synthesizes all results
    yield sse("status", {"msg": f"Coordinator ({AI_NAMES[coordinator]}) is synthesizing results..."})

    worker_outputs = "\n\n".join(
        f"**{AI_NAMES[ai]}** worked on: _{subtasks[i % len(subtasks)]}_\n{results.get(ai, 'No output.')}"
        for i, ai in enumerate(workers or [coordinator])
    )

    synthesis_prompt = (
        f"Original task: {prompt}\n\n"
        f"Your team produced the following results:\n\n{worker_outputs}\n\n"
        "Synthesize all of this into one comprehensive, well-organized final answer. "
        "Combine the best insights from each result. Fix any contradictions. "
        "Format your answer clearly for the user."
    )

    synth_parts: list[str] = []
    async for token in stream_fn([{"role": "user", "content": synthesis_prompt}]):
        yield sse("token", {"ai": coordinator, "text": token, "phase": "synthesis"})
        synth_parts.append(token)

    yield sse("done_ai", {"ai": coordinator, "phase": "synthesis"})
    yield sse("complete", {"msg": "Group work complete!", "synthesis": "".join(synth_parts)})


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    prompt: str = body.get("prompt", "").strip()
    mode: str = body.get("mode", "parallel")
    enabled: dict = body.get("enabled", {"claude": True, "chatgpt": True, "perplexity": True})
    rounds: int = int(body.get("rounds", 1))

    if not prompt:
        return {"error": "No prompt provided."}

    async def generate():
        if mode == "parallel":
            async for event in mode_parallel(prompt, enabled):
                yield event
        elif mode == "discussion":
            async for event in mode_discussion(prompt, enabled, rounds):
                yield event
        elif mode == "group":
            async for event in mode_group_work(prompt, enabled):
                yield event
        else:
            yield sse("error", {"msg": f"Unknown mode: {mode}"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/status")
async def status():
    return {
        "claude": bool(get_anthropic_client()),
        "chatgpt": bool(get_openai_client()),
        "perplexity": bool(get_perplexity_client()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
