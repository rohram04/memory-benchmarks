"""Bridge to the custom MemoryManager (MM) system for the LongMemEval runner.

MM lives in a separate repo (default ~/MemoryManager, override with
MEMORYMANAGER_PATH). Unlike mem0's stateless extract→search pipeline, MM is a
stateful managed-context system: we ingest the conversation turn-by-turn, then
at answer time surface relevant memory into its context (PREP), let the SAME
harness answerer generate the answer from those surfaced blocks, and persist the
exchange (PERSIST). Only the surfaced memory differs from the mem0 path — the
answerer prompt + model and the judge stay identical, so the memory system is
the sole variable.

Construction mirrors MM's own eval/agent_server.py wiring. The LLM backend
(OpenRouter, gpt-4o family) and the embedder are process-wide singletons built
once and shared across per-question agents (both are safe for concurrent
inference); each question gets its own Agent with an isolated in-memory
LongTermStore.
"""

from __future__ import annotations

import os
import sys
import threading

_MM_PATH = os.environ.get("MEMORYMANAGER_PATH", os.path.expanduser("~/MemoryManager"))
if _MM_PATH not in sys.path:
    sys.path.insert(0, _MM_PATH)

# MM imports are resolved lazily inside _ensure()/make_mm_agent so that importing
# this module (e.g. for a syntax/import smoke test) does not require API keys or
# pull in heavy deps until an agent is actually built.

_LOCK = threading.Lock()
_BACKEND = None
_EMBEDDER = None


def _ensure(embedding_model: str):
    """Build (once) and return the shared (backend, embedder) singletons."""
    global _BACKEND, _EMBEDDER
    with _LOCK:
        if _BACKEND is None:
            from llm import make_llm_backend  # MM's llm package

            # OpenRouter chat-completions backend (gpt-4o / gpt-4o-mini). Reads
            # OPENROUTER_API_KEY from env / MM's repo-root .env.
            _BACKEND = make_llm_backend(backend="openrouter")
        if _EMBEDDER is None:
            from memory.embeddings import make_embedder  # MM's embedder factory

            # Resolves a spec to the right backend: "text-embedding-3-small" /
            # "openai:<m>" (OpenAI direct), "openrouter:<m>" (via OpenRouter), or
            # a sentence-transformers name (local). Matches mem0 OSS's embedder.
            _EMBEDDER = make_embedder(embedding_model)
    return _BACKEND, _EMBEDDER


def make_mm_agent(
    max_tokens: int,
    model: str = "openai/gpt-4o",
    util_model: str = "openai/gpt-4o-mini",
    embedding_model: str = "openrouter:openai/text-embedding-3-small",
    novelty_mode=None,
    mode: str = "llm",
):
    """Build a fresh, isolated MemoryManager Agent for one benchmark question."""
    from agent import Agent, MemoryMode
    from controller import MemoryController
    from ContextManager import ContextManager
    from functions.llm_fns import make_compress_fn, make_merge_fn
    from memory.longterm import LongTermStore
    from memory.novelty import NoveltyMode
    from memory.store import ContextStore

    backend, embedder = _ensure(embedding_model)

    store = ContextStore(max_tokens=max_tokens)
    lt = LongTermStore("sqlite:///:memory:")  # isolated per agent; StaticPool = thread-safe
    cm = ContextManager(store, lt, embedding_model=embedder)
    controller = MemoryController(
        cm,
        compress_fn=make_compress_fn(backend, util_model),
        merge_fn=make_merge_fn(backend, cm, util_model),
    )
    mm_mode = MemoryMode.LLM if str(mode).lower() == "llm" else MemoryMode.ALGORITHMIC
    return Agent(
        controller,
        backend,
        model=model,
        mode=mm_mode,
        novelty_mode=(novelty_mode or NoveltyMode.EMBEDDING),
        novelty_model=util_model,
    )


def _join_chunk(chunk) -> str:
    """Render one ingestion chunk (a user+assistant pair) as plain text."""
    if isinstance(chunk, str):
        return chunk
    parts = []
    for msg in chunk:
        if isinstance(msg, dict):
            parts.append(f"{msg.get('role', '')}: {msg.get('content', '')}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def mm_ingest(agent, pairs) -> None:
    """Ingest each conversation chunk into MM's memory lifecycle (no LLM reply)."""
    for chunk in pairs:
        text = _join_chunk(chunk)
        if text.strip():
            agent.ingest(text)


def mm_surface_and_format(agent, question_text: str) -> list[dict]:
    """PREP: surface relevant memory into context, then return the in-context
    blocks shaped like the harness's search results ({memory, score, created_at})
    so they feed the unchanged get_answer_generation_prompt.

    created_at is None for now — threading session dates onto blocks is a tracked
    follow-up in MemoryManager (disadvantages MM on temporal-reasoning questions).
    """
    agent._llm_prep_phase(question_text)
    blocks = agent._controller._cm._store.all_blocks()
    return [
        {
            "memory": b.content,
            "score": float(getattr(b, "novelty_score", 0.0)),
            "created_at": None,
        }
        for b in blocks
    ]


def mm_persist(agent, question_text: str, answer: str) -> None:
    """PERSIST: store the exchange and re-score novelty."""
    agent._llm_persist_phase(question_text, answer)
