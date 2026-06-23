"""Bridge to the custom MemoryManager (MM) system for the LongMemEval runner.

MM lives in a separate repo (default ~/MemoryManager, override with
MEMORYMANAGER_PATH). Unlike mem0's stateless extract→search pipeline, MM is a
stateful managed-context system. Each benchmark question gets a fresh Agent with
an isolated in-memory LongTermStore.

Two memory modes mirror Agent's native turn structures (REPLY is always the
external harness answerer, not MM's model):

  llm mode — ingest:  PREP(user) → PERSIST(user, assistant) per haystack pair
             query:   PREP(question) → harness answer → PERSIST(question, answer)

  algorithmic mode — ingest:  receive(user) → receive(assistant) per pair
                     query:   receive(question) → harness answer → receive(answer)

Only the surfaced memory differs from the mem0 path — the answerer prompt,
model, and judge stay identical, so the memory system is the sole variable.

Each pair is ingested under its real-world session date (via the scoped
ContextManager.using_source_date), so surfaced blocks carry a created_at the
answerer can date-order — parity with mem0's per-memory timestamps.

LLM-mode ingest is expensive (~2 tool-loop calls per haystack pair).

The LLM backend (OpenRouter, gpt-4o family) and embedder are process-wide
singletons built once and shared across per-question agents.
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
    clock_seconds_per_turn: float = 0.0,
):
    """Build a fresh, isolated MemoryManager Agent for one benchmark question."""
    from agent import Agent, MemoryMode
    from controller import MemoryController
    from ContextManager import ContextManager
    from functions.llm_fns import make_compress_fn, make_merge_fn
    from memory.config import MemoryConfig
    from memory.longterm import LongTermStore
    from memory.novelty import NoveltyMode
    from memory.store import ContextStore

    backend, embedder = _ensure(embedding_model)

    cfg = MemoryConfig(clock_seconds_per_turn=clock_seconds_per_turn)
    store = ContextStore(max_tokens=max_tokens, config=cfg)
    lt = LongTermStore("sqlite:///:memory:")  # isolated per agent; StaticPool = thread-safe
    cm = ContextManager(store, lt, embedding_model=embedder, config=cfg)
    controller = MemoryController(
        cm,
        compress_fn=make_compress_fn(backend, util_model),
        merge_fn=make_merge_fn(backend, cm, util_model),
        config=cfg,
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


def _split_pair(chunk) -> tuple[str, str]:
    """Extract user and assistant content from one haystack pair."""
    if isinstance(chunk, str):
        return chunk, ""
    user = assistant = ""
    for msg in chunk:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    return user, assistant


def _algorithmic_receive(agent, content: str) -> None:
    """One receive() pass — mirrors the memory half of _algorithmic_turn."""
    if not content.strip():
        return
    embedding = agent._controller.embed(content)
    agent._controller.receive(
        content, embedding, agent._novelty_fn(content, embedding)
    )


def _format_blocks(agent) -> list[dict]:
    """Return in-context blocks shaped like mem0 search hits for the harness.

    created_at is the block's source_date (the real-world date of the memory's
    content) as an ISO string, so the answerer prompt can date-group/-order
    memories just like it does for mem0's per-memory timestamps.
    """
    blocks = agent._controller._cm._store.all_blocks()
    out: list[dict] = []
    for b in blocks:
        sd = getattr(b, "source_date", None)
        out.append(
            {
                "memory": b.content,
                "score": float(getattr(b, "novelty_score", 0.0)),
                "created_at": sd.isoformat() if sd else None,
            }
        )
    return out


def mm_ingest(agent, pairs, dates=None) -> None:
    """Ingest haystack pairs using the agent's native mode lifecycle (no REPLY).

    ``dates`` is an optional list parallel to ``pairs`` of datetimes (the real-world
    session date of each pair); when given, each pair's blocks are stamped with that
    source_date via the scoped using_source_date() so the answerer can date-order
    them. None entries (or dates=None) leave blocks undated.
    """
    from agent import MemoryMode

    cm = agent._controller._cm
    for i, chunk in enumerate(pairs):
        user, assistant = _split_pair(chunk)
        if not user.strip() and not assistant.strip():
            continue
        source_date = dates[i] if dates is not None else None
        with cm.using_source_date(source_date):
            if agent._mode == MemoryMode.LLM:
                if user.strip():
                    agent._llm_prep_phase(user)
                if user.strip() or assistant.strip():
                    agent._llm_persist_phase(user, assistant)
            else:
                _algorithmic_receive(agent, user)
                _algorithmic_receive(agent, assistant)


def mm_surface_and_format(agent, question_text: str) -> list[dict]:
    """Surface relevant memory into context, then return blocks for the harness.

    Surfaced blocks carry the source_date stamped at ingest (see mm_ingest), which
    _format_blocks emits as created_at for date-ordering in the answerer prompt.
    """
    from agent import MemoryMode

    if agent._mode == MemoryMode.LLM:
        agent._llm_prep_phase(question_text)
    else:
        _algorithmic_receive(agent, question_text)
    return _format_blocks(agent)


def mm_persist(agent, question_text: str, answer: str) -> None:
    """Persist the Q+A exchange after the harness answer."""
    from agent import MemoryMode

    if agent._mode == MemoryMode.LLM:
        agent._llm_persist_phase(question_text, answer)
    else:
        _algorithmic_receive(agent, answer)
