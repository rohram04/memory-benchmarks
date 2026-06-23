# memory-benchmarks — CLAUDE.md

Harness for memory-system benchmarks (LongMemEval, LoCoMo, BEAM). Upstream it
benchmarks **mem0**; this repo has been extended with a **MemoryManager (MM)**
backend so the two can be compared head-to-head on LongMemEval, varying only the
memory system. See the repo `README.md` for the original mem0 usage.

## The comparison goal

Fair LongMemEval head-to-head: **only the memory system differs.** Conversation
chunks, the answerer (`gpt-4o`), the judge (`gpt-4o-mini`), and the embedder
(`text-embedding-3-small`) are identical across both backends. mem0 OSS's defaults
(`gpt-4o-mini` extraction + `text-embedding-3-small`) already match, and MM is
configured to the same models.

**Everything routes through OpenRouter on one key — no OpenAI key.** MM chat, MM
embeddings, the harness answerer/judge, and the mem0 server all hit OpenRouter's
OpenAI-compatible endpoint.

## How the MM integration works

MM lives in a **separate repo** (`~/MemoryManager`, override `MEMORYMANAGER_PATH`)
and is a stateful managed-context system, not a stateless extract→search store. So
the integration is **at the runner's call sites**, not a drop-in `Mem0Client`:

- `benchmarks/common/mm_bridge.py` — builds a fresh, isolated `Agent` per question
  (in-memory `LongTermStore`), and exposes `mm_ingest` / `mm_surface_and_format`
  (PREP) / `mm_persist` (PERSIST). The **harness answerer** generates the answer
  from MM's surfaced blocks (MM's own REPLY phase is bypassed — the fairness crux);
  the **judge is unchanged**.
- `benchmarks/longmemeval/run.py` — `--backend memorymanager` branch at the
  ingest/answer call sites; all mem0 paths untouched. MM yields one managed-context
  window (no top-k cutoffs), so it reports a single cutoff.

MM-side dependencies (the OpenRouter `LLMBackend` and the `memory/embeddings.py`
Embedder abstraction) are on MM's **`main`** branch — run against a `main` checkout.

## All-OpenRouter routing (3 points + env)

| Component | How it routes to OpenRouter |
|---|---|
| MM chat (gpt-4o / gpt-4o-mini) | `make_llm_backend(backend="openrouter")` (reads `OPENROUTER_API_KEY` from MM's `.env`) |
| MM embedder | `--mm-embedding-model openrouter:openai/text-embedding-3-small` (default) → `OpenAIEmbedder(base_url=OpenRouter)` |
| Harness answerer + judge | `LLMClient` (provider `openai`) honors `OPENAI_BASE_URL` + `OPENAI_API_KEY` env |
| mem0 OSS server | `mem0-config.yaml` (`openai_base_url: …openrouter…`) mounted via `docker-compose.yml`; key via `OPENAI_API_KEY` |

`.env` (repo root, gitignored — **create it yourself**, the key comes from MM's `.env`):
```
OPENAI_API_KEY=<your OpenRouter key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=<your OpenRouter key>
```
One-liner to create it from MM's `.env`:
```
K=$(grep -E '^OPENROUTER_API_KEY=' ~/MemoryManager/.env | head -1 | cut -d= -f2-) && \
printf 'OPENAI_API_KEY=%s\nOPENAI_BASE_URL=https://openrouter.ai/api/v1\nOPENROUTER_API_KEY=%s\n' "$K" "$K" > .env
```

## Environment

No `.venv` is committed here; the default `python3` may be too new for some wheels.
**Reuse MM's venv** (Python 3.12, already has every dep + `aiolimiter` added):
`~/MemoryManager/.venv/bin/python`. mem0 is reached over raw HTTP, so no mem0 SDK
is needed. Ensure `~/MemoryManager` is on `main` (has `memory/embeddings.py`).

## Running

```bash
cd ~/memory-benchmarks
PY=~/MemoryManager/.venv/bin/python

# mem0 OSS server (OpenRouter-backed via mem0-config.yaml), at localhost:8888
docker compose up -d

# MemoryManager backend
$PY -m benchmarks.longmemeval.run --backend memorymanager \
  --answerer-model openai/gpt-4o --judge-model openai/gpt-4o-mini --provider openai \
  --mm-max-tokens 8000 --mm-embedding-model openrouter:openai/text-embedding-3-small \
  --per-type 1 --max-workers 2 --project-name mm_smoke

# mem0 OSS backend (same flags)
$PY -m benchmarks.longmemeval.run --backend oss \
  --answerer-model openai/gpt-4o --judge-model openai/gpt-4o-mini --provider openai \
  --per-type 1 --max-workers 4 --project-name mem0_smoke
```

MM fires several gpt-4o calls per question (PREP/PERSIST tool loops, and LLM-mode
ingest is ~2 calls/haystack-pair) → **expensive**. Scale up deliberately:
`--per-type 1` smoke → `--per-type 5` → `--all-questions` (500) only when intended.
Use a lower `--max-workers` for MM than for mem0.

### Key MM flags (`memorymanager` backend)
- `--mm-max-tokens` (default 8000) — MM context-window budget; set near the token
  size of the mem0 cutoff you compare against.
- `--mm-model` / `--mm-util-model` — default `openai/gpt-4o` / `openai/gpt-4o-mini`.
- `--mm-embedding-model` — default `openrouter:openai/text-embedding-3-small`; also
  `openai:<m>`/`text-embedding-3-small` (OpenAI direct) or a sentence-transformers
  name (local, keyless).

## Fairness notes / known caveats
- **Single managed window vs mem0 cutoffs:** MM has no top-k; it reports one cutoff.
  Compare against the mem0 cutoff whose token footprint ≈ `--mm-max-tokens`.
- **Per-memory dates:** MM blocks carry `created_at: None`, so the answer prompt
  can't date-group MM memories (mem0 does) — a temporal-reasoning disadvantage.
  Tracked follow-up in MM (thread session timestamps onto blocks).
- **`--mode answerer` only** for the MM backend (retrieval mode is rejected).
