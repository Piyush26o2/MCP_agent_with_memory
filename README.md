# Agent with MCP Memory — NVIDIA NIM edition

## Tech stack

- **CrewAI** — the 3-agent crew (Web Search → Memory Manager → Response Generator)
- **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`) — the "brain" for both the crew *and* the memory layer
- **Zep Graphiti** — knowledge-graph memory (Neo4j-backed)
- **Linkup** — web search via MCP
- **MCP** — Linkup server on :8080, Graphiti memory server on :8000 (both SSE)
- **Opik** (optional) — observability, enable with `OPIK_ENABLED=true`

## Setup

```bash
uv sync
cp .env.example .env   # then fill in NVIDIA_API_KEY + LINKUP_API_KEY
```

Get keys: [build.nvidia.com](https://build.nvidia.com) (starts with `nvapi-`) · [linkup.so](https://www.linkup.so/)

## Run

```bash
# 1. Start Neo4j (Docker Desktop must be running)
docker compose up -d

# 2. Terminal 1 — Linkup web-search MCP server
python server.py

# 3. Terminal 2 — Graphiti memory MCP server
python memory_server.py

# 4. Terminal 3 — the agent
python agent.py
```

## How it works

1. User query → **Assistant Agent** calls `web_search` (Linkup MCP, :8080)
2. **Memory Manager Agent** stores query + findings via `add_memory` (Graphiti MCP, :8000 → Neo4j)
3. **Response Generator Agent** recalls context via `search_memory_nodes` and writes the final answer

All memory extractions/embeddings run on NVIDIA NIM:
- Chat: `nvidia/nemotron-3-super-120b-a12b` (verified working on your key)
- Embeddings: `nvidia/nemotron-3-embed-1b` (dim 2048, verified working)

> Note: several older models (e.g. `meta/llama-3.3-70b-instruct`) have been retired from
> NVIDIA's API; `.env` already points to live models. `SEMAPHORE_LIMIT` caps concurrent
> Graphiti LLM calls to respect NVIDIA rate limits.

### Extras
- Clear memory: run the `clear_graph` snippet in `demo_cells.py` or call the tool directly.
- Inspect the graph: open http://localhost:7474 (neo4j / demodemo).
