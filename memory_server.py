"""Graphiti memory MCP server (SSE on port 8000), powered by NVIDIA NIM.

Exposes the same three tools the original project used from the getzep/graphiti
MCP server: add_memory, search_memory_nodes, clear_graph.

All LLM + embedding calls go to NVIDIA's OpenAI-compatible endpoint
(https://integrate.api.nvidia.com/v1) using NVIDIA_API_KEY — no OpenAI key.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from mcp.server.fastmcp import FastMCP

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")
SMALL_MODEL = os.getenv("SMALL_MODEL", "nvidia/nemotron-3-super-120b-a12b")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "nvidia/nemotron-3-embed-1b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", "2"))  # NVIDIA free tier is rate-limited

class PassthroughCrossEncoder(CrossEncoderClient):
    """Reranker stub: preserves the engine's RRF ordering (no extra LLM calls)."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0 - i * 1e-6) for i, p in enumerate(passages)]


_bootstrap_done = False


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Bootstrap the graph schema once, inside the server's own event loop.

    Two constraints this has to respect:

    1. The Neo4j async driver pools sockets on the event loop that first uses
       them, so the bootstrap must run in the loop that serves requests.
       Running it via asyncio.run() beforehand closes that loop and every later
       query fails with "'NoneType' object has no attribute 'send'".
    2. FastMCP enters this lifespan *once per SSE connection*. The driver is a
       module-level singleton shared by every connection, so it must never be
       closed here: closing it would leave the process permanently broken for
       all later clients. It stays open for the lifetime of the process (the OS
       reclaims the sockets on exit).
    """
    global _bootstrap_done
    if not _bootstrap_done:
        await graphiti.build_indices_and_constraints()
        _bootstrap_done = True
    yield {}


mcp = FastMCP('graphiti-memory-server', port=8000, lifespan=lifespan)

# --- Graphiti wired to NVIDIA (OpenAI-compatible endpoint) ---
llm_config = LLMConfig(
    api_key=NVIDIA_API_KEY,
    model=MODEL_NAME,
    small_model=SMALL_MODEL,
    base_url=NIM_BASE_URL,
    temperature=0.0,
    max_tokens=4096,
)

embedder = OpenAIEmbedder(
    config=OpenAIEmbedderConfig(
        api_key=NVIDIA_API_KEY,
        base_url=NIM_BASE_URL,
        embedding_model=EMBEDDER_MODEL,
        embedding_dim=EMBEDDING_DIM,
    )
)

graphiti = Graphiti(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "demodemo"),
    llm_client=OpenAIClient(config=llm_config),
    embedder=embedder,
    cross_encoder=PassthroughCrossEncoder(),  # avoid the default OpenAI reranker (needs OPENAI_API_KEY)
    max_coroutines=SEMAPHORE_LIMIT,
)


@mcp.tool()
async def add_memory(episode_text: str, source_description: str = "conversation") -> str:
    """Store information in the knowledge graph memory for future reference."""
    await graphiti.add_episode(
        name=f"episode_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        episode_body=episode_text,
        source=EpisodeType.text,
        source_description=source_description,
        reference_time=datetime.now(timezone.utc),
    )
    return f"Memory stored: {source_description}"


@mcp.tool()
async def search_memory_nodes(query: str) -> str:
    """Search the knowledge graph memory for relevant facts and entities."""
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    results = await graphiti.search_(query, config=EDGE_HYBRID_SEARCH_RRF)
    if not results.edges:
        return "No relevant memories found."
    return "\n".join(f"- {edge.fact}" for edge in results.edges[:10])


@mcp.tool()
async def clear_graph() -> str:
    """Clear all data from the memory graph."""
    await graphiti.build_indices_and_constraints()
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data

    await clear_data(graphiti.driver)
    return "Memory graph cleared."


if __name__ == "__main__":
    mcp.run(transport="sse")
