"""Agent with MCP web search + Graphiti memory, powered by NVIDIA NIM.

Run order:
  1. docker compose up -d            (Neo4j for memory)
  2. python server.py                (Linkup web-search MCP, port 8080)
  3. python memory_server.py         (Graphiti memory MCP, port 8000)
  4. python agent.py
"""
import os
import sys

# CrewAI's verbose formatter prints emoji through a Rich console that resolves
# sys.stdout lazily. On Windows sys.stdout defaults to the ANSI code page
# (cp1252), so every panel raises UnicodeEncodeError and the events bus swallows
# it -- leaving verbose=True output completely empty. Force UTF-8 first.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Crew, Task, LLM
from crewai_tools import MCPServerAdapter

# --- NVIDIA NIM as the crew brain (replaces ollama/llama3.2) ---
llm = LLM(
    model=f"openai/{os.getenv('MODEL_NAME', 'nvidia/nemotron-3-super-120b-a12b')}",
    base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.0,
)

# --- Opik tracing (optional, off by default) ---
if os.getenv("OPIK_ENABLED", "false").lower() == "true":
    import opik
    from opik.integrations.crewai import track_crewai

    opik.configure(use_local=False)
    track_crewai(project_name="crewai-nvidia-mcp-memory")

# --- Both MCP servers: Graphiti memory (8000) + Linkup search (8080) ---
server_params = [
    {"url": "http://localhost:8000/sse", "transport": "sse"},
    {"url": "http://localhost:8080/sse", "transport": "sse"},
]

QUERY = "What is happening in FIFA club world cup 2025?"

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    # Assistant Agent — searches the web when needed
    assistant_agent = Agent(
        role="Helpful Assistant",
        goal="Provide accurate and helpful responses to user queries",
        backstory="I am a helpful AI assistant that can search the web when needed to provide accurate information.",
        allow_delegation=False,
        tools=[mcp_tools["web_search"]],
        llm=llm,
    )

    # Memory Manager Agent — stores conversation context in Graphiti
    memory_manager = Agent(
        role="Memory Manager",
        goal="Store and manage conversation history",
        backstory="I am responsible for maintaining the conversation memory by storing relevant information.",
        allow_delegation=False,
        tools=[mcp_tools["add_memory"]],
        llm=llm,
    )

    # Response Generator Agent — answers using memory context
    response_generator = Agent(
        role="Response Generator",
        goal="Generate coherent responses using memory context",
        backstory="I analyze memory nodes to generate contextually relevant responses.",
        allow_delegation=False,
        tools=[mcp_tools["search_memory_nodes"]],
        llm=llm,
    )

    assistant_task = Task(
        description="Process the user query '{query}' and provide a helpful response. Search the web if needed.",
        agent=assistant_agent,
        expected_output="A detailed response addressing the user's query",
    )

    memory_task = Task(
        description="Store the query '{query}' and the key findings in memory for future reference.",
        agent=memory_manager,
        expected_output="Confirmation of memory storage",
    )

    response_gen_task = Task(
        description="Generate a response for '{query}' using relevant information from memory.",
        agent=response_generator,
        expected_output="A coherent response incorporating context from memory",
    )

    crew = Crew(
        agents=[assistant_agent, memory_manager, response_generator],
        tasks=[assistant_task, memory_task, response_gen_task],
        verbose=True,
    )

    result = crew.kickoff(inputs={"query": QUERY})
    print(result)
