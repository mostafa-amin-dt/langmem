# LangMem

Build memory systems for your agents with full control over memory extraction, maintenance, and persistence. LangMem provides:

1. Memory extraction from LLM interactions
2. Storage-agnostic memory management
3. Ready-to-deploy LangGraph integration

LangMem sits between raw LLM calls and high-level frameworks. You get type-safe memory management primitives you can customize and use in any application. If you _are_ using LangGraph platform, LangMem can integrate directly with your app's storage layer for less hassle.

## Installation

```bash
pip install -U langmem
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

Here's how to create an agent that actively manages its own long-term memory in just a few lines:

```python
# Import core components (1)
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up storage
store = InMemoryStore() # (2)

# Create an agent with memory capabilities (3)
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence (4)
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)
```

1. The memory tools work in any LangGraph app. Here we use [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.create_react_agent) to run an LLM with tools, but you can add these tools to your existing agents or build [custom memory systems](concepts/conceptual_guide.md#functional-core) without agents.

2. [`InMemoryStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.memory.InMemoryStore) keeps memories in process memory—they'll be lost on restart. For production, use the [AsyncPostgresStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore) or a similar DB-backed store to persist memories across server restarts.

3. The memory tools ([`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) and [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool)) let you control what gets stored. The agent extracts key information from conversations, maintains memory consistency, and knows when to search past interactions. See [Memory Tools](guides/memory_tools.md) for configuration options.

Then use the agent:

```python
# Store a new memory (1)
agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode."}]}
)

# Retrieve the stored memory (2)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my lighting preferences?"}]}
)
print(response["messages"][-1].content)
# Output: "You've told me that you prefer dark mode."
```

1. The agent gets to decide what and when to store the memory. No special commands needed—just chat normally and the agent uses [`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) to store relevant details.

2. The agent maintains context between chats. When you ask about previous interactions, the LLM can invoke [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool) to search for memories with similar content. See [Memory Tools](guides/memory_tools.md) to customize memory storage and retrieval, and see the [agent quickstart](quickstart.md) for a more complete example on how to include memories without the agent having to expliictly search.

The agent can now store important information from conversations, search its memory when relevant, and persist knowledge across conversations.

## Next Steps

For more examples and detailed documentation:

- [Quickstart Guide](quickstart.md) - Get up and running
- [Core Concepts](concepts/conceptual_guide.md#memory-in-llm-applications) - Learn key ideas
- [API Reference](reference/index.md) - Full function documentation
- [Integration Guides](guides/memory_tools.md) - Common patterns and best practices

## Requirements

- Python 3.11+
- Access to a supported LLM provider (Anthropic, OpenAI, etc.)
- Optional: a [LangGraph BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) implementation for persistent storage (for the stateful primitives); if you're deploying on LangGraph Platform, this is included without any additional configuration.
