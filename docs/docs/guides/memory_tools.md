---
title: How to Use Memory Tools
---

# How to Use Memory Tools

LangMem provides tools that let your agent store and search memories in a LangGraph store.


## Basic Usage

Create an agent with memory tools:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up store and memory saver
store = InMemoryStore()

# Create agent with memory tools
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        # {user_id} is filled from config at runtime
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ],
    store=store,
)

# Use the agent
config = {"configurable": {"user_id": "user-1"}}

# Store a preference
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Remember I prefer dark mode"}]},
    config=config,
)

# Search preferences
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my preferences?"}]},
    config=config,
)


agent_a_tools = [
    # Write to agent-specific namespace
    create_manage_memory_tool(namespace=("memories", "team_a", "agent_a")),
    # Read from shared team namespace
    create_search_memory_tool(namespace=("memories", "team_a"))
]



# Agents with different prompts sharing read access
agent_a = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=agent_a_tools,
    store=store,
    prompt="You are a research assistant"
)

# Create tools for agent B with different write space
agent_b_tools = [
    create_manage_memory_tool(namespace=("memories", "team_a", "agent_b")),
    create_search_memory_tool(namespace=("memories", "team_a"))
]
agent_b = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=agent_b_tools,
    store=store,
    prompt="You are a report writer."
)

agent_b.invoke({"messages": [{"role": "user", "content": "Hi"}]})
```


## Shared Storage {#storage}

The store is shared within a given deployment. This lets you do things like create namespaced memories to share data between agents in a team.

```python

agent_a_tools = [
    # Write to agent-specific namespace
    create_manage_memory_tool(namespace=("memories", "team_a", "agent_a")),
    # Read from shared team namespace
    create_search_memory_tool(namespace=("memories", "team_a"))
]



# Agents with different prompts sharing read access
agent_a = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=agent_a_tools,
    store=store,
    prompt="You are a research assistant"
)

# Create tools for agent B with different write space
agent_b_tools = [
    create_manage_memory_tool(namespace=("memories", "team_a", "agent_b")),
    create_search_memory_tool(namespace=("memories", "team_a"))
]
agent_b = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=agent_b_tools,
    store=store,
    prompt="You are a report writer."
)

agent_b.invoke({"messages": [{"role": "user", "content": "Hi"}]})
```

For storage patterns, see [Storage System](../concepts/conceptual_guide.md#storage-system).
