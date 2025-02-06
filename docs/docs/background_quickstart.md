---
title: Reflection Quickstart Guide
description: Get started processing memories "in the background".
---

# Reflection Quickstart Guide

Memories can be created in two ways:

1. In the hot path: the agent consciously saves notes using tools (see [Hot path quickstart](hot_path_quickstart.md)).
2. In the background: memories are "subconsciously" extracted automatically from conversations.

This guide shows you how to extract and consolidate memories in the background using [`create_memory_store_enricher`](). The agent will continue as normal while memories are processed in the background.

# Prerequisites

First, install LangMem:

```bash
pip install -U langmem
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

## Basic Usage

```python
from langmem import create_memory_store_enricher, ReflectionExecutor
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

# Create enricher to extract memories from conversations
enricher = create_memory_store_enricher(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories",)  # (1)
)

store = InMemoryStore() # (4)
# Create agent (no memory tools needed)
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    store=store
)

# Process memories in background thread (2)
# highlight-next-line
executor = ReflectionExecutor(enricher) 

# Run conversation as normal
response = agent.invoke(
    {"messages": [{"role": "user", "content": "I prefer dark mode"}]},
)

# Schedule memory extraction in background
# highlight-next-line
executor.submit({"messages": response["messages"]})  # (3)
```

1. The `namespace` parameter lets you isolate memories that are stored and retrieved. In this example, we store memories in a global "memories" path, but you could instead use template variables to scope to a user-specific path based on configuration. See [how to dynamically configure namespaces](guides/dynamically_configure_namespaces.md) for more information.

2. The [`ReflectionExecutor`](reference/utils.md#langmem.ReflectionExecutor) schedules memories either in a background thread (as we are doing in this case), or remotely via a LangGraph Platform instance. This reduces impact on the main conversation.

    A common concern with background processing is that it could get expensive if you're running it on every interaction turn. You often don't know when a conversatino completes, so a common pattern is to schedule processing of memories for some time in the future. If a new input arrives for a particular thread before memories are processed, you can just cancel and reschedule it.

    The `ReflectionExecutor` offers an `after_seconds` argument for this purpose. Note that for this type of use case, the **local** thread version wouldn't be useful if you're trying to deploy using serverless functions, since the thread would be terminated.
    Check out the [`ReflectionExecutor`](reference/utils.md#langmem.ReflectionExecutor) reference docs for more information.


3. You can also process memories directly with `enricher.process(messages)` if you don't need background processing

4. What's a store? It's a document store you can add vector-search too. The "InMemoryStore" is, as it says, saved in-memory and not persistent.

    ???+ tip "For Production"
        Use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore) instead of `InMemoryStore` to persist data between restarts.


## What's Next

- [Configure Dynamic Namespaces](guides/dynamically_configure_namespaces.md) - Learn more ways to organize memories by user, agent, or other values.