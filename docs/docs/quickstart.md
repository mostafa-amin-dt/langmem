---
title: Quickstart Guide
description: Get started with LangMem
---

# Quickstart Guide

This guide will help you get started with LangMem, showing you how to create an agent with long-term memory.

## Installation & Setup

First, install LangMem:

```bash
pip install -U langmem
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

## Basic Example

Here's a complete example showing how to create an agent with memory that persists across conversations:

``` python hl_lines="14-19 35-38"
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store

from langmem import create_manage_memory_tool, create_search_memory_tool

# Create a store and memory saver
store = InMemoryStore() # (1)
memory = MemorySaver()
```

1. For production deployments, use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore). `InMemoryStore` works fine for development but doesn't persist data between restarts.

```python
def prompt(state):
    store = get_store()
    memories = store.search(
        # Search within the same namespace as the one
        # we've configured for the agent (2)
        namespace=("memories", state["configurable"]["user_id"]),
        query=state["messages"][-1].content,
    )
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant.\n\n" f"## Memories\n\n{memories}",
        },
        *state["messages"],
    ]


# Create an agent with memory tools
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[
        # The {user_id} is configurable at runtime (1)
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ],
    store=store,
    checkpointer=memory,  # Make each conversation stateful
)
```

1.  The `{user_id}` placeholder lets you configure LangGraph's [BaseStore]() namespace at runtime to scope memories per user, agent, organization, etc.

    This is a common pattern in LangMem. See the  [conceptual guide](concepts/conceptual_guide.md#memory-namespaces) for more information. Below are a couple of examples of how you'd provide the values:

    ```python
    # Example 1: Store User A's preference
    response_a1 = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember I like dark mode"}]},
        # Uses namespace ("memories", "user-a")
        config={"configurable": {"thread_id": "thread-1", "user_id": "user-a"}}
    )  
    
    # Example 2: Access User A's preference from another thread
    response_a2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What mode do I prefer?"}]},
        config={"configurable": {"thread_id": "thread-2", "user_id": "user-a"}}
    )  # Same namespace ("memories", "user-a")
    ```
    
    The namespace template `("memories", "{user_id}")` is filled with values from the config dict at runtime. You can use LangGraph's BaseStore namespacing patterns:
    
    ```python
    # Organization-level sharing
    namespace=("memories", "org-123")
    
    # User-specific storage
    namespace=("memories", "user-a")
    
    # Feature partitioning
    namespace=("memories", "user-a", "settings")
    ```

2.  We're searching within all items in namespace `("memories", configurable["user_id"])` here, which is the one we configured for our [memory tools](guides/memory_tools.md).

This agent can now be used to interact with different users. For example:

```python
# Configuration for the conversation thread
thread_a = "thread-a"
user_a = "user-a"
# Our 'user_id' value will determine the namespace. In this case, it will be ("memories", "user-a").
config = {"configurable": {"thread_id": thread_a, "user_id": user_a}}

# Use the agent
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config,
)
print(response["messages"][-1].content)
# Output: "No preference specified."

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "dark. Remember that."}
        ]
    },
    config=config,
)

# New thread.
thread_b = "thread-b"
config = {"configurable": {"thread_id": thread_b, "user_id": user_a}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    config=config,
)
print(response["messages"][-1].content)
# Output: "You've told me that you prefer dark mode."

# New thread and new user
thread_c = "thread-c"
user_c = "user-c"
new_config = {"configurable": {"thread_id": thread_c, "user_id": user_c}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    config=config,
)
print(response["messages"][-1].content)
# Output: "I don't have any information about your preferences yet."
```

This example demonstrates memory persistence across conversations and thread isolation between users. The agent stores the user's dark mode preference in one thread but cannot access it from another thread.

## Storage and Memory

The [`InMemoryStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.PostgresStore.asearch) provides ephemeral storage suitable for development. In production, replace this with a DB-backed [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/stores/#basestore) implementation for persistence. When deploying on the LangGraph platform, a postgres-backed store is automatically provided. This store enables saving and retrieving information from any namespace, letting you scope memories by user, agent, organization, or other arbitrary categories.

The [`MemorySaver`](https://langchain-ai.github.io/langgraph/reference/checkpoints/) checkpointer maintains conversation history within each thread. This "short-term" memory is isolated by `thread_id`, ensuring that conversations remain independent.

## Memory Tools

The tools [`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) and [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool) allow agents to store and retrieve information from their memory. The `namespace` parameter scopes the memories, ensuring that data is kept separate for different users or contexts.

## Next Steps

To learn more about LangMem's features and further configuration, refer to the following guides:

- [Memory Management](guides/memory_tools.md) – Learn how to use LangMem's core memory utilities for reflection and memory management.
- [Persistent Storage](guides/memory_tools.md#storage) – Learn more about the durable storage options.
- [Memory Tools](guides/memory_tools.md) – Learn more about the memory tools used in this quickstart.
