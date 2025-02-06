---
title: Background Quickstart Guide
description: Get started processing memories "in the background".
---

# Background Quickstart Guide

Memories can be created in two ways:

1. In the hot path: the agent consciously saves notes using tools (see [Hot path quickstart](hot_path_quickstart.md)).
2. 👉**In the background (this guide)**: memories are "subconsciously" extracted automatically from conversations.

![Hot Path Quickstart Diagram](concepts/img/hot_path_vs_background.png)

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
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem import ReflectionExecutor, create_memory_store_enricher

# Create memory manager Runnable to extract memories from conversations
memory_manager = create_memory_store_enricher(
    "anthropic:claude-3-5-sonnet-latest",
    # Store memories in the "memories" namespace (aka directory)
    namespace=("memories",),  # (1)
)

# Run memory management in a background thread to avoid slowing down our app (2)
# highlight-next-line
executor = ReflectionExecutor(memory_manager)

store = InMemoryStore()  # (4)
# Create agent (no memory tools needed)
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


@entrypoint(store=store)  # Create a LangGraph workflow
def chat(message: str):
    response = llm.invoke(message)

    # Our `memory_manager` expects a conversation history. We'll provide it in OpenAI's message format.
    # highlight-next-line
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}
    # This calls memory_manager.invoke(to_process) in the background
    # If **new** messages arrive for this conversation before `after_seconds` completes,
    # the management task is canceled and replaced with the new one
    # Typically you'd choose a larger value (like 30 minutes)
    executor.submit(to_process, after_seconds=0.5)  # (3)
    return response.content


# Run conversation as normal
response = chat.invoke(
    "I like dogs. My dog's name is Fido.",
)
print(response)
# Output: That's nice! Dogs make wonderful companions. Fido is a classic dog name. What kind of dog is Fido?
```
1. The `namespace` parameter lets you isolate memories that are stored and retrieved. In this example, we store memories in a global "memories" path, but you could instead use template variables to scope to a user-specific path based on configuration. See [how to dynamically configure namespaces](guides/dynamically_configure_namespaces.md) for more information.

2. The [`ReflectionExecutor`](reference/utils.md#langmem.ReflectionExecutor) schedules memories either in a background thread (as we are doing in this case), or remotely via a LangGraph Platform instance. This reduces impact on the main conversation.

   A common concern with background processing is that it could get expensive if you're running it on every interaction turn. You often don't know when a conversatino completes, so a common pattern is to schedule processing of memories for some time in the future. If a new input arrives for a particular thread before memories are processed, you can just cancel and reschedule it.

   The `ReflectionExecutor` offers an `after_seconds` argument for this purpose. Note that for this type of use case, the **local** thread version wouldn't be useful if you're trying to deploy using serverless functions, since the thread would be terminated.
   Check out the [`ReflectionExecutor`](reference/utils.md#langmem.ReflectionExecutor) reference docs for more information.

3. You can also process memories directly with `memory_manager.process(messages)` if you don't need background processing

4. What's a store? It's a document store you can add vector-search too. The "InMemoryStore" is, as it says, saved in-memory and not persistent.

   ???+ tip "For Production"
   Use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore) instead of `InMemoryStore` to persist data between restarts.


If you want to see what memories have been extracted, you can search the store:

```python
# (in case our memory manager is still running)
executor.shutdown(wait=True)  # block until all tasks are done
print(store.search(("memories",)))
# [
#     Item(
#         namespace=["memories"],
#         key="0145905e-2b78-4675-9a54-4cb13099bd0b",
#         value={"kind": "Memory", "content": {"content": "User likes dogs as pets"}},
#         created_at="2025-02-06T18:54:32.568595+00:00",
#         updated_at="2025-02-06T18:54:32.568596+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="19cc4024-999a-4380-95b1-bb9dddc22d22",
#         value={"kind": "Memory", "content": {"content": "User has a dog named Fido"}},
#         created_at="2025-02-06T18:54:32.568680+00:00",
#         updated_at="2025-02-06T18:54:32.568682+00:00",
#         score=None,
#     ),
# ]
```


## What's Next

- [Configure Dynamic Namespaces](guides/dynamically_configure_namespaces.md) - Learn more ways to organize memories by user, agent, or other values.
