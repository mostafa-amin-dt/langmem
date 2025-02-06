---
title: Core Concepts
---

# Memory in LLM Applications

Memory allows agents to remember important information across conversations. LangMem provides ways to extract meaningful details from chats, store them, and use them to improve future interactions. At its core, each memory operation in LangMem follows the same pattern:

1. Accept conversation(s) and current memory state
2. Prompt an LLM to determine how to expand or consolidate the memory state
3. Respond with the updated memory state

How you represent and manage memories depends on what your agent needs to learn. Read on to learn about the different types of memory and why they are useful.

## Types of Memory

Memory in LLM applications can reflect some of the structure of human memory, with each type serving a distinct purpose in building adaptive, context-aware systems:


| Memory Type | Purpose | Storage Pattern | When to Use |
|-------------|---------|-----------------|-------------|
| Semantic | Facts & Knowledge | Profile or Collection | For storing facts, relationships, and evolving context |
| Episodic | Past Experiences | Collection | For learning from successful examples and adapting responses |
| Procedural | System Behavior | Prompts or Collection | For defining and evolving agent personality and capabilities |

### Semantic Memory: Facts and Knowledge

Semantic memory stores the essential facts and other information that ground an agent's responses. Two common representations of semantic memory are:

#### Profile

Semantic memories can be managed in different ways. For example, memories can be a single, continuously updated "profile" of well-scoped and specific information about a user, organization, or other entity (including the agent itself).

![Collection update process](img/update-list.png)

??? example "Extracting semantic memories as collections"

      ```python
      from langmem import create_memory_enricher
      
      # highlight-next-line
      enricher = create_memory_enricher(
         "anthropic:claude-3-5-sonnet-latest",
         instructions="Extract all noteworthy facts, events, and relationships. Indicate their importance.",
         # highlight-next-line
         enable_inserts=True,
      )

      # Process a conversation to extract semantic memories
      conversation = [
         {"role": "user", "content": "I work at Acme Corp in the ML team"},
         {"role": "assistant", "content": "I'll remember that. What kind of ML work do you do?"},
         {"role": "user", "content": "Mostly NLP and large language models"}
      ]

      memories = enricher.invoke({"messages": conversation})
      # Example memories:
      # [
      #     {"fact": "User works at Acme Corp", "type": "employment", "importance": "high"},
      #     {"fact": "User is in the ML team", "type": "role", "importance": "high"},
      #     {"fact": "User works on NLP and LLMs", "type": "expertise", "importance": "high"}
      # ]
      ```

2. **Profiles** maintain a single document that represents the current state, like a user's active preferences or system settings. When new information arrives, it updates the existing document rather than creating a new one. This approach is ideal when you only care about the latest state, such as a user's current theme preference or language setting.

![Profile update process](img/update-profile.png)

??? example "Managing user preferences with profiles"

      ```python
      from langmem import create_memory_enricher
      from pydantic import BaseModel
      
      class UserProfile(BaseModel):
         preferences: dict[str, str]
         settings: dict[str, str]
      
      enricher = create_memory_enricher(
         "anthropic:claude-3-5-sonnet-latest",
         # highlight-next-line
         schemas=[UserProfile],
         instructions="Extract user preferences and settings",
         # highlight-next-line
         enable_inserts=False,
      )

      # Extract user preferences from a conversation
      conversation = [
         {"role": "user", "content": "I prefer dark mode and using vim keybindings"},
         {"role": "assistant", "content": "I'll set those preferences for you"},
         {"role": "user", "content": "Also set the language to Python"}
      ]

      profile = enricher.invoke({"messages": conversation})
      # Example profile:
      # UserProfile(
      #     preferences={
      #         "theme": "dark",
      #         "editor": "vim",
      #         "language": "python"
      #     },
      #     settings={}
      # )
      ```

Choose between profiles and collections based on how you'll use the data: profiles excel when you need quick access to current state and when you have data requirements about what type of information you can store. They are also easy to present to a user for manual ediing. Collections shine when you want to track knowledge across many interactions without loss of information, and it exceeds when you want to recall certain information contextually rather than every time..

### Episodic Memory: Past Experiences

Episodic memory preserves successful interactions as learning examples that guide future behavior. Unlike semantic memory which stores facts, episodic memory captures the full context of an interaction—the situation, the thought process that led to success, and why that approach worked. These memories help the agent learn from experience, adapting its responses based on what has worked before.

??? example "Defining and extracting episodes"

    ```python
    from pydantic import BaseModel, Field
    from langmem import create_memory_enricher

    class Episode(BaseModel):
        """An episode captures how to handle a specific situation, including the reasoning process
        and what made it successful."""
        
        observation: str = Field(
            ..., 
            description="The situation and relevant context"
        )
        thoughts: str = Field(
            ...,
            description="Key considerations and reasoning process"
        )
        action: str = Field(
            ...,
            description="What was done in response"
        )
        result: str = Field(
            ...,
            description="What happened and why it worked"
        )

    # highlight-next-line
    enricher = create_memory_enricher(
        "anthropic:claude-3-5-sonnet-latest",
        schemas=[Episode],
        instructions="Extract examples of successful interactions. Include the context, thought process, and why the approach worked.",
        enable_inserts=True,
    )

    # Example conversation
    conversation = [
        {"role": "user", "content": "What's a binary tree? I work with family trees if that helps"},
        {"role": "assistant", "content": "A binary tree is like a family tree, but each parent has at most 2 children. Here's a simple example:\n   Bob\n  /  \\\nAmy  Carl\n\nJust like in family trees, we call Bob the 'parent' and Amy and Carl the 'children'."},
        {"role": "user", "content": "Oh that makes sense! So in a binary search tree, would it be like organizing a family by age?"},
    ]

    # Extract episode
    episodes = enricher.invoke({"messages": conversation})
    # Example episode:
    # {
    #     "observation": "User asks about binary trees, mentions familiarity with family trees",
    #     "thoughts": "Can use family tree analogy since user has that background",
    #     "action": "Explained binary trees using family tree analogy with a visual example",
    #     "result": "User understood and extended analogy to binary search trees"
    # }
    ```


### Procedural Memory: System Instructions

Procedural memory encodes how an agent should behave and respond. It starts with system prompts that define core behavior, then evolves through feedback and experience. As the agent interacts with users, it refines these instructions, learning which approaches work best for different situations.

![Instructions update process](img/update-instructions.png)

??? example "Optimizing prompts based on feedback"

      ```python
      from langmem import create_prompt_optimizer

      # highlight-next-line
      optimizer = create_prompt_optimizer(
         "anthropic:claude-3-5-sonnet-latest",
         kind="metaprompt",
         config={"max_reflection_steps": 3}
      )

      # Optimize prompt based on user feedback
      prompt = "You are a helpful assistant."
      trajectory = [
         {"role": "user", "content": "Explain inheritance in Python"},
         {"role": "assistant", "content": "Here's a detailed theoretical explanation..."},
         {"role": "user", "content": "Show me a practical example instead"},
      ]
      optimized = optimizer.invoke({
         "trajectories": [(trajectory, {"user_score": 0})], 
         "prompt": prompt
      })
      ```

## Memory Formation

Memories can form in two ways, each suited for different needs. Active formation happens during conversations, enabling immediate updates when critical context emerges. Background formation occurs between interactions, allowing deeper pattern analysis without impacting response time. This dual approach lets you balance responsiveness with thorough learning.

| Formation Type | Latency Impact | Update Speed | Processing Load | Use Case |
|----------------|----------------|--------------|-----------------|-----------|
| Active | Higher | Immediate | During Response | Critical Context Updates |
| Background | None | Delayed | Between/After Calls | Pattern Analysis, Summaries |

![Hot path vs background memory processing](img/hot_path_vs_background.png)

### Conscious Formation

In active formation, the agent makes real-time decisions about what to remember during a conversation. This approach excels when immediate context is crucial, like capturing user preferences or important details that affect the current interaction.


??? example "Active memory formation"

    ```python
    from langgraph.prebuilt import create_react_agent
    from langmem import create_manage_memory_tool

    # highlight-next-line
    agent = create_react_agent(
        "anthropic:claude-3-5-sonnet-latest",
        tools=[create_manage_memory_tool(namespace=("memories",))],
        store=store
    )

    # Agent processes and stores memory during the conversation
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Remember I prefer dark mode"}]
    })
    ```

### Subconcious Formation

"Subconcious" memory formation refers to the technique of prompting an LLM to reflect on a conversation after it occurs (or after it has been inactive for some period), finding patterns and extracting insights without slowing down the immediate interaction or adding complexity to the agent's tool choice decisions. This approach is perfect for ensuring higher recall of exracted information.


??? example "Background pattern extraction"

    ```python
    from langmem import create_memory_store_enricher

    # highlight-next-line
    enricher = create_memory_store_enricher(
        "anthropic:claude-3-5-sonnet-latest",
        namespace=("memories",),
        instructions="Extract key preferences and facts from conversations"
    )

    enricher.invoke({"messages": conversation_history})
    ```

## Integration Patterns

LangMem's memory utilities are organized in three layers of increasing abstraction, each serving different integration needs:

### 1. Functional Core {#functional-core}

At its heart, LangMem provides functions that transform memory state without side effects. These primitives are the building blocks for memory operations:

- [**Memory Enrichers**](../reference/memory.md#langmem.create_memory_enricher): Extract and structure information from conversations
- [**Prompt Optimizers**](../reference/prompt_optimization.md#langmem.create_prompt_optimizer): Learn and improve system behavior from feedback
- [**Multi-Prompt Optimizers**](../reference/prompt_optimization.md#langmem.create_multi_prompt_optimizer): Coordinate learning across a system composed of multiple LLM steps or agents

These core functions are right for you if you need maximum control and want to integrate memory management into your own persistence layer.

??? example "Without storage"

    ```python
    from langmem import create_memory_enricher

    # highlight-next-line
    enricher = create_memory_enricher(
        "anthropic:claude-3-5-sonnet-latest",
        schemas=[UserProfile],  # Optional: structure your memories
        enable_inserts=True     # Allow creating new memories
    )
    memories = enricher.invoke(messages)
    ```

### 2. Stateful Integration

The next layer adds persistence through LangGraph's storage primitives. These operators automatically handle saving and retrieving memories:

- [**Store Enrichers**](../reference/memory.md#langmem.create_memory_store_enricher): Automatically persist extracted memories
- [**Memory Management Tools**](../reference/tools.md#langmem.create_manage_memory_tool): Give agents direct access to memory operations

Use these when you need persistent memory without managing storage yourself.

??? example "Using stateful operators"

    ```python
    from langmem import create_memory_store_enricher

    # highlight-next-line
    store_enricher = create_memory_store_enricher(
        "anthropic:claude-3-5-sonnet-latest",
        namespace=("memories",)  # Organize memories hierarchically
    )
    # Automatically persists memories to store
    store_enricher.invoke({"messages": messages})
    ```

## Storage System {#storage-system}

??? note "Storage is optional"
    
    Remember that LangMem's core functionality is built around that don't require any specific storage layer. The storage features described here are part of LangMem's higher-level integration with LangGraph, useful when you want built-in persistence.


When using LangMem's stateful operators or platform services, the storage system is built on LangGraph's storage primitives, providing a flexible and powerful way to organize and access memories. The storage system is designed around three key concepts:

### Memory Namespaces {#memory-namespaces}

Memories are organized into namespaces that allows for natural segmentation of data:

- **Multi-Level Namespaces**: Group memories by organization, user, application, or any other hierarchical structure
- **Contextual Keys**: Identify memories uniquely within their namespace
- **Structured Content**: Store rich, structured data with metadata for better organization

??? example "Organizing memories hierarchically"

    ```python
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore() # (1)
    # Organize memories by organization -> user -> context
    namespace = ("acme_corp", "user_123", "code_assistant")
    ```

    1. For production use cases, use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore). `InMemoryStore` is great for testing and development but loses data on restart.
    
    # highlight-next-line
    # Store structured memory with metadata
    memory = {
        "content": "User prefers dark mode",
        "type": "preference",
        "confidence": 0.95,
        "updated_at": "2025-02-04T04:29:22-08:00"
    }
    store.put(namespace, "ui_preferences", memory)
    ```

Namespaces can include template variables (such as `"{user_id}"`) to be populated at runtime from `configurable` fields in the `RunnableConfig`.

```python
    from langmem import create_manage_memory_tool
    from langgraph.prebuilt import create_react_agent

    # highlight-next-line
    tool =create_manage_memory_tool(namespace=("memories", "{user_id}"))
    agent = create_react_agent("anthropic:claude-3-5-sonnet-latest", tools=[tool])
    agent.invoke({
        "messages": [{"role": "user", "content": "I work at Acme Corp in the ML team"}],
        # Any memories for this run will be stored under ("memories", "user_123")
        "configurable": {"user_id": "user_123"}
    })
```

See the [NamespaceTemplate](../reference/utils.md#langmem.utils.NamespaceTemplate) class reference docs for more details.

### Flexible Retrieval

If you use one of the managed APIs, LangMem will integrate directly with LangGraph's [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) interface for memory storage and retrieval. The storage system supports multiple ways to retrieve memories:

- [**Direct Access**](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get): Get a specific memory by key
- [**Semantic Search**](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search): Find memories by semantic similarity
- [**Metadata Filtering**](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search): Filter memories by their attributes

For more details on storage capabilities, see the [LangGraph Storage documentation](https://langchain-ai.github.io/langgraph/reference/store/).
