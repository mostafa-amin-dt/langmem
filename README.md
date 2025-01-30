# LangMem: Memory Utilities

LLM apps work best if they can learn from mistakes and user preferences. LangMem provides utilities commonly used to build memory systems that:

1. Extract useful information from conversations
2. Store in structured or unstructured form
3. Retrieve when relevant (using semantic search or directly saving in the system prompt)

```bash
pip install langmem==0.0.5rc8
```

We include functions at different levels of abstraction for better customization. Below, we include quick-start examples for using the higher-level APIs. We then include references for the main utility functions below.

## Agent quickstart

Let's build a tool-calling agent that can manually manage its memory to track user preferences:

```python
import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain.chat_models import init_chat_model

store = InMemoryStore()

# These tools are stateful and let the agent "consciously"
# 1. Create/update/delete memories
# 2. Search for relevant memories
# It saves these to the configured "BaseStore" (in our case an ephemeral "InMemoryStore")
tools = {
    t.name: t
    for t in (
        create_manage_memory_tool(),
        create_search_memory_tool(),
    )
}
model = init_chat_model("anthropic:claude-3-sonnet-20240229").bind_tools(tools.values())


@task
async def execute_tool(tool_call: dict):
    result = await tools[tool_call["name"]].ainvoke(tool_call["args"])
    return {
        "role": "tool",
        "content": result,
        "tool_call_id": tool_call["id"],
    }


system_prompt = (
    """You are a helpful assistant. Save memories whenever you learn something new."""
)


@entrypoint(checkpointer=MemorySaver(), store=store)
async def assistant(messages: list[dict], *, previous=None):
    messages = (previous or []) + messages
    response = None
    while True:
        if response:
            if response.tool_calls:
                tool_messages = await asyncio.gather(
                    *(execute_tool(tool_call) for tool_call in response.tool_calls)
                )
                messages.extend(tool_messages)
            else:
                break
        response = await model.ainvoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        messages.append(response)

    return entrypoint.final(value=response, save=messages)


async def main():
    config = {"configurable": {"thread_id": "user123"}}
    messages = [{"role": "user", "content": "I really like shoo-fly pie."}]
    async for step in assistant.astream(messages, config, stream_mode="messages"):
        print(step)


asyncio.run(main())
```

The agent will automatically:

1. Save preferences when mentioned
2. Search past memories for context
3. Maintain conversation history

## Reflection Quickstart

TODO.

## Package Overview

LangMem provides utilities organized by their state management and deployment patterns:

### Stateless Functions

These functions process data without maintaining state:

- `create_prompt_optimizer`: Learns from trajectories and feedback to store instructions in your system prompt
- `create_multi_prompt_optimizer`: Optimizes multiple prompts (for a multi-agent or chained system) simultaneously
- `create_memory_enricher`: Extracts and updates structured information from conversations.

### Stateful Functions

These functions read from and write to the configured store:

- `create_memory_store_enricher`: End-to-end memory management with storage integration

### Tools

These are LangChain tools that agents can use to manage memory:

- `create_manage_memory_tool`: Create/update/delete memories during conversations
- `create_search_memory_tool`: Search through existing memories

### Deployable Graphs

Ready-to-use LangGraph components for production:

- `langmem.graphs.semantic:graph`: Memory enrichment graph for extracting and updating memories from conversations
- `langmem.graphs.prompts:optimize_prompts`: Prompt optimization graph for improving system prompts based on conversation history

## Core Functions Reference

Below is a reference of the main utility functions exported by this package, organized by their state management patterns.

### Stateless Functions

These functions process data without maintaining state.

#### `create_prompt_optimizer(model: str | BaseChatModel, kind: Literal["gradient", "prompt_memory", "metaprompt"] = "gradient", config: Optional[OptimizerConfig] = None)`

Improves prompts based on successful conversations. Supports multiple optimization strategies:

- `gradient`: Uses gradient-based techniques to iteratively improve prompt effectiveness
- `prompt_memory`: Leverages past successful prompts to suggest improvements
- `metaprompt`: Uses meta-learning to generate optimized prompts

```python
import asyncio
from langmem import create_prompt_optimizer

optimizer = create_prompt_optimizer(
    "anthropic:claude-3-sonnet-20240229", kind="metaprompt"
)


async def main():
    conversation = [
        {
            "role": "user",
            "content": "How do I write a bash script to find all python files in a directory?",
        },
        {
            "role": "assistant",
            "content": "Ah, that's an interesting question about file searching!"
            " You know, there are several ways we could approach this in bash. Let me think..."
            " One common approach would be to use the find command - "
            "it's quite versatile for this sort of task. "
            "Would you like me to explain how we could use it? "
            "Basically, you'd want to structure it something like this: you can use 'find . -name \"*.py\"'"
            "where the dot represents the current directory. "
            "Of course, you could also use other options with find to make it more specific to your needs. "
            "Is there any particular aspect of the script you'd like me to elaborate on?",
        },
    ]
    # Feedback is optional. The optimizer can also try to learn from the trajectory itself.
    feedback = "You should respond directly with a code snippet, not be so chatty."
    annotated_trajectories = [(conversation, feedback)]
    better_prompt = await optimizer(
        annotated_trajectories, "You are a coding assistant."
    )
    print(f"Improved prompt: {better_prompt}")


asyncio.run(main())
```

Output:

```text
Improved prompt: You are a coding assistant that provides concise code solutions to programming
questions without extraneous explanations or chatter. When asked for code, directly provide
the relevant code snippet to address the request.
Do not provide lengthy preambles or follow-up questions - simply give the code.
```

#### `create_multi_prompt_optimizer(model: str | BaseChatModel, kind: Literal["gradient", "prompt_memory", "metaprompt"] = "gradient", config: Optional[OptimizerConfig] = None)`

Optimizes multiple related prompts while maintaining consistency and leveraging shared context.

```python
import asyncio
from langmem import create_multi_prompt_optimizer

# Single prompt example
multi_optimizer = create_multi_prompt_optimizer(
    "anthropic:claude-3-sonnet-20240229", kind="metaprompt"
)


async def multi_agent_example():
    # Conversation with a multi-agent system
    conversation = [
        {
            "role": "user",
            "content": "Can you research and write a report about the impact of remote work on employee productivity?",
        },
        {
            "role": "assistant",
            "content": "Here's what I found in my research:"
            "\n\n- Studies show mixed results on productivity"
            "\n- Some employees report fewer distractions"
            "\n- Communication can be challenging"
            "\n- Home office setup matters"
            "\n\nBased on this research, here's my report:"
            "\n\nKey Findings on Remote Work Productivity:"
            "\n- Productivity varies by individual"
            "\n- Communication needs special attention"
            "\n- Office setup is a critical factor"
            "\n\nRecommendations:"
            "\n- Invest in proper equipment"
            "\n- Establish clear communication channels"
            "\n- Monitor individual performance",
        },
        {
            "role": "user",
            "content": "Can you dig deeper into the actual impact on different types of work?",
        },
        {
            "role": "assistant",
            "content": "Here's what I found about different work types:"
            "\n\nCreative Work:"
            "\n- Less office noise"
            "\n- Flexible hours"
            "\n\nCollaborative Work:"
            "\n- Video calls required"
            "\n- Time zone challenges"
            "\n\nIndividual Tasks:"
            "\n- Better focus"
            "\n- Fewer interruptions"
            "\n\nRecommended Actions:"
            "\n- Use different tools for different work types"
            "\n- Adjust schedules based on work type"
            "\n- Provide appropriate support for each category",
        },
    ]

    feedback = "The research seems quite surface-level and generic. Also, the report is just a list of bullet points - it would be better as a proper narrative that tells a story about how remote work affects different types of employees and their work patterns."

    prompts = [
        {
            "name": "research",
            "prompt": "Conduct deep research on the topic, finding specific studies, statistics, and detailed examples",
        },
        {
            "name": "email",
            "prompt": "Write professional and concise emails summarizing the key points",
        },
        {
            "name": "report",
            "prompt": "Write a narrative report that tells a coherent story about the findings and their implications",
        },
    ]

    better_prompts = await multi_optimizer([(conversation, feedback)], prompts)
    print("Optimized prompts based on conversation feedback:")
    for p in better_prompts:
        if p["name"] == "email":
            print(
                f'\n{p["name"]}: {p["prompt"]}  # This prompt remains unchanged since the email agent wasn\'t used'
            )
        else:
            print(f'\n{p["name"]}: {p["prompt"]}')
        print("-" * 80)

asyncio.run(multi_agent_example())
```

Output: (note the email prompt remains the same)

```text
Optimized prompts based on conversation feedback:

research: Conduct in-depth research on the impact of remote work on employee productivity for different types of work using credible sources like published research studies, analysis reports, surveys, statistics and real-world examples. Focus your research and findings on the specific context provided in the query, such as particular industries, work types or organizational structures.

Structure your response as a comprehensive narrative report with the following sections:

1. Introduction - Provide an overview of the context and key areas covered in the report.

2. Findings by Work Type - Discuss the productivity impacts, challenges and benefits of remote work for different categories of work like creative work, collaborative projects, individual tasks, client-facing roles, etc. Support the findings with data, statistics and real examples.

3. Cross-Cutting Themes - Examine common factors influencing remote productivity across work types, such as communication tools, home office setup, organizational policies, worker personality types etc.

4. Key Recommendations - Based on the research findings, provide tailored recommendations for organizations to optimize remote productivity, grouped by areas like technology, policies, training, role types, etc.

5. Conclusion - Summarize the main takeaways and overall impact of remote work on productivity.

Ensure the report has a coherent narrative flow, is tailored to the specific query context, and leverages in-depth research from reputable sources.
--------------------------------------------------------------------------------

email: Write professional and concise emails summarizing the key points  # This prompt remains unchanged since the email agent wasn't used
--------------------------------------------------------------------------------

report: Write a narrative report that provides an in-depth analysis of the impact of remote work on employee productivity, tailored to different types of work like creative work, collaborative projects, individual tasks, etc.

For each work category, research and analyze the specific challenges and benefits of remote work on productivity. Go beyond surface-level findings and dig into the nuanced factors that influence productivity in that domain.

Then, structure your report as a coherent story that:

1) Introduces the different work categories being analyzed
2) Describes the key research findings on productivity impacts for each category
3) Explores the underlying reasons and implications behind those impacts
4) Provides actionable recommendations for optimizing productivity in a remote setup based on the work type

Ensure your narrative flows logically and provides valuable insights through an engaging, story-driven structure rather than just listing key points. Here's an example narrative structure:

Introduction - Brief overview of work categories and the goal of analyzing remote productivity impacts

Chapter 1 - Creative Work
   Challenges: Distractions at home, motivation barriers, etc.
   Benefits: Flexible schedules, quieter environment, etc.
   Recommended Approach: ...

Chapter 2 - Collaborative Projects
   Challenges: Communication barriers, time zone issues, etc.
   Benefits: Geographic flexibility, reduced commute, etc.
   Recommended Approach: ...

(Continue for other categories...)

Conclusion - Summary of key findings and overarching recommended strategies for thriving in remote work based on work type.
--------------------------------------------------------------------------------
```

#### `create_memory_enricher(model: str | BaseChatModel, schemas: Optional[list[BaseModel]] = None)`

Creates a function that extracts information from conversations. The simplest usage automatically extracts unstructured memories:

```python
import asyncio
from langmem import create_memory_enricher

enricher = create_memory_enricher(
    "anthropic:claude-3-sonnet-20240229",
    instructions="""Extract all memorable information from the following sessions as distinct memories.

1. If new information is provided, consider whether it should update an existing memory or whether it should fit in a new memory
2. If existing memories are incorrect or outdated, patch them based on the new information.
3. For distinct information, save them as new memories for better organization.

Invoke all patch and memory calls in a single generation before completing.""",
)


async def main():
    # Example conversation showing implicit preferences and behaviors
    conversation = [
        {
            "role": "user",
            "content": "This codebase is a mess. There are no type hints, the functions are huge, and there's zero documentation.",
        },
        {
            "role": "assistant",
            "content": "I understand your concerns. Let's add type hints, break down the functions, and add docstrings.",
        },
        {
            "role": "user",
            "content": "Perfect. I also wanna add linting. I always use black with line length 88.",
        },
    ]

    # The enricher extracts implicit information about code quality preferences
    memories = await enricher(conversation)
    print("Extracted preferences:")
    for memory_id, content in memories:
        print(f"{memory_id}: {content}")

    # Second conversation showing evolution of preferences
    conversation_2 = [
        {
            "role": "user",
            "content": "I've been working with JavaScript lately and really missing type safety - prefer to use the | types instead of union.",
        },
        {
            "role": "assistant",
            "content": "We could migrate to TypeScript. It would give us strong typing similar to what you're used to from Python.",
        },
        {
            "role": "user",
            "content": "Ya that sounds great. You work on that. I gotta go take my dog Sparky for a walk.",
        },
    ]

    # Include the previous memories to enrich or extend
    updated_memories = await enricher(conversation_2, memories)
    print("\nUpdated preferences:")
    for memory_id, content in updated_memories:
        print(f"{memory_id}: {content}")

asyncio.run(main())
```

Output: 
```text
Extracted preferences:
f1d47111-6055-459c-a7f1-816a40ec5a33: content='The codebase had the following issues:\n\n1. No type hints\n2. Large, complex functions \n3. Lack of documentation\n\nTo improve it, the following actions were recommended:\n\n1. Add type hints\n2. Break down larger functions into smaller, more focused functions\n3. Add docstrings to document code\n4. Set up linting with Black formatter, using a line length of 88'

Updated preferences:
f1d47111-6055-459c-a7f1-816a40ec5a33: content='The codebase had the following issues:\n\n1. No type hints\n2. Large, complex functions \n3. Lack of documentation\n4. Lack of type safety in JavaScript\n\nTo improve it, the following actions were recommended:\n\n1. Add type hints\n2. Break down larger functions into smaller, more focused functions\n3. Add docstrings to document code\n4. Set up linting with Black formatter, using a line length of 88\n5. Migrate to TypeScript for better type safety'
dbb57acb-6d64-4aa6-adaa-bbb73d7dec43: content='The person mentioned having a dog named Sparky that they needed to take for a walk.'
```

### Stateful Functions

These functions read from and write to the configured BaseStore. They are usable in the LangGraph context.

#### `create_memory_store_enricher(model: str | BaseChatModel, schemas: Optional[list[BaseModel]] = None, enable_inserts: bool = True, enable_deletes: bool = False)`

End-to-end memory management system that combines automatic search, extraction, and storage operations.

```python
import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem.knowledge import create_memory_store_enricher
from langchain.chat_models import init_chat_model

enricher = create_memory_store_enricher(
    "anthropic:claude-3-sonnet-20240229",
    instructions="""Extract all memorable information from the following sessions as distinct memories.

1. If new information is provided, consider whether it should update an existing memory or whether it should fit in a new memory
2. If existing memories are incorrect or outdated, patch them based on the new information.
3. For distinct information, save them as new memories for better organization.

Invoke all patch and memory calls in a single generation before completing.""",
)
llm = init_chat_model("anthropic:claude-3-sonnet-20240229")

store = InMemoryStore()


@entrypoint(checkpointer=MemorySaver(), store=store)
async def agent(state: dict, *, previous=None):
    conversation = (previous or []) + state["messages"]
    response = await llm.ainvoke(conversation)
    if state.get("trigger"):
        await enricher(conversation)
    return entrypoint.final(value=response, save=conversation)


async def main():
    config = {"configurable": {"thread_id": "convo-1"}}
    conversation_1 = [
        {
            "role": "user",
            "content": "This codebase is a mess. There are no type hints, the functions are huge, and there's zero documentation.",
        },
        {
            "role": "user",
            "content": "Perfect. I also wanna add linting. I always use black with line length 88.",
        },
    ]
    for i, user_turn in enumerate(conversation_1):
        _ = await agent.ainvoke(
            {"messages": [user_turn], "trigger": i == len(conversation_1) - 1},
            config=config,
        )

    print("After conversation 1")
    print(store.search(()))

    # Second conversation showing evolution of preferences
    config_2 = {"configurable": {"thread_id": "convo-2"}}
    conversation_2 = [
        {
            "role": "user",
            "content": "I've been working with JavaScript lately and really missing type safety - prefer to use the | types instead of union.",
        },
        {
            "role": "user",
            "content": "Ya that sounds great. You work on that. I gotta go take my dog Sparky for a walk.",
        },
    ]

    for i, user_turn in enumerate(conversation_2):
        await agent.ainvoke(
            {"messages": [user_turn], "trigger": i == len(conversation_2) - 1},
            config=config_2,
        )

    print("After conversation 2")
    print(store.search(()))


asyncio.run(main())
```

Output:
```text
After conversation 1
[Item(namespace=['memories', '{user_id}'], key='29199c91-6705-471e-80a3-c72d599bfc15', value={'kind': 'Memory', 'content': {'content': 'The Python codebase being worked on has several issues:\n\n- No type hints\n- Huge/long functions\n- Zero documentation\n\nTo improve the codebase, the following steps were planned:\n\n- Add type hints\n- Refactor long functions into smaller units\n- Add docstrings/documentation\n- Setup linting with Black formatter with line length 88'}}, created_at='2025-01-31T01:10:36.978793+00:00', updated_at='2025-01-31T01:10:36.978795+00:00', score=None)]
After conversation 2
[Item(namespace=['memories', '{user_id}'], key='29199c91-6705-471e-80a3-c72d599bfc15', value={'kind': 'Memory', 'content': {'content': 'The Python codebase being worked on has several issues:\n\n- No type hints\n- Huge/long functions\n- Zero documentation\n\nTo improve the codebase, the following steps were planned:\n\n- Add type hints\n- Refactor long functions into smaller units\n- Add docstrings/documentation\n- Setup linting with Black formatter with line length 88'}}, created_at='2025-01-31T01:10:36.978793+00:00', updated_at='2025-01-31T01:10:36.978795+00:00', score=None), Item(namespace=['memories', '{user_id}'], key='3e542944-70f3-49dc-b4cd-4bd17502b091', value={'kind': 'Memory', 'content': {'content': 'Prefers to use pipe (|) types instead of union types when working with JavaScript for better type safety.'}}, created_at='2025-01-31T01:10:49.097310+00:00', updated_at='2025-01-31T01:10:49.097312+00:00', score=None)]
```


### Tools

These are tools that LangGraph agents can use to "consciously" manage memory. Memories are persisted to a LangGraph BaseStore and searchable.

#### `create_manage_memory_tool(instructions: str = DEFAULT_INSTRUCTIONS, namespace_prefix: tuple[str, ...] | NamespaceTemplate = ("memories", "{user_id}"), kind: Literal["single", "multi"] = "multi")`

Creates a tool for explicit memory management during conversations.

```python
from langmem import create_manage_memory_tool

memory_tool = create_manage_memory_tool(
    instructions="Custom instructions for when to create memories",
    # The brackets will be populated by the configurable values
    namespace_prefix=("app", "user_memories", "{user_id}"),
    kind="multi"
)
await memory_tool.ainvoke({
    "action": "create",
    "content": "User dislikes notifications",
    "tags": ["preferences", "notifications"]
})
```

### Deployable Graphs

LangMem provides ready-to-use graphs that can be deployed on the LangGraph platform. We maintain a hosted version at `https://langmem-v0-544fccf4898a5e3c87bdca29b5f9ab21.us.langgraph.app` that you can use immediately, or you can deploy your own modified version.

To use the hosted version, first set up authentication:

```python
import os
from langgraph_sdk import get_client

# Required: LangSmith API key (US region)
os.environ["LANGSMITH_API_KEY"] = "<your key>"

# Connect to the hosted service
url = "https://langmem-v0-544fccf4898a5e3c87bdca29b5f9ab21.us.langgraph.app"
client = get_client(url=url)
```

#### Learn prompt instructions

Automatically learn instructions and core memories to store in your prompts based on conversation history and feedback:

```python
# Simple example conversation
conversation = [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]
feedback = {"user_feedback": "Please provide more context in your answers"}

# Update a single prompt
results = await client.runs.wait(
    None,
    "optimize_prompts",
    input={
        "threads": [[conversation, feedback]],
        "prompts": [{
            "name": "assistant_prompt",
            "prompt": "You are a helpful assistant.",
            "when_to_update": "When user requests more detail",
            "update_instructions": "Add instructions about providing context",
        }]
    },
    config={"configurable": {"model": "claude-3-5-sonnet-latest"}}
)
```

#### Learn semantic memory

Extract and store semantic knowledge from conversations:

```python
# Example conversation
conversation = [
    {"role": "user", "content": "I prefer dark mode and minimalist interfaces"},
    {"role": "assistant", "content": "I'll remember your UI preferences."},
]

# Extract memories with optional schema
results = await client.runs.wait(
    None,
    "extract_memories",
    input={
        "messages": conversation,
        "schemas": [{  # Optional: define memory structure
            "title": "UserPreference",
            "type": "object",
            "properties": {
                "preference": {"type": "string"},
                "category": {"type": "string"},
            }
        }]
    },
    config={"configurable": {"model": "claude-3-5-sonnet-latest"}}
)

# Search memories
memories = await client.store.search_items((), query="UI preferences")
```

## Conceptual guide

Below is a high-level conceptual overview of a way to think about memory management.

### 1. Formation Pattern

How memories are created:

| Pattern                   | Description                                        | Best For                                                                           | Tools                                                          |
| ------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Conscious (Agent Tools)   | Agent actively decides to save during conversation | - Direct user feedback<br>- Explicit rules and preferences<br>- Teaching the agent | - `create_manage_memory_tool`<br>- `create_search_memory_tool` |
| Subconscious (Background) | Separate LLM analyzes conversations/trajectories   | - Pattern discovery<br>- Learning from experience<br>- Complex relationships       | - `create_memory_enricher`<br>- `create_memory_store_enricher` |

Example of conscious memory formation:

```python
from langmem import create_manage_memory_tool, create_search_memory_tool

tools = {
    "manage_memory": create_manage_memory_tool(),
    "search_memory": create_search_memory_tool()
}
model = init_chat_model("model_name").bind_tools(tools.values())
```

Example of subconscious memory formation:

```python
from langmem import create_memory_store_enricher

enricher = create_memory_store_enricher(
    "model_name",
    schemas=[UserProfile],
    enable_inserts=True
)
memories = await enricher.manage_memories(conversation)
```

### 2. Storage Pattern

How memories are structured:

| Pattern             | Description                         | Best For                                                                 | Implementation                                    |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- |
| Constrained Profile | Single schema, continuously updated | - User preferences<br>- System settings<br>- Current state               | Use `create_memory_enricher` with defined schemas |
| Event Stream        | Expansive list of discrete memories | - Conversation history<br>- Learning experiences<br>- Evolving knowledge | Use `create_manage_memory_tool` with kind="multi" |

Example of constrained profile:

```python
from pydantic import BaseModel
from langmem import create_memory_enricher

class UserProfile(BaseModel):
    preferences: dict[str, str]
    settings: dict[str, Any]

enricher = create_memory_enricher(
    "model_name",
    schemas=[UserProfile],
    kind="single"
)
```

Example saving semantic facts

```python
from langmem import create_manage_memory_tool

memory_tool = create_manage_memory_tool(
    kind="multi",
    namespace_prefix=("user", "experiences")
)
```

### 3. Retrieval Pattern

How memories are accessed:

| Pattern                   | Description                                    | When to Use                                                              | Implementation                                              |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| Always-On (System Prompt) | Critical context included in every interaction | - Core rules<br>- User preferences<br>- Session state                    | Use `create_prompt_optimizer` with memory integration       |
| Associative (Search)      | Contextually searched when needed              | - Historical conversations<br>- Specific knowledge<br>- Past experiences | Use `create_search_memory_tool` or `create_memory_searcher` |
