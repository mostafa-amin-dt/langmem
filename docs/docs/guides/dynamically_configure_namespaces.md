---
title: How to configure dynamic namespaces
---

# How to configure dynamic namespaces

Langmem's stateful components store memories within "namespaces" so you can isolate data by user, agent, or other values. These namespaces can contain template variables to be populated from configurable values at runtime. Below is a quick example:

```python
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool

# Create tool with {user_id} template
tool = create_manage_memory_tool(namespace=("memories", "{user_id}"))
# Agent just sees that it has memory. It doesn't know where it's stored.
app = create_react_agent("anthropic:claude-3-5-sonnet-latest", tools=[tool])
# Use with different users
app.invoke(
    "Remember I like dark mode",
    # highlight-next-line
    config={"configurable": {"user_id": "user-123"}}
)  # Stores in ("memories", "user-123")
```

Namespace templates can be used in any of LangMem stateful components, such as `create_memory_store_enricher` and `create_manage_memory_tool`.
Below is a simple example:

## Common Patterns

Organize memories by user, organization, or feature:

```python
# Organization-level
tool = create_manage_memory_tool(
    namespace=("memories", "{org_id}")
)
# ... re-create app, then:
app.invoke(
    "Set company policy",
    config={"configurable": {"org_id": "acme"}}
)

# User within organization
tool = create_manage_memory_tool(
    namespace=("memories", "{org_id}", "{user_id}")
)
# If you wanted to, you could let the agent
# search over all users within an organization
tool = create_search_memory_tool(
    namespace=("memories", "{org_id}")
)
# ... re-create app, then:
app.invoke(
    "Remember user settings",
    config={"configurable": {
        "org_id": "acme",
        "user_id": "alice"
    }}
)

# You could also organize memories by type or category if you prefer 
tool = create_manage_memory_tool(
    namespace=("agent_smith", "memories", "{user_id}", "preferences")
)
# ... re-create app, then:
app.invoke(
    "Save theme preference",
    config={"configurable": {"user_id": "alice"}}
)
```

???+ note "Template Variables"
    Template variables (like `{user_id}`) must be present in your runtime config's `configurable` dict. If they are no