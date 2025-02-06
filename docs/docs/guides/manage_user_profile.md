---
title: How to Manage User Profiles
---

# How to Manage User Profile

Need to maintain consistent user data across conversations? Here's how to manage profile information without data loss.

## Without storage

Extract profile data:

```python
from langmem import create_memory_enricher
from pydantic import BaseModel
from typing import Optional


# Define profile structure
class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


# Configure extraction
enricher = create_memory_enricher(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[UserProfile],
    instructions="Extract user profile information",
    enable_inserts=False,
)

# First conversation
conversation1 = [{"role": "user", "content": "I'm Alice from California"}]
memories = enricher.invoke({"messages": conversation1})
print(memories[0])
# ExtractedMemory(id='80999209-3456-4016-90de-c8473e3b0ea5', content=UserProfile(name='Alice', language=None, timezone='America/Los_Angeles'))

# Second conversation
conversation2 = [{"role": "user", "content": "I speak Spanish too!"}]
update = enricher.invoke({"messages": conversation2, "existing": memories})
print(update[0])
# ExtractedMemory(id='80999209-3456-4016-90de-c8473e3b0ea5', content=UserProfile(name='Alice', language='Spanish', timezone='America/Los_Angeles'))

```

For more about profiles, see [Semantic Memory](../concepts/conceptual_guide.md#semantic-memory-facts-and-knowledge).

## With storage

To persist profiles across conversations, use `create_memory_store_enricher` with LangGraph's store system:

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_enricher

# Set up store and models
store = InMemoryStore() # (1)
```

1. For production deployments, use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore). `InMemoryStore` works fine for development but doesn't persist data between restarts.

        ```python
        enricher = create_memory_store_enricher(
            "anthropic:claude-3-5-sonnet-latest",
            namespace=("users", "{user_id}", "profile"),
            schemas=[UserProfile],
            enable_inserts=False
        )
        my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
        ```
        
        Note too that in this example, the `{user_id}` placeholder lets you manage user profiles in LangGraph's BaseStore namespace, creating isolated storage for each user's information.

        ```python
        # Example 1: Update User A's profile
        # enricher.invoke(
        #     {"messages": [{"role": "user", "content": "I'm John, an engineer at Acme"}]},
        #     config={"configurable": {"user_id": "user-a"}}
        # )  
        ```
        
        The namespace structure `("users", "{user_id}", "profile")` supports different profile management patterns:
        
        ```python
        # Individual user profiles
        namespace=("users", "user-123", "profile")
        
        # Team/department profiles
        namespace=("users", "team-sales", "profile")
        
        # Role-based profiles
        namespace=("users", "admin-1", "profile")
        ```

```python
# Define app with store context
@entrypoint(store=store)
def app(messages: list):
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ]
        + messages
    )

    # Update profile with new information (Uses store from @entrypoint context)
    # Existing memories are updated automatically
    # Note: in prod, you would do this asynchronously in the background
    enricher.invoke({"messages": messages})

    return response


# Use the app
app.invoke(
    [
        {"role": "user", "content": "I'm alice and I speak English"},
    ],
    config={"configurable": {"user_id": "user-a"}},
)

app.invoke(
    [
        {"role": "user", "content": "I'm austin and I speak Tamil"},
    ],
    config={"configurable": {"user_id": "user-b"}},
)

# Check stored profiles
for item in store.search(("users",)):
    print(item.namespace, item.value)

# Output:
# ('users', 'user-a', 'profile') {'kind': 'UserProfile', 'content': {'name': 'alice', 'language': 'English', 'timezone': None}}
# ('users', 'user-b', 'profile') {'kind': 'UserProfile', 'content': {'name': 'austin', 'language': 'Tamil', 'timezone': None}}
```

See [Storage System](../concepts/conceptual_guide.md#storage-system) for more about store configuration.
