---
title: How to Extract Semantic Memories
---

# How to Extract Semantic Memories

Need to extract multiple related facts from conversations? Here's how to use LangMem's collection pattern for semantic memories. For single-document patterns like user profiles, see [Manage User Profile](./manage_user_profile.md).

## Using Functional Primitives

Extract semantic memories:

```python
from langmem import create_memory_enricher
from pydantic import BaseModel

# Define what to extract
class Triple(BaseModel):
    """Store all new facts, preferences, and relationships as riples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# Configure extraction
enricher = create_memory_enricher(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Triple],
    instructions="Extract user preferences",
    enable_inserts=True,
    enable_deletes=True,
)
```
After the first short ineraction, the system has extravted some semantic triples:
```python
# First conversation - extract triples
conversation1 = [
    {"role": "user", "content": "Alice manages the ML team and mentors Bob, who is also on the team."}
]
memories = enricher.invoke({"messages": conversation1})
print("After first conversation:")
for m in memories:
    print(m)
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
```

The second conversation updates some existing memories. Since we have enabled "deletes", the enricher will return `RemoveDoc` objects to indicate that the memory should be removed, and a new memory will be created in its place. Since this is the funcitonal API, you can control what "removal" means, be that a soft or hard delete, or simply a down-weighting of the memory.

```python
# Second conversation - update and add triples
conversation2 = [
    {"role": "user", "content": "Bob now leads the ML team and the NLP project."}
]
update = enricher.invoke({"messages": conversation2, "existing": memories})
print("After second conversation:")
for m in update:
    print(m)
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
existing = [m for m in update if isinstance(m.content, Triple)]
```

The third conversation overwrites even more memories.
```python
# Delete triples about an entity
conversation3 = [
    {"role": "user", "content": "Alice left the company."}
]
final = enricher.invoke({"messages": conversation3, "existing": existing})
print("After third conversation:")
for m in final:
    print(m)
# ExtractedMemory(id='7ca76217-66a4-4041-ba3d-46a03ea58c1b', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='35b443c7-49e2-4007-8624-f1d6bcb6dc69', content=RemoveDoc(json_doc_id='0214f151-b0c5-40c4-b621-db36b845956c'))
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
```

For more about semantic memories, see [Memory Types](../concepts/conceptual_guide.md#types-of-memory).

## Managed in LangGraph's BaseStore

The same extraction can be managed automatically by LangGraph's BaseStore:

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_enricher

# Set up store and models
store = InMemoryStore()
enricher = create_memory_store_enricher(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("chat", "{user_id}", "triples"),
    schemas=[Triple],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


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

    # Extract and store triples (Uses store from @entrypoint context)
    # Note: in prod, you would do this asynchronously in the background
    enricher.invoke({"messages": messages})

    return response


# First conversation
app.invoke(
    [
        {
            "role": "user",
            "content": "Alice manages the ML team and mentors Bob, who is also on the team.",
        },
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Second conversation
app.invoke(
    [
        {"role": "user", "content": "Bob now leads the ML team and the NLP project."},
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Third conversation
app.invoke(
    [
        {"role": "user", "content": "Alice left the company."},
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Check stored triples
for item in store.search(("chat", "user123")):
    print(item.namespace, item.value)

# Output:
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'is_member_of', 'object': 'ML_team', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'leads', 'object': 'ML_team', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'leads', 'object': 'NLP_project', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Alice', 'predicate': 'employment_status', 'object': 'left_company', 'context': None}}

```


See [Storage System](../concepts/conceptual_guide.md#storage-system) for namespace patterns.
