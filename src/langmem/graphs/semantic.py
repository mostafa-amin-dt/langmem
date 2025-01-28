from langmem import create_memory_store_enricher
from langgraph.graph.state import StateGraph
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage
import typing


class InputState(TypedDict, total=False):
    messages: typing.Required[
        list[AnyMessage] | list[tuple[list[AnyMessage], dict[str, str]]]
    ]
    schemas: None | list[dict] | dict
    namespace: tuple[str, ...] | None


class OutputState(TypedDict):
    updated_memories: list
    root_namesapace: tuple[str, ...]


class Config(TypedDict):
    model: str
    query_model: str | None = None
    enable_inserts: bool = True
    enable_deletions: bool = True


async def enrich(state: InputState, config: RunnableConfig):
    messages = state.get("messages", [])
    if not messages:
        return {"updated_memories": []}
    if isinstance(messages[0], list):
        messages = [m[0] for m in messages]
    namespace = state.get("namespace", ())
    configurable = config.get("configurable", {})
    model = configurable.get("model", "claude-3-5-sonnet-latest")
    schemas = state.get("schemas", None)
    enricher = create_memory_store_enricher(
        model,
        query_model=configurable.get("query_model", model),
        schemas=[schemas] if isinstance(schemas, dict) else schemas,
        enable_inserts=configurable.get("enable_inserts", True),
        enable_deletions=configurable.get("enable_deletions", True),
        namespace_prefix=("semantic", "{langgraph_auth_user_id}", *namespace),
    )

    updated_memories = await enricher(messages)
    return {
        "updated_memories": updated_memories,
        "root_namespace": ("semantic", configurable.get("langgraph_auth_user_id", "")),
    }


graph = (
    StateGraph(input=InputState, output=OutputState, config_schema=Config)
    .add_node("enrich", enrich)
    .add_edge("__start__", "enrich")
    .compile()
)
graph.name = "enrich_memories"
