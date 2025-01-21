from langchain_core.runnables import Runnable
from typing import Optional, Union, Dict, Any, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from langmem.utils import NamespaceTemplate

class MemoryLayer(Runnable):
    __slots__ = ("model", "variable", "label", "configurable", "update_instructions", "scope", "schema", "kind", "config")

    def __init__(   
        self,
        label: Optional[str] = None,
        variable: Optional[str] = None, 
        update_instructions: Optional[str] = None,
        namespace: Optional[tuple[str, ...]] = None,
        schemas: Union[type[str], list[type[BaseModel]], list[dict]] = str,
        kind: Literal["incontext", "vector"] = "incontext",
    ) -> None:
        """Initialize a memory layer.
        
        Args:
            label: Optional[str]: Human readable label for this memory layer
            variable: Optional[str]:  Where in the prompt this will be inserted. If not provided, it will be appended to the system prompt in the configured order
            configurable: Optional[str]: Field that can be configured per request
            update_instructions: Optional[str]: Instructions for when to update this memory layer
            namespace: Optional[tuple[str, ...]]: Scope for this memory layer (e.g. user_id, org_id)
            schemas: Union[type[str], list[type[BaseModel]], list[dict]]: Schema for validating memories
            kind: Literal["incontext", "vector"]: Type of memory storage/lookup - "incontext" or "vector"
        """
        self.variable = variable
        self.label = label
        self.update_instructions = update_instructions
        self.namespace = NamespaceTemplate(namespace)
        self.schemas = schemas
        self.kind = kind

    def invoke(self, messages: list[AnyMessage]) -> str:
        if self.kind == "incontext":
            # cached lookup
        elif self.kind == "vector":
            # vector lookup
        else:
            raise ValueError(f"Unknown kind: {self.kind}")

class MemoryPrompt(Runnable):
    __slots__ = ("model",)

    def __init__(self, 
    model: str | BaseChatModel) -> None:
