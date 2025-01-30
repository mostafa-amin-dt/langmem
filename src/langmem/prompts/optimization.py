import asyncio
import typing

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langmem import utils
from langmem.prompts.stateless import PromptMemoryMultiple
from langmem.prompts.types import Prompt
from langmem.prompts.gradient import (
    GradientOptimizerConfig,
    create_gradient_prompt_optimizer,
)
from langmem.prompts.metaprompt import (
    MetapromptOptimizerConfig,
    create_metaprompt_optimizer,
)
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor
from typing import Protocol, runtime_checkable

KINDS = typing.Literal["gradient", "metaprompt", "prompt_memory"]


@runtime_checkable
class PromptOptimizerProto(Protocol):
    """
    Protocol for a single-prompt optimizer that can be called as:
       await optimizer(sessions, prompt)
    or
       await optimizer.ainvoke({"sessions": ..., "prompt": ...})
    returning an updated prompt string.
    """

    async def __call__(
        self,
        sessions: list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]] | str,
        prompt: str | Prompt,
    ) -> str: ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional[GradientOptimizerConfig] = None,
) -> PromptOptimizerProto:
    """Create a prompt optimizer that improves prompt effectiveness.

    This function creates an optimizer that can analyze and improve prompts for better
    performance with language models. It uses gradient-based techniques to suggest
    improvements to prompt structure, clarity, and effectiveness.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["gradient"], optional): The optimization strategy to use.
            Currently only supports "gradient". Defaults to "gradient".
        config (Optional[GradientOptimizerConfig], optional): Configuration options
            for the gradient optimizer. Defaults to None.

    Returns:
        PromptOptimizerProto: A callable that takes a prompt and returns an optimized version.

    Example:
        >>> optimizer = create_prompt_optimizer("anthropic:claude-3-sonnet-20240229")
        >>> original_prompt = "Tell me about the solar system"
        >>> optimized = await optimizer(original_prompt)
        >>> print(optimized)
        'Provide a comprehensive overview of the solar system...'
    """
    ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional[MetapromptOptimizerConfig] = None,
) -> PromptOptimizerProto:
    """Create a prompt optimizer that improves prompt effectiveness.

    This function creates an optimizer that can analyze and improve prompts for better
    performance with language models. It uses meta-prompting techniques to suggest
    improvements to prompt structure, clarity, and effectiveness.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["metaprompt"], optional): The optimization strategy to use.
            Currently only supports "metaprompt". Defaults to "metaprompt".
        config (Optional[MetapromptOptimizerConfig], optional): Configuration options
            for the meta-prompt optimizer. Defaults to None.

    Returns:
        PromptOptimizerProto: A callable that takes a prompt and returns an optimized version.

    Example:
        >>> optimizer = create_prompt_optimizer("anthropic:claude-3-sonnet-20240229")
        >>> original_prompt = "Tell me about the solar system"
        >>> optimized = await optimizer(original_prompt)
        >>> print(optimized)
        'Provide a comprehensive overview of the solar system...'
    """
    ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
    config: None = None,
) -> PromptOptimizerProto:
    """Create a prompt optimizer that improves prompt effectiveness.

    This function creates an optimizer that can analyze and improve prompts for better
    performance with language models. It uses prompt memory techniques to suggest
    improvements to prompt structure, clarity, and effectiveness.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["prompt_memory"], optional): The optimization strategy to use.
            Currently only supports "prompt_memory". Defaults to "prompt_memory".
        config (None, optional): Configuration options for the prompt memory optimizer.
            Defaults to None.

    Returns:
        PromptOptimizerProto: A callable that takes a prompt and returns an optimized version.

    Example:
        >>> optimizer = create_prompt_optimizer("anthropic:claude-3-sonnet-20240229")
        >>> original_prompt = "Tell me about the solar system"
        >>> optimized = await optimizer(original_prompt)
        >>> print(optimized)
        'Provide a comprehensive overview of the solar system...'
    """
    ...


def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: KINDS = "gradient",
    config: typing.Union[
        GradientOptimizerConfig, MetapromptOptimizerConfig, None
    ] = None,
) -> PromptOptimizerProto:
    """
    Returns an object that can be awaited or .ainvoke(...)'d to produce an updated prompt.

    Example usage:
        optimizer = create_prompt_optimizer(..., kind="gradient")
        new_prompt = await optimizer(sessions, prompt)
    """
    if kind == "gradient":
        return create_gradient_prompt_optimizer(model, config)  # type: ignore
    elif kind == "metaprompt":
        return create_metaprompt_optimizer(model, config)  # type: ignore
    elif kind == "prompt_memory":
        return PromptMemoryMultiple(model)  # type: ignore
    else:
        raise NotImplementedError(
            f"Unsupported optimizer kind: {kind}.\nExpected one of {KINDS}"
        )


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional[GradientOptimizerConfig] = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    """Create an optimizer for multiple prompts with shared context.

    This function creates an optimizer that can improve multiple related prompts
    while maintaining consistency and leveraging shared context between them.
    It's particularly useful for optimizing chains of prompts or prompt templates
    that work together.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["gradient"], optional): The optimization strategy to use.
            Currently only supports "gradient". Defaults to "gradient".
        config (Optional[GradientOptimizerConfig], optional): Configuration options
            for the gradient optimizer. Defaults to None.

    Returns:
        Callable: An async function that takes multiple prompts or messages and returns
            a list of optimized prompts.

    Example:
        >>> optimizer = create_multi_prompt_optimizer("anthropic:claude-3-sonnet-20240229")
        >>> prompts = [
        ...     "Extract key information from the text",
        ...     "Summarize the extracted information",
        ...     "Generate insights based on the summary"
        ... ]
        >>> optimized = await optimizer(prompts)
        >>> for orig, opt in zip(prompts, optimized):
        ...     print(f"Original: {orig}")
        ...     print(f"Optimized: {opt}\n")
    """
    ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional[MetapromptOptimizerConfig] = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    """Create an optimizer for multiple prompts with shared context.

    This function creates an optimizer that can improve multiple related prompts
    while maintaining consistency and leveraging shared context between them.
    It's particularly useful for optimizing chains of prompts or prompt templates
    that work together.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["metaprompt"], optional): The optimization strategy to use.
            Currently only supports "metaprompt". Defaults to "metaprompt".
        config (Optional[MetapromptOptimizerConfig], optional): Configuration options
            for the meta-prompt optimizer. Defaults to None.

    Returns:
        Callable: An async function that takes multiple prompts or messages and returns
            a list of optimized prompts.

    Example:
        >>> optimizer = create_multi_prompt_optimizer("anthropic:claude-3-sonnet-20240229")
        >>> prompts = [
        ...     "Extract key information from the text",
        ...     "Summarize the extracted information",
        ...     "Generate insights based on the summary"
        ... ]
        >>> optimized = await optimizer(prompts)
        >>> for orig, opt in zip(prompts, optimized):
        ...     print(f"Original: {orig}")
        ...     print(f"Optimized: {opt}\n")
    """
    ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
    config: None = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    """Create an optimizer for multiple prompts with shared context.

    This function creates an optimizer that can improve multiple related prompts
    while maintaining consistency and leveraging shared context between them.
    It's particularly useful for optimizing chains of prompts or prompt templates
    that work together.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["prompt_memory"], optional): The optimization strategy to use.
            Currently only supports "prompt_memory". Defaults to "prompt_memory".
        config (None, optional): Configuration options for the prompt memory optimizer.
            Defaults to None.

    Returns:
        Callable: An async function that takes multiple prompts or messages and returns
            a list of optimized prompts.

    Example:
        >>> optimizer = create_multi_prompt_optimizer("anthropic:claude-3-sonnet-20240229")
        >>> prompts = [
        ...     "Extract key information from the text",
        ...     "Summarize the extracted information",
        ...     "Generate insights based on the summary"
        ... ]
        >>> optimized = await optimizer(prompts)
        >>> for orig, opt in zip(prompts, optimized):
        ...     print(f"Original: {orig}")
        ...     print(f"Optimized: {opt}\n")
    """
    ...


def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: KINDS = "gradient",
    config: typing.Union[
        GradientOptimizerConfig, MetapromptOptimizerConfig, None
    ] = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    """
    Returns an async function that:
       1) Classifies which prompts should be updated
       2) Calls the single-prompt optimizer for each
       3) Returns the updated list of prompts
    """
    _optimizer = create_prompt_optimizer(model, kind, config)

    @ls.traceable
    async def process_multi_prompt_sessions(
        sessions: (
            list[list[AnyMessage]]
            | list[AnyMessage]
            | list[tuple[list[AnyMessage], str]]
            | str
        ),
        prompts: list[Prompt],
    ) -> list[Prompt]:
        choices = [p["name"] for p in prompts]
        sessions_str = utils.format_sessions(sessions)
        if (
            isinstance(prompts, list)
            and len(prompts) == 1
            and prompts[0].get("when_to_update") is None
        ):
            updated_prompt = await _optimizer(sessions, prompts[0])
            return [
                {
                    **prompts[0],
                    "prompt": updated_prompt,
                }
            ]

        class Classify(BaseModel):
            """Classify which prompts merit updating for this conversation."""

            reasoning: str = Field(description="Reasoning for which prompts to update.")

            which: list[str] = Field(
                description=f"List of prompt names that should be updated. Must be among {choices}"
            )

            @model_validator(mode="after")
            def validate_choices(self) -> "Classify":
                invalid = set(self.which) - set(choices)
                if invalid:
                    raise ValueError(
                        f"Invalid choices: {invalid}. Must be among: {choices}"
                    )
                return self

        classifier = create_extractor(model, tools=[Classify], tool_choice="Classify")
        prompt_joined_content = "".join(
            f"{p['name']}: {p['prompt']}\n" for p in prompts
        )
        classification_prompt = f"""Analyze the following sessions and decide which prompts 
ought to be updated to improve the performance on future sessions:

{sessions_str}

Below are the prompts being optimized:
{prompt_joined_content}

Return JSON with "which": [...], listing the names of prompts that need updates."""
        result = await classifier.ainvoke(classification_prompt)
        to_update = result["responses"][0].which

        which_to_update = [p for p in prompts if p["name"] in to_update]

        # For each chosen prompt, call the single-prompt optimizer
        updated_results = await asyncio.gather(
            *[_optimizer(sessions, prompt=p) for p in which_to_update]
        )

        # Merge updated prompt text back into original prompt objects
        updated_map = {
            p["name"]: new_text for p, new_text in zip(which_to_update, updated_results)
        }

        final_list = []
        for p in prompts:
            if p["name"] in updated_map:
                final_list.append({**p, "prompt": updated_map[p["name"]]})
            else:
                final_list.append(p)

        return final_list

    return process_multi_prompt_sessions
