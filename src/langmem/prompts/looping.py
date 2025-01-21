import asyncio
import re
import typing

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langmem import utils
from langmem.prompts.stateless import PromptMemoryMultiple
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor
from typing_extensions import TypedDict

KINDS = typing.Literal["gradient", "metaprompt", "prompt_memory"]


class Prompt(TypedDict, total=False):
    name: str
    prompt: str
    update_instructions: str
    when_to_update: str | None


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional["GradientOptimizerConfig"] = None,
): ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional["MetapromptOptimizerConfig"] = None,
): ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
): ...


def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: KINDS = "gradient",
    config: typing.Union[
        "GradientOptimizerConfig", "MetapromptOptimizerConfig", None
    ] = None,
) -> typing.Callable[
    [list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]], str | Prompt],
    typing.Awaitable[str],
]:
    if kind == "gradient":
        return create_gradient_prompt_optimizer(model, config)  # type: ignore
    elif kind == "metaprompt":
        return create_metaprompt_optimizer(model, config)  # type: ignore
    elif kind == "prompt_memory":
        return PromptMemoryMultiple(model).areflect  # type: ignore
    else:
        raise NotImplementedError(
            f"Unsupported optimizer kind: {kind}.\nExpected one of {KINDS}"
        )


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional["GradientOptimizerConfig"] = None,
): ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional["MetapromptOptimizerConfig"] = None,
): ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
): ...


def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: KINDS = "gradient",
    config: typing.Union[
        "GradientOptimizerConfig", "MetapromptOptimizerConfig", None
    ] = None,
) -> typing.Callable[
    [
        list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]],
        typing.Sequence[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    _optimizer = create_prompt_optimizer(model, kind, config)

    @ls.traceable
    async def process_multi_prompt_sessions(
        sessions: (
            list[list[AnyMessage]]
            | list[AnyMessage]
            | list[tuple[list[AnyMessage], str]]
            | tuple[list[AnyMessage], str]
            | str
        ),
        prompts: list[Prompt],
    ):
        choices = [p["name"] for p in prompts]
        sessions = utils.format_sessions(sessions)

        class Classify(BaseModel):
            """Classify which prompts merit updating for this conversation."""

            reasoning: str = Field(
                description="Reasoning for classifying which prompts merit updating. Cite any relevant evidence."
            )

            which: list[str] = Field(
                description=f"List of prompt names that should be updated. Must be one or more of: {choices}"
            )

            @model_validator(mode="after")
            def validate_choices(self) -> "Classify":
                invalid = set(self.which) - set(choices)
                if invalid:
                    raise ValueError(
                        f"Invalid choices: {invalid}. Must be one of: {choices}"
                    )
                return self

        classifier = create_extractor(model, tools=[Classify], tool_choice="Classify")
        prompts_str = "\n\n".join(f"{p['name']}: {p['prompt']}" for p in prompts)
        result = await classifier.ainvoke(
            f"""Analyze the following sessions and decide which prompts ought to be updated to improve the performance on future sessions:
{sessions}

Below are the prompts being optimized:
{prompts_str}
Consider any instructions on when_to_update when making a decision.
"""
        )
        to_update = result["responses"][0].which
        which_to_update = [p for p in prompts if p["name"] in to_update]
        results = await asyncio.gather(
            *(_optimizer(sessions, prompt=p) for p in which_to_update)
        )
        updated = {p["name"]: r for p, r in zip(which_to_update, results)}
        # Return the final prompts
        final = []
        for p in prompts:
            if p["name"] in updated:
                final.append({**p, "prompt": updated[p["name"]]})
            else:
                final.append(p)
        return final

    return process_multi_prompt_sessions


DEFAULT_MAX_REFLECTION_STEPS = 5
DEFAULT_MIN_REFLECTION_STEPS = 1

DEFAULT_GRADIENT_PROMPT = """You are reviewing the performance of an AI assistant in a given interaction. 

## Instructions

The current prompt that was used for the session is provided below.

<current_prompt>
{prompt}
</current_prompt>

The developer provided the following instructions around when and how to update the prompt:

<update_instructions>
{update_instructions}
</update_instructions>

## Session data

Analyze the following sessions (and any associated user feedback) (either conversations with a user or other work that was performed by the assistant):

<sessions>
{sessions}
</sessions>

## Feedback

The following feedback is provided for this session:

<feedback>
{feedback}
</feedback>

## Task

Analyze the conversation, including the user’s request and the assistant’s response, and evaluate:
1. How effectively the assistant fulfilled the user’s intent.
2. Where the assistant might have deviated from user expectations or the desired outcome.
3. Specific areas (correctness, completeness, style, tone, alignment, etc.) that need improvement.

If the prompt seems to do well, then no further action is needed. We ONLY recommend updates if there is evidence of failures.
When failures occur, we want to recommend the minimal required changes to fix the problem.

Focus on actionable changes and be concrete.

1. Summarize the key successes and failures in the assistant’s response. 
2. Identify which failure mode(s) best describe the issues (examples: style mismatch, unclear or incomplete instructions, flawed logic or reasoning, hallucination, etc.).
3. Based on these failure modes, recommend the most suitable edit strategy. For example, consider::
   - Use synthetic few-shot examples for style or clarifying decision boundaries.
   - Use explicit instruction updates for conditionals, rules, or logic fixes.
   - Provide step-by-step reasoning guidelines for multi-step logic problems.
4. Provide detailed, concrete suggestions for how to update the prompt accordingly.

But remember, the final updated prompt should only be changed if there is evidence of poor performance, and our recommendations should be minimally invasive.
Do not recommend generic changes that aren't clearly linked to failure modes.

First think through the conversation and critique the current behavior.
If you believe the prompt needs to further adapt to the target context, provide precise recommendations.
Otherwise, mark `warrants_adjustment` as False and respond with 'No recommendations.'"""


DEFAULT_GRADIENT_METAPROMPT = """You are optimizing a prompt to handle its target task more effectively.

<current_prompt>
{current_prompt}
</current_prompt>

We hypothesize the current prompt underperforms for these reasons:

<hypotheses>
{hypotheses}
</hypotheses>

Based on these hypotheses, we recommend the following adjustments:

<recommendations>
{recommendations}
</recommendations>

Respond with the updated prompt. Remember to ONLY make changes that are clearly necessary. Aim to be minimally invasive:"""


class GradientOptimizerConfig(TypedDict, total=False):
    """Configuration for the gradient optimizer."""

    gradient_prompt: str
    metaprompt: str
    max_reflection_steps: int
    min_reflection_steps: int


def create_gradient_prompt_optimizer(
    model: str | BaseChatModel, config: GradientOptimizerConfig | None = None
):
    config = config or GradientOptimizerConfig()
    config = GradientOptimizerConfig(
        gradient_prompt=config.get("gradient_prompt", DEFAULT_GRADIENT_PROMPT),
        metaprompt=config.get("metaprompt", DEFAULT_GRADIENT_METAPROMPT),
        max_reflection_steps=config.get(
            "max_reflection_steps", DEFAULT_MAX_REFLECTION_STEPS
        ),
        min_reflection_steps=config.get(
            "min_reflection_steps", DEFAULT_MIN_REFLECTION_STEPS
        ),
    )

    @ls.traceable
    async def react_agent(
        model: str | BaseChatModel, inputs: str, max_steps: int, min_steps: int
    ):
        messages = [
            {"role": "user", "content": inputs},
        ]
        just_think = create_extractor(
            model,
            tools=[think, critique],
            tool_choice="any",
        )
        any_chain = create_extractor(
            model,
            tools=[think, critique, recommend],
            tool_choice="any",
        )
        final_chain = create_extractor(
            model,
            tools=[recommend],
            tool_choice="recommend",
        )
        for ix in range(max_steps):
            if ix == max_steps - 1:
                chain = final_chain
            elif ix < min_steps:
                chain = just_think
            else:
                chain = any_chain
            response = await chain.ainvoke(messages)
            final_response = next(
                (r for r in response["responses"] if r.__repr_name__() == "recommend"),
                None,
            )
            if final_response:
                return final_response
            msg: AIMessage = response["messages"][-1]
            messages.append(msg)
            ids = [tc["id"] for tc in (msg.tool_calls or [])]
            for id_ in ids:
                messages.append({"role": "tool", "content": "", "tool_call_id": id_})

        raise ValueError(f"Failed to generate response after {n} attempts")

    def think(thought: str):
        """First call this to reason over complicated domains, uncover hidden input/output patterns, theorize why previous hypotheses failed, and creatively conduct error analyses (e.g., deep diagnostics/recursively analyzing "why" something failed). List characteristics of the data generating process you failed to notice before. Hypothesize fixes, prioritize, critique, and repeat calling this tool until you are confident in your next solution."""
        return "Take as much time as you need! If you're stuck, take a step back and try something new."

    def critique(criticism: str):
        """Then, critique your thoughts and hypotheses. Identify flaws in your previous hypotheses and current thinking. Forecast why the hypotheses won't work. Get to the bottom of what is really driving the problem. This tool returns no new information but gives you more time to plan."""
        return "Take as much time as you need. It's important to think through different strategies."

    def recommend(
        warrants_adjustment: bool,
        hypotheses: str | None = None,
        full_recommendations: str | None = None,
    ):
        """Once you've finished thinking, decide whether the session indicates the prompt should be adjusted.
        If so, hypothesize why the prompt is inadequate and provide a clear and specific recommendation for how to improve the prompt.
        Specify the precise changes and edit strategy. Specify what things not to touch.
        If not, respond with 'No recommendations.'"""

    @ls.traceable
    async def update_prompt(
        hypotheses: str,
        recommendations: str,
        current_prompt: str,
        update_instructions: str,
    ):
        schema = _prompt_schema(current_prompt)

        extractor = create_extractor(
            model,
            tools=[schema],
            tool_choice="OptimizedPromptOutput",
        )
        result = await extractor.ainvoke(
            config["metaprompt"].format(
                current_prompt=current_prompt,
                recommendations=recommendations,
                hypotheses=hypotheses,
                update_instructions=update_instructions,
            )
        )
        return result["responses"][0].improved_prompt

    @ls.traceable(metadata={"kind": "gradient"})
    async def optimize_prompt(
        sessions: (
            list[list[AnyMessage]]
            | list[AnyMessage]
            | list[tuple[list[AnyMessage], dict[str, str]]]
            | tuple[list[AnyMessage], dict[str, str]]
            | str
        ),
        prompt: str | Prompt,
    ):
        prompt_str = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
        if not sessions:
            return prompt_str
        elif isinstance(sessions, str):
            sessions = sessions
        else:
            sessions = utils.format_sessions(sessions)

        feedback = "" if isinstance(prompt, str) else prompt.get("feedback", "")
        update_instructions = (
            "" if isinstance(prompt, str) else prompt.get("update_instructions", "")
        )

        inputs = config["gradient_prompt"].format(
            sessions=sessions,
            feedback=feedback,
            prompt=prompt_str,
            update_instructions=update_instructions,
        )
        result = await react_agent(
            model,
            inputs,
            max_steps=config["max_reflection_steps"],
            min_steps=config["min_reflection_steps"],
        )
        if result.warrants_adjustment:
            return await update_prompt(
                result.hypotheses,
                result.full_recommendations,
                prompt_str,
                update_instructions,
            )
        return prompt_str

    return optimize_prompt


class MetapromptOptimizerConfig(TypedDict, total=False):
    """Configuration for the metaprompt optimizer."""

    metaprompt: str
    max_reflection_steps: int
    min_reflection_steps: int


DEFAULT_METAPROMPT = """You are helping an AI assistant learn by optimizing its prompt.

## Background

Below is the current prompt:

<current_prompt>
{prompt}
</current_prompt>

The developer provided these instructions regarding when/how to update:

<update_instructions>
{update_instructions}
</update_instructions>

## Session Data
Analyze the session(s) (and any user feedback) below:

<sessions>
{sessions}
</sessions>

Here is the user's feedback:
<feedback>
{feedback}
</feedback>

## Instructions

1. Reflect on the agent's performance on the given session(s) and identify any real failure modes (e.g., style mismatch, unclear or incomplete instructions, flawed reasoning, etc.).
2. Recommend the minimal changes necessary to address any real failures. If the prompt performs perfectly, simply respond with the original prompt without making any changes.
3. Retain any f-string variables in the existing prompt exactly as they are (e.g. {{variable_name}}).

IFF changes are warranted, focus on actionable edits. Be concrete. Edits should be appropriate for the identified failure modes. For example, consider synthetic few-shot examples for style or clarifying decision boundaries, or adding or modifying explicit instructions for conditionals, rules, or logic fixes; or provide step-by-step reasoning guidelines for multi-step logic problems if the model is failing to reason appropriately."""


def create_metaprompt_optimizer(
    model: str | BaseChatModel, config: MetapromptOptimizerConfig | None = None
):
    """
    Creates a single-step prompt-updater.  If reflect_and_critique=True and max_reflection_steps>1,
    it does some "think/critique" calls before the final 'optimized prompt' call.
    Otherwise it just does one direct call to produce the updated prompt.
    """
    # Default config
    config = config or {}
    final_config = MetapromptOptimizerConfig(
        metaprompt=config.get("metaprompt", DEFAULT_METAPROMPT),
        max_reflection_steps=config.get(
            "max_reflection_steps", DEFAULT_MAX_REFLECTION_STEPS
        ),
        min_reflection_steps=config.get(
            "min_reflection_steps", DEFAULT_MIN_REFLECTION_STEPS
        ),
    )

    # Tools used if reflect_and_critique=True
    def think(thought: str):
        """Reflect deeply or hypothesize reasons why the prompt might fail and how to fix it. Think about what the model may perpetually misunderstand. Think about what instructions or examples would be helpful for the model to ALWAYS see so it doesn't respond incorrectly."""
        return "Reflecting on possible weaknesses, analyzing them in detail..."

    def critique(criticism: str):
        """Critique or refine prior thoughts; identify flaws or improvements in the line of reasoning. Critique unnecessary changes or flaws in logic."""
        return "Critiquing prior reasoning to refine or correct mistakes..."

    @ls.traceable
    async def reflect_then_update(
        sessions_str: str,
        prompt: str,
        feedback: str,
        update_instructions: str,
        max_steps: int,
        min_steps: int,
    ):
        # We'll store the conversation messages
        messages = [
            {
                "role": "user",
                "content": final_config["metaprompt"].format(
                    prompt=prompt,
                    update_instructions=update_instructions,
                    sessions=sessions_str,
                    feedback=feedback,
                ),
            },
        ]

        reflect_chain = create_extractor(
            model, tools=[think, critique], tool_choice="any"
        )
        schema_tool = _prompt_schema(prompt)
        any_chain = create_extractor(
            model, tools=[think, critique, schema_tool], tool_choice="any"
        )

        final_chain = create_extractor(
            model,
            tools=[schema_tool],
            tool_choice="OptimizedPromptOutput",
        )

        for ix in range(max_steps):
            if ix < max_steps - 1:
                if ix < min_steps - 1:
                    response = await reflect_chain.ainvoke(messages)
                else:
                    response = await any_chain.ainvoke(messages)
            else:
                # final pass calls the prompt schema
                response = await final_chain.ainvoke(messages)
                return response["responses"][0]

            # Add last AI message to conversation
            ai_msg: AIMessage = response["messages"][-1]
            messages.append(ai_msg)

            # Also add any tool calls
            for tc in ai_msg.tool_calls or []:
                messages.append(
                    {"role": "tool", "content": "", "tool_call_id": tc["id"]}
                )

        # If we somehow exit the loop without returning, raise an error
        raise RuntimeError("Exceeded reflection steps without final output")

    @ls.traceable(metadata={"kind": "metaprompt"})
    async def optimize_prompt(
        sessions: (
            list[list[AnyMessage]]
            | list[AnyMessage]
            | list[tuple[list[AnyMessage], dict[str, str]]]
            | tuple[list[AnyMessage], dict[str, str]]
            | str
        ),
        prompt: str | Prompt,
    ) -> str:
        """
        Main user-facing function:
        1. Formats sessions into a string if needed.
        2. Optionally does reflection steps,
        3. Then final single-step call to produce "analysis" & "improved_prompt".
        4. If improved_prompt says "No recommendations." or is empty,
           we return original prompt; else we return the new prompt.
        """
        prompt_str = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
        if not sessions:
            return prompt_str
        feedback = "" if isinstance(prompt, str) else prompt.get("feedback", "")
        update_instructions = (
            "" if isinstance(prompt, str) else prompt.get("update_instructions", "")
        )

        if isinstance(sessions, str):
            sessions_str = sessions
        else:
            sessions_str = utils.format_sessions(sessions)
        result_obj = await reflect_then_update(
            sessions_str,
            prompt_str,
            feedback,
            update_instructions,
            final_config["max_reflection_steps"],
            final_config["min_reflection_steps"],
        )

        improved_prompt = result_obj.improved_prompt
        if not improved_prompt or improved_prompt.strip().lower().startswith(
            "no recommend"
        ):
            # No real changes
            return prompt_str
        else:
            return improved_prompt

    return optimize_prompt


def _prompt_schema(
    original_prompt: str,
):
    required_variables = set(re.findall(r"\{(.+?)\}", original_prompt, re.MULTILINE))
    if required_variables:
        variables_str = ", ".join(f"{{{var}}}" for var in required_variables)
        prompt_description = (
            f" The prompt section being optimized contains the following f-string variables to be templated in: {variables_str}."
            " You must retain all of these variables in your improved prompt. No other input variables are allowed."
        )
    else:
        prompt_description = (
            " The prompt section being optimized contains no input f-string variables."
            " Any brackets {{ foo }} you emit will be escaped and not used."
        )

    pipeline = utils._get_var_healer(set(required_variables), all_required=True)

    class OptimizedPromptOutput(BaseModel):
        """Schema for the optimized prompt output."""

        analysis: str = Field(
            description="First, analyze the current results and plan improvements to reconcile them."
        )
        improved_prompt: typing.Optional[str] = Field(
            description="Finally, generate the full updated prompt to address the identified issues. "
            f" <TO_OPTIMIZE> and </TO_OPTIMIZE> tags, in f-string format. Do not include <TO_OPTIMIZE> in your response. {prompt_description}"
        )

        @model_validator(mode="before")
        @classmethod
        def validate_input_variables(cls, data: typing.Any) -> typing.Any:
            assert "improved_prompt" in data
            data["improved_prompt"] = pipeline(data["improved_prompt"])
            return data

    return OptimizedPromptOutput
