from typing import Any, Optional

from opentelemetry.trace import Span, Status, StatusCode

from paid.logger import logger
from paid.tracing.tracing import (
    get_paid_tracer,
)

try:
    from agents import RunHooks
    from agents.models import get_default_model
except ImportError:
    raise ImportError(
        "openai-agents package is a peer-dependency. To use the Paid wrapper around openai-agents "
        "you're assumed to already have openai-agents package installed."
    )

# Global dictionary to store spans keyed by context object ID
# This avoids polluting user's context.context and works across async boundaries
_paid_span_store: dict[int, Span] = {}


class PaidOpenAIAgentsHook(RunHooks[Any]):
    """
    Hook that traces individual LLM calls for OpenAI Agents SDK with Paid tracking.

    Can optionally wrap user-provided hooks to combine Paid tracking with custom behavior.
    """

    def __init__(self, user_hooks: Optional[RunHooks[Any]] = None):
        """
        Initialize PaidAgentsHook.

        Args:
            user_hooks: Optional user-provided RunHooks to combine with Paid tracking

        Usage:
            @paid_tracing("<ext_customer_id>", "<ext_agent_id>")
            def run_agent():
                hook = PaidAgentsHook()
                return Runner.run_streamed(agent, input, hooks=hook)
            run_agent()

            # With user hooks
            class MyHook(RunHooks):
                async def on_llm_start(self, context, agent, system_prompt, input_items):
                    print("Starting LLM call!")

            my_hook = MyHook()
            hook = PaidAgentsHook(user_hooks=my_hook)
        """
        super().__init__()
        self.user_hooks = user_hooks

    def _start_span(self, context, agent, hook_name) -> None:
        try:
            tracer = get_paid_tracer()

            # Get model name from agent
            model_name = str(agent.model if agent.model else get_default_model())

            # Start span for this LLM call
            span = tracer.start_span(f"openai.agents.{hook_name}")

            # Set initial attributes
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": f"{hook_name}",
                "gen_ai.request.model": model_name,
            }

            span.set_attributes(attributes)

            # Store span in global dict keyed by context object ID
            # This works across async boundaries without polluting user's context
            context_id = id(context)
            _paid_span_store[context_id] = span

        except Exception as error:
            logger.error(f"Error while starting span in PaidAgentsHook.{hook_name}: {error}")

    def _end_span(self, context, hook_name):
        try:
            # Retrieve span from global dict using context object ID
            context_id = id(context)
            span = _paid_span_store.get(context_id)

            if span:
                # Get usage data from the response
                if hasattr(context, "usage") and context.usage:
                    usage = context.usage

                    usage_attributes = {
                        "gen_ai.usage.input_tokens": usage.input_tokens,
                        "gen_ai.usage.output_tokens": usage.output_tokens,
                    }

                    # Add detailed usage if available
                    if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                        usage_attributes["gen_ai.usage.cached_input_tokens"] = usage.input_tokens_details.cached_tokens

                    if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
                        usage_attributes["gen_ai.usage.reasoning_output_tokens"] = (
                            usage.output_tokens_details.reasoning_tokens
                        )

                    span.set_attributes(usage_attributes)
                    span.set_status(Status(StatusCode.OK))
                else:
                    # No usage data available
                    span.set_status(Status(StatusCode.ERROR, "No usage available"))

                span.end()

                # Clean up from global dict
                del _paid_span_store[context_id]

        except Exception as error:
            # Try to end span on error
            logger.error(f"Error while ending span in PaidAgentsHook.{hook_name}: {error}")
            try:
                context_id = id(context)
                span = _paid_span_store.get(context_id)
                if span:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(error)
                    span.end()
                    del _paid_span_store[context_id]
            except:
                logger.error(f"Failed to end span after error in PaidAgentsHook.{hook_name}")

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        if self.user_hooks and hasattr(self.user_hooks, "on_llm_start"):
            await self.user_hooks.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(self, context, agent, response) -> None:
        if self.user_hooks and hasattr(self.user_hooks, "on_llm_end"):
            await self.user_hooks.on_llm_end(context, agent, response)

    async def on_agent_start(self, context, agent) -> None:
        """Start a span for agent operations and call user hooks."""
        if self.user_hooks and hasattr(self.user_hooks, "on_agent_start"):
            await self.user_hooks.on_agent_start(context, agent)

        self._start_span(context, agent, "on_agent")

    async def on_agent_end(self, context, agent, output) -> None:
        """End the span for agent operations and call user hooks."""
        self._end_span(context, "on_agent")

        if self.user_hooks and hasattr(self.user_hooks, "on_agent_end"):
            await self.user_hooks.on_agent_end(context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        if self.user_hooks and hasattr(self.user_hooks, "on_handoff"):
            await self.user_hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context, agent, tool) -> None:
        if self.user_hooks and hasattr(self.user_hooks, "on_tool_start"):
            await self.user_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        if self.user_hooks and hasattr(self.user_hooks, "on_tool_end"):
            await self.user_hooks.on_tool_end(context, agent, tool, result)
