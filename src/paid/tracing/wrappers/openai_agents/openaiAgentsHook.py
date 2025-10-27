from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from paid.tracing.tracing import (
    get_paid_tracer,
    logger,
    paid_external_agent_id_var,
    paid_external_customer_id_var,
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
_paid_span_store: dict[int, trace.Span] = {}


class PaidOpenAIAgentsHook(RunHooks[Any]):
    """
    Hook that traces individual LLM calls for OpenAI Agents SDK with Paid tracking.

    Can optionally wrap user-provided hooks to combine Paid tracking with custom behavior.
    """

    def __init__(self, user_hooks: Optional[RunHooks[Any]] = None, optional_tracing: bool = False):
        """
        Initialize PaidAgentsHook.

        Args:
            user_hooks: Optional user-provided RunHooks to combine with Paid tracking
            optional_tracing: If True, gracefully skip tracing when context is missing.
                             If False, raise errors when tracing context is not available.

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

            # Optional tracing (won't raise errors if context missing)
            hook = PaidAgentsHook(optional_tracing=True)
        """
        super().__init__()
        self.tracer = get_paid_tracer()
        self.optional_tracing = optional_tracing
        self.user_hooks = user_hooks

    def _get_context_vars(self):
        """Get tracing context from context variables set by Paid.trace()."""
        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        return external_customer_id, external_agent_id

    def _should_skip_tracing(self, external_customer_id: Optional[str]) -> bool:
        """Check if tracing should be skipped."""
        # Check if there's an active span (from Paid.trace())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, skipping LLM tracking.")
                return True
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        if not external_customer_id:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id, skipping LLM tracking")
                return True
            raise RuntimeError(
                "Missing required tracing information: external_customer_id."
                " Make sure to call this method from Paid.trace()."
            )
        return False

    def _start_span(self, context, agent, hook_name) -> None:
        try:
            external_customer_id, external_agent_id = self._get_context_vars()

            # Skip tracing if required context is missing
            if self._should_skip_tracing(external_customer_id):
                return

            # Get model name from agent
            model_name = str(agent.model if agent.model else get_default_model())

            # Start span for this LLM call
            span = self.tracer.start_span(f"openai.agents.{hook_name}")
            logger.debug(f"{hook_name} : started span")

            # Set initial attributes
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": f"{hook_name}",
                "external_customer_id": external_customer_id,
                "gen_ai.request.model": model_name,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id

            span.set_attributes(attributes)

            # Store span in global dict keyed by context object ID
            # This works across async boundaries without polluting user's context
            context_id = id(context)
            _paid_span_store[context_id] = span
            logger.debug(f"_start_span: Stored span for context ID {context_id}")

        except Exception as error:
            logger.error(f"Error while starting span in PaidAgentsHook.{hook_name}: {error}")

    def _end_span(self, context, hook_name):
        try:
            # Retrieve span from global dict using context object ID
            context_id = id(context)
            span = _paid_span_store.get(context_id)
            logger.debug(f"_end_span: Retrieved span for context ID {context_id}: {span}")

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
                logger.debug(f"{hook_name} : ended span")

                # Clean up from global dict
                del _paid_span_store[context_id]
                logger.debug(f"_end_span: Cleaned up span for context ID {context_id}")
            else:
                logger.warning(f"_end_span: No span found for context ID {context_id}")

        except Exception as error:
            logger.error(f"Error while ending span in PaidAgentsHook.{hook_name}_end: {error}")
            # Try to end span on error
            try:
                context_id = id(context)
                span = _paid_span_store.get(context_id)
                if span:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(error)
                    span.end()
                    del _paid_span_store[context_id]
            except:
                pass

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        logger.debug(f"on_llm_start : context_usage : {getattr(context, 'usage', None)}")

        if self.user_hooks and hasattr(self.user_hooks, "on_llm_start"):
            await self.user_hooks.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(self, context, agent, response) -> None:
        logger.debug(
            f"on_llm_end : context_usage : {getattr(context, 'usage', None)} : response_usage : {getattr(response, 'usage', None)}"
        )

        if self.user_hooks and hasattr(self.user_hooks, "on_llm_end"):
            await self.user_hooks.on_llm_end(context, agent, response)

    async def on_agent_start(self, context, agent) -> None:
        """Start a span for agent operations and call user hooks."""
        logger.debug(f"on_agent_start : context_usage : {getattr(context, 'usage', None)}")

        if self.user_hooks and hasattr(self.user_hooks, "on_agent_start"):
            await self.user_hooks.on_agent_start(context, agent)

        self._start_span(context, agent, "on_agent")

    async def on_agent_end(self, context, agent, output) -> None:
        """End the span for agent operations and call user hooks."""
        logger.debug(f"on_agent_end : context_usage : {getattr(context, 'usage', None)}")

        self._end_span(context, "on_agent")

        if self.user_hooks and hasattr(self.user_hooks, "on_agent_end"):
            await self.user_hooks.on_agent_end(context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        logger.debug(f"on_handoff : context_usage : {getattr(context, 'usage', None)}")
        if self.user_hooks and hasattr(self.user_hooks, "on_handoff"):
            await self.user_hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context, agent, tool) -> None:
        logger.debug(f"on_tool_start : context_usage : {getattr(context, 'usage', None)}")

        if self.user_hooks and hasattr(self.user_hooks, "on_tool_start"):
            await self.user_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        logger.debug(f"on_tool_end : context_usage : {getattr(context, 'usage', None)}")

        if self.user_hooks and hasattr(self.user_hooks, "on_tool_end"):
            await self.user_hooks.on_tool_end(context, agent, tool, result)
