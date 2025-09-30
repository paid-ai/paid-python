from typing import Any, Optional

from ..tracing import get_paid_tracer, logger, paid_external_agent_id_var, paid_external_customer_id_var, paid_token_var
from agents import RunHooks
from agents.models import get_default_model
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


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
        token = paid_token_var.get()
        return external_customer_id, external_agent_id, token

    def _should_skip_tracing(self, external_customer_id: Optional[str], token: Optional[str]) -> bool:
        """Check if tracing should be skipped."""
        # Check if there's an active span (from Paid.trace())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, skipping LLM tracking.")
                return True
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, skipping LLM tracking")
                return True
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )
        return False

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        """Start a span for each individual LLM call and call user hooks."""
        # Call user hook first
        if self.user_hooks and hasattr(self.user_hooks, "on_llm_start"):
            await self.user_hooks.on_llm_start(context, agent, system_prompt, input_items)

        # Then handle Paid tracing
        try:
            external_customer_id, external_agent_id, token = self._get_context_vars()

            # Skip tracing if required context is missing
            if self._should_skip_tracing(external_customer_id, token):
                return

            # Get model name from agent
            model_name = str(agent.model if agent.model else get_default_model())

            # Start span for this LLM call
            span = self.tracer.start_span("trace.openai.agents")

            # Set initial attributes
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "agent_generation",
                "external_customer_id": external_customer_id,
                "token": token,
                "gen_ai.request.model": model_name,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id

            span.set_attributes(attributes)

            # Store span in context.context safely (check if user is using it)
            if not hasattr(context, "context"):
                context.context = {}
            elif not isinstance(context.context, dict):
                # User is using context.context for something else, create a nested dict
                context.context = {"_user_context": context.context}

            context.context["_paid_span"] = span

        except Exception as error:
            logger.error(f"Error in PaidAgentsHook.on_llm_start: {error}")

    async def on_llm_end(self, context, agent, response) -> None:
        """End the span and capture usage data for this LLM call, then call user hooks."""
        # Handle Paid tracing first
        try:
            # Get the span we stored in on_llm_start
            span = context.context.get("_paid_span") if hasattr(context, "context") else None
            if span:
                # Get usage data from the response
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage

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

        except Exception as error:
            logger.error(f"Error in PaidAgentsHook.on_llm_end: {error}")
            # Try to end span on error
            try:
                span = context.context.get("_paid_span") if hasattr(context, "context") else None
                if span:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(error)
                    span.end()
            except:
                pass
        finally:
            if hasattr(context, "context") and isinstance(context.context, dict):
                # Clean up span from context
                if "_paid_span" in context.context:
                    del context.context["_paid_span"]
                if "_user_context" in context.context:
                    context.context = context.context["_user_context"]

        # Call user hook after tracing
        if self.user_hooks and hasattr(self.user_hooks, "on_llm_end"):
            await self.user_hooks.on_llm_end(context, agent, response)

    async def on_agent_start(self, context, agent) -> None:
        """Call user hooks' on_agent_start methods."""
        if self.user_hooks and hasattr(self.user_hooks, "on_agent_start"):
            await self.user_hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context, agent, output) -> None:
        """Call user hooks' on_agent_end methods."""
        if self.user_hooks and hasattr(self.user_hooks, "on_agent_end"):
            await self.user_hooks.on_agent_end(context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        """Call user hooks' on_handoff methods."""
        if self.user_hooks and hasattr(self.user_hooks, "on_handoff"):
            await self.user_hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context, agent, tool) -> None:
        """Call user hooks' on_tool_start methods."""
        if self.user_hooks and hasattr(self.user_hooks, "on_tool_start"):
            await self.user_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Call user hooks' on_tool_end methods."""
        if self.user_hooks and hasattr(self.user_hooks, "on_tool_end"):
            await self.user_hooks.on_tool_end(context, agent, tool, result)
