import time
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from ..tracing import logger, paid_external_customer_id_var, paid_token_var
from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
from langchain_core.outputs import LLMResult  # type: ignore
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class PaidLangChainCallback(BaseCallbackHandler):
    """
    LangChain callback handler that integrates with Paid tracing infrastructure.

    This handler creates OpenTelemetry spans for LangChain operations including:
    - LLM calls (chat completions, completions)
    - Chain executions
    - Tool usage
    - Retrieval operations
    - Agent actions

    Usage:
        # Initialize with your existing Paid setup
        callback_handler = PaidCallbackHandler()

        # Use with LangChain
        llm = ChatOpenAI(callbacks=[callback_handler])
        response = llm.invoke("Hello world")

        # Or with chains
        chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback_handler])
        result = chain.run("input text")
    """

    def __init__(self):
        """Initialize the callback handler."""
        super().__init__()
        self.tracer = trace.get_tracer("paid.python")
        self._spans: Dict[str, Any] = {}  # Track active spans by run_id
        self._start_times: Dict[str, float] = {}  # Track start times

    def _get_span_name(self, operation: str, name: Optional[str] = None) -> str:
        """Generate a consistent span name."""
        if name:
            return f"{operation} {name}"
        return f"{operation}"

    def _start_span(self, run_id: UUID, span_name: str, **attributes: Any) -> Optional[Any]:
        """Start a new span and store it."""
        # Check if there's an active span (from capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            logger.warning("No active span found - LangChain operations will not be traced")
            return None

        # Get context variables
        external_customer_id = paid_external_customer_id_var.get()
        token = paid_token_var.get()

        # Create child span
        span = self.tracer.start_span(span_name)

        # Set common attributes
        base_attributes = {
            "langchain.operation": span_name.split()[0].split(".")[-1],
            "langchain.run_id": str(run_id),
        }

        if external_customer_id:
            base_attributes["external_customer_id"] = external_customer_id
        if token:
            base_attributes["token"] = token

        # Add custom attributes
        base_attributes.update(attributes)
        span.set_attributes(base_attributes)

        # Store span and start time
        self._spans[str(run_id)] = span
        self._start_times[str(run_id)] = time.time()

        return span

    def _end_span(self, run_id: UUID, error: Optional[BaseException] = None, **attributes):
        """End a span and clean up."""
        span_key = str(run_id)
        span = self._spans.get(span_key)

        if not span:
            return

        try:
            # Add duration
            if span_key in self._start_times:
                duration = time.time() - self._start_times[span_key]
                span.set_attribute("langchain.duration_ms", int(duration * 1000))
                del self._start_times[span_key]

            # Add final attributes
            span.set_attributes(attributes)

            # Set status
            if error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
            else:
                span.set_status(Status(StatusCode.OK))

        finally:
            span.end()
            del self._spans[span_key]

    # LLM Callbacks
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts running."""
        if not metadata:
            logger.warning("No metadata provided for LLM start")
            return None

        model_type = metadata.get("ls_model_type", "unknown")
        model_name = metadata.get("ls_model_name", "unknown")
        logger.info(f"LLM start: {metadata}")
        logger.info(f"model_name: {model_name}")
        span_name = self._get_span_name(f"trace.{model_type}", model_name)

        attributes = {
            "gen_ai.system": self._extract_provider(serialized),
            "gen_ai.operation.name": metadata["ls_model_type"],
            "gen_ai.request.model": model_name,
            "langchain.prompts.count": len(prompts),
        }

        # Add prompt content (be careful with size)
        if prompts:
            # Only add first prompt and truncate if too long
            first_prompt = prompts[0]
            if len(first_prompt) > 1000:
                first_prompt = first_prompt[:1000] + "..."
            attributes["langchain.prompt"] = first_prompt

        if tags:
            attributes["langchain.tags"] = ",".join(tags)

        if metadata:
            # Add safe metadata (avoid large objects)
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[f"langchain.metadata.{key}"] = value

        self._start_span(run_id, span_name, **attributes)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM ends running."""
        attributes = {}

        # Add token usage if available
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            if "prompt_tokens" in usage:
                attributes["gen_ai.usage.input_tokens"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                attributes["gen_ai.usage.output_tokens"] = usage["completion_tokens"]
            if "cached_input_tokens" in usage:
                attributes["gen_ai.usage.cached_input_tokens"] = usage["cached_input_tokens"]
            if "reasoning_output_tokens" in usage:
                attributes["gen_ai.usage.reasoning_output_tokens"] = usage["reasoning_output_tokens"]
            if "total_tokens" in usage:
                attributes["gen_ai.usage.total_tokens"] = usage["total_tokens"]

        # Add response count
        attributes["langchain.generations.count"] = len(response.generations)

        # Add model from response if available
        if response.llm_output and "model_name" in response.llm_output:
            attributes["gen_ai.response.model"] = response.llm_output["model_name"]

        self._end_span(run_id, **attributes)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM errors."""
        self._end_span(run_id, error=error)

    # Chain Callbacks
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain starts running."""
        chain_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        span_name = self._get_span_name("chain", chain_name)

        attributes = {
            "langchain.chain.name": chain_name,
            "langchain.inputs.count": len(inputs),
        }

        # Add input keys (but not values for privacy)
        if inputs:
            attributes["langchain.input_keys"] = ",".join(inputs.keys())

        if tags:
            attributes["langchain.tags"] = ",".join(tags)

        self._start_span(run_id, span_name, **attributes)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain ends running."""
        attributes: Dict[str, Any] = {
            "langchain.outputs.count": len(outputs),
        }

        # Add output keys (but not values for privacy)
        if outputs:
            attributes["langchain.output_keys"] = ",".join(outputs.keys())

        self._end_span(run_id, **attributes)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain errors."""
        self._end_span(run_id, error=error)

    # Tool Callbacks
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts running."""
        tool_name = serialized.get("name", "unknown")
        span_name = self._get_span_name("tool", tool_name)

        attributes = {
            "langchain.tool.name": tool_name,
        }

        # Add input (truncated for size)
        if input_str:
            if len(input_str) > 500:
                input_str = input_str[:500] + "..."
            attributes["langchain.tool.input"] = input_str

        if tags:
            attributes["langchain.tags"] = ",".join(tags)

        self._start_span(run_id, span_name, **attributes)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool ends running."""
        attributes: Dict[str, Any] = {}

        # Add output (truncated for size)
        if output:
            if len(output) > 500:
                output = output[:500] + "..."
            attributes["langchain.tool.output"] = output

        self._end_span(run_id, **attributes)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool errors."""
        self._end_span(run_id, error=error)

    # Retriever Callbacks
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when retriever starts running."""
        retriever_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        span_name = self._get_span_name("retriever", retriever_name)

        attributes = {
            "langchain.retriever.name": retriever_name,
        }

        # Add query (truncated for size)
        if query:
            if len(query) > 500:
                query = query[:500] + "..."
            attributes["langchain.retriever.query"] = query

        self._start_span(run_id, span_name, **attributes)

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when retriever ends running."""
        attributes: Dict[str, Any] = {
            "langchain.retriever.documents_count": len(documents),
        }

        self._end_span(run_id, **attributes)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when retriever errors."""
        self._end_span(run_id, error=error)

    def _extract_provider(self, serialized: Dict[str, Any]) -> str:
        """Extract the LLM provider from serialized data."""
        if not serialized or "id" not in serialized:
            return "unknown"

        id_parts = serialized["id"]
        if not id_parts:
            return "unknown"

        # Common patterns
        for part in id_parts:
            if "openai" in part.lower():
                return "openai"
            elif "anthropic" in part.lower():
                return "anthropic"
            elif "cohere" in part.lower():
                return "cohere"
            elif "huggingface" in part.lower():
                return "huggingface"

        return "unknown"


# Convenience function to create callback
def create_paid_callback() -> PaidLangChainCallback:
    """
    Create a PaidCallbackHandler instance.

    Returns:
        PaidCallbackHandler instance

    Example:
        callback = create_paid_callback()
        llm = ChatOpenAI(callbacks=[callback])
    """
    return PaidLangChainCallback()
