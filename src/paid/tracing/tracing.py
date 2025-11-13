# Initializing tracing for OTLP
import asyncio
import atexit
import os
import signal
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar, Union

import dotenv
from . import distributed_tracing
from .context_data import ContextData
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import NonRecordingSpan, NoOpTracerProvider, SpanContext, Status, StatusCode, TraceFlags

from paid.logger import logger

_ = dotenv.load_dotenv()
DEFAULT_COLLECTOR_ENDPOINT = (
    os.environ.get("PAID_OTEL_COLLECTOR_ENDPOINT") or "https://collector.agentpaid.io:4318/v1/traces"
)

T = TypeVar("T")


class _TokenStore:
    """Private token storage to enforce access through getter/setter."""

    __token: Optional[str] = None

    @classmethod
    def get(cls) -> Optional[str]:
        """Get the stored API token."""
        return cls.__token

    @classmethod
    def set(cls, token: str) -> None:
        """Set the API token."""
        cls.__token = token


def get_token() -> Optional[str]:
    """Get the stored API token."""
    return _TokenStore.get()


def set_token(token: str) -> None:
    """Set the API token."""
    _TokenStore.set(token)


# Isolated tracer provider for Paid - separate from any user OTEL setup
# Initialized at module load with defaults, never None (uses no-op provider if not initialized or API key isn't available)
paid_tracer_provider: Union[TracerProvider, NoOpTracerProvider] = NoOpTracerProvider()


class PaidSpanProcessor(SpanProcessor):
    """
    Span processor that:
    1. Prefixes all span names with 'paid.trace.'
    2. Automatically adds external_customer_id and external_agent_id attributes
       to all spans based on context variables set by the tracing decorator.
    """

    SPAN_NAME_PREFIX = "paid.trace."
    PROMPT_ATTRIBUTES_PREFIXES = {
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.request.messages",
        "gen_ai.response.messages",
        "llm.output_message",
        "llm.input_message",
        "output.value",
        "input.value",
    }

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span is started. Prefix the span name and add attributes."""
        # Prefix the span name
        if span.name and not span.name.startswith(self.SPAN_NAME_PREFIX):
            span.update_name(f"{self.SPAN_NAME_PREFIX}{span.name}")

        # Add customer and agent IDs from context
        customer_id = ContextData.get_context_key("external_customer_id")
        if customer_id:
            span.set_attribute("external_customer_id", customer_id)

        agent_id = ContextData.get_context_key("external_agent_id")
        if agent_id:
            span.set_attribute("external_agent_id", agent_id)

        metadata = ContextData.get_context_key("user_metadata")
        if metadata:
            metadata_attributes: dict[str, Any] = {}

            def flatten_dict(d: dict[str, Any], parent_key: str = "") -> None:
                """Recursively flatten nested dictionaries into dot-notation keys."""
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        flatten_dict(v, new_key)
                    else:
                        metadata_attributes[new_key] = v

            flatten_dict(metadata)

            # Add all flattened metadata attributes to the span
            for key, value in metadata_attributes.items():
                span.set_attribute(f"metadata.{key}", value)

    def on_end(self, span: ReadableSpan) -> None:
        """Filter out prompt and response contents unless explicitly asked to store"""
        store_prompt = ContextData.get_context_key("store_prompt")
        if store_prompt:
            return

        original_attributes = span.attributes

        if original_attributes:
            # Filter out prompt-related attributes
            filtered_attrs = {
                k: v
                for k, v in original_attributes.items()
                if not any(k.startswith(prefix) for prefix in self.PROMPT_ATTRIBUTES_PREFIXES)
            }
            # This works because the exporter reads attributes during serialization
            object.__setattr__(span, "_attributes", filtered_attrs)

    def shutdown(self) -> None:
        """Called when the processor is shut down. No action needed."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Called to force flush. Always returns True since there's nothing to flush."""
        return True


def setup_graceful_termination():
    def flush_traces():
        try:
            if not isinstance(paid_tracer_provider, NoOpTracerProvider) and not paid_tracer_provider.force_flush(10000):
                logger.error("OTEL force flush : timeout reached")
        except Exception as e:
            logger.error(f"Error flushing traces: {e}")

    def create_chained_signal_handler(signum: int):
        current_handler = signal.getsignal(signum)

        def chained_handler(_signum, frame):
            logger.warning(f"Received signal {_signum}, flushing traces")
            flush_traces()
            # Restore the original handler
            signal.signal(_signum, current_handler)
            # Re-raise the signal to let the original handler (or default) handle it
            os.kill(os.getpid(), _signum)

        return chained_handler

    try:
        # This is already done by default OTEL shutdown,
        # but user might turn that off - so register it explicitly
        atexit.register(flush_traces)

        # signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, create_chained_signal_handler(sig))
    except Exception as e:
        logger.warning(
            f"Could not set up termination handlers: {e}"
            "\nConsider calling initialize_tracing() from the main thread during app initialization if you don't already"
        )


def initialize_tracing(api_key: Optional[str] = None, collector_endpoint: Optional[str] = DEFAULT_COLLECTOR_ENDPOINT):
    """
    Initialize OpenTelemetry with OTLP exporter for Paid backend.

    Args:
        api_key: The API key for authentication. If not provided, will try to get from PAID_API_KEY environment variable.
        collector_endpoint: The OTLP collector endpoint URL.
    """
    global paid_tracer_provider

    if not collector_endpoint:
        collector_endpoint = DEFAULT_COLLECTOR_ENDPOINT

    try:
        if get_token() is not None:
            logger.warning("Tracing is already initialized - skipping re-initialization")
            return

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("PAID_API_KEY")
            if api_key is None:
                logger.error("API key must be provided via PAID_API_KEY environment variable")
                # don't throw - tracing should not break the app
                return

        set_token(api_key)

        resource = Resource(attributes={"api.key": api_key})
        # Create isolated tracer provider for Paid - don't use or modify global provider
        paid_tracer_provider = TracerProvider(resource=resource)

        # Add span processor to prefix span names and add customer/agent ID attributes
        paid_tracer_provider.add_span_processor(PaidSpanProcessor())

        # Set up OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=collector_endpoint,
            headers={},  # No additional headers needed for OTLP
        )

        # Use SimpleSpanProcessor for immediate span export.
        # There are problems with BatchSpanProcessor in some environments - ex. Airflow.
        # Airflow terminates processes before the batch is sent, losing traces.
        span_processor = SimpleSpanProcessor(otlp_exporter)
        paid_tracer_provider.add_span_processor(span_processor)

        setup_graceful_termination()  # doesn't throw

        logger.info("Paid tracing initialized successfully - collector at %s", collector_endpoint)
    except Exception as e:
        logger.error(f"Failed to initialize Paid tracing: {e}")
        # don't throw - tracing should not break the app


def get_paid_tracer() -> trace.Tracer:
    """
    Get the tracer from the isolated Paid tracer provider.

    Returns:
        The Paid tracer instance.

    Raises:
        RuntimeError: If the tracer provider is not initialized.

    Notes:
        Tracing is automatically initialized when using @paid_tracing decorator or context manager.
    """
    global paid_tracer_provider
    return paid_tracer_provider.get_tracer("paid.python")


def trace_sync_(
    external_customer_id: Optional[str],
    fn: Callable[..., T],
    external_agent_id: Optional[str] = None,
    tracing_token: Optional[int] = None,
    store_prompt: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
) -> T:
    """
    Internal function for synchronous tracing. Use @paid_tracing decorator instead.

    This is a low-level internal function. Users should use the @paid_tracing decorator
    or context manager for a more Pythonic interface.

    Parameters:
        external_customer_id: The external customer ID to associate with the trace.
        fn: The function to execute and trace.
        external_agent_id: Optional external agent ID.
        tracing_token: Optional token for distributed tracing.
        store_prompt: Whether to store prompt/completion contents.
        metadata: Optional metadata to attach to the trace.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.

    Returns:
        The result of executing fn(*args, **kwargs).

    Raises:
        Only when user callback raises.
    """
    args = args or ()
    kwargs = kwargs or {}

    # Set context variables for access by nested spans
    ContextData.set_context_key("external_customer_id", external_customer_id)
    ContextData.set_context_key("external_agent_id", external_agent_id)
    ContextData.set_context_key("store_prompt", store_prompt)
    ContextData.set_context_key("user_metadata", metadata)

    # If user set trace context manually
    override_trace_id = tracing_token
    if not override_trace_id:
        override_trace_id = ContextData.get_context_key("trace_id")
    ctx: Optional[Context] = None
    if override_trace_id is not None:
        span_context = SpanContext(
            trace_id=override_trace_id,
            span_id=distributed_tracing.otel_id_generator.generate_span_id(),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(span_context))

    try:
        tracer = get_paid_tracer()
        logger.info(f"Creating span for external_customer_id: {external_customer_id}")
        with tracer.start_as_current_span("parent_span", context=ctx) as span:
            try:
                result = fn(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                logger.info(f"Function {fn.__name__} executed successfully")
                return result
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise
    finally:
        ContextData.reset_context()


async def trace_async_(
    external_customer_id: Optional[str],
    fn: Callable[..., Union[T, Awaitable[T]]],
    external_agent_id: Optional[str] = None,
    tracing_token: Optional[int] = None,
    store_prompt: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
) -> Union[T, Awaitable[T]]:
    """
    Internal function for asynchronous tracing. Use @paid_tracing decorator instead.

    This is a low-level internal function. Users should use the @paid_tracing decorator
    or context manager for a more Pythonic interface.

    Parameters:
        external_customer_id: The external customer ID to associate with the trace.
        fn: The async function to execute and trace.
        external_agent_id: Optional external agent ID.
        tracing_token: Optional token for distributed tracing.
        store_prompt: Whether to store prompt/completion contents.
        metadata: Optional metadata to attach to the trace.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.

    Returns:
        The result of executing fn(*args, **kwargs).

    Raises:
        Only when user callback raises.
    """
    args = args or ()
    kwargs = kwargs or {}

    # Set context variables for access by nested spans
    ContextData.set_context_key("external_customer_id", external_customer_id)
    ContextData.set_context_key("external_agent_id", external_agent_id)
    ContextData.set_context_key("store_prompt", store_prompt)
    ContextData.set_context_key("user_metadata", metadata)

    # If user set trace context manually
    override_trace_id = tracing_token
    if not override_trace_id:
        override_trace_id = ContextData.get_context_key("trace_id")
    ctx: Optional[Context] = None
    if override_trace_id is not None:
        span_context = SpanContext(
            trace_id=override_trace_id,
            span_id=distributed_tracing.otel_id_generator.generate_span_id(),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(span_context))

    try:
        tracer = get_paid_tracer()
        logger.info(f"Creating span for external_customer_id: {external_customer_id}")
        with tracer.start_as_current_span("parent_span", context=ctx) as span:
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                logger.info(f"Async function {fn.__name__} executed successfully")
                return result
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise
    finally:
        ContextData.reset_context()
