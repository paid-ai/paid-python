# Initializing tracing for OTLP
import asyncio
import atexit
import contextvars
import functools
import logging
import os
import signal
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar, Union

import dotenv
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

# Configure logging
dotenv.load_dotenv()
log_level_name = os.environ.get("PAID_LOG_LEVEL")
if log_level_name is not None:
    log_level = getattr(logging, log_level_name.upper())
else:
    log_level = logging.ERROR  # Default to show errors
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Context variables for passing data to nested spans (e.g., in openAiWrapper)
paid_external_customer_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "paid_external_customer_id", default=None
)
paid_external_agent_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "paid_external_agent_id", default=None
)
# trace id storage (generated from token)
paid_trace_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("paid_trace_id", default=None)
# flag to enable storing prompt contents
paid_store_prompt_var: contextvars.ContextVar[Optional[bool]] = contextvars.ContextVar(
    "paid_store_prompt", default=False
)
# user metadata
paid_user_metadata_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "paid_user_metadata", default=None
)

T = TypeVar("T")

_token: Optional[str] = None


def get_token() -> Optional[str]:
    """Get the stored API token."""
    global _token
    return _token


def set_token(token: str) -> None:
    """Set the API token."""
    global _token
    _token = token


otel_id_generator = RandomIdGenerator()

# Isolated tracer provider for Paid - separate from any user OTEL setup
paid_tracer_provider: Optional[TracerProvider] = None


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
        customer_id = paid_external_customer_id_var.get()
        if customer_id:
            span.set_attribute("external_customer_id", customer_id)

        agent_id = paid_external_agent_id_var.get()
        if agent_id:
            span.set_attribute("external_agent_id", agent_id)

        metadata = paid_user_metadata_var.get()
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
        store_prompt = paid_store_prompt_var.get()
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


def initialize_tracing_(
    api_key: Optional[str] = None, collector_endpoint: Optional[str] = "https://collector.agentpaid.io:4318/v1/traces"
):
    """
    Initialize OpenTelemetry with OTLP exporter for Paid backend.

    Args:
        api_key: The API key for authentication. If not provided, will try to get from PAID_API_KEY environment variable.
        collector_endpoint: The OTLP collector endpoint URL.
    """
    global paid_tracer_provider

    try:
        if _token is not None:
            logger.warn("Tracing is already initialized - skipping re-initialization")
            return

        # Get API key from parameter or environment
        if api_key is None:
            import dotenv

            dotenv.load_dotenv()
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

        # Terminate gracefully and don't lose traces
        def flush_traces():
            try:
                if paid_tracer_provider is not None and not paid_tracer_provider.force_flush(10000):
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

        # This is already done by default OTEL shutdown,
        # but user might turn that off - so register it explicitly
        atexit.register(flush_traces)

        # Handle signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, create_chained_signal_handler(sig))

        logger.info("Paid tracing initialized successfully - collector at %s", collector_endpoint)
    except Exception as e:
        logger.error(f"Failed to initialize Paid tracing: {e}")
        # don't throw - tracing should not break the app


def get_paid_tracer() -> Optional[trace.Tracer]:
    """
    Get the tracer from the isolated Paid tracer provider.

    Returns:
        The Paid tracer instance.

    Raises:
        RuntimeError: If the tracer provider is not initialized.

    Notes:
        Tracing is automatically initialized when using @paid_tracing decorator or context manager.
    """
    try:
        return paid_tracer_provider.get_tracer("paid.python")
    except Exception as e:
        logger.error(f"Failed to get Paid tracer: {e}")
        # don't throw - tracing should not break the app
        return None

def trace_sync_(
    external_customer_id: str,
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
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_agent_id_ctx_token = paid_external_agent_id_var.set(external_agent_id)
    reset_store_prompt_ctx_token = paid_store_prompt_var.set(store_prompt)
    reset_user_metadata_ctx_token = paid_user_metadata_var.set(metadata)

    # If user set trace context manually
    override_trace_id = tracing_token
    if not override_trace_id:
        override_trace_id = paid_trace_id.get()
    ctx: Optional[Context] = None
    if override_trace_id is not None:
        span_context = SpanContext(
            trace_id=override_trace_id,
            span_id=otel_id_generator.generate_span_id(),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(span_context))

    try:
        tracer = get_paid_tracer()
        if not tracer:
            logger.error("Can't trace, no tracer available")
            return fn(*args, **kwargs)
        logger.info(f"Creating span for external_customer_id: {external_customer_id}")
        with tracer.start_as_current_span("parent_span", context=ctx) as span:
            span.set_attribute("external_customer_id", external_customer_id)
            if external_agent_id:
                span.set_attribute("external_agent_id", external_agent_id)
            try:
                result = fn(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                logger.info(f"Function {fn.__name__} executed successfully")
                return result
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise
    finally:
        paid_external_customer_id_var.reset(reset_id_ctx_token)
        paid_external_agent_id_var.reset(reset_agent_id_ctx_token)
        paid_store_prompt_var.reset(reset_store_prompt_ctx_token)
        paid_user_metadata_var.reset(reset_user_metadata_ctx_token)


async def trace_async_(
    external_customer_id: str,
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
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_agent_id_ctx_token = paid_external_agent_id_var.set(external_agent_id)
    reset_store_prompt_ctx_token = paid_store_prompt_var.set(store_prompt)
    reset_user_metadata_ctx_token = paid_user_metadata_var.set(metadata)

    # If user set trace context manually
    override_trace_id = tracing_token
    if not override_trace_id:
        override_trace_id = paid_trace_id.get()
    ctx: Optional[Context] = None
    if override_trace_id is not None:
        span_context = SpanContext(
            trace_id=override_trace_id,
            span_id=otel_id_generator.generate_span_id(),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(span_context))

    try:
        tracer = get_paid_tracer()
        logger.info(f"Creating span for external_customer_id: {external_customer_id}")
        if not tracer:
            logger.error("Can't trace, no tracer available")
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        with tracer.start_as_current_span("parent_span", context=ctx) as span:
            span.set_attribute("external_customer_id", external_customer_id)
            if external_agent_id:
                span.set_attribute("external_agent_id", external_agent_id)
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
        paid_external_customer_id_var.reset(reset_id_ctx_token)
        paid_external_agent_id_var.reset(reset_agent_id_ctx_token)
        paid_store_prompt_var.reset(reset_store_prompt_ctx_token)
        paid_user_metadata_var.reset(reset_user_metadata_ctx_token)


def generate_tracing_token() -> int:
    """
    Generate a unique tracing token without setting it in the context.

    Use this when you want to generate a trace ID to store or pass to another
    process/service without immediately associating it with the current tracing context.
    The token can later be used with set_tracing_token() to link traces across
    different execution contexts.

    Returns:
        int: A unique OpenTelemetry trace ID.

    Notes:
        - This function only generates and returns the token; it does NOT set it in the context.
        - For most use cases, use generate_and_set_tracing_token() instead.
        - Use this when you need to store the token separately before setting it.

    Examples:
        Generate token to store for later use:

            from paid.tracing import generate_tracing_token, set_tracing_token

            # Process 1: Generate and store
            token = generate_tracing_token()
            save_to_database("task_123", token)

            # Process 2: Retrieve and use
            token = load_from_database("task_123")
            set_tracing_token(token)

            @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
            def process_task():
                # This trace is now linked to the same token
                pass

    See Also:
        generate_and_set_tracing_token: Generate and immediately set the token.
        set_tracing_token: Set a previously generated token.
    """
    return otel_id_generator.generate_trace_id()


def generate_and_set_tracing_token() -> int:
    """
    Deprecated: Pass tracing_token directly to @paid_tracing() decorator instead.

    This function is deprecated and will be removed in a future version.
    Use the tracing_token parameter in @paid_tracing() to link traces across processes.

    Instead of:
        token = generate_and_set_tracing_token()
        @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
        def my_function():
            ...

    Use:
        from paid.tracing import generate_tracing_token
        token = generate_tracing_token()

        @paid_tracing(
            external_customer_id="cust_123",
            external_agent_id="agent_456",
            tracing_token=token
        )
        def my_function():
            ...

    Old behavior (for reference):
        This function generated a tracing token and set it in the context,
        so all subsequent @paid_tracing() calls would use it automatically.

    Returns:
        int: A unique OpenTelemetry trace ID (for backward compatibility).
    """
    import warnings
    warnings.warn(
        "generate_and_set_tracing_token() is deprecated and will be removed in a future version. "
        "Pass tracing_token directly to @paid_tracing(tracing_token=...) decorator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    random_trace_id = otel_id_generator.generate_trace_id()
    _ = paid_trace_id.set(random_trace_id)
    return random_trace_id


def set_tracing_token(token: int):
    """
    Deprecated: Pass tracing_token directly to @paid_tracing() decorator instead.

    This function is deprecated and will be removed in a future version.
    Use the tracing_token parameter in @paid_tracing() to link traces across processes.

    Instead of:
        token = load_from_storage("workflow_123")
        set_tracing_token(token)
        @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
        def process_workflow():
            ...
        unset_tracing_token()

    Use:
        token = load_from_storage("workflow_123")

        @paid_tracing(
            external_customer_id="cust_123",
            external_agent_id="agent_456",
            tracing_token=token
        )
        def process_workflow():
            ...

    Parameters:
        token (int): A tracing token (for backward compatibility only).

    Old behavior (for reference):
        This function set a token in the context, so all subsequent @paid_tracing() calls
        would use it automatically until unset_tracing_token() was called.
    """
    import warnings
    warnings.warn(
        "set_tracing_token() is deprecated and will be removed in a future version. "
        "Pass tracing_token directly to @paid_tracing(tracing_token=...) decorator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _ = paid_trace_id.set(token)


def unset_tracing_token():
    """
    Deprecated: No longer needed. Use tracing_token parameter in @paid_tracing() instead.

    This function is deprecated and will be removed in a future version.
    Since tracing_token is now passed directly to @paid_tracing(), there's no need
    to manually set/unset tokens in the context.

    Old behavior (for reference):
        This function unset a token previously set by set_tracing_token() or
        generate_and_set_tracing_token(), allowing subsequent @paid_tracing() calls
        to have independent traces.

    Migration:
        If you were using set_tracing_token() + unset_tracing_token() pattern,
        simply pass the token directly to @paid_tracing(tracing_token=...) instead.
    """
    import warnings
    warnings.warn(
        "unset_tracing_token() is deprecated and will be removed in a future version. "
        "Use tracing_token parameter in @paid_tracing(tracing_token=...) decorator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _ = paid_trace_id.set(None)


class paid_tracing:
    """
    Decorator and context manager for tracing with Paid.

    This class can be used both as a decorator and as a context manager (with/async with),
    providing flexible tracing capabilities for both functions and code blocks.

    Parameters
    ----------
    external_customer_id : str
        The external customer ID to associate with the trace.
    external_agent_id : Optional[str], optional
        The external agent ID to associate with the trace, by default None.
    tracing_token : Optional[int], optional
        Optional tracing token for distributed tracing, by default None.
    store_prompt : bool, optional
        Whether to store prompt contents in span attributes, by default False.
    collector_endpoint: Optional[str], optional
        OTEL collector HTTP endpoint, by default "https://collector.agentpaid.io:4318/v1/traces".
    metadata : Optional[Dict[str, Any]], optional
        Optional metadata to attach to the trace, by default None.

    Examples
    --------
    As a decorator (sync):
    >>> @paid_tracing(external_customer_id="customer123", external_agent_id="agent456")
    ... def my_function(arg1, arg2):
    ...     return arg1 + arg2

    As a decorator (async):
    >>> @paid_tracing(external_customer_id="customer123")
    ... async def my_async_function(arg1, arg2):
    ...     return arg1 + arg2

    As a context manager (sync):
    >>> with paid_tracing(external_customer_id="customer123", external_agent_id="agent456"):
    ...     result = expensive_computation()

    As a context manager (async):
    >>> async with paid_tracing(external_customer_id="customer123"):
    ...     result = await async_operation()

    Notes
    -----
    If tracing is not already initialized, the decorator will automatically
    initialize it using the PAID_API_KEY environment variable.
    """

    def __init__(
        self,
        external_customer_id: str,
        *,
        external_agent_id: Optional[str] = None,
        tracing_token: Optional[int] = None,
        store_prompt: bool = False,
        collector_endpoint: Optional[str] = "https://collector.agentpaid.io:4318/v1/traces",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.external_customer_id = external_customer_id
        self.external_agent_id = external_agent_id
        self.tracing_token = tracing_token
        self.store_prompt = store_prompt
        self.collector_endpoint = collector_endpoint
        self.metadata = metadata
        self._span: Any = None
        self._reset_tokens: Optional[
            Tuple[
                contextvars.Token[Optional[str]],
                contextvars.Token[Optional[str]],
                contextvars.Token[Optional[bool]],
                contextvars.Token[Optional[Dict[str, Any]]],
            ]
        ] = None

    def _setup_context(self) -> Optional[Context]:
        """Set up context variables and return OTEL context if needed."""

        # Set context variables
        reset_id_ctx_token = paid_external_customer_id_var.set(self.external_customer_id)
        reset_agent_id_ctx_token = paid_external_agent_id_var.set(self.external_agent_id)
        reset_store_prompt_ctx_token = paid_store_prompt_var.set(self.store_prompt)
        reset_user_metadata_ctx_token = paid_user_metadata_var.set(self.metadata)

        # Store reset tokens for cleanup
        self._reset_tokens = (
            reset_id_ctx_token,
            reset_agent_id_ctx_token,
            reset_store_prompt_ctx_token,
            reset_user_metadata_ctx_token,
        )

        # Handle distributed tracing token
        override_trace_id = self.tracing_token
        if not override_trace_id:
            override_trace_id = paid_trace_id.get()

        ctx: Optional[Context] = None
        if override_trace_id is not None:
            span_context = SpanContext(
                trace_id=override_trace_id,
                span_id=otel_id_generator.generate_span_id(),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            ctx = trace.set_span_in_context(NonRecordingSpan(span_context))

        return ctx

    def _cleanup_context(self):
        """Reset all context variables."""
        if self._reset_tokens:
            (
                reset_id_ctx_token,
                reset_agent_id_ctx_token,
                reset_store_prompt_ctx_token,
                reset_user_metadata_ctx_token,
            ) = self._reset_tokens
            paid_external_customer_id_var.reset(reset_id_ctx_token)
            paid_external_agent_id_var.reset(reset_agent_id_ctx_token)
            paid_store_prompt_var.reset(reset_store_prompt_ctx_token)
            paid_user_metadata_var.reset(reset_user_metadata_ctx_token)
            self._reset_tokens = None

    # Context manager methods for sync
    def __enter__(self):
        """Enter synchronous context."""
        ctx = self._setup_context()

        tracer = get_paid_tracer()
        logger.info(f"Creating span for external_customer_id: {self.external_customer_id}")
        self._span = tracer.start_as_current_span("parent_span", context=ctx)
        span = self._span.__enter__()

        span.set_attribute("external_customer_id", self.external_customer_id)
        if self.external_agent_id:
            span.set_attribute("external_agent_id", self.external_agent_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit synchronous context."""
        try:
            if self._span:
                if exc_type is not None:
                    # Get the actual span object to set status
                    span_obj = trace.get_current_span()
                    if span_obj:
                        span_obj.set_status(Status(StatusCode.ERROR, str(exc_val)))
                else:
                    span_obj = trace.get_current_span()
                    if span_obj:
                        span_obj.set_status(Status(StatusCode.OK))
                        logger.info("Context block executed successfully")

                self._span.__exit__(exc_type, exc_val, exc_tb)
                self._span = None
        finally:
            self._cleanup_context()

        return False  # Don't suppress exceptions

    # Context manager methods for async
    async def __aenter__(self):
        """Enter asynchronous context."""
        ctx = self._setup_context()

        tracer = get_paid_tracer()
        logger.info(f"Creating span for external_customer_id: {self.external_customer_id}")
        self._span = tracer.start_as_current_span("parent_span", context=ctx)
        span = self._span.__enter__()

        span.set_attribute("external_customer_id", self.external_customer_id)
        if self.external_agent_id:
            span.set_attribute("external_agent_id", self.external_agent_id)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit asynchronous context."""
        try:
            if self._span:
                if exc_type is not None:
                    # Get the actual span object to set status
                    span_obj = trace.get_current_span()
                    if span_obj:
                        span_obj.set_status(Status(StatusCode.ERROR, str(exc_val)))
                else:
                    span_obj = trace.get_current_span()
                    if span_obj:
                        span_obj.set_status(Status(StatusCode.OK))
                        logger.info("Async context block executed successfully")

                self._span.__exit__(exc_type, exc_val, exc_tb)
                self._span = None
        finally:
            self._cleanup_context()

        return False  # Don't suppress exceptions

    # Decorator functionality
    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator."""
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Auto-initialize tracing if not done
                if get_token() is None:
                    try:
                        initialize_tracing_(None, self.collector_endpoint)
                    except Exception as e:
                        logger.error(f"Failed to auto-initialize tracing: {e}")
                        # Fall back to executing function without tracing
                        return await func(*args, **kwargs)

                try:
                    return await trace_async_(
                        external_customer_id=self.external_customer_id,
                        fn=func,
                        external_agent_id=self.external_agent_id,
                        tracing_token=self.tracing_token,
                        store_prompt=self.store_prompt,
                        metadata=self.metadata,
                        args=args,
                        kwargs=kwargs,
                    )
                except Exception as e:
                    logger.error(f"Failed to trace async function {func.__name__}: {e}")
                    raise e

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Auto-initialize tracing if not done
                if get_token() is None:
                    try:
                        initialize_tracing_(None, self.collector_endpoint)
                    except Exception as e:
                        logger.error(f"Failed to auto-initialize tracing: {e}")
                        # Fall back to executing function without tracing
                        return func(*args, **kwargs)

                try:
                    return trace_sync_(
                        external_customer_id=self.external_customer_id,
                        fn=func,
                        external_agent_id=self.external_agent_id,
                        tracing_token=self.tracing_token,
                        store_prompt=self.store_prompt,
                        metadata=self.metadata,
                        args=args,
                        kwargs=kwargs,
                    )
                except Exception as e:
                    logger.error(f"Failed to trace sync function {func.__name__}: {e}")
                    raise e

            return sync_wrapper
