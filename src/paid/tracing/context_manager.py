import asyncio
import functools
from typing import Any, Callable, Dict, Optional

from . import distributed_tracing, tracing
from .context_data import ContextData
from .tracing import get_paid_tracer, get_token, initialize_tracing, trace_async_, trace_sync_
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, Status, StatusCode, TraceFlags

from paid.logger import logger


class paid_tracing:
    """
    Decorator and context manager for tracing with Paid.

    This class can be used both as a decorator and as a context manager (with/async with),
    providing flexible tracing capabilities for both functions and code blocks.

    Parameters
    ----------
    external_customer_id : str
        The external customer ID to associate with the trace.
    external_product_id : Optional[str], optional
        The external product ID to associate with the trace, by default None.
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
    >>> @paid_tracing(external_customer_id="customer123", external_product_id="product456")
    ... def my_function(arg1, arg2):
    ...     return arg1 + arg2

    As a decorator (async):
    >>> @paid_tracing(external_customer_id="customer123")
    ... async def my_async_function(arg1, arg2):
    ...     return arg1 + arg2

    As a context manager (sync):
    >>> with paid_tracing(external_customer_id="customer123", external_product_id="product456"):
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
        external_customer_id: Optional[str] = None,
        *,
        external_product_id: Optional[str] = None,
        tracing_token: Optional[int] = None,
        store_prompt: bool = False,
        collector_endpoint: Optional[str] = tracing.DEFAULT_COLLECTOR_ENDPOINT,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.external_customer_id = external_customer_id
        # Map external_product_id to external_agent_id for internal context
        self.external_agent_id = external_product_id
        self.tracing_token = tracing_token
        self.store_prompt = store_prompt
        self.collector_endpoint = collector_endpoint
        self.metadata = metadata
        self.span: Optional[Span] = None
        self.span_ctx: Optional[Any] = None  # Context manager for the span

        if not get_token():
            initialize_tracing(None, self.collector_endpoint)

    def _setup_context(self) -> Optional[Context]:
        """Set up context variables and return OTEL context if needed."""

        # Only override context variables if explicitly provided
        if self.external_customer_id is not None:
            ContextData.set_context_key("external_customer_id", self.external_customer_id)
        if self.external_agent_id is not None:
            ContextData.set_context_key("external_agent_id", self.external_agent_id)
        # Only override store_prompt if explicitly set to True
        if self.store_prompt:
            ContextData.set_context_key("store_prompt", True)
        # Merge metadata with existing instead of replacing
        if self.metadata is not None:
            existing_metadata = ContextData.get_context_key("user_metadata") or {}
            merged_metadata = {**existing_metadata, **self.metadata}
            ContextData.set_context_key("user_metadata", merged_metadata)

        # Handle distributed tracing token
        override_trace_id = self.tracing_token
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

        return ctx

    def _cleanup_context(self):
        """Reset all context variables."""
        ContextData.reset_context()

    # Context manager methods for sync
    def __enter__(self):
        return self._enter_ctx()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._exit_ctx(exc_type, exc_val, exc_tb)

    # Context manager methods for async
    async def __aenter__(self):
        return self._enter_ctx()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self._exit_ctx(exc_type, exc_val, exc_tb)

    def _enter_ctx(self):
        ctx = self._setup_context()
        tracer = get_paid_tracer()
        logger.info(f"Creating span for external_customer_id: {self.external_customer_id}")
        self.span_ctx = tracer.start_as_current_span("parent_span", context=ctx)
        self.span = self.span_ctx.__enter__()
        return self

    def _exit_ctx(self, exc_type, exc_val, exc_tb):
        """Exit synchronous context."""
        try:
            if self.span and self.span_ctx:
                if exc_type is not None:
                    self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                else:
                    self.span.set_status(Status(StatusCode.OK))
                    logger.info("Context block executed successfully")

                self.span_ctx.__exit__(exc_type, exc_val, exc_tb)
                self.span_ctx = None
                self.span = None

        finally:
            self._cleanup_context()

        return False  # Don't suppress exceptions

    # Decorator functionality
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Use as a decorator."""
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Auto-initialize tracing if not done
                if get_token() is None:
                    try:
                        initialize_tracing(None, self.collector_endpoint)
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
                        initialize_tracing(None, self.collector_endpoint)
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
