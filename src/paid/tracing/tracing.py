# Initializing tracing for OTLP
import asyncio
import atexit
import contextvars
import logging
import os
import signal
from typing import Awaitable, Callable, Dict, Optional, Tuple, TypeVar, Union

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

# Configure logging
log_level_name = os.environ.get("PAID_LOG_LEVEL")
if log_level_name is not None:
    log_level = getattr(logging, log_level_name.upper())
else:
    log_level = 100  # Default to no logging
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
# api_key storage
paid_token_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("paid_token", default=None)
# trace id storage (generated from token)
paid_trace_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("paid_trace_id", default=None)

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


def _initialize_tracing(api_key: str, collector_endpoint: str):
    """
    Initialize OpenTelemetry with OTLP exporter for Paid backend.

    Args:
        api_key: The API key for authentication
    """
    try:
        if _token is not None:
            raise RuntimeError("Tracing is already initialized.")

        set_token(api_key)

        # Set up tracer provider
        tracer_provider = trace.get_tracer_provider()
        if (
            not tracer_provider
            or tracer_provider.__class__.__name__ == "NoOpTracerProvider"
            or tracer_provider.__class__.__name__ == "ProxyTracerProvider"
        ):
            logger.info("No existing tracer provider found, creating a new one.")
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        # Fix static type checkers that don't understand the above logic
        if not isinstance(tracer_provider, TracerProvider):
            raise RuntimeError("Failed to create a valid TracerProvider instance.")

        # Set up OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=collector_endpoint,
            headers={},  # No additional headers needed for OTLP
        )

        # Use SimpleSpanProcessor for immediate span export.
        # There are problems with BatchSpanProcessor in some environments - ex. Airflow.
        # Airflow terminates processes before the batch is sent, losing traces.
        span_processor = SimpleSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Terminate gracefully and don't lose traces
        def flush_traces():
            try:
                if not tracer_provider.force_flush(10000):
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
    except Exception:
        logger.exception("Failed to initialize Paid tracing")
        raise


def _trace_sync(
    external_customer_id: str,
    fn: Callable[..., T],
    external_agent_id: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
) -> T:
    args = args or ()
    kwargs = kwargs or {}
    token = get_token()
    if not token:
        raise RuntimeError(
            "No token found - tracing is not initialized and will not be captured. Call Paid.initialize_tracing() first."
        )

    # Set context variables for access by nested spans
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_agent_id_ctx_token = paid_external_agent_id_var.set(external_agent_id)
    reset_token_ctx_token = paid_token_var.set(token)

    # If user set trace context manually
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
        tracer = trace.get_tracer("paid.python")
        logger.info(f"Creating span for external_customer_id: {external_customer_id}")
        with tracer.start_as_current_span(f"paid.python:{external_customer_id}", context=ctx) as span:
            span.set_attribute("external_customer_id", external_customer_id)
            if external_agent_id:
                span.set_attribute("external_agent_id", external_agent_id)
            span.set_attribute("token", token)
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
        paid_token_var.reset(reset_token_ctx_token)


async def _trace_async(
    external_customer_id: str,
    fn: Callable[..., Union[T, Awaitable[T]]],
    external_agent_id: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
) -> Union[T, Awaitable[T]]:
    args = args or ()
    kwargs = kwargs or {}
    token = get_token()
    if not token:
        raise RuntimeError(
            "No token found - tracing is not initialized and will not be captured. Call Paid.initialize_tracing() first."
        )

    # Set context variables for access by nested spans
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_agent_id_ctx_token = paid_external_agent_id_var.set(external_agent_id)
    reset_token_ctx_token = paid_token_var.set(token)

    # If user set trace context manually
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
        tracer = trace.get_tracer("paid.python")
        logger.info(f"Creating span for external_customer_id: {external_customer_id}")
        with tracer.start_as_current_span(f"paid.python:{external_customer_id}", context=ctx) as span:
            span.set_attribute("external_customer_id", external_customer_id)
            if external_agent_id:
                span.set_attribute("external_agent_id", external_agent_id)
            span.set_attribute("token", token)
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
        paid_token_var.reset(reset_token_ctx_token)


def _generate_and_set_tracing_token() -> int:
    random_trace_id = otel_id_generator.generate_trace_id()
    _ = paid_trace_id.set(random_trace_id)
    return random_trace_id


def _set_tracing_token(token: int):
    _ = paid_trace_id.set(token)


def _unset_tracing_token():
    _ = paid_trace_id.set(None)
