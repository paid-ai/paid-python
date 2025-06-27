# Initializing tracing for OTLP
import asyncio
import contextvars
import logging
import os
from typing import Optional, TypeVar, Callable, Union, Awaitable, Tuple, Dict
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.openai import OpenAIInstrumentor

# Configure logging
log_level_name = os.environ.get("PAID_LOG_LEVEL")
if log_level_name is not None:
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
else:
    log_level = 100  # No logs by default
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

_token: Optional[str] = None
# Context variables for passing data to nested spans (e.g., in openAiWrapper)
paid_external_customer_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("paid_external_customer_id", default=None)
paid_external_agent_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("paid_external_agent_id", default=None)
paid_token_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("paid_token", default=None)

T = TypeVar('T')

def get_token() -> Optional[str]:
    """Get the stored API token."""
    global _token
    return _token

def set_token(token: str) -> None:
    """Set the API token."""
    global _token
    _token = token


def _initialize_tracing(api_key: str):
    """
    Initialize OpenTelemetry with OTLP exporter for Paid backend.

    Args:
        api_key: The API key for authentication
    """
    endpoint = "https://collector.agentpaid.io:4318/v1/traces"
    # endpoint = "http://localhost:4318/v1/traces"
    try:
        set_token(api_key)

        # Set up tracer provider
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        # Set up OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={},  # No additional headers needed for OTLP
        )

        # Set up span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Don't need auto-instrumentation,
        # will add it when implementing automatic tracing (w/o explicit capture)
        # instrumentor = OpenAIInstrumentor(
        #     # exception_logger=lambda e: Telemetry().log_exception(e),
        #     # enrich_assistant=True,
        #     enrich_token_usage=True,
        #     # get_common_metrics_attributes=metrics_common_attributes,
        #     # upload_base64_image=base64_image_uploader,
        # )
        # if not instrumentor.is_instrumented_by_opentelemetry:
        #     instrumentor.instrument()

        logger.info("Paid tracing initialized successfully")
    except Exception:
        logger.exception("Failed to initialize Paid tracing")
        raise

def _trace(
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
        logger.warning(
            "No token found - tracing is not initialized and will not be captured"
        )
        return fn(*args, **kwargs)

    # Set context variables for access by nested spans
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_agent_id_ctx_token = paid_external_agent_id_var.set(external_agent_id)
    reset_token_ctx_token = paid_token_var.set(token)
    try:
        if asyncio.iscoroutinefunction(fn):
            return _trace_async(external_customer_id, fn, token, external_agent_id, args, kwargs)
        else:
            return _trace_sync(external_customer_id, fn, token, external_agent_id, args, kwargs)
    finally:
        paid_external_customer_id_var.reset(reset_id_ctx_token)
        paid_external_agent_id_var.reset(reset_agent_id_ctx_token)
        paid_token_var.reset(reset_token_ctx_token)

def _trace_sync(
    external_customer_id: str,
    fn: Callable[..., T],
    token: str,
    external_agent_id: Optional[str] = None,
    args: Tuple = (),
    kwargs: Dict = {},
) -> T:
    tracer = trace.get_tracer("paid.python")
    logger.info(f"Creating span for external_customer_id: {external_customer_id}")
    with tracer.start_as_current_span(f"paid.python:{external_customer_id}") as span:
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


async def _trace_async(
    external_customer_id: str,
    fn: Callable[..., Awaitable[T]],
    token: str,
    external_agent_id: Optional[str] = None,
    args: Tuple = (),
    kwargs: Dict = {},
) -> T:
    tracer = trace.get_tracer("paid.python")
    logger.info(f"Creating span for external_customer_id: {external_customer_id}")
    with tracer.start_as_current_span(f"paid.python:{external_customer_id}") as span:
        span.set_attribute("external_customer_id", external_customer_id)
        if external_agent_id:
            span.set_attribute("external_agent_id", external_agent_id)
        span.set_attribute("token", token)
        try:
            result = await fn(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Async function {fn.__name__} executed successfully")
            return result
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise
