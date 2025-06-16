# Initializing tracing for OTLP
import asyncio
import contextvars
import logging
from typing import Optional, TypeVar, Callable, Union, Awaitable, Tuple, Dict
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.openai import OpenAIInstrumentor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_token: Optional[str] = None
# Context variables for passing data to nested spans (e.g., in openAiWrapper)
paid_external_customer_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("paid_external_customer_id", default=None)
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



def _capture(
    external_customer_id: str,
    fn: Callable[..., Union[T, Awaitable[T]]],
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
) -> Union[T, Awaitable[T]]:
    """
    Executes a function within an OpenTelemetry span for tracing purposes.

    Args:
        external_customer_id (str): Backend won't record the trace if the id isn't recognized.
        fn (Callable[..., Union[T, Awaitable[T]]]): The function to be executed, supposedly calls a Paid wrapper around an LLM API.
        args (Optional[Tuple], optional): Positional arguments to pass to the function.
        kwargs (Optional[Dict], optional): Keyword arguments to pass to the function.

    Returns:
        Union[T, Awaitable[T]]: The result of the executed function, either synchronously or asynchronously.

    Notes:
        - If no token is available, tracing will not be initialized, and the function will execute without tracing.
        - Automatically determines whether the function is synchronous or asynchronous and handles it accordingly.
    """
    args = args or ()
    kwargs = kwargs or {}
    token = get_token()
    if not token:
        logger.warning(
            "No token found - tracing is not initialized and will not be captured"
        )
        return fn(*args, **kwargs)

    if asyncio.iscoroutinefunction(fn):
        return _capture_async(external_customer_id, fn, token, args, kwargs)
    else:
        return _capture_sync(external_customer_id, fn, token, args, kwargs)


def _capture_sync(
    external_customer_id: str,
    fn: Callable[..., T],
    token: str,
    args: Tuple = (),
    kwargs: Dict = {},
) -> T:
    """Handle synchronous function capture."""
    # Set context variables for access by nested spans
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_token_ctx_token = paid_token_var.set(token)

    tracer = trace.get_tracer("paid.python")
    logger.info(f"Creating span for external_customer_id: {external_customer_id}")
    try:
        with tracer.start_as_current_span(f"paid.python:{external_customer_id}") as span:
            span.set_attribute("external_customer_id", external_customer_id)
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
        # Reset context variables to their previous state
        paid_external_customer_id_var.reset(reset_id_ctx_token)
        paid_token_var.reset(reset_token_ctx_token)


async def _capture_async(
    external_customer_id: str,
    fn: Callable[..., Awaitable[T]],
    token: str,
    args: Tuple = (),
    kwargs: Dict = {},
) -> T:
    """Handle asynchronous function capture."""
    # Set context variables for access by nested spans
    reset_id_ctx_token = paid_external_customer_id_var.set(external_customer_id)
    reset_token_ctx_token = paid_token_var.set(token)

    tracer = trace.get_tracer("paid.python")
    logger.info(f"Creating span for external_customer_id: {external_customer_id}")
    try:
        with tracer.start_as_current_span(f"paid.python:{external_customer_id}") as span:
            span.set_attribute("external_customer_id", external_customer_id)
            span.set_attribute("token", token)
            try:
                result = await fn(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                logger.info(f"Async function {fn.__name__} executed successfully")
                return result
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise
    finally:
        # Reset context variables to their previous state
        paid_external_customer_id_var.reset(reset_id_ctx_token)
        paid_token_var.reset(reset_token_ctx_token)
