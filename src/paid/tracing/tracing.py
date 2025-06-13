# Initializing tracing for OTLP

import asyncio
import logging
from typing import Optional, TypeVar, Callable, Union, Awaitable, Tuple, Dict
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.openai import OpenAIInstrumentor

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
_token: Optional[str] = None

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
        endpoint: The OTLP endpoint URL (defaults to localhost for development)
    """
    # endpoint = "https://collector.agentpaid.io:4318/v1/traces"
    endpoint = "http://localhost:4318/v1/traces"
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



# Capturing the traces with Auto-intstrumented OTLP

def _capture(
    external_customer_id: str, 
    fn: Callable[[], Union[T, Awaitable[T]]],
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None
) -> Union[T, Awaitable[T]]:
    """Capture the execution of a function with OpenTelemetry tracing."""
    # Check if function is async
    if asyncio.iscoroutinefunction(fn):
        return _capture_async(external_customer_id, fn, args, kwargs)
    else:
        return _capture_sync(external_customer_id, fn, args, kwargs)

def _capture_sync(external_customer_id: str,
                  fn: Callable[..., T],
                  args: Optional[Tuple] = None,
                  kwargs: Optional[Dict] = None) -> T:
    """Handle synchronous function capture."""
    args = args or ()
    kwargs = kwargs or {}

    token = get_token()
    if not token:
        logger.warning('No token found - tracing is not initialized and will not be captured')
        return fn(*args, **kwargs)

    tracer = trace.get_tracer("paid.python")
    
    logger.info(f"Creating span for external_customer_id: {external_customer_id}")
    with tracer.start_as_current_span(f"paid.python:{external_customer_id}") as span:
        span.set_attribute('external_customer_id', external_customer_id)
        span.set_attribute('token', token)
        try:
            result = fn(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Function {fn.__name__} executed successfully")
            return result
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise

async def _capture_async(external_customer_id: str,
                         fn: Callable[..., Awaitable[T]],
                         args: Optional[Tuple] = None,
                         kwargs: Optional[Dict] = None) -> T:
    """Handle asynchronous function capture."""
    args = args or ()
    kwargs = kwargs or {}

    token = get_token()
    if not token:
        logger.warning('No token found - tracing is not initialized and async function will not be captured')
        return await fn(*args, **kwargs)

    tracer = trace.get_tracer("paid.python")
    with tracer.start_as_current_span(f"paid.python:{external_customer_id}") as span:
        span.set_attribute('external_customer_id', external_customer_id)
        span.set_attribute('token', token)
        try:
            result = await fn(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Async function {fn.__name__} executed successfully")
            return result
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise
