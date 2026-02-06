# Initializing tracing for OTLP
import asyncio
import atexit
import os
import signal
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar, Union

from . import distributed_tracing
from .context_data import ContextData
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import NonRecordingSpan, NoOpTracerProvider, SpanContext, Status, StatusCode, TraceFlags
from opentelemetry.util.types import Attributes

from paid.logger import logger

DEFAULT_COLLECTOR_ENDPOINT = (
    os.environ.get("PAID_OTEL_COLLECTOR_ENDPOINT") or "https://collector.agentpaid.io:4318/v1/traces"
)

T = TypeVar("T")


@dataclass
class PydanticProcessorSettings:
    """Settings for Pydantic AI span processing."""

    track_usage: bool = True
    """If False, filters out usage and cost attributes from spans. Default is True."""


@dataclass
class ProcessorSettings:
    """Configuration for span processors."""

    pydantic: Optional[PydanticProcessorSettings] = None
    """Settings for Pydantic AI span processing. If provided, enables Pydantic-specific filtering."""


# Scope name constants for library-specific tracers
PYDANTIC_SCOPE_NAME = "paid.pydantic"


class _PydanticSettingsRegistry:
    """Registry for Pydantic tracer settings, keyed by scope name."""

    _settings: Dict[str, PydanticProcessorSettings] = {}

    @classmethod
    def register(cls, scope_name: str, settings: PydanticProcessorSettings) -> None:
        """Register settings for a scope."""
        cls._settings[scope_name] = settings

    @classmethod
    def get(cls, scope_name: str) -> Optional[PydanticProcessorSettings]:
        """Get settings for a scope, or None if not registered."""
        return cls._settings.get(scope_name)

    @classmethod
    def has_scope(cls, scope_name: str) -> bool:
        """Check if a scope is registered."""
        return scope_name in cls._settings


class PydanticTracerProvider:
    """
    A TracerProvider wrapper for Pydantic AI that registers settings when get_tracer() is called.

    This allows Pydantic AI's InstrumentationSettings to receive a "TracerProvider-like" object
    that will register the appropriate settings for any scope name Pydantic AI uses internally.

    Spans created through tracers from this provider will have the PydanticSpanProcessor
    applied with the configured settings.
    """

    def __init__(self, settings: PydanticProcessorSettings):
        """
        Initialize the Pydantic tracer provider wrapper.

        Args:
            settings: The settings to apply to spans created through this provider.
        """
        self._settings = settings

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[Attributes] = None,
    ) -> trace.Tracer:
        """
        Get a tracer that will have Pydantic span processing applied.

        Registers the settings for the given scope name, then delegates to
        the real Paid tracer provider.

        Args:
            instrumenting_module_name: The name of the instrumenting module (scope name).
            instrumenting_library_version: Optional version of the instrumenting library.
            schema_url: Optional schema URL for the instrumentation.
            attributes: Optional attributes for the instrumentation.

        Returns:
            A tracer from the Paid tracer provider with settings registered for this scope.
        """
        global paid_tracer_provider

        # Register settings for this scope so PydanticSpanProcessor can find them
        _PydanticSettingsRegistry.register(instrumenting_module_name, self._settings)

        return paid_tracer_provider.get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            schema_url,
            attributes,
        )


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

def get_paid_tracer_provider() -> Optional[TracerProvider]:
    """Export the tracer provider to the user.
    Initialize tracing if not already. Never return NoOpTracerProvider.

    Returns:
        The tracer provider instance.
    """
    global paid_tracer_provider

    if get_token() is None:
        initialize_tracing()

    if not isinstance(paid_tracer_provider, TracerProvider):
        return None

    return paid_tracer_provider

class PaidSpanProcessor(SpanProcessor):
    """
    Span processor that:
    1. Prefixes all span names with 'paid.trace.'
    2. Automatically adds external_customer_id and external_agent_id attributes
       to all spans based on context variables set by the tracing decorator.
    3. Filters out prompt/response data unless store_prompt=True.
    4. Filters out duplicate LangChain spans that may duplicate information from other instrumentations.
    """

    SPAN_NAME_PREFIX = "paid.trace."
    PROMPT_ATTRIBUTES_SUBSTRINGS = {
        "gen_ai.completion",
        "gen_ai.request.messages",
        "gen_ai.response.messages",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "llm.output_message",
        "llm.input_message",
        "llm.invocation_parameters",
        "gen_ai.prompt",
        "langchain.prompt",
        "output.value",
        "input.value",
        "model_request_parameters",
        "logfire.json_schema",
    }

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span is started. Prefix the span name and add attributes."""

        LANGCHAIN_SPAN_FILTERS = ["ChatOpenAI", "ChatAnthropic"]
        if any(f in span.name for f in LANGCHAIN_SPAN_FILTERS):
            # HACK TO FILTER DUPLICATE SPANS CREATED BY LANGCHAIN INSTRUMENTATION.
            # Langchain instrumentation creates spans, that are created by other instrumentations (ex. OpenAI, Anthropic).
            # Not all spans need filtering (ex. ChatGoogleGenerativeAI), so first test actual telemetry before adding filters.
            # TODO: maybe consider a dropping sampler for such spans instead of raising an exception?
            logger.debug("[paid:span] Dropping duplicate LangChain span: %s", span.name)
            raise Exception(f"Dropping Langchain span: {span.name}")

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

        logger.debug(
            "[paid:span] on_start: name=%s, customer_id=%s, agent_id=%s",
            span.name, customer_id, agent_id,
        )

        metadata = ContextData.get_context_key("user_metadata")
        if metadata:
            metadata_attributes: dict[str, Any] = {}

            # OTEL attributes only accept: bool, str, bytes, int, float
            _OTEL_SAFE_TYPES = (bool, str, bytes, int, float)

            def _sanitize_value(v: Any) -> Any:
                """Convert non-OTEL-safe values (e.g. UUID) to str."""
                if isinstance(v, _OTEL_SAFE_TYPES):
                    return v
                if isinstance(v, (list, tuple)):
                    return [
                        el if isinstance(el, _OTEL_SAFE_TYPES) else str(el)
                        for el in v
                    ]
                return str(v)

            def flatten_dict(d: dict[str, Any], parent_key: str = "") -> None:
                """Recursively flatten nested dictionaries into dot-notation keys."""
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        flatten_dict(v, new_key)
                    else:
                        metadata_attributes[new_key] = _sanitize_value(v)

            flatten_dict(metadata)

            # Add all flattened metadata attributes to the span
            for key, value in metadata_attributes.items():
                span.set_attribute(f"metadata.{key}", value)
            logger.debug("[paid:span] on_start: attached metadata keys=%s", list(metadata_attributes.keys()))

    def on_end(self, span: ReadableSpan) -> None:
        if span.name and not span.name.startswith(self.SPAN_NAME_PREFIX):
            # Note: ReadableSpan is immutable, need to use internal attribute
            object.__setattr__(span, "_name", f"{self.SPAN_NAME_PREFIX}{span.name}")
            
        """Filter out prompt and response contents unless explicitly asked to store"""
        store_prompt = ContextData.get_context_key("store_prompt")
        if store_prompt:
            logger.debug("[paid:span] on_end: name=%s, store_prompt=True, keeping all attributes", span.name)
            return

        original_attributes = span.attributes

        if original_attributes:
            # Filter out prompt-related attributes
            filtered_attrs = {
                k: v
                for k, v in original_attributes.items()
                if not any(substr in k for substr in self.PROMPT_ATTRIBUTES_SUBSTRINGS)
            }
            filtered_count = len(original_attributes) - len(filtered_attrs)
            if filtered_count > 0:
                logger.debug(
                    "[paid:span] on_end: name=%s, filtered %d prompt attribute(s)",
                    span.name, filtered_count,
                )
            # This works because the exporter reads attributes during serialization
            object.__setattr__(span, "_attributes", filtered_attrs)

    def shutdown(self) -> None:
        """Called when the processor is shut down. No action needed."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Called to force flush. Always returns True since there's nothing to flush."""
        return True


class PydanticSpanProcessor(SpanProcessor):
    """
    Span processor that filters usage and cost attributes from Pydantic AI spans.

    Only processes spans from tracers with registered Pydantic settings (via get_paid_tracer_pydantic).
    Settings are looked up from the _PydanticSettingsRegistry by instrumentation scope name.
    """

    USAGE_ATTRIBUTES_SUBSTRINGS = {
        "gen_ai.usage",
        "operation.cost",
        "usage",
        "cost",
    }

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span is started. No action needed - PaidSpanProcessor handles this."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Filter out usage/cost data based on settings for this span's scope."""
        # Only process spans from registered pydantic scopes
        scope_name = span.instrumentation_scope.name if span.instrumentation_scope else None
        if not scope_name or not _PydanticSettingsRegistry.has_scope(scope_name):
            return

        settings = _PydanticSettingsRegistry.get(scope_name)
        if settings is None or settings.track_usage:
            # No filtering needed - keep usage/cost data
            logger.debug("[paid:span] PydanticSpanProcessor on_end: scope=%s, track_usage=True, keeping usage data", scope_name)
            return

        original_attributes = span.attributes
        if not original_attributes:
            return

        filtered_attrs = {
            k: v
            for k, v in original_attributes.items()
            if not any(substr in k for substr in self.USAGE_ATTRIBUTES_SUBSTRINGS)
        }
        filtered_count = len(original_attributes) - len(filtered_attrs)
        logger.debug(
            "[paid:span] PydanticSpanProcessor on_end: scope=%s, filtered %d usage/cost attribute(s)",
            scope_name, filtered_count,
        )
        # This works because the exporter reads attributes during serialization
        object.__setattr__(span, "_attributes", filtered_attrs)

    def shutdown(self) -> None:
        """Called when the processor is shut down. No action needed."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Called to force flush. Always returns True since there's nothing to flush."""
        return True


def setup_graceful_termination(paid_tracer_provider: TracerProvider):
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
        logger.debug("[paid:init] Registered atexit and signal handlers (SIGINT, SIGTERM)")
    except Exception as e:
        logger.warning(
            f"Could not set up termination handlers: {e}"
            "\nConsider calling initialize_tracing() from the main thread during app initialization if you don't already"
        )


def initialize_tracing(
    api_key: Optional[str] = None,
    collector_endpoint: Optional[str] = DEFAULT_COLLECTOR_ENDPOINT,
    processor_settings: Optional[ProcessorSettings] = None,
):
    """
    Initialize OpenTelemetry with OTLP exporter for Paid backend.

    Args:
        api_key: The API key for authentication. If not provided, will try to get from PAID_API_KEY environment variable.
        collector_endpoint: The OTLP collector endpoint URL.
        processor_settings: Optional configuration for span processors (deprecated for Pydantic - use get_paid_tracer_pydantic instead).

    Example:
        # Basic initialization
        initialize_tracing()

    """
    global paid_tracer_provider

    if not collector_endpoint:
        collector_endpoint = DEFAULT_COLLECTOR_ENDPOINT

    logger.debug("[paid:init] initialize_tracing called, endpoint=%s, api_key=%s",
                 collector_endpoint, "provided" if api_key else "not provided (will check env)")

    try:
        # Check if tracing is disabled via environment variable
        paid_enabled = os.environ.get("PAID_ENABLED", "true").lower()
        if paid_enabled == "false":
            logger.info("Paid tracing is disabled via PAID_ENABLED environment variable")
            return

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
            logger.debug("[paid:init] API key resolved from PAID_API_KEY environment variable")
        else:
            logger.debug("[paid:init] API key provided via parameter")

        set_token(api_key)

        resource = Resource(attributes={"api.key": api_key})
        # Create isolated tracer provider for Paid - don't use or modify global provider
        # Pass explicit sampler and span_limits to avoid inheriting from OTEL env vars
        # (OTEL_TRACES_SAMPLER=always_off or OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=1
        #  set by the client app would silently break this)
        paid_tracer_provider = TracerProvider(
            resource=resource,
            sampler=ALWAYS_ON,
            span_limits=SpanLimits(
                max_span_attributes=128,
                max_event_attributes=128,
                max_link_attributes=128,
                max_events=128,
                max_links=128,
            ),
        )
        # Override OTEL_SDK_DISABLED - the client may set it to disable their own OTEL
        paid_tracer_provider._disabled = False
        logger.debug("[paid:init] TracerProvider created (sampler=ALWAYS_ON, OTEL_SDK_DISABLED overridden)")

        # Add span processor to prefix span names and add customer/agent ID attributes
        paid_tracer_provider.add_span_processor(PaidSpanProcessor())
        logger.debug("[paid:init] Added PaidSpanProcessor")

        # Add Pydantic span processor - it self-filters by scope using the settings registry
        paid_tracer_provider.add_span_processor(PydanticSpanProcessor())
        logger.debug("[paid:init] Added PydanticSpanProcessor")

        # Legacy support: if processor_settings.pydantic is provided, register it for the default scope
        if processor_settings and processor_settings.pydantic:
            _PydanticSettingsRegistry.register(PYDANTIC_SCOPE_NAME, processor_settings.pydantic)
            logger.debug("[paid:init] Registered legacy PydanticProcessorSettings for scope=%s", PYDANTIC_SCOPE_NAME)

        # Set up OTLP exporter with explicit settings to avoid inheriting from
        # client OTEL env vars (e.g. OTEL_EXPORTER_OTLP_HEADERS, OTEL_EXPORTER_OTLP_TIMEOUT)
        otlp_exporter = OTLPSpanExporter(
            endpoint=collector_endpoint,
            headers={"_paid": "1"},  # Non-empty to prevent env var OTEL_EXPORTER_OTLP_HEADERS leak (empty dict is falsy)
            timeout=10,  # Explicit timeout to prevent env var OTEL_EXPORTER_OTLP_TIMEOUT override
        )

        # Use SimpleSpanProcessor for immediate span export.
        # There are problems with BatchSpanProcessor in some environments - ex. Airflow.
        # Airflow terminates processes before the batch is sent, losing traces.
        span_processor = SimpleSpanProcessor(otlp_exporter)
        paid_tracer_provider.add_span_processor(span_processor)
        logger.debug("[paid:init] Added SimpleSpanProcessor with OTLPSpanExporter -> %s", collector_endpoint)

        setup_graceful_termination(paid_tracer_provider)  # doesn't throw

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
    logger.debug("[paid:init] get_paid_tracer: provider_type=%s", type(paid_tracer_provider).__name__)
    return paid_tracer_provider.get_tracer("paid.python")



def get_paid_tracer_provider_pydantic(
    settings: Optional[PydanticProcessorSettings] = None,
) -> PydanticTracerProvider:
    """
    Get a TracerProvider for Pydantic AI with custom processing settings.

    This is designed to work with Pydantic AI's InstrumentationSettings, which expects
    a TracerProvider. The returned provider registers settings when
    get_tracer() is called, so spans from any scope name Pydantic AI uses internally
    will have the PydanticSpanProcessor applied.

    Spans created through this provider share the same trace context as other Paid tracers,
    so they will be properly linked within traces.

    Args:
        settings: Optional settings for Pydantic span processing.
                  If not provided, defaults to PydanticProcessorSettings() (track_usage=True).

    Returns:
        A TracerProvider configured for Pydantic AI with the specified settings.

    Example:
        from pydantic_ai.models.instrumented import InstrumentationSettings
        from paid.tracing import get_paid_tracer_provider_pydantic, PydanticProcessorSettings

        # Get a provider that tracks usage (default)
        instrumentation = InstrumentationSettings(
            tracer_provider=get_paid_tracer_provider_pydantic(),
        )

        # Get a provider that filters out usage/cost data
        instrumentation = InstrumentationSettings(
            tracer_provider=get_paid_tracer_provider_pydantic(
                PydanticProcessorSettings(track_usage=False)
            ),
        )
    """
    effective_settings = settings if settings is not None else PydanticProcessorSettings()
    return PydanticTracerProvider(effective_settings)


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
        logger.debug("[paid:distributed] trace_sync_ using override trace_id=%s",
                     format(override_trace_id, '032x') if isinstance(override_trace_id, int) else str(override_trace_id))
    else:
        logger.debug("[paid:distributed] trace_sync_ no override trace_id, using auto-generated")

    try:
        tracer = get_paid_tracer()
        logger.debug("[paid:span] trace_sync_ creating parent_span for customer_id=%s, agent_id=%s, fn=%s",
                      external_customer_id, external_agent_id, getattr(fn, '__name__', repr(fn)))
        with tracer.start_as_current_span("parent_span", context=ctx) as span:
            logger.debug("[paid:span] trace_sync_ span created, trace_id=%s",
                         format(span.get_span_context().trace_id, '032x') if isinstance(span.get_span_context().trace_id, int) else str(span.get_span_context().trace_id))
            try:
                result = fn(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                logger.debug("[paid:span] trace_sync_ fn=%s completed successfully", getattr(fn, '__name__', repr(fn)))
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
        logger.debug("[paid:distributed] trace_async_ using override trace_id=%s",
                     format(override_trace_id, '032x') if isinstance(override_trace_id, int) else str(override_trace_id))
    else:
        logger.debug("[paid:distributed] trace_async_ no override trace_id, using auto-generated")

    try:
        tracer = get_paid_tracer()
        logger.debug("[paid:span] trace_async_ creating parent_span for customer_id=%s, agent_id=%s, fn=%s",
                      external_customer_id, external_agent_id, getattr(fn, '__name__', repr(fn)))
        with tracer.start_as_current_span("parent_span", context=ctx) as span:
            logger.debug("[paid:span] trace_async_ span created, trace_id=%s",
                         format(span.get_span_context().trace_id, '032x') if isinstance(span.get_span_context().trace_id, int) else str(span.get_span_context().trace_id))
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                logger.debug("[paid:span] trace_async_ fn=%s completed successfully", getattr(fn, '__name__', repr(fn)))
                return result
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise
    finally:
        ContextData.reset_context()
