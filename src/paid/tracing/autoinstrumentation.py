"""
Auto-instrumentation for AI libraries using OpenTelemetry instrumentors.

This module provides automatic instrumentation for supported AI libraries,
sending traces to the Paid collector endpoint.
"""

from typing import List, Optional

from . import tracing
from .tracing import initialize_tracing
from opentelemetry.trace import NoOpTracerProvider

from paid.logger import logger

# Safe imports for instrumentation libraries
try:
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

    ANTHROPIC_AVAILABLE = True
except Exception:
    logger.debug("Anthropic instrumentation library not available, skipping instrumentation")
    ANTHROPIC_AVAILABLE = False

try:
    from openinference.instrumentation.openai import OpenAIInstrumentor

    OPENAI_AVAILABLE = True
except Exception:
    logger.debug("OpenAI instrumentation library not available, skipping instrumentation")
    OPENAI_AVAILABLE = False

try:
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    OPENAI_AGENTS_AVAILABLE = True
except Exception:
    logger.debug("OpenAI Agents instrumentation library not available, skipping instrumentation")
    OPENAI_AGENTS_AVAILABLE = False

try:
    from openinference.instrumentation.bedrock import BedrockInstrumentor

    BEDROCK_AVAILABLE = True
except Exception:
    logger.debug("Bedrock instrumentation library not available, skipping instrumentation")
    BEDROCK_AVAILABLE = False

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LANGCHAIN_AVAILABLE = True
except Exception:
    logger.debug("LangChain instrumentation library not available, skipping instrumentation")
    LANGCHAIN_AVAILABLE = False

try:
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    logger.debug("Google GenAI instrumentation library not available, skipping instrumentation")
    GOOGLE_GENAI_AVAILABLE = False

try:
    from openinference.instrumentation.instructor import InstructorInstrumentor

    INSTRUCTOR_AVAILABLE = True
except Exception:
    logger.debug("Instructor instrumentation library not available, skipping instrumentation")
    INSTRUCTOR_AVAILABLE = False

# Track which instrumentors have been initialized
_initialized_instrumentors: List[str] = []


def paid_autoinstrument(libraries: Optional[List[str]] = None) -> None:
    """
    Enable automatic instrumentation for AI libraries.

    This function sets up OpenTelemetry auto-instrumentation for the specified AI libraries,
    sending traces to the Paid collector. Traces will use Paid's isolated tracer provider
    to avoid interfering with any existing user OTEL setup.

    Args:
        libraries: List of library names to instrument. Currently supported:
                  - "anthropic": Anthropic library
                  - "openai": OpenAI library
                  - "openai-agents": OpenAI Agents SDK
                  - "bedrock": AWS Bedrock
                  - "langchain": LangChain library
                  - "google-genai": Google GenAI library
                  - "instructor": Instructor library
                  If None, all supported libraries that are installed will be instrumented.

    Note:
        If tracing is not already initialized, this function will automatically call
        _initialize_tracing() to set it up.

    Example:
        >>> from paid import Paid
        >>> from paid.tracing import paidAutoInstrument
        >>> from anthropic import Anthropic
        >>>
        >>> client = Paid(token="YOUR_API_KEY")
        >>> client.initialize_tracing()
        >>>
        >>> # Instrument only Anthropic
        >>> paidAutoInstrument(libraries=["anthropic"]) # empty args will instrument all it can
        >>>
        >>> # Now all Anthropic API calls will be automatically traced
        >>> anthropic_client = Anthropic()
        >>> # This call will be traced automatically
        >>> anthropic_client.messages.create(...)
    """
    # Initialize tracing if not already initialized
    if isinstance(tracing.paid_tracer_provider, NoOpTracerProvider):
        logger.info("Tracing not initialized, initializing automatically")
        initialize_tracing()

    logger.debug(
        "[paid:autoinstrument] tracer_provider type=%s, requested libraries=%s",
        type(tracing.paid_tracer_provider).__name__, libraries,
    )

    # Default to all supported libraries if none specified
    if libraries is None:
        libraries = ["anthropic", "openai", "openai-agents", "bedrock", "langchain", "google-genai", "instructor"]
        logger.debug("[paid:autoinstrument] No libraries specified, defaulting to all: %s", libraries)

    for library in libraries:
        if library in _initialized_instrumentors:
            logger.warning(f"Instrumentation for {library} is already enabled, skipping.")
            continue

        if library == "anthropic":
            _instrument_anthropic()
        elif library == "openai":
            _instrument_openai()
        elif library == "openai-agents":
            _instrument_openai_agents()
        elif library == "bedrock":
            _instrument_bedrock()
        elif library == "langchain":
            _instrument_langchain()
        elif library == "google-genai":
            _instrument_google_genai()
        elif library == "instructor":
            _instrument_instructor()
        else:
            logger.warning(
                f"Unknown library '{library}' - supported libraries: anthropic, openai, openai-agents, bedrock, langchain, google-genai, instructor"
            )

    logger.info(f"Auto-instrumentation enabled for: {', '.join(_initialized_instrumentors)}")


def _instrument_anthropic() -> None:
    """
    Instrument the Anthropic library using opentelemetry-instrumentation-anthropic.
    """
    if not ANTHROPIC_AVAILABLE:
        logger.warning("Anthropic library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting anthropic with AnthropicInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    # Instrument Anthropic with Paid's tracer provider
    AnthropicInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("anthropic")
    logger.info("Anthropic auto-instrumentation enabled")


def _instrument_openai() -> None:
    """
    Instrument the OpenAI library using opentelemetry-instrumentation-openai.
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting openai with OpenAIInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    # Instrument OpenAI with Paid's tracer provider
    OpenAIInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("openai")
    logger.info("OpenAI auto-instrumentation enabled")


def _instrument_openai_agents() -> None:
    """
    Instrument the OpenAI Agents SDK using openinference-instrumentation-openai-agents.
    """
    if not OPENAI_AGENTS_AVAILABLE:
        logger.warning("OpenAI Agents library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting openai-agents with OpenAIAgentsInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    # Instrument OpenAI Agents with Paid's tracer provider
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("openai-agents")
    logger.info("OpenAI Agents auto-instrumentation enabled")


def _instrument_bedrock() -> None:
    """
    Instrument AWS Bedrock using openinference-instrumentation-bedrock.
    """
    if not BEDROCK_AVAILABLE:
        logger.warning("Bedrock instrumentation library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting bedrock with BedrockInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    # Instrument Bedrock with Paid's tracer provider
    BedrockInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("bedrock")
    logger.info("Bedrock auto-instrumentation enabled")


def _instrument_langchain() -> None:
    """
    Instrument LangChain using openinference-instrumentation-langchain.
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain instrumentation library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting langchain with LangChainInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    # Instrument LangChain with Paid's tracer provider
    LangChainInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("langchain")
    logger.info("LangChain auto-instrumentation enabled")


def _instrument_google_genai() -> None:
    """
    Instrument Google GenAI using openinference-instrumentation-google-genai.
    """
    if not GOOGLE_GENAI_AVAILABLE:
        logger.warning("Google GenAI instrumentation library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting google-genai with GoogleGenAIInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)
    _initialized_instrumentors.append("google-genai")
    logger.info("Google GenAI auto-instrumentation enabled")


def _instrument_instructor() -> None:
    """
    Instrument the Instructor library using openinference-instrumentation-instructor.
    """
    if not INSTRUCTOR_AVAILABLE:
        logger.warning("Instructor library not available, skipping instrumentation")
        return

    logger.debug("[paid:autoinstrument] Instrumenting instructor with InstructorInstrumentor, provider=%s",
                 type(tracing.paid_tracer_provider).__name__)
    InstructorInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)
    _initialized_instrumentors.append("instructor")
    logger.info("Instructor auto-instrumentation enabled")
