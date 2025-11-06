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
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    OPENAI_AGENTS_AVAILABLE = False

try:
    from openinference.instrumentation.bedrock import BedrockInstrumentor

    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False


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
                  - "gemini": Gemini library
                  - "openai": OpenAI library
                  - "openai-agents": OpenAI Agents SDK
                  - "bedrock": AWS Bedrock
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

    # Default to all supported libraries if none specified
    if libraries is None:
        libraries = ["anthropic", "gemini", "openai", "openai-agents", "bedrock"]

    for library in libraries:
        if library in _initialized_instrumentors:
            logger.warning(f"Instrumentation for {library} is already enabled, skipping.")
            continue

        if library == "anthropic":
            _instrument_anthropic()
        elif library == "gemini":
            _instrument_gemini()
        elif library == "openai":
            _instrument_openai()
        elif library == "openai-agents":
            _instrument_openai_agents()
        elif library == "bedrock":
            _instrument_bedrock()
        else:
            logger.warning(
                f"Unknown library '{library}' - supported libraries: anthropic, gemini, openai, openai-agents, bedrock"
            )

    logger.info(f"Auto-instrumentation enabled for: {', '.join(_initialized_instrumentors)}")


def _instrument_anthropic() -> None:
    """
    Instrument the Anthropic library using opentelemetry-instrumentation-anthropic.
    """
    if not ANTHROPIC_AVAILABLE:
        logger.warning("Anthropic library not available, skipping instrumentation")
        return

    # Instrument Anthropic with Paid's tracer provider
    AnthropicInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("anthropic")
    logger.info("Anthropic auto-instrumentation enabled")


def _instrument_gemini() -> None:
    """
    Instrument the Google Gemini library using opentelemetry-instrumentation-google-generativeai.
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini library not available, skipping instrumentation")
        return

    # Instrument Gemini with Paid's tracer provider
    GoogleGenerativeAiInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("gemini")
    logger.info("Gemini auto-instrumentation enabled")


def _instrument_openai() -> None:
    """
    Instrument the OpenAI library using opentelemetry-instrumentation-openai.
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI library not available, skipping instrumentation")
        return

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

    # Instrument Bedrock with Paid's tracer provider
    BedrockInstrumentor().instrument(tracer_provider=tracing.paid_tracer_provider)

    _initialized_instrumentors.append("bedrock")
    logger.info("Bedrock auto-instrumentation enabled")
