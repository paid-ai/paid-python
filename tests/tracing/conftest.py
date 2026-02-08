import os
from typing import Generator

import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from paid.tracing import autoinstrumentation, tracing
from paid.tracing.context_data import ContextData
from paid.tracing.tracing import PaidSpanProcessor, PydanticSpanProcessor


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["x-api-key", "anthropic-api-key", "authorization"],
        "cassette_library_dir": os.path.join(os.path.dirname(__file__), "cassettes"),
    }


@pytest.fixture()
def in_memory_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracing_provider(in_memory_exporter: InMemorySpanExporter) -> Generator[TracerProvider, None, None]:
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(PaidSpanProcessor())
    provider.add_span_processor(PydanticSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))

    original_provider = tracing.paid_tracer_provider
    tracing.paid_tracer_provider = provider
    yield provider
    provider.shutdown()
    tracing.paid_tracer_provider = original_provider


@pytest.fixture()
def tracing_setup(
    in_memory_exporter: InMemorySpanExporter,
    tracing_provider: TracerProvider,
) -> Generator[InMemorySpanExporter, None, None]:
    yield in_memory_exporter

    ContextData.reset_context()
    autoinstrumentation._initialized_instrumentors.clear()

    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from paid.tracing.anthropic_patches import uninstrument_anthropic
        uninstrument_anthropic()
    except Exception:
        pass


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "test-key-for-cassettes")
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY


@pytest.fixture()
def anthropic_client() -> Anthropic:
    return Anthropic(api_key=ANTHROPIC_API_KEY)


@pytest.fixture()
def async_anthropic_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


SIMPLE_MESSAGE_PARAMS = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
}

TOOL_USE_PARAMS = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
                },
                "required": ["location"],
            },
        }
    ],
}

SYSTEM_PROMPT_PARAMS = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "system": "You are a pirate. Respond in pirate speak only.",
    "messages": [{"role": "user", "content": "Say hello."}],
}

MULTI_TURN_PARAMS = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "messages": [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ],
}

COUNT_TOKENS_PARAMS = {
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}

CACHE_CONTROL_PARAMS = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "system": [
        {
            "type": "text",
            "text": "You are a helpful assistant that answers questions concisely.",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    "messages": [{"role": "user", "content": "What is 2+2?"}],
}

ANTHROPIC_MODEL = "anthropic:claude-sonnet-4-20250514"
