import os
from collections.abc import AsyncGenerator
from typing import Any, Generator

import pytest
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from paid.tracing import autoinstrumentation, tracing
from paid.tracing.context_data import ContextData
from paid.tracing.tracing import PaidSpanProcessor, PydanticSpanProcessor


def _scrub_response_headers(response):
    headers = response.get("headers", {})
    for h in ["anthropic-organization-id"]:
        headers.pop(h, None)
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["x-api-key", "anthropic-api-key", "authorization", "x-goog-api-key"],
        "before_record_response": _scrub_response_headers,
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

    active = list(autoinstrumentation._initialized_instrumentors)
    ContextData.reset_context()
    autoinstrumentation._initialized_instrumentors.clear()

    if "anthropic" in active:
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

    if "google-genai" in active:
        try:
            from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
            GoogleGenAIInstrumentor().uninstrument()
        except Exception:
            pass

        try:
            from paid.tracing.gemini_patches import uninstrument_google_genai
            uninstrument_google_genai()
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


SIMPLE_MESSAGE_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
}

TOOL_USE_PARAMS: dict[str, Any] = {
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

SYSTEM_PROMPT_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "system": "You are a pirate. Respond in pirate speak only.",
    "messages": [{"role": "user", "content": "Say hello."}],
}

MULTI_TURN_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 32,
    "messages": [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ],
}

COUNT_TOKENS_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}

CACHE_CONTROL_PARAMS: dict[str, Any] = {
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

ANTHROPIC_TOOL_CHOICE_ANY_PARAMS: dict[str, Any] = {
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
    "tool_choice": {"type": "any"},
}

ANTHROPIC_TOOL_CHOICE_SPECIFIC_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Tell me about the weather in London."}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. London, UK"}
                },
                "required": ["location"],
            },
        }
    ],
    "tool_choice": {"type": "tool", "name": "get_weather"},
}

ANTHROPIC_STOP_SEQUENCES_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Count from 1 to 10, separated by commas."}],
    "stop_sequences": [","],
}

ANTHROPIC_MODEL = "anthropic:claude-sonnet-4-20250514"
GEMINI_PYDANTIC_MODEL = "google-gla:gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Gemini fixtures & params
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "test-key-for-cassettes")
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

GEMINI_MODEL = "gemini-2.5-flash"


@pytest.fixture()
async def gemini_client() -> AsyncGenerator[genai.Client, None]:
    # Pass httpx_async_client to force the SDK to use httpx instead of aiohttp
    # for async requests.  VCR.py intercepts httpx reliably, whereas its
    # aiohttp stubs don't support SSE streaming which Gemini uses.
    import httpx

    async_client = httpx.AsyncClient()
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"httpx_async_client": async_client},  # type: ignore[arg-type]
    )
    yield client
    await async_client.aclose()


GEMINI_EMBED_MODEL = "gemini-embedding-001"

GEMINI_SIMPLE_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "Say hello in exactly 3 words.",
}

GEMINI_SYSTEM_PROMPT_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "Say hello.",
    "config": {"system_instruction": "You are a pirate. Respond in pirate speak only."},
}

GEMINI_MULTI_TURN_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": [
        genai_types.Content(role="user", parts=[genai_types.Part(text="My name is Alice.")]),
        genai_types.Content(role="model", parts=[genai_types.Part(text="Hello Alice! Nice to meet you.")]),
        genai_types.Content(role="user", parts=[genai_types.Part(text="What is my name?")]),
    ],
}

GEMINI_TOOL_USE_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "What is the weather in San Francisco?",
    "config": {
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ]
    },
}

GEMINI_COUNT_TOKENS_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "Hello, world!",
}

GEMINI_EMBED_PARAMS: dict[str, Any] = {
    "model": GEMINI_EMBED_MODEL,
    "contents": "Hello, world!",
}

GEMINI_STRUCTURED_OUTPUT_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "What is the capital of France? Return the country and its capital.",
    "config": {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["capital", "country"],
        },
    },
}

GEMINI_THINKING_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "What is 15 * 37?",
    "config": {"thinking_config": {"thinking_budget": 1024}},
}

GEMINI_FORCED_TOOL_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "What is the weather in San Francisco?",
    "config": {
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ],
        "tool_config": {"function_calling_config": {"mode": "ANY"}},
    },
}

GEMINI_STOP_SEQUENCES_PARAMS: dict[str, Any] = {
    "model": GEMINI_MODEL,
    "contents": "Count from 1 to 10, separated by commas.",
    "config": {"stop_sequences": [","]},
}
