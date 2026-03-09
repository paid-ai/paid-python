"""Tests for tool-call and tool-execution child spans.

Verifies that `paid_autoinstrument` emits TOOL-kind child spans when:
  - the model responds with a tool_use / function_call block  (tool-call span)
  - the next request includes a tool_result / tool-role message (tool-execution span)

Record cassettes:
    source .env && poetry run pytest tests/tracing/test_tool_spans.py --record-mode=once -rP
"""

import json

import pytest

from paid.tracing import autoinstrumentation, tracing
from paid.tracing.tool_spans import emit_tool_call_span, emit_tool_execution_span
from paid.tracing.tracing import PaidSpanProcessor, PydanticSpanProcessor

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

# ---------------------------------------------------------------------------
# Attribute constants
# ---------------------------------------------------------------------------

ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_TOOL_NAME = "tool_call.function.name"
ATTR_TOOL_CALL_ID = "tool_call.id"
ATTR_TOOL_ARGS = "tool_call.function.arguments"
ATTR_GEN_AI_SYSTEM = "gen_ai.system"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_exporter_and_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(PaidSpanProcessor())
    provider.add_span_processor(PydanticSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def _get_tool_spans(exporter):
    return [
        s for s in exporter.get_finished_spans()
        if s.attributes and s.attributes.get(ATTR_SPAN_KIND) == "TOOL"
    ]


def _get_llm_spans(exporter):
    return [
        s for s in exporter.get_finished_spans()
        if s.attributes and s.attributes.get(ATTR_SPAN_KIND) == "LLM"
    ]


# ===========================================================================
# Unit tests for emit_tool_call_span / emit_tool_execution_span helpers
# ===========================================================================

class TestEmitToolCallSpan:
    """Direct unit tests for the shared tool_spans.py helpers."""

    def setup_method(self):
        self.exporter, self.provider = _make_exporter_and_provider()
        self._orig = tracing.paid_tracer_provider
        tracing.paid_tracer_provider = self.provider

    def teardown_method(self):
        self.provider.shutdown()
        tracing.paid_tracer_provider = self._orig

    def test_emits_tool_call_span_with_correct_attributes(self):
        emit_tool_call_span(
            tool_name="get_weather",
            tool_call_id="call_abc123",
            arguments_json='{"location": "San Francisco, CA"}',
            gen_ai_system="anthropic",
        )
        spans = _get_tool_spans(self.exporter)
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs[ATTR_SPAN_KIND] == "TOOL"
        assert attrs[ATTR_TOOL_NAME] == "get_weather"
        assert attrs[ATTR_TOOL_CALL_ID] == "call_abc123"
        assert attrs[ATTR_TOOL_ARGS] == '{"location": "San Francisco, CA"}'
        assert attrs[ATTR_GEN_AI_SYSTEM] == "anthropic"
        assert spans[0].status.status_code.name == "OK"

    def test_tool_call_span_name_ends_with_tool_name(self):
        emit_tool_call_span(tool_name="search_web")
        spans = _get_tool_spans(self.exporter)
        # PaidSpanProcessor prefixes span names with "paid.trace."
        assert spans[0].name.endswith("search_web")

    def test_tool_call_span_optional_fields_absent_when_empty(self):
        emit_tool_call_span(tool_name="noop")
        attrs = dict(_get_tool_spans(self.exporter)[0].attributes)
        assert ATTR_TOOL_CALL_ID not in attrs
        assert ATTR_TOOL_ARGS not in attrs

    def test_emits_tool_execution_span(self):
        emit_tool_execution_span(
            tool_name="get_weather",
            tool_call_id="call_abc123",
            output_value="72F and sunny",
            gen_ai_system="openai",
        )
        spans = _get_tool_spans(self.exporter)
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs[ATTR_SPAN_KIND] == "TOOL"
        assert attrs[ATTR_TOOL_NAME] == "get_weather"
        assert attrs[ATTR_TOOL_CALL_ID] == "call_abc123"
        assert attrs[ATTR_GEN_AI_SYSTEM] == "openai"
        assert spans[0].status.status_code.name == "OK"

    def test_emitter_never_raises(self):
        from unittest.mock import MagicMock
        tracing.paid_tracer_provider = MagicMock(side_effect=RuntimeError("boom"))
        emit_tool_call_span(tool_name="safe")
        emit_tool_execution_span(tool_name="safe")
        tracing.paid_tracer_provider = self.provider


# ===========================================================================
# Anthropic – tool-call spans
# ===========================================================================

def _anthropic_setup(tracing_setup):
    from paid.tracing.autoinstrumentation import paid_autoinstrument
    paid_autoinstrument(libraries=["anthropic"])
    return tracing_setup


ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
        },
        "required": ["location"],
    },
}

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


class TestAnthropicToolCallSpans:

    @pytest.mark.vcr()
    def test_tool_call_span_emitted_on_tool_use_response(self, tracing_setup):
        """One TOOL span per tool_use block in the model response."""
        from anthropic import Anthropic

        exporter = _anthropic_setup(tracing_setup)
        client = Anthropic()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            tools=[ANTHROPIC_TOOL],
        )

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1

        tool_spans = _get_tool_spans(exporter)
        assert len(tool_spans) == len(tool_blocks)

        for span, block in zip(tool_spans, tool_blocks):
            attrs = dict(span.attributes)
            assert attrs[ATTR_SPAN_KIND] == "TOOL"
            assert attrs[ATTR_TOOL_NAME] == block.name
            assert attrs.get(ATTR_TOOL_CALL_ID) == block.id
            assert json.loads(attrs[ATTR_TOOL_ARGS]) == block.input

    @pytest.mark.vcr()
    def test_tool_call_span_is_child_of_llm_span(self, tracing_setup):
        """TOOL span must be a child of the parent LLM span."""
        from anthropic import Anthropic

        exporter = _anthropic_setup(tracing_setup)
        client = Anthropic()
        client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[ANTHROPIC_TOOL],
            tool_choice={"type": "any"},
        )

        llm_spans = _get_llm_spans(exporter)
        tool_spans = _get_tool_spans(exporter)
        assert len(llm_spans) >= 1
        assert len(tool_spans) >= 1
        llm_span = llm_spans[0]
        for ts in tool_spans:
            assert ts.parent is not None
            assert ts.parent.span_id == llm_span.context.span_id

    @pytest.mark.vcr()
    def test_tool_execution_span_emitted_for_tool_result(self, tracing_setup):
        """One TOOL span per tool_result block submitted back to the model."""
        from anthropic import Anthropic

        exporter = _anthropic_setup(tracing_setup)
        client = Anthropic()

        # Turn 1 – model calls the tool
        first = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[ANTHROPIC_TOOL],
            tool_choice={"type": "any"},
        )
        tool_block = next(b for b in first.content if b.type == "tool_use")
        exporter.clear()

        # Turn 2 – submit the tool result
        client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=256,
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": first.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "18°C and sunny",
                        }
                    ],
                },
            ],
            tools=[ANTHROPIC_TOOL],
        )

        tool_spans = _get_tool_spans(exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)
        assert attrs[ATTR_SPAN_KIND] == "TOOL"
        assert attrs.get(ATTR_TOOL_CALL_ID) == tool_block.id


# ===========================================================================
# OpenAI – tool-call spans
# ===========================================================================

def _openai_setup(tracing_setup):
    from paid.tracing.autoinstrumentation import paid_autoinstrument
    paid_autoinstrument(libraries=["openai"])
    return tracing_setup


OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
            },
            "required": ["location"],
        },
    },
}

OPENAI_MODEL = "gpt-4o-mini"


class TestOpenAIToolCallSpans:

    @pytest.mark.vcr()
    def test_tool_call_span_emitted_on_tool_call_response(self, tracing_setup):
        """One TOOL span per tool_call in the assistant message."""
        import openai

        exporter = _openai_setup(tracing_setup)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            tools=[OPENAI_TOOL],
            tool_choice="required",
        )

        tool_calls = [
            tc
            for choice in response.choices
            for tc in (choice.message.tool_calls or [])
        ]
        assert len(tool_calls) >= 1

        tool_spans = _get_tool_spans(exporter)
        assert len(tool_spans) == len(tool_calls)

        for span, tc in zip(tool_spans, tool_calls):
            attrs = dict(span.attributes)
            assert attrs[ATTR_SPAN_KIND] == "TOOL"
            assert attrs[ATTR_TOOL_NAME] == tc.function.name
            assert attrs.get(ATTR_TOOL_CALL_ID) == tc.id
            assert ATTR_TOOL_ARGS in attrs

    @pytest.mark.vcr()
    def test_tool_call_span_is_child_of_llm_span(self, tracing_setup):
        """TOOL span must be a child of the parent LLM span."""
        import openai

        exporter = _openai_setup(tracing_setup)
        client = openai.OpenAI()
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[OPENAI_TOOL],
            tool_choice="required",
        )

        llm_spans = _get_llm_spans(exporter)
        tool_spans = _get_tool_spans(exporter)
        assert len(llm_spans) >= 1
        assert len(tool_spans) >= 1
        llm_span = llm_spans[0]
        for ts in tool_spans:
            assert ts.parent is not None
            assert ts.parent.span_id == llm_span.context.span_id

    @pytest.mark.vcr()
    def test_tool_execution_span_emitted_for_tool_role_message(self, tracing_setup):
        """One TOOL span per role='tool' message submitted back to the model."""
        import openai

        exporter = _openai_setup(tracing_setup)
        client = openai.OpenAI()

        # Turn 1 – model calls the tool
        first = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[OPENAI_TOOL],
            tool_choice="required",
        )
        tool_call = first.choices[0].message.tool_calls[0]
        exporter.clear()

        # Turn 2 – submit the tool result
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                first.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "content": "18°C and sunny"},
            ],
            tools=[OPENAI_TOOL],
        )

        tool_spans = _get_tool_spans(exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)
        assert attrs[ATTR_SPAN_KIND] == "TOOL"
        assert attrs.get(ATTR_TOOL_CALL_ID) == tool_call.id


# ===========================================================================
# Gemini – tool-call spans
# ===========================================================================

def _gemini_setup(tracing_setup):
    from paid.tracing.autoinstrumentation import paid_autoinstrument
    paid_autoinstrument(libraries=["google-genai"])
    return tracing_setup


GEMINI_TOOL = {
    "function_declarations": [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
                },
                "required": ["location"],
            },
        }
    ]
}

GEMINI_MODEL = "gemini-2.5-flash"


class TestGeminiToolCallSpans:

    @pytest.mark.vcr()
    async def test_tool_call_span_emitted_on_function_call_response(self, tracing_setup, gemini_client):
        """One TOOL span per function_call part in the model response."""
        from .conftest import GEMINI_FORCED_TOOL_PARAMS

        exporter = _gemini_setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_FORCED_TOOL_PARAMS)

        fc_parts = [
            p
            for c in (response.candidates or [])
            if c.content and c.content.parts
            for p in c.content.parts
            if p.function_call is not None
        ]
        assert len(fc_parts) >= 1

        tool_spans = _get_tool_spans(exporter)
        assert len(tool_spans) == len(fc_parts)

        for span, part in zip(tool_spans, fc_parts):
            attrs = dict(span.attributes)
            assert attrs[ATTR_SPAN_KIND] == "TOOL"
            assert attrs[ATTR_TOOL_NAME] == part.function_call.name
            if part.function_call.args:
                assert ATTR_TOOL_ARGS in attrs

    @pytest.mark.vcr()
    async def test_tool_call_span_is_child_of_llm_span(self, tracing_setup, gemini_client):
        """TOOL span must be a child of the parent LLM span."""
        from .conftest import GEMINI_FORCED_TOOL_PARAMS

        exporter = _gemini_setup(tracing_setup)
        await gemini_client.aio.models.generate_content(**GEMINI_FORCED_TOOL_PARAMS)

        llm_spans = _get_llm_spans(exporter)
        tool_spans = _get_tool_spans(exporter)
        assert len(llm_spans) >= 1
        assert len(tool_spans) >= 1
        llm_span = llm_spans[0]
        for ts in tool_spans:
            assert ts.parent is not None
            assert ts.parent.span_id == llm_span.context.span_id

    @pytest.mark.vcr()
    async def test_tool_execution_span_emitted_for_function_response(self, tracing_setup, gemini_client):
        """One TOOL span per function_response part submitted back to the model."""
        from google.genai import types as genai_types

        exporter = _gemini_setup(tracing_setup)
        client = gemini_client

        # Turn 1 – model calls the tool
        first = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents="What is the weather in Paris?",
            config={"tools": [GEMINI_TOOL], "tool_config": {"function_calling_config": {"mode": "ANY"}}},
        )
        fc_part = next(
            p
            for c in (first.candidates or [])
            if c.content and c.content.parts
            for p in c.content.parts
            if p.function_call is not None
        )
        exporter.clear()

        # Turn 2 – submit the function response
        await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Content(role="user", parts=[genai_types.Part(text="What is the weather in Paris?")]),
                genai_types.Content(role="model", parts=[genai_types.Part(function_call=fc_part.function_call)]),
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=fc_part.function_call.name,
                                response={"result": "18°C and sunny"},
                            )
                        )
                    ],
                ),
            ],
            config={"tools": [GEMINI_TOOL]},
        )

        tool_spans = _get_tool_spans(exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)
        assert attrs[ATTR_SPAN_KIND] == "TOOL"
        assert attrs[ATTR_TOOL_NAME] == fc_part.function_call.name
