import os

import pytest
from agents import Agent, Runner, function_tool

from paid.tracing.autoinstrumentation import paid_autoinstrument

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "test-key-for-cassettes")
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def _setup(tracing_setup):
    paid_autoinstrument(libraries=["openai-agents"])
    return tracing_setup


def _find_span(spans, *, kind: str, name: str | None = None):
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("openinference.span.kind") != kind:
            continue
        if name is not None and attrs.get("tool.name") != name and span.name != name:
            continue
        return span
    return None


class TestOpenAIAgentsAutoinstrumentation:

    @pytest.mark.vcr()
    def test_tool_span_includes_tool_metadata(self, tracing_setup):
        exporter = _setup(tracing_setup)

        @function_tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"It is 12C in {city}."

        agent = Agent(
            name="weather-agent",
            instructions="Use the get_weather tool for the user's city and stop after using it.",
            tools=[get_weather],
            model="gpt-4.1-nano-2025-04-14",
            tool_use_behavior="stop_on_first_tool",
        )

        result = Runner.run_sync(agent, "What's the weather in London?")
        assert "12C" in str(result.final_output)

        spans = exporter.get_finished_spans()
        llm_span = _find_span(spans, kind="LLM")
        tool_span = _find_span(spans, kind="TOOL", name="get_weather")

        assert llm_span is not None
        assert tool_span is not None

        llm_attrs = dict(llm_span.attributes or {})
        tool_attrs = dict(tool_span.attributes or {})

        assert any(key.startswith("llm.tools.") for key in llm_attrs)
        assert tool_attrs["tool.name"] == "get_weather"
        assert tool_attrs["tool.description"] == "Get the current weather for a city."
        assert tool_attrs["tool.parameters"] == (
            '{"properties": {"city": {"title": "City", "type": "string"}}, '
            '"required": ["city"], "title": "get_weather_args", "type": "object", '
            '"additionalProperties": false}'
        )
        assert "tool.json_schema" in tool_attrs
