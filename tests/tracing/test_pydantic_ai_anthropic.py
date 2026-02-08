"""Pydantic-AI + Anthropic + Paid tracing tests. Verifies span attributes match agent results.

Record cassettes: ANTHROPIC_API_KEY=sk-... poetry run pytest tests/tracing/ --record-mode=once -rP
"""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from paid.tracing.autoinstrumentation import paid_autoinstrument
from paid.tracing.tracing import PydanticProcessorSettings, get_paid_tracer_provider_pydantic, trace_sync_

from .conftest import ANTHROPIC_MODEL

ATTR_OPERATION = "gen_ai.operation.name"
ATTR_GEN_SYSTEM = "gen_ai.system"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_TOOL_DEFS = "gen_ai.tool.definitions"


def _instr(provider):
    return InstrumentationSettings(tracer_provider=get_paid_tracer_provider_pydantic())


def _instr_no_usage(provider):
    return InstrumentationSettings(tracer_provider=get_paid_tracer_provider_pydantic(PydanticProcessorSettings(track_usage=False)))


def _gen_ai_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.attributes and s.attributes.get(ATTR_OPERATION)]


def _assert_span_matches(span, result):
    """Cross-check span attributes against AgentRunResult."""
    attrs = dict(span.attributes) if span.attributes else {}
    usage = result.usage()
    assert attrs.get(ATTR_GEN_SYSTEM) == "anthropic"
    if usage.input_tokens > 0:
        assert attrs.get(ATTR_INPUT_TOKENS) == usage.input_tokens
    if usage.output_tokens > 0:
        assert attrs.get(ATTR_OUTPUT_TOKENS) == usage.output_tokens
    assert span.status.status_code.name in ("OK", "UNSET")


class TestBasicRuns:

    @pytest.mark.vcr()
    def test_run_sync(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        result = agent.run_sync("Say hello in exactly 3 words.")
        assert isinstance(result.output, str) and len(result.output) > 0
        assert result.usage().input_tokens > 0 and result.usage().output_tokens > 0
        spans = _gen_ai_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches(spans[0], result)

    @pytest.mark.vcr()
    async def test_run_async(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        result = await agent.run("Say hello in exactly 3 words.")
        assert isinstance(result.output, str) and len(result.output) > 0
        spans = _gen_ai_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches(spans[0], result)

    @pytest.mark.vcr()
    async def test_run_stream_text(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        async with agent.run_stream("Say hello in exactly 3 words.") as stream:
            chunks = [t async for t in stream.stream_text()]
        assert len(chunks[-1] if chunks else "") > 0
        assert stream.usage().input_tokens > 0 and stream.usage().output_tokens > 0
        assert len(_gen_ai_spans(exporter)) >= 1

    @pytest.mark.vcr()
    async def test_run_stream_get_output(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        async with agent.run_stream("Say hello in exactly 3 words.") as stream:
            output = await stream.get_output()
        assert isinstance(output, str) and len(output) > 0
        assert len(_gen_ai_spans(exporter)) >= 1

    @pytest.mark.vcr()
    async def test_iter_nodes(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        async with agent.iter("Say hello in exactly 3 words.") as run:
            async for _ in run:
                pass
        assert run.result is not None and len(run.result.output) > 0
        assert len(_gen_ai_spans(exporter)) >= 1


class CityInfo(BaseModel):
    city: str
    country: str


class TestStructuredOutput:

    @pytest.mark.vcr()
    def test_structured_output_pydantic_model(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, output_type=CityInfo, instrument=_instr(tracing_provider))
        result = agent.run_sync("Where were the 2012 Olympics held?")
        assert isinstance(result.output, CityInfo) and result.output.city.lower() == "london"
        spans = _gen_ai_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches(spans[0], result)

    @pytest.mark.vcr()
    def test_structured_output_list_of_strings(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, output_type=list[str], instrument=_instr(tracing_provider))
        result = agent.run_sync("List 3 colors. Return only the list.")
        assert isinstance(result.output, list) and len(result.output) >= 2
        assert len(_gen_ai_spans(exporter)) >= 1


class TestToolCalling:

    @pytest.mark.vcr()
    def test_tool_plain(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, system_prompt="Use the get_temperature tool to answer.", instrument=_instr(tracing_provider))

        @agent.tool_plain
        def get_temperature(city: str) -> str:
            """Get the current temperature for a city."""
            return f"72F in {city}"

        result = agent.run_sync("What is the temperature in Paris?")
        assert "72" in result.output or "paris" in result.output.lower()
        assert result.usage().tool_calls >= 1
        spans = _gen_ai_spans(exporter)
        assert len(spans) >= 1
        assert dict(spans[0].attributes).get(ATTR_TOOL_DEFS) is not None

    @pytest.mark.vcr()
    def test_tool_with_deps(self, tracing_setup, tracing_provider):
        from pydantic_ai import RunContext

        @dataclass
        class MyDeps:
            user_name: str

        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, deps_type=MyDeps, system_prompt="Use the greet tool.", instrument=_instr(tracing_provider))

        @agent.tool
        def greet(ctx: RunContext[MyDeps]) -> str:
            """Greet the user by name."""
            return f"Hello, {ctx.deps.user_name}!"

        result = agent.run_sync("Please greet me.", deps=MyDeps(user_name="Alice"))
        assert "alice" in result.output.lower()
        assert len(_gen_ai_spans(exporter)) >= 1

    @pytest.mark.vcr()
    def test_multiple_tool_calls(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, system_prompt="Always use both tools to answer completely.", instrument=_instr(tracing_provider))

        @agent.tool_plain
        def get_population(city: str) -> str:
            """Get population of a city."""
            return f"{city}: 2.1 million"

        @agent.tool_plain
        def get_area(city: str) -> str:
            """Get area of a city in sq km."""
            return f"{city}: 105 sq km"

        result = agent.run_sync("What are the population and area of Paris?")
        assert isinstance(result.output, str)
        assert len(_gen_ai_spans(exporter)) >= 2


class TestSystemPrompts:

    @pytest.mark.vcr()
    def test_static_system_prompt(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, system_prompt="You are a pirate. Always respond in pirate speak.", instrument=_instr(tracing_provider))
        result = agent.run_sync("Say hello.")
        assert isinstance(result.output, str) and result.usage().input_tokens > 0
        _assert_span_matches(_gen_ai_spans(exporter)[0], result)

    @pytest.mark.vcr()
    def test_dynamic_system_prompt(self, tracing_setup, tracing_provider):
        from pydantic_ai import RunContext

        @dataclass
        class UserDeps:
            language: str

        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, deps_type=UserDeps, instrument=_instr(tracing_provider))

        @agent.system_prompt
        def custom_prompt(ctx: RunContext[UserDeps]) -> str:
            return f"Always respond in {ctx.deps.language}."

        result = agent.run_sync("Say hello.", deps=UserDeps(language="French"))
        assert isinstance(result.output, str) and result.usage().input_tokens > 0
        assert len(_gen_ai_spans(exporter)) >= 1


class TestMultiTurn:

    @pytest.mark.vcr()
    def test_multi_turn_with_history(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        result1 = agent.run_sync("My name is Alice.")
        result2 = agent.run_sync("What is my name?", message_history=result1.all_messages())
        assert "alice" in result2.output.lower()
        assert len(_gen_ai_spans(exporter)) >= 2


class TestPydanticSpanProcessor:

    @pytest.mark.vcr()
    def test_track_usage_true(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        agent.run_sync("Say hello.")
        attrs = dict(_gen_ai_spans(exporter)[0].attributes)
        assert attrs.get(ATTR_INPUT_TOKENS) is not None and attrs[ATTR_INPUT_TOKENS] > 0
        assert attrs.get(ATTR_OUTPUT_TOKENS) is not None and attrs[ATTR_OUTPUT_TOKENS] > 0

    @pytest.mark.vcr()
    def test_track_usage_false(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr_no_usage(tracing_provider))
        agent.run_sync("Say hello.")
        attrs = dict(_gen_ai_spans(exporter)[0].attributes)
        assert attrs.get(ATTR_INPUT_TOKENS) is None
        assert attrs.get(ATTR_OUTPUT_TOKENS) is None


class TestContextPropagation:

    @pytest.mark.vcr()
    def test_spans_have_customer_id(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        result = trace_sync_(external_customer_id="cust-pydantic", fn=lambda: agent.run_sync("Say hello."))
        assert result.output
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_customer_id") == "cust-pydantic"

    @pytest.mark.vcr()
    def test_spans_have_agent_id(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        result = trace_sync_(external_customer_id="cust", fn=lambda: agent.run_sync("Say hello."), external_agent_id="agent-pydantic")
        assert result.output
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_agent_id") == "agent-pydantic"


class TestPromptFiltering:

    @pytest.mark.vcr()
    def test_prompt_content_filtered_by_default(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        trace_sync_(external_customer_id="cust", fn=lambda: agent.run_sync("Say hello."))
        attrs = dict(_gen_ai_spans(exporter)[0].attributes)
        for key in attrs:
            assert "gen_ai.input.messages" not in key and "gen_ai.output.messages" not in key
        assert attrs.get(ATTR_INPUT_TOKENS) is not None

    @pytest.mark.vcr()
    def test_prompt_content_kept_with_store_prompt(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        trace_sync_(external_customer_id="cust", fn=lambda: agent.run_sync("Say hello."), store_prompt=True)
        attrs = dict(_gen_ai_spans(exporter)[0].attributes)
        assert any("gen_ai.input.messages" in k or "gen_ai.output.messages" in k or "model_request_parameters" in k for k in attrs)


class TestCombinedInstrumentation:

    @pytest.mark.vcr()
    def test_pydantic_and_anthropic_autoinstrumentation(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        paid_autoinstrument(libraries=["anthropic"])
        agent = Agent(ANTHROPIC_MODEL, instrument=_instr(tracing_provider))
        result = agent.run_sync("Say hello.")
        assert result.output
        assert len(_gen_ai_spans(exporter)) >= 1
        assert len(exporter.get_finished_spans()) >= len(_gen_ai_spans(exporter))
