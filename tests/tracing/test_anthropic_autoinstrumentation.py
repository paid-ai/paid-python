"""Anthropic autoinstrumentation tests. Verifies OTEL span attributes match actual API responses.

Record cassettes: ANTHROPIC_API_KEY=sk-... poetry run pytest tests/tracing/ --record-mode=once -rP
"""

import pytest
from .conftest import (
    ANTHROPIC_STOP_SEQUENCES_PARAMS,
    ANTHROPIC_TOOL_CHOICE_ANY_PARAMS,
    ANTHROPIC_TOOL_CHOICE_SPECIFIC_PARAMS,
    CACHE_CONTROL_PARAMS,
    COUNT_TOKENS_PARAMS,
    MULTI_TURN_PARAMS,
    SIMPLE_MESSAGE_PARAMS,
    SYSTEM_PROMPT_PARAMS,
    TOOL_USE_PARAMS,
)
from anthropic import Anthropic, AsyncAnthropic

from paid.tracing.autoinstrumentation import paid_autoinstrument
from paid.tracing.tracing import trace_async_, trace_sync_

ATTR_MODEL = "llm.model_name"
ATTR_TOKENS_PROMPT = "llm.token_count.prompt"
ATTR_TOKENS_COMPLETION = "llm.token_count.completion"
ATTR_TOKENS_TOTAL = "llm.token_count.total"
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_PROVIDER = "llm.provider"
ATTR_SYSTEM = "llm.system"
ATTR_CACHE_READ = "llm.token_count.prompt_details.cache_read"
ATTR_CACHE_WRITE = "llm.token_count.prompt_details.cache_write"
ATTR_RESPONSE_ID = "gen_ai.response.id"


def _get_message_spans(exporter):
    return [s for s in exporter.get_finished_spans() if "essage" in s.name]


def _setup(tracing_setup):
    paid_autoinstrument(libraries=["anthropic"])
    return tracing_setup


def _assert_span_matches_response(span, response):
    """Cross-check span attributes against the Anthropic response."""
    from paid.version import __version__

    attrs = dict(span.attributes) if span.attributes else {}
    usage = response.usage

    # SDK version must be stamped on every span
    assert attrs.get("paid.sdk.version") == f"python-{__version__}"

    assert attrs.get(ATTR_MODEL) == response.model
    assert attrs.get(ATTR_PROVIDER) == "anthropic"
    assert attrs.get(ATTR_SYSTEM) == "anthropic"
    assert attrs.get(ATTR_SPAN_KIND) == "LLM"

    # Verify response ID is captured
    assert attrs.get(ATTR_RESPONSE_ID) == response.id, (
        f"Expected response ID '{response.id}', got '{attrs.get(ATTR_RESPONSE_ID)}'"
    )

    expected_prompt = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    assert attrs.get(ATTR_TOKENS_PROMPT) == expected_prompt
    assert attrs.get(ATTR_TOKENS_COMPLETION) == usage.output_tokens
    if attrs.get(ATTR_TOKENS_TOTAL) is not None:
        assert attrs[ATTR_TOKENS_TOTAL] == expected_prompt + usage.output_tokens
    if usage.cache_read_input_tokens:
        assert attrs.get(ATTR_CACHE_READ) == usage.cache_read_input_tokens
    if usage.cache_creation_input_tokens:
        assert attrs.get(ATTR_CACHE_WRITE) == usage.cache_creation_input_tokens
    assert span.status.status_code.name == "OK"


def _assert_streaming_span_has_token_counts(span, expect_response_id: bool = True):
    attrs = dict(span.attributes) if span.attributes else {}
    assert attrs.get(ATTR_TOKENS_PROMPT) is not None and attrs[ATTR_TOKENS_PROMPT] > 0
    assert attrs.get(ATTR_TOKENS_COMPLETION) is not None and attrs[ATTR_TOKENS_COMPLETION] > 0
    if expect_response_id:
        assert attrs.get(ATTR_RESPONSE_ID) is not None, (
            f"Expected response ID in streaming span, got attrs: {list(attrs.keys())}"
        )
        assert attrs[ATTR_RESPONSE_ID].startswith("msg_"), (
            f"Expected response ID to start with 'msg_', got '{attrs[ATTR_RESPONSE_ID]}'"
        )


class TestSyncMessagesCreate:

    @pytest.mark.vcr()
    def test_basic_create(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS)
        assert response.content and response.usage.input_tokens > 0
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    def test_create_with_tools(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**TOOL_USE_PARAMS)
        assert response.content and response.stop_reason in ("tool_use", "end_turn")
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)
        assert any("llm.tools" in k for k in dict(spans[0].attributes))


class TestSyncMessagesCreateStream:

    @pytest.mark.vcr()
    def test_stream_iteration(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        events = list(anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS, stream=True))
        assert len(events) > 0 and "message_start" in {e.type for e in events}
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    def test_stream_as_context_manager(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        with anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS, stream=True) as stream:
            events = list(stream)
        assert len(events) > 0
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_streaming_span_has_token_counts(spans[0])


class TestSyncMessagesStream:

    @pytest.mark.vcr()
    def test_stream_text_stream(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        with anthropic_client.messages.stream(**SIMPLE_MESSAGE_PARAMS) as stream:
            text = "".join(stream.text_stream)
        assert len(text) > 0
        assert len(exporter.get_finished_spans()) >= 1

    @pytest.mark.vcr()
    def test_stream_get_final_message(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        with anthropic_client.messages.stream(**SIMPLE_MESSAGE_PARAMS) as stream:
            for _ in stream.text_stream:
                pass
            final = stream.get_final_message()
        assert final.content and final.usage.input_tokens > 0 and final.usage.output_tokens > 0
        assert len(exporter.get_finished_spans()) >= 1


class TestAsyncMessagesCreate:

    @pytest.mark.vcr()
    async def test_basic_create(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS)
        assert response.content and response.usage.input_tokens > 0
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_create_with_tools(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**TOOL_USE_PARAMS)
        assert response.content and response.stop_reason in ("tool_use", "end_turn")
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)
        assert any("llm.tools" in k for k in dict(spans[0].attributes))


class TestAsyncMessagesCreateStream:

    @pytest.mark.vcr()
    async def test_stream_iteration(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        stream = await async_anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS, stream=True)
        events = [e async for e in stream]
        assert len(events) > 0 and "message_start" in {e.type for e in events}
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    async def test_stream_as_async_context_manager(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS, stream=True)
        async with response as stream:
            events = [e async for e in stream]
        assert len(events) > 0
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_streaming_span_has_token_counts(spans[0])


class TestAsyncMessagesStream:

    @pytest.mark.vcr()
    async def test_stream_text_stream(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        async with async_anthropic_client.messages.stream(**SIMPLE_MESSAGE_PARAMS) as stream:
            text = "".join([t async for t in stream.text_stream])
        assert len(text) > 0
        assert len(exporter.get_finished_spans()) >= 1

    @pytest.mark.vcr()
    async def test_stream_get_final_message(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        async with async_anthropic_client.messages.stream(**SIMPLE_MESSAGE_PARAMS) as stream:
            async for _ in stream.text_stream:
                pass
            final = await stream.get_final_message()
        assert final.content and final.usage.input_tokens > 0 and final.usage.output_tokens > 0
        assert len(exporter.get_finished_spans()) >= 1


class TestContextPropagation:

    @pytest.mark.vcr()
    def test_autoinstrumented_spans_have_customer_id(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        result = trace_sync_(external_customer_id="cust-autoinstr", fn=lambda: anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS))
        assert result.content
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_customer_id") == "cust-autoinstr"

    @pytest.mark.vcr()
    def test_autoinstrumented_spans_have_agent_id(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        result = trace_sync_(external_customer_id="cust", fn=lambda: anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS), external_agent_id="agent-007")
        assert result.content
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_agent_id") == "agent-007"

    @pytest.mark.vcr()
    async def test_async_autoinstrumented_spans_have_customer_id(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)

        async def call():
            return await async_anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS)

        result = await trace_async_(external_customer_id="cust-async", fn=call)
        assert result.content  # type: ignore[union-attr]
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_customer_id") == "cust-async"


class TestPromptFiltering:

    @pytest.mark.vcr()
    def test_prompt_filtered_by_default(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = trace_sync_(external_customer_id="cust", fn=lambda: anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS))
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        for key in attrs:
            assert "input.value" not in key and "output.value" not in key
        assert attrs.get(ATTR_TOKENS_PROMPT) == (response.usage.input_tokens + (response.usage.cache_creation_input_tokens or 0) + (response.usage.cache_read_input_tokens or 0))

    @pytest.mark.vcr()
    def test_prompt_kept_with_store_prompt(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        trace_sync_(external_customer_id="cust", fn=lambda: anthropic_client.messages.create(**SIMPLE_MESSAGE_PARAMS), store_prompt=True)
        attrs = dict(_get_message_spans(exporter)[0].attributes)
        assert any("input.value" in k or "output.value" in k for k in attrs)


class TestCountTokens:

    @pytest.mark.vcr()
    def test_sync_count_tokens(self, tracing_setup, anthropic_client: Anthropic):
        _setup(tracing_setup)
        assert anthropic_client.messages.count_tokens(**COUNT_TOKENS_PARAMS).input_tokens > 0

    @pytest.mark.vcr()
    async def test_async_count_tokens(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        _setup(tracing_setup)
        assert (await async_anthropic_client.messages.count_tokens(**COUNT_TOKENS_PARAMS)).input_tokens > 0


class TestSystemPrompt:

    @pytest.mark.vcr()
    def test_sync_system_prompt(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**SYSTEM_PROMPT_PARAMS)
        assert response.content
        _assert_span_matches_response(_get_message_spans(exporter)[0], response)

    @pytest.mark.vcr()
    async def test_async_system_prompt(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**SYSTEM_PROMPT_PARAMS)
        assert response.content
        _assert_span_matches_response(_get_message_spans(exporter)[0], response)


class TestMultiTurn:

    @pytest.mark.vcr()
    def test_sync_multi_turn(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**MULTI_TURN_PARAMS)
        assert "alice" in response.content[0].text.lower()
        _assert_span_matches_response(_get_message_spans(exporter)[0], response)

    @pytest.mark.vcr()
    async def test_async_multi_turn(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**MULTI_TURN_PARAMS)
        assert "alice" in response.content[0].text.lower()
        _assert_span_matches_response(_get_message_spans(exporter)[0], response)


class TestPromptCaching:

    @pytest.mark.vcr()
    def test_sync_cache_control(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**CACHE_CONTROL_PARAMS)
        assert response.content and response.usage.input_tokens > 0
        _assert_span_matches_response(_get_message_spans(exporter)[0], response)

    @pytest.mark.vcr()
    async def test_async_cache_control(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**CACHE_CONTROL_PARAMS)
        assert response.content and response.usage.input_tokens > 0
        _assert_span_matches_response(_get_message_spans(exporter)[0], response)


class TestStreamingVariants:

    @pytest.mark.vcr()
    def test_sync_stream_with_system_prompt(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        events = list(anthropic_client.messages.create(**SYSTEM_PROMPT_PARAMS, stream=True))
        assert len(events) > 0
        _assert_streaming_span_has_token_counts(_get_message_spans(exporter)[0])

    @pytest.mark.vcr()
    async def test_async_stream_with_multi_turn(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        stream = await async_anthropic_client.messages.create(**MULTI_TURN_PARAMS, stream=True)
        events = [e async for e in stream]
        assert len(events) > 0
        _assert_streaming_span_has_token_counts(_get_message_spans(exporter)[0])


class TestBetaMessages:

    @pytest.mark.vcr()
    def test_sync_beta_messages_create(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        r = anthropic_client.beta.messages.create(**SIMPLE_MESSAGE_PARAMS)
        assert r.content and r.usage.input_tokens > 0 and r.usage.output_tokens > 0
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], r)

    @pytest.mark.vcr()
    async def test_async_beta_messages_create(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        r = await async_anthropic_client.beta.messages.create(**SIMPLE_MESSAGE_PARAMS)
        assert r.content and r.usage.input_tokens > 0 and r.usage.output_tokens > 0
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], r)

    @pytest.mark.vcr()
    def test_sync_beta_messages_create_stream(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        events = list(anthropic_client.beta.messages.create(**SIMPLE_MESSAGE_PARAMS, stream=True))
        assert len(events) > 0 and "message_start" in {e.type for e in events}
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    async def test_async_beta_messages_create_stream(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        events = [e async for e in await async_anthropic_client.beta.messages.create(**SIMPLE_MESSAGE_PARAMS, stream=True)]
        assert len(events) > 0 and "message_start" in {e.type for e in events}
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_streaming_span_has_token_counts(spans[0])


class TestToolChoice:

    @pytest.mark.vcr()
    def test_sync_tool_choice_any(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**ANTHROPIC_TOOL_CHOICE_ANY_PARAMS)
        # With tool_choice=any, the model must use a tool
        assert response.stop_reason == "tool_use"
        assert any(b.type == "tool_use" for b in response.content)
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)
        assert any("llm.tools" in k for k in dict(spans[0].attributes))

    @pytest.mark.vcr()
    async def test_async_tool_choice_any(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**ANTHROPIC_TOOL_CHOICE_ANY_PARAMS)
        assert response.stop_reason == "tool_use"
        assert any(b.type == "tool_use" for b in response.content)
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    def test_sync_tool_choice_specific(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**ANTHROPIC_TOOL_CHOICE_SPECIFIC_PARAMS)
        # With tool_choice=tool(name=get_weather), must call exactly get_weather
        assert response.stop_reason == "tool_use"
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1
        assert tool_blocks[0].name == "get_weather"
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_async_tool_choice_specific(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**ANTHROPIC_TOOL_CHOICE_SPECIFIC_PARAMS)
        assert response.stop_reason == "tool_use"
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1
        assert tool_blocks[0].name == "get_weather"
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)


class TestStopSequences:

    @pytest.mark.vcr()
    def test_sync_stop_sequences(self, tracing_setup, anthropic_client: Anthropic):
        exporter = _setup(tracing_setup)
        response = anthropic_client.messages.create(**ANTHROPIC_STOP_SEQUENCES_PARAMS)
        # Model should stop at the comma
        assert response.stop_reason == "stop_sequence"
        assert response.content
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_async_stop_sequences(self, tracing_setup, async_anthropic_client: AsyncAnthropic):
        exporter = _setup(tracing_setup)
        response = await async_anthropic_client.messages.create(**ANTHROPIC_STOP_SEQUENCES_PARAMS)
        assert response.stop_reason == "stop_sequence"
        assert response.content
        spans = _get_message_spans(exporter)
        assert len(spans) == 1
        _assert_span_matches_response(spans[0], response)
