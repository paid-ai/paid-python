"""Google GenAI (Gemini) autoinstrumentation tests. Verifies OTEL span attributes match actual API responses.

Record cassettes:
    GEMINI_API_KEY=... poetry run pytest tests/tracing/test_gemini_autoinstrumentation.py --record-mode=all -rP
"""

import pytest
from google import genai

from paid.tracing.autoinstrumentation import paid_autoinstrument
from paid.tracing.tracing import trace_async_, trace_sync_

from .conftest import (
    GEMINI_COUNT_TOKENS_PARAMS,
    GEMINI_EMBED_PARAMS,
    GEMINI_FORCED_TOOL_PARAMS,
    GEMINI_MULTI_TURN_PARAMS,
    GEMINI_SIMPLE_PARAMS,
    GEMINI_STOP_SEQUENCES_PARAMS,
    GEMINI_STRUCTURED_OUTPUT_PARAMS,
    GEMINI_SYSTEM_PROMPT_PARAMS,
    GEMINI_THINKING_PARAMS,
    GEMINI_TOOL_USE_PARAMS,
)

ATTR_MODEL = "llm.model_name"
ATTR_TOKENS_PROMPT = "llm.token_count.prompt"
ATTR_TOKENS_COMPLETION = "llm.token_count.completion"
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_PROVIDER = "llm.provider"
ATTR_SYSTEM = "llm.system"
ATTR_RESPONSE_ID = "gen_ai.response.id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_generate_content_spans(exporter):
    """Get spans related to GenerateContent (non-streaming)."""
    return [
        s for s in exporter.get_finished_spans()
        if "GenerateContent" in s.name and "Stream" not in s.name
    ]


def _get_stream_spans(exporter):
    """Get spans related to GenerateContentStream."""
    return [s for s in exporter.get_finished_spans() if "GenerateContentStream" in s.name]



def _setup(tracing_setup):
    paid_autoinstrument(libraries=["google-genai"])
    return tracing_setup


def _assert_span_matches_response(span, response):
    """Cross-check span attributes against the Gemini response."""
    from paid.version import __version__

    attrs = dict(span.attributes) if span.attributes else {}

    # SDK version must be stamped on every span
    assert attrs.get("paid.sdk.version") == f"python-{__version__}"

    # Verify response ID is captured
    assert attrs.get(ATTR_RESPONSE_ID) == response.response_id, (
        f"Expected response ID '{response.response_id}', got '{attrs.get(ATTR_RESPONSE_ID)}'"
    )

    # Verify model name
    assert attrs.get(ATTR_MODEL) == response.model_version

    # Verify token counts
    if response.usage_metadata:
        if response.usage_metadata.prompt_token_count:
            assert attrs.get(ATTR_TOKENS_PROMPT) == response.usage_metadata.prompt_token_count
        # Completion tokens = candidates + thoughts (reasoning) tokens
        expected_completion = (
            (response.usage_metadata.candidates_token_count or 0)
            + (response.usage_metadata.thoughts_token_count or 0)
        )
        if expected_completion:
            assert attrs.get(ATTR_TOKENS_COMPLETION) == expected_completion


def _assert_streaming_span_has_token_counts(span):
    """Assert a streaming span has prompt/completion token counts and a response ID."""
    attrs = dict(span.attributes) if span.attributes else {}
    assert attrs.get(ATTR_TOKENS_PROMPT) is not None and attrs[ATTR_TOKENS_PROMPT] > 0, (
        f"Expected prompt tokens > 0, got {attrs.get(ATTR_TOKENS_PROMPT)}"
    )
    assert attrs.get(ATTR_TOKENS_COMPLETION) is not None and attrs[ATTR_TOKENS_COMPLETION] > 0, (
        f"Expected completion tokens > 0, got {attrs.get(ATTR_TOKENS_COMPLETION)}"
    )
    # Verify response ID is captured in streaming spans
    assert attrs.get(ATTR_RESPONSE_ID) is not None, (
        f"Expected gen_ai.response.id in streaming span, got attrs: {list(attrs.keys())}"
    )
    assert isinstance(attrs[ATTR_RESPONSE_ID], str) and len(attrs[ATTR_RESPONSE_ID]) > 0, (
        f"Expected non-empty response ID string, got '{attrs.get(ATTR_RESPONSE_ID)}'"
    )


# ===========================================================================
# Sync generate_content
# ===========================================================================

class TestSyncGenerateContent:

    @pytest.mark.vcr()
    def test_basic_generate_content(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_SIMPLE_PARAMS)
        assert response.text
        assert response.usage_metadata and response.usage_metadata.prompt_token_count
        assert response.usage_metadata.prompt_token_count > 0
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    def test_with_system_prompt(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_SYSTEM_PROMPT_PARAMS)
        assert response.text
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    def test_with_tools(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_TOOL_USE_PARAMS)
        assert response.candidates
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    def test_multi_turn(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_MULTI_TURN_PARAMS)
        assert response.text and "alice" in response.text.lower()
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)


# ===========================================================================
# Async generate_content
# ===========================================================================

class TestAsyncGenerateContent:

    @pytest.mark.vcr()
    async def test_basic_generate_content(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_SIMPLE_PARAMS)
        assert response.text
        assert response.usage_metadata and response.usage_metadata.prompt_token_count
        assert response.usage_metadata.prompt_token_count > 0
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_with_system_prompt(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_SYSTEM_PROMPT_PARAMS)
        assert response.text
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_with_tools(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_TOOL_USE_PARAMS)
        assert response.candidates
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_multi_turn(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_MULTI_TURN_PARAMS)
        assert response.text and "alice" in response.text.lower()
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)


# ===========================================================================
# Sync generate_content_stream
# ===========================================================================

class TestSyncGenerateContentStream:

    @pytest.mark.vcr()
    def test_stream_iteration(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        chunks = list(gemini_client.models.generate_content_stream(**GEMINI_SIMPLE_PARAMS))
        assert len(chunks) > 0
        # At least one chunk should have text
        assert any(c.text for c in chunks if c.text is not None)
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    def test_stream_collect_text(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        text_parts = []
        for chunk in gemini_client.models.generate_content_stream(**GEMINI_SIMPLE_PARAMS):
            if chunk.text:
                text_parts.append(chunk.text)
        full_text = "".join(text_parts)
        assert len(full_text) > 0
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])


# ===========================================================================
# Async generate_content_stream
# ===========================================================================

class TestAsyncGenerateContentStream:

    @pytest.mark.vcr()
    async def test_stream_iteration(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        chunks = []
        async for chunk in await gemini_client.aio.models.generate_content_stream(**GEMINI_SIMPLE_PARAMS):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert any(c.text for c in chunks if c.text is not None)
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    async def test_stream_collect_text(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        text_parts = []
        async for chunk in await gemini_client.aio.models.generate_content_stream(**GEMINI_SIMPLE_PARAMS):
            if chunk.text:
                text_parts.append(chunk.text)
        full_text = "".join(text_parts)
        assert len(full_text) > 0
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])


# ===========================================================================
# Streaming variants (different param combinations)
# ===========================================================================

class TestGeminiStreamingVariants:

    @pytest.mark.vcr()
    def test_sync_stream_with_system_prompt(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        chunks = list(gemini_client.models.generate_content_stream(**GEMINI_SYSTEM_PROMPT_PARAMS))
        assert len(chunks) > 0
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    async def test_async_stream_with_multi_turn(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        chunks = []
        async for chunk in await gemini_client.aio.models.generate_content_stream(**GEMINI_MULTI_TURN_PARAMS):
            chunks.append(chunk)
        assert len(chunks) > 0
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])

    @pytest.mark.vcr()
    def test_sync_stream_with_tools(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        chunks = list(gemini_client.models.generate_content_stream(**GEMINI_TOOL_USE_PARAMS))
        assert len(chunks) > 0
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])


# ===========================================================================
# Context propagation
# ===========================================================================

class TestGeminiContextPropagation:

    @pytest.mark.vcr()
    def test_spans_have_customer_id(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        result = trace_sync_(
            external_customer_id="cust-gemini",
            fn=lambda: gemini_client.models.generate_content(**GEMINI_SIMPLE_PARAMS),
        )
        assert result.text
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_customer_id") == "cust-gemini"

    @pytest.mark.vcr()
    def test_spans_have_agent_id(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        result = trace_sync_(
            external_customer_id="cust",
            external_agent_id="agent-gemini",
            fn=lambda: gemini_client.models.generate_content(**GEMINI_SIMPLE_PARAMS),
        )
        assert result.text
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_agent_id") == "agent-gemini"

    @pytest.mark.vcr()
    async def test_async_spans_have_customer_id(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)

        async def call():
            return await gemini_client.aio.models.generate_content(**GEMINI_SIMPLE_PARAMS)

        result = await trace_async_(external_customer_id="cust-async-gemini", fn=call)
        assert result.text  # type: ignore[union-attr]
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_customer_id") == "cust-async-gemini"

    @pytest.mark.vcr()
    def test_streaming_spans_have_customer_id(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)

        def call():
            chunks = list(gemini_client.models.generate_content_stream(**GEMINI_SIMPLE_PARAMS))
            return chunks

        result = trace_sync_(
            external_customer_id="cust-stream-gemini",
            fn=call,
        )
        assert len(result) > 0
        for span in exporter.get_finished_spans():
            assert span.attributes.get("external_customer_id") == "cust-stream-gemini"


# ===========================================================================
# Prompt filtering
# ===========================================================================

class TestGeminiPromptFiltering:

    @pytest.mark.vcr()
    def test_prompt_filtered_by_default(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = trace_sync_(
            external_customer_id="cust",
            fn=lambda: gemini_client.models.generate_content(**GEMINI_SIMPLE_PARAMS),
        )
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        for key in attrs:
            assert "input.value" not in key and "output.value" not in key

    @pytest.mark.vcr()
    def test_prompt_kept_with_store_prompt(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        trace_sync_(
            external_customer_id="cust",
            fn=lambda: gemini_client.models.generate_content(**GEMINI_SIMPLE_PARAMS),
            store_prompt=True,
        )
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        attrs = dict(spans[0].attributes)
        assert any("input.value" in k or "output.value" in k for k in attrs)


# ===========================================================================
# Count tokens
# ===========================================================================

class TestGeminiCountTokens:

    @pytest.mark.vcr()
    def test_sync_count_tokens(self, tracing_setup, gemini_client: genai.Client):
        _setup(tracing_setup)
        result = gemini_client.models.count_tokens(**GEMINI_COUNT_TOKENS_PARAMS)
        assert result.total_tokens and result.total_tokens > 0

    @pytest.mark.vcr()
    async def test_async_count_tokens(self, tracing_setup, gemini_client: genai.Client):
        _setup(tracing_setup)
        result = await gemini_client.aio.models.count_tokens(**GEMINI_COUNT_TOKENS_PARAMS)
        assert result.total_tokens and result.total_tokens > 0


# ===========================================================================
# Embed content
# ===========================================================================

class TestGeminiEmbedContent:

    @pytest.mark.vcr()
    def test_sync_embed_content(self, tracing_setup, gemini_client: genai.Client):
        _setup(tracing_setup)
        result = gemini_client.models.embed_content(**GEMINI_EMBED_PARAMS)
        assert result.embeddings and len(result.embeddings) > 0
        values = result.embeddings[0].values
        assert values and len(values) > 0

    @pytest.mark.vcr()
    async def test_async_embed_content(self, tracing_setup, gemini_client: genai.Client):
        _setup(tracing_setup)
        result = await gemini_client.aio.models.embed_content(**GEMINI_EMBED_PARAMS)
        assert result.embeddings and len(result.embeddings) > 0
        values = result.embeddings[0].values
        assert values and len(values) > 0


# ===========================================================================
# Structured output (JSON mode)
# ===========================================================================

class TestGeminiStructuredOutput:

    @pytest.mark.vcr()
    def test_sync_json_mode(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_STRUCTURED_OUTPUT_PARAMS)
        assert response.text
        import json
        parsed = json.loads(response.text)
        assert "capital" in parsed and "country" in parsed
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_async_json_mode(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_STRUCTURED_OUTPUT_PARAMS)
        assert response.text
        import json
        parsed = json.loads(response.text)
        assert "capital" in parsed and "country" in parsed
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    def test_sync_stream_json_mode(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        text_parts = []
        for chunk in gemini_client.models.generate_content_stream(**GEMINI_STRUCTURED_OUTPUT_PARAMS):
            if chunk.text:
                text_parts.append(chunk.text)
        full_text = "".join(text_parts)
        assert len(full_text) > 0
        import json
        parsed = json.loads(full_text)
        assert "capital" in parsed and "country" in parsed
        spans = _get_stream_spans(exporter)
        assert len(spans) >= 1
        _assert_streaming_span_has_token_counts(spans[0])


# ===========================================================================
# Thinking config
# ===========================================================================

class TestGeminiThinkingConfig:

    @pytest.mark.vcr()
    def test_sync_with_thinking_budget(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_THINKING_PARAMS)
        assert response.text
        # Verify that thinking tokens were used
        assert response.usage_metadata
        assert response.usage_metadata.thoughts_token_count and response.usage_metadata.thoughts_token_count > 0
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)
        # Explicitly verify thinking tokens contribute to completion count
        attrs = dict(spans[0].attributes) if spans[0].attributes else {}
        expected_completion = (
            (response.usage_metadata.candidates_token_count or 0)
            + (response.usage_metadata.thoughts_token_count or 0)
        )
        assert attrs.get(ATTR_TOKENS_COMPLETION) == expected_completion

    @pytest.mark.vcr()
    async def test_async_with_thinking_budget(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_THINKING_PARAMS)
        assert response.text
        assert response.usage_metadata
        assert response.usage_metadata.thoughts_token_count and response.usage_metadata.thoughts_token_count > 0
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)


# ===========================================================================
# Forced tool use (tool_config mode=ANY)
# ===========================================================================

class TestGeminiForcedToolUse:

    @pytest.mark.vcr()
    def test_sync_forced_tool_call(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_FORCED_TOOL_PARAMS)
        assert response.candidates
        # With mode=ANY, the model must call a function
        content = response.candidates[0].content
        assert content and content.parts
        assert any(p.function_call is not None for p in content.parts), "Expected a function_call part with tool_config mode=ANY"
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_async_forced_tool_call(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_FORCED_TOOL_PARAMS)
        assert response.candidates
        content = response.candidates[0].content
        assert content and content.parts
        assert any(p.function_call is not None for p in content.parts), "Expected a function_call part with tool_config mode=ANY"
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)


# ===========================================================================
# Stop sequences
# ===========================================================================

class TestGeminiStopSequences:

    @pytest.mark.vcr()
    def test_sync_stop_sequences(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = gemini_client.models.generate_content(**GEMINI_STOP_SEQUENCES_PARAMS)
        # Model should have stopped early due to the comma stop sequence
        assert response.text is not None
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)

    @pytest.mark.vcr()
    async def test_async_stop_sequences(self, tracing_setup, gemini_client: genai.Client):
        exporter = _setup(tracing_setup)
        response = await gemini_client.aio.models.generate_content(**GEMINI_STOP_SEQUENCES_PARAMS)
        assert response.text is not None
        spans = _get_generate_content_spans(exporter)
        assert len(spans) >= 1
        _assert_span_matches_response(spans[0], response)
