"""Tests for the core tracing infrastructure.

Covers:
- initialize_tracing()
- PaidSpanProcessor (name prefixing, attribute injection, prompt filtering)
- PydanticSpanProcessor (usage filtering)
- trace_sync_ / trace_async_ (span creation, exception propagation, context cleanup)
"""

import os
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import NoOpTracerProvider

from paid.tracing import tracing
from paid.tracing.context_data import ContextData
from paid.tracing.tracing import (
    PaidSpanProcessor,
    PydanticProcessorSettings,
    PydanticSpanProcessor,
    _PydanticSettingsRegistry,
    _TokenStore,
    get_paid_tracer,
    initialize_tracing,
    trace_async_,
    trace_sync_,
)

# ===================================================================
# Helpers
# ===================================================================

def _make_test_provider_and_exporter():
    """Create a standalone provider + in-memory exporter for isolated tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(PaidSpanProcessor())
    provider.add_span_processor(PydanticSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


# ===================================================================
# initialize_tracing()
# ===================================================================

class TestInitializeTracing:

    def setup_method(self):
        """Reset global state before each test."""
        self._original_provider = tracing.paid_tracer_provider
        self._original_token = _TokenStore.get()

    def teardown_method(self):
        """Restore global state after each test."""
        tracing.paid_tracer_provider = self._original_provider
        # Reset token store
        if self._original_token is not None:
            _TokenStore.set(self._original_token)
        else:
            # Force token to None by reaching into internals
            _TokenStore._TokenStore__token = None

    @patch.dict(os.environ, {"PAID_API_KEY": "test-api-key-123"}, clear=False)
    def test_initialize_creates_tracer_provider(self):
        """After init, paid_tracer_provider should be a real TracerProvider."""
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None

        initialize_tracing()

        assert isinstance(tracing.paid_tracer_provider, TracerProvider)
        assert not isinstance(tracing.paid_tracer_provider, NoOpTracerProvider)

    @patch.dict(os.environ, {"PAID_API_KEY": "test-api-key-123"}, clear=False)
    def test_initialize_prevents_reinit(self):
        """Calling initialize_tracing() twice should be a no-op on the second call."""
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None

        initialize_tracing()
        first_provider = tracing.paid_tracer_provider

        initialize_tracing()
        assert tracing.paid_tracer_provider is first_provider

    @patch.dict(os.environ, {"PAID_API_KEY": "from-env-var"}, clear=False)
    def test_initialize_from_env_var(self):
        """Reads PAID_API_KEY from env when no api_key param."""
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None

        initialize_tracing()

        assert _TokenStore.get() == "from-env-var"
        assert isinstance(tracing.paid_tracer_provider, TracerProvider)

    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_missing_api_key(self):
        """No key anywhere results in graceful degradation, not an exception."""
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None

        # Clear the env var if present
        os.environ.pop("PAID_API_KEY", None)
        os.environ.pop("PAID_ENABLED", None)
        os.environ.pop("PAID_OTEL_COLLECTOR_ENDPOINT", None)

        initialize_tracing()

        # Provider should still be NoOp since no API key
        assert isinstance(tracing.paid_tracer_provider, NoOpTracerProvider)

    @patch.dict(os.environ, {"PAID_ENABLED": "false", "PAID_API_KEY": "should-not-use"}, clear=False)
    def test_initialize_disabled_via_env(self):
        """PAID_ENABLED=false prevents initialization."""
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None

        initialize_tracing()

        assert isinstance(tracing.paid_tracer_provider, NoOpTracerProvider)

    def test_initialize_with_explicit_api_key(self):
        """Passing api_key directly should work."""
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None

        initialize_tracing(api_key="explicit-key")

        assert _TokenStore.get() == "explicit-key"
        assert isinstance(tracing.paid_tracer_provider, TracerProvider)


# ===================================================================
# PaidSpanProcessor
# ===================================================================

class TestPaidSpanProcessor:

    def test_span_name_prefixed(self, tracing_setup, tracing_provider):
        """Span names should get 'paid.trace.' prefix."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("foo", "bar")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "paid.trace.my_operation"

    def test_span_name_not_double_prefixed(self, tracing_setup, tracing_provider):
        """Already-prefixed spans should not get double-prefixed."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        with tracer.start_as_current_span("paid.trace.already_prefixed"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "paid.trace.already_prefixed"

    def test_customer_id_added_to_span(self, tracing_setup, tracing_provider):
        """external_customer_id from context should appear in span attributes."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        ContextData.set_context_key("external_customer_id", "cust-123")
        try:
            with tracer.start_as_current_span("test_op"):
                pass

            spans = exporter.get_finished_spans()
            assert spans[0].attributes["external_customer_id"] == "cust-123"
        finally:
            ContextData.reset_context()

    def test_agent_id_added_to_span(self, tracing_setup, tracing_provider):
        """external_agent_id from context should appear in span attributes."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        ContextData.set_context_key("external_agent_id", "agent-456")
        try:
            with tracer.start_as_current_span("test_op"):
                pass

            spans = exporter.get_finished_spans()
            assert spans[0].attributes["external_agent_id"] == "agent-456"
        finally:
            ContextData.reset_context()

    def test_metadata_flattened_on_span(self, tracing_setup, tracing_provider):
        """Nested metadata should be flattened to dot-notation keys."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        metadata = {"user": {"id": 123, "name": "Alice"}, "version": "v2"}
        ContextData.set_context_key("user_metadata", metadata)
        try:
            with tracer.start_as_current_span("test_op"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes)
            assert attrs["metadata.user.id"] == 123  # int is OTEL-safe, stays as int
            assert attrs["metadata.user.name"] == "Alice"
            assert attrs["metadata.version"] == "v2"
        finally:
            ContextData.reset_context()

    def test_prompt_attributes_filtered_by_default(self, tracing_setup, tracing_provider):
        """Prompt-related attributes should be stripped on_end by default."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        with tracer.start_as_current_span("test_op") as span:
            span.set_attribute("gen_ai.completion", "some completion text")
            span.set_attribute("gen_ai.prompt", "some prompt text")
            span.set_attribute("llm.input_messages", "user message")
            span.set_attribute("gen_ai.usage.input_tokens", 100)  # should survive

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert "gen_ai.completion" not in attrs
        assert "gen_ai.prompt" not in attrs
        assert "llm.input_messages" not in attrs
        # Usage attributes should survive prompt filtering
        assert attrs["gen_ai.usage.input_tokens"] == 100

    def test_prompt_attributes_kept_when_store_prompt(self, tracing_setup, tracing_provider):
        """When store_prompt=True, prompt attributes should be preserved."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        ContextData.set_context_key("store_prompt", True)
        try:
            with tracer.start_as_current_span("test_op") as span:
                span.set_attribute("gen_ai.completion", "some completion text")
                span.set_attribute("gen_ai.prompt", "some prompt text")

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes)
            assert "gen_ai.completion" in attrs
            assert "gen_ai.prompt" in attrs
        finally:
            ContextData.reset_context()

    def test_langchain_duplicate_spans_dropped(self, tracing_setup, tracing_provider):
        """Spans containing 'ChatOpenAI' or 'ChatAnthropic' should be dropped."""
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")

        # These should be dropped (the span processor raises to drop them)
        for name in ["ChatOpenAI", "ChatAnthropic"]:
            try:
                with tracer.start_as_current_span(name):
                    pass
            except Exception:
                pass

        # This should survive
        with tracer.start_as_current_span("legitimate_span"):
            pass

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert any("legitimate_span" in n for n in span_names)
        assert not any("ChatOpenAI" in n for n in span_names)
        assert not any("ChatAnthropic" in n for n in span_names)


# ===================================================================
# PydanticSpanProcessor
# ===================================================================

class TestPydanticSpanProcessor:

    def test_usage_attributes_filtered_when_track_usage_false(self, tracing_setup, tracing_provider):
        """Usage/cost attributes should be removed when track_usage=False."""
        exporter = tracing_setup

        scope_name = "test.pydantic.scope"
        _PydanticSettingsRegistry.register(scope_name, PydanticProcessorSettings(track_usage=False))

        try:
            tracer = tracing_provider.get_tracer(scope_name)
            with tracer.start_as_current_span("pydantic_op") as span:
                span.set_attribute("gen_ai.usage.input_tokens", 100)
                span.set_attribute("gen_ai.usage.output_tokens", 50)
                span.set_attribute("operation.cost", 0.01)
                span.set_attribute("gen_ai.system", "anthropic")  # should survive

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes)
            assert "gen_ai.usage.input_tokens" not in attrs
            assert "gen_ai.usage.output_tokens" not in attrs
            assert "operation.cost" not in attrs
            assert attrs.get("gen_ai.system") == "anthropic"
        finally:
            _PydanticSettingsRegistry._settings.pop(scope_name, None)

    def test_usage_attributes_kept_when_track_usage_true(self, tracing_setup, tracing_provider):
        """Usage attributes should be preserved when track_usage=True."""
        exporter = tracing_setup

        scope_name = "test.pydantic.scope.keep"
        _PydanticSettingsRegistry.register(scope_name, PydanticProcessorSettings(track_usage=True))

        try:
            tracer = tracing_provider.get_tracer(scope_name)
            with tracer.start_as_current_span("pydantic_op") as span:
                span.set_attribute("gen_ai.usage.input_tokens", 100)

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes)
            assert attrs.get("gen_ai.usage.input_tokens") == 100
        finally:
            _PydanticSettingsRegistry._settings.pop(scope_name, None)

    def test_non_pydantic_spans_untouched(self, tracing_setup, tracing_provider):
        """Spans from non-registered scopes should pass through unchanged."""
        exporter = tracing_setup

        tracer = tracing_provider.get_tracer("not.a.pydantic.scope")
        with tracer.start_as_current_span("regular_op") as span:
            span.set_attribute("gen_ai.usage.input_tokens", 100)
            span.set_attribute("operation.cost", 0.01)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        # PydanticSpanProcessor should not touch these since scope is not registered
        # (PaidSpanProcessor doesn't filter usage, only prompt content)
        assert attrs.get("gen_ai.usage.input_tokens") == 100


# ===================================================================
# trace_sync_ / trace_async_
# ===================================================================

class TestTraceSync:

    def test_trace_sync_creates_parent_span(self, tracing_setup):
        """Wrapping a sync function should create a parent span with OK status."""
        exporter = tracing_setup

        def my_func():
            return "hello"

        result = trace_sync_(
            external_customer_id="cust-1",
            fn=my_func,
            external_agent_id="agent-1",
        )

        assert result == "hello"
        spans = exporter.get_finished_spans()
        assert len(spans) >= 1
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        assert parent_span.status.status_code.name == "OK"
        assert parent_span.attributes.get("external_customer_id") == "cust-1"
        assert parent_span.attributes.get("external_agent_id") == "agent-1"

    def test_trace_sync_propagates_exception(self, tracing_setup):
        """Exception in wrapped fn should set ERROR status and re-raise."""
        exporter = tracing_setup

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            trace_sync_(
                external_customer_id="cust-err",
                fn=failing_func,
            )

        spans = exporter.get_finished_spans()
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        assert parent_span.status.status_code.name == "ERROR"

    def test_trace_sync_with_tracing_token(self, tracing_setup):
        """Override trace_id should propagate through span context."""
        exporter = tracing_setup
        custom_trace_id = 0x1234567890ABCDEF1234567890ABCDEF

        def my_func():
            return "traced"

        result = trace_sync_(
            external_customer_id="cust-token",
            fn=my_func,
            tracing_token=custom_trace_id,
        )

        assert result == "traced"
        spans = exporter.get_finished_spans()
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        assert parent_span.context.trace_id == custom_trace_id

    def test_context_reset_after_trace_sync(self, tracing_setup):
        """Context variables should be cleaned up after trace completes."""
        def my_func():
            # During execution, context should be set
            assert ContextData.get_context_key("external_customer_id") == "cust-ctx"
            return "done"

        trace_sync_(
            external_customer_id="cust-ctx",
            fn=my_func,
            external_agent_id="agent-ctx",
        )

        # After trace, context should be reset
        assert ContextData.get_context_key("external_customer_id") is None
        assert ContextData.get_context_key("external_agent_id") is None


class TestTraceAsync:

    async def test_trace_async_creates_parent_span(self, tracing_setup):
        """Wrapping an async function should create a parent span with OK status."""
        exporter = tracing_setup

        async def my_async_func():
            return "async hello"

        result = await trace_async_(
            external_customer_id="cust-async",
            fn=my_async_func,
            external_agent_id="agent-async",
        )

        assert result == "async hello"
        spans = exporter.get_finished_spans()
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        assert parent_span.status.status_code.name == "OK"
        assert parent_span.attributes.get("external_customer_id") == "cust-async"

    async def test_trace_async_propagates_exception(self, tracing_setup):
        """Exception in async wrapped fn should set ERROR status and re-raise."""
        exporter = tracing_setup

        async def failing_async():
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await trace_async_(
                external_customer_id="cust-async-err",
                fn=failing_async,
            )

        spans = exporter.get_finished_spans()
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        assert parent_span.status.status_code.name == "ERROR"

    async def test_trace_async_with_tracing_token(self, tracing_setup):
        """Override trace_id should propagate through span context for async."""
        exporter = tracing_setup
        custom_trace_id = 0xABCDEF1234567890ABCDEF1234567890

        async def my_async_func():
            return "traced async"

        result = await trace_async_(
            external_customer_id="cust-async-token",
            fn=my_async_func,
            tracing_token=custom_trace_id,
        )

        assert result == "traced async"
        spans = exporter.get_finished_spans()
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        assert parent_span.context.trace_id == custom_trace_id

    async def test_context_reset_after_trace_async(self, tracing_setup):
        """Context variables should be cleaned up after async trace completes."""

        async def my_async_func():
            assert ContextData.get_context_key("external_customer_id") == "cust-async-ctx"
            return "done"

        await trace_async_(
            external_customer_id="cust-async-ctx",
            fn=my_async_func,
            external_agent_id="agent-async-ctx",
        )

        assert ContextData.get_context_key("external_customer_id") is None
        assert ContextData.get_context_key("external_agent_id") is None


# ===================================================================
# trace_sync_ / trace_async_ with metadata and store_prompt
# ===================================================================

class TestTraceWithMetadata:

    def test_trace_sync_with_metadata(self, tracing_setup):
        """Metadata should be flattened and attached to spans."""
        exporter = tracing_setup

        def my_func():
            return "ok"

        trace_sync_(
            external_customer_id="cust-meta",
            fn=my_func,
            metadata={"env": "test", "nested": {"key": "val"}},
        )

        spans = exporter.get_finished_spans()
        parent_span = [s for s in spans if "parent_span" in s.name][0]
        attrs = dict(parent_span.attributes)
        assert attrs.get("metadata.env") == "test"
        assert attrs.get("metadata.nested.key") == "val"

    def test_trace_sync_with_store_prompt(self, tracing_setup):
        """store_prompt=True should preserve prompt attributes."""
        exporter = tracing_setup

        def my_func():
            # Simulate an inner span with prompt attributes
            tracer = get_paid_tracer()
            with tracer.start_as_current_span("inner") as span:
                span.set_attribute("gen_ai.prompt", "test prompt")
            return "ok"

        trace_sync_(
            external_customer_id="cust-prompt",
            fn=my_func,
            store_prompt=True,
        )

        spans = exporter.get_finished_spans()
        inner_span = [s for s in spans if "inner" in s.name][0]
        attrs = dict(inner_span.attributes)
        assert "gen_ai.prompt" in attrs
