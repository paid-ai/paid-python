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


class TestInitializeTracing:

    def setup_method(self):
        self._original_provider = tracing.paid_tracer_provider
        self._original_token = _TokenStore.get()

    def teardown_method(self):
        tracing.paid_tracer_provider = self._original_provider
        if self._original_token is not None:
            _TokenStore.set(self._original_token)
        else:
            _TokenStore._TokenStore__token = None

    @patch.dict(os.environ, {"PAID_API_KEY": "test-api-key-123"}, clear=False)
    def test_initialize_creates_tracer_provider(self):
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None
        initialize_tracing()
        assert isinstance(tracing.paid_tracer_provider, TracerProvider)
        assert not isinstance(tracing.paid_tracer_provider, NoOpTracerProvider)

    @patch.dict(os.environ, {"PAID_API_KEY": "test-api-key-123"}, clear=False)
    def test_initialize_prevents_reinit(self):
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None
        initialize_tracing()
        first_provider = tracing.paid_tracer_provider
        initialize_tracing()
        assert tracing.paid_tracer_provider is first_provider

    @patch.dict(os.environ, {"PAID_API_KEY": "from-env-var"}, clear=False)
    def test_initialize_from_env_var(self):
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None
        initialize_tracing()
        assert _TokenStore.get() == "from-env-var"
        assert isinstance(tracing.paid_tracer_provider, TracerProvider)

    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_missing_api_key(self):
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None
        os.environ.pop("PAID_API_KEY", None)
        os.environ.pop("PAID_ENABLED", None)
        os.environ.pop("PAID_OTEL_COLLECTOR_ENDPOINT", None)
        initialize_tracing()
        assert isinstance(tracing.paid_tracer_provider, NoOpTracerProvider)

    @patch.dict(os.environ, {"PAID_ENABLED": "false", "PAID_API_KEY": "should-not-use"}, clear=False)
    def test_initialize_disabled_via_env(self):
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None
        initialize_tracing()
        assert isinstance(tracing.paid_tracer_provider, NoOpTracerProvider)

    def test_initialize_with_explicit_api_key(self):
        tracing.paid_tracer_provider = NoOpTracerProvider()
        _TokenStore._TokenStore__token = None
        initialize_tracing(api_key="explicit-key")
        assert _TokenStore.get() == "explicit-key"
        assert isinstance(tracing.paid_tracer_provider, TracerProvider)


class TestPaidSpanProcessor:

    def test_span_name_prefixed(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("foo", "bar")
        assert exporter.get_finished_spans()[0].name == "paid.trace.my_operation"

    def test_span_name_not_double_prefixed(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        with tracer.start_as_current_span("paid.trace.already_prefixed"):
            pass
        assert exporter.get_finished_spans()[0].name == "paid.trace.already_prefixed"

    def test_customer_id_added_to_span(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        ContextData.set_context_key("external_customer_id", "cust-123")
        try:
            with tracer.start_as_current_span("test_op"):
                pass
            assert exporter.get_finished_spans()[0].attributes["external_customer_id"] == "cust-123"
        finally:
            ContextData.reset_context()

    def test_agent_id_added_to_span(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        ContextData.set_context_key("external_agent_id", "agent-456")
        try:
            with tracer.start_as_current_span("test_op"):
                pass
            assert exporter.get_finished_spans()[0].attributes["external_agent_id"] == "agent-456"
        finally:
            ContextData.reset_context()

    def test_metadata_flattened_on_span(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        ContextData.set_context_key("user_metadata", {"user": {"id": 123, "name": "Alice"}, "version": "v2"})
        try:
            with tracer.start_as_current_span("test_op"):
                pass
            attrs = dict(exporter.get_finished_spans()[0].attributes)
            assert attrs["metadata.user.id"] == 123
            assert attrs["metadata.user.name"] == "Alice"
            assert attrs["metadata.version"] == "v2"
        finally:
            ContextData.reset_context()

    def test_prompt_attributes_filtered_by_default(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        with tracer.start_as_current_span("test_op") as span:
            span.set_attribute("gen_ai.completion", "some text")
            span.set_attribute("gen_ai.prompt", "some text")
            span.set_attribute("llm.input_messages", "user message")
            span.set_attribute("gen_ai.usage.input_tokens", 100)
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert "gen_ai.completion" not in attrs
        assert "gen_ai.prompt" not in attrs
        assert "llm.input_messages" not in attrs
        assert attrs["gen_ai.usage.input_tokens"] == 100  # usage survives

    def test_prompt_attributes_kept_when_store_prompt(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        ContextData.set_context_key("store_prompt", True)
        try:
            with tracer.start_as_current_span("test_op") as span:
                span.set_attribute("gen_ai.completion", "some text")
                span.set_attribute("gen_ai.prompt", "some text")
            attrs = dict(exporter.get_finished_spans()[0].attributes)
            assert "gen_ai.completion" in attrs
            assert "gen_ai.prompt" in attrs
        finally:
            ContextData.reset_context()

    def test_langchain_duplicate_spans_dropped(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("test")
        for name in ["ChatOpenAI", "ChatAnthropic"]:
            try:
                with tracer.start_as_current_span(name):
                    pass
            except Exception:
                pass
        with tracer.start_as_current_span("legitimate_span"):
            pass
        span_names = [s.name for s in exporter.get_finished_spans()]
        assert any("legitimate_span" in n for n in span_names)
        assert not any("ChatOpenAI" in n for n in span_names)
        assert not any("ChatAnthropic" in n for n in span_names)


class TestPydanticSpanProcessor:

    def test_usage_attributes_filtered_when_track_usage_false(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        scope_name = "test.pydantic.scope"
        _PydanticSettingsRegistry.register(scope_name, PydanticProcessorSettings(track_usage=False))
        try:
            tracer = tracing_provider.get_tracer(scope_name)
            with tracer.start_as_current_span("pydantic_op") as span:
                span.set_attribute("gen_ai.usage.input_tokens", 100)
                span.set_attribute("gen_ai.usage.output_tokens", 50)
                span.set_attribute("operation.cost", 0.01)
                span.set_attribute("gen_ai.system", "anthropic")
            attrs = dict(exporter.get_finished_spans()[0].attributes)
            assert "gen_ai.usage.input_tokens" not in attrs
            assert "gen_ai.usage.output_tokens" not in attrs
            assert "operation.cost" not in attrs
            assert attrs.get("gen_ai.system") == "anthropic"
        finally:
            _PydanticSettingsRegistry._settings.pop(scope_name, None)

    def test_usage_attributes_kept_when_track_usage_true(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        scope_name = "test.pydantic.scope.keep"
        _PydanticSettingsRegistry.register(scope_name, PydanticProcessorSettings(track_usage=True))
        try:
            tracer = tracing_provider.get_tracer(scope_name)
            with tracer.start_as_current_span("pydantic_op") as span:
                span.set_attribute("gen_ai.usage.input_tokens", 100)
            assert dict(exporter.get_finished_spans()[0].attributes).get("gen_ai.usage.input_tokens") == 100
        finally:
            _PydanticSettingsRegistry._settings.pop(scope_name, None)

    def test_non_pydantic_spans_untouched(self, tracing_setup, tracing_provider):
        exporter = tracing_setup
        tracer = tracing_provider.get_tracer("not.a.pydantic.scope")
        with tracer.start_as_current_span("regular_op") as span:
            span.set_attribute("gen_ai.usage.input_tokens", 100)
            span.set_attribute("operation.cost", 0.01)
        assert dict(exporter.get_finished_spans()[0].attributes).get("gen_ai.usage.input_tokens") == 100


class TestTraceSync:

    def test_trace_sync_creates_parent_span(self, tracing_setup):
        exporter = tracing_setup
        result = trace_sync_(external_customer_id="cust-1", fn=lambda: "hello", external_agent_id="agent-1")
        assert result == "hello"
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        assert parent.status.status_code.name == "OK"
        assert parent.attributes.get("external_customer_id") == "cust-1"
        assert parent.attributes.get("external_agent_id") == "agent-1"

    def test_trace_sync_propagates_exception(self, tracing_setup):
        exporter = tracing_setup
        with pytest.raises(ValueError, match="test error"):
            trace_sync_(external_customer_id="cust-err", fn=lambda: (_ for _ in ()).throw(ValueError("test error")))
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        assert parent.status.status_code.name == "ERROR"

    def test_trace_sync_with_tracing_token(self, tracing_setup):
        exporter = tracing_setup
        tid = 0x1234567890ABCDEF1234567890ABCDEF
        trace_sync_(external_customer_id="cust", fn=lambda: "traced", tracing_token=tid)
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        assert parent.context.trace_id == tid

    def test_context_reset_after_trace_sync(self, tracing_setup):
        def check():
            assert ContextData.get_context_key("external_customer_id") == "cust-ctx"
            return "done"

        trace_sync_(external_customer_id="cust-ctx", fn=check, external_agent_id="agent-ctx")
        assert ContextData.get_context_key("external_customer_id") is None
        assert ContextData.get_context_key("external_agent_id") is None


class TestTraceAsync:

    async def test_trace_async_creates_parent_span(self, tracing_setup):
        exporter = tracing_setup

        async def fn():
            return "async hello"

        result = await trace_async_(external_customer_id="cust-async", fn=fn, external_agent_id="agent-async")
        assert result == "async hello"
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        assert parent.status.status_code.name == "OK"
        assert parent.attributes.get("external_customer_id") == "cust-async"

    async def test_trace_async_propagates_exception(self, tracing_setup):
        exporter = tracing_setup

        async def fail():
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await trace_async_(external_customer_id="cust-err", fn=fail)
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        assert parent.status.status_code.name == "ERROR"

    async def test_trace_async_with_tracing_token(self, tracing_setup):
        exporter = tracing_setup
        tid = 0xABCDEF1234567890ABCDEF1234567890

        async def fn():
            return "traced"

        await trace_async_(external_customer_id="cust", fn=fn, tracing_token=tid)
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        assert parent.context.trace_id == tid

    async def test_context_reset_after_trace_async(self, tracing_setup):
        async def check():
            assert ContextData.get_context_key("external_customer_id") == "cust-async-ctx"
            return "done"

        await trace_async_(external_customer_id="cust-async-ctx", fn=check, external_agent_id="agent-async-ctx")
        assert ContextData.get_context_key("external_customer_id") is None
        assert ContextData.get_context_key("external_agent_id") is None


class TestTraceWithMetadata:

    def test_trace_sync_with_metadata(self, tracing_setup):
        exporter = tracing_setup
        trace_sync_(external_customer_id="cust", fn=lambda: "ok", metadata={"env": "test", "nested": {"key": "val"}})
        parent = [s for s in exporter.get_finished_spans() if "parent_span" in s.name][0]
        attrs = dict(parent.attributes)
        assert attrs.get("metadata.env") == "test"
        assert attrs.get("metadata.nested.key") == "val"

    def test_trace_sync_with_store_prompt(self, tracing_setup):
        exporter = tracing_setup

        def fn():
            tracer = get_paid_tracer()
            with tracer.start_as_current_span("inner") as span:
                span.set_attribute("gen_ai.prompt", "test prompt")
            return "ok"

        trace_sync_(external_customer_id="cust", fn=fn, store_prompt=True)
        inner = [s for s in exporter.get_finished_spans() if "inner" in s.name][0]
        assert "gen_ai.prompt" in dict(inner.attributes)
