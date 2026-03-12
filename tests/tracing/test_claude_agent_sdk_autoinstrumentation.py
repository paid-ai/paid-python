from paid.tracing import autoinstrumentation, tracing


def _setup(tracing_setup):
    autoinstrumentation.paid_autoinstrument(libraries=["claude-agent-sdk"])
    return tracing_setup


class TestClaudeAgentSDKAutoinstrumentation:
    def test_instruments_with_paid_tracer_provider(self, tracing_setup, monkeypatch):
        calls: list[object] = []

        class FakeClaudeAgentSDKInstrumentor:
            def instrument(self, *, tracer_provider):
                calls.append(tracer_provider)

        monkeypatch.setattr(autoinstrumentation, "CLAUDE_AGENT_SDK_AVAILABLE", True)
        monkeypatch.setattr(
            autoinstrumentation,
            "ClaudeAgentSDKInstrumentor",
            FakeClaudeAgentSDKInstrumentor,
        )

        autoinstrumentation._initialized_instrumentors.clear()
        exporter = _setup(tracing_setup)

        assert len(exporter.get_finished_spans()) == 0
        assert calls == [tracing.paid_tracer_provider]
        assert autoinstrumentation._initialized_instrumentors == ["claude-agent-sdk"]
