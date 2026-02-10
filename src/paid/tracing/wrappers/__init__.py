# Tracing module for OpenTelemetry integration

# Use lazy imports to avoid requiring peer dependencies when not in use
def __getattr__(name):
    """Lazy import wrappers to avoid requiring peer dependencies."""
    if name == "PaidLangChainCallback":
        from .langchain.paidLangChainCallback import PaidLangChainCallback

        return PaidLangChainCallback
    elif name == "PaidOpenAIAgentsHook":
        from .openai_agents.openaiAgentsHook import PaidOpenAIAgentsHook

        return PaidOpenAIAgentsHook

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
