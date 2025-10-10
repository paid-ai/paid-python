# Tracing module for OpenTelemetry integration

# Use lazy imports to avoid requiring peer dependencies when not in use
def __getattr__(name):
    """Lazy import wrappers to avoid requiring peer dependencies."""
    if name == "PaidOpenAI":
        from .openai.openAiWrapper import PaidOpenAI

        return PaidOpenAI
    elif name == "PaidAsyncOpenAI":
        from .openai.openAiWrapper import PaidAsyncOpenAI

        return PaidAsyncOpenAI
    elif name == "PaidAnthropic":
        from .anthropic.anthropicWrapper import PaidAnthropic

        return PaidAnthropic
    elif name == "PaidAsyncAnthropic":
        from .anthropic.anthropicWrapper import PaidAsyncAnthropic

        return PaidAsyncAnthropic
    elif name == "PaidMistral":
        from .mistral.mistralWrapper import PaidMistral

        return PaidMistral
    elif name == "PaidBedrock":
        from .bedrock.bedrockWrapper import PaidBedrock

        return PaidBedrock
    elif name == "PaidGemini":
        from .gemini.geminiWrapper import PaidGemini

        return PaidGemini
    elif name == "PaidLlamaIndexOpenAI":
        from .llamaindex.llamaIndexWrapper import PaidLlamaIndexOpenAI

        return PaidLlamaIndexOpenAI
    elif name == "PaidLangChainCallback":
        from .langchain.paidLangChainCallback import PaidLangChainCallback

        return PaidLangChainCallback
    elif name == "PaidOpenAIAgentsHook":
        from .openai_agents.openaiAgentsHook import PaidOpenAIAgentsHook
        return PaidOpenAIAgentsHook

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
