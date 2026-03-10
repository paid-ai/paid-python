"""Patches for openinference-instrumentation-openai-agents.

Adds tool metadata to function/tool spans by carrying schema details from
response spans onto the later function execution spans.
"""

from .patches import (
    instrument_openai_agents,
    uninstrument_openai_agents,
)

__all__ = [
    "instrument_openai_agents",
    "uninstrument_openai_agents",
]
