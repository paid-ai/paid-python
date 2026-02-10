"""Patches for openinference-instrumentation-google-genai.

Adds response ID extraction (gen_ai.response.id) to both non-streaming
and streaming Google GenAI spans.
"""

from .patches import (
    instrument_google_genai,
    uninstrument_google_genai,
)

__all__ = [
    "instrument_google_genai",
    "uninstrument_google_genai",
]
