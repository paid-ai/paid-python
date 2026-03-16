"""Patches for openinference-instrumentation-openai.

Removes raw embedding vectors from telemetry while preserving model and token
usage attributes for embedding spans.
"""

from .patches import (
    instrument_openai,
    uninstrument_openai,
)

__all__ = [
    "instrument_openai",
    "uninstrument_openai",
]
