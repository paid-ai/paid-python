"""Patches for openinference-instrumentation-anthropic.

Fixes: missing stream context managers, broken MessageStreamManager.__enter__,
and missing AsyncMessages.stream() instrumentation.
"""

from .patches import (
    instrument_anthropic,
    uninstrument_anthropic,
)

__all__ = [
    "instrument_anthropic",
    "uninstrument_anthropic",
]
