"""Patches for openinference-instrumentation-anthropic.

Fixes: missing stream context managers, broken MessageStreamManager.__enter__,
and missing AsyncMessages.stream() instrumentation.
"""

from .patches import (
    _AsyncMessageStreamManagerProxy,
    _original_async_messages_stream,
    _patch_message_stream_manager,
    _patch_stream_context_managers,
    _wrap_async_messages_stream,
    _wrap_beta_messages,
    instrument_anthropic,
    uninstrument_anthropic,
)

__all__ = [
    "instrument_anthropic",
    "uninstrument_anthropic",
    "_patch_stream_context_managers",
    "_patch_message_stream_manager",
    "_wrap_async_messages_stream",
    "_wrap_beta_messages",
    "_AsyncMessageStreamManagerProxy",
    "_original_async_messages_stream",
]
