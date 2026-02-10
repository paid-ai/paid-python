"""Monkey-patches for openinference-instrumentation-anthropic."""

from __future__ import annotations

from typing import Any, Dict

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy, wrap_function_wrapper  # type: ignore[import-untyped]

from paid.logger import logger
from paid.tracing import tracing

_original_async_messages_stream = None

# Originals for the beta path, keyed by method name.
_beta_originals: Dict[str, Any] = {}

# Originals for response-ID patches.
_response_id_originals: Dict[str, Any] = {}

_BETA_MODULE = "anthropic.resources.beta.messages.messages"

_ATTR_RESPONSE_ID = "gen_ai.response.id"


def instrument_anthropic() -> None:
    """Apply all Anthropic patches. Call after AnthropicInstrumentor().instrument()."""
    _patch_stream_context_managers()
    _patch_message_stream_manager()
    _wrap_async_messages_stream()
    _patch_response_accumulator_for_beta()
    _patch_response_id_extraction()
    _patch_streaming_response_id_extraction()
    _wrap_beta_messages()


def uninstrument_anthropic() -> None:
    global _original_async_messages_stream
    if _original_async_messages_stream is not None:
        try:
            from anthropic.resources.messages import AsyncMessages
            AsyncMessages.stream = _original_async_messages_stream  # type: ignore[method-assign]
        except Exception:
            pass
        _original_async_messages_stream = None

    _uninstrument_beta_messages()
    _uninstrument_response_id_patches()

def _patch_stream_context_managers() -> None:
    try:
        from openinference.instrumentation.anthropic._stream import (
            _MessagesStream,
            _Stream,
        )
    except ImportError:
        logger.debug("Could not import stream classes for patching, skipping")
        return

    for cls in (_MessagesStream, _Stream):
        if not hasattr(cls, "__aenter__"):

            async def _aenter(self):  # type: ignore[misc]
                if hasattr(self.__wrapped__, "__aenter__"):
                    await self.__wrapped__.__aenter__()
                return self

            cls.__aenter__ = _aenter  # type: ignore[attr-defined]

        if not hasattr(cls, "__aexit__"):

            async def _aexit(self, exc_type, exc_val, exc_tb):  # type: ignore[misc]
                if hasattr(self.__wrapped__, "__aexit__"):
                    return await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)

            cls.__aexit__ = _aexit  # type: ignore[attr-defined]

        # Always override: ObjectProxy's defaults return the wrapped object, bypassing instrumentation.
        def _enter(self):  # type: ignore[misc]
            if hasattr(self.__wrapped__, "__enter__"):
                self.__wrapped__.__enter__()
            return self

        cls.__enter__ = _enter  # type: ignore[attr-defined]

        def _exit(self, exc_type, exc_val, exc_tb):  # type: ignore[misc]
            if hasattr(self.__wrapped__, "__exit__"):
                return self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)

        cls.__exit__ = _exit  # type: ignore[attr-defined]

    logger.debug("Patched stream proxies with context manager support")


# Fix 2: _MessageStreamManager.__enter__ calls __api_request() returning a raw stream,
# losing the high-level MessageStream (.text_stream, .get_final_message()).
# We delegate to the real __enter__ instead, then wrap in _MessagesStream for tracing.

def _patch_message_stream_manager() -> None:
    try:
        from openinference.instrumentation.anthropic._stream import _MessagesStream
        from openinference.instrumentation.anthropic._wrappers import _MessageStreamManager
    except ImportError:
        logger.debug("Could not import _MessageStreamManager for patching, skipping")
        return

    def _fixed_enter(self):  # type: ignore[misc]
        message_stream = self.__wrapped__.__enter__()
        return _MessagesStream(message_stream, self._self_with_span)

    def _fixed_exit(self, exc_type, exc_val, exc_tb):  # type: ignore[misc]
        try:
            return self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._self_with_span.finish_tracing()  # idempotent

    _MessageStreamManager.__enter__ = _fixed_enter  # type: ignore[attr-defined, method-assign]
    _MessageStreamManager.__exit__ = _fixed_exit  # type: ignore[attr-defined, method-assign]

    logger.debug("Patched _MessageStreamManager with fixed __enter__/__exit__")


# Fix 3: openinference only wraps sync Messages.stream, not AsyncMessages.stream.

class _AsyncMessageStreamManagerProxy(ObjectProxy):  # type: ignore[misc]
    """Wraps AsyncMessageStreamManager with span lifecycle management."""

    def __init__(self, manager: Any, span: trace_api.Span) -> None:
        super().__init__(manager)
        self._self_span = span
        self._self_stream: Any = None

    async def __aenter__(self):  # type: ignore[misc]
        stream = await self.__wrapped__.__aenter__()
        self._self_stream = stream
        return stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # type: ignore[misc]
        try:
            return await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            try:
                # Try to capture response ID from the stream's final message snapshot
                stream = self._self_stream
                if stream is not None:
                    final_msg = getattr(stream, "_MessageStream__final_message_snapshot", None)
                    if final_msg is not None and hasattr(final_msg, "id") and final_msg.id:
                        self._self_span.set_attribute(_ATTR_RESPONSE_ID, final_msg.id)
            except Exception:
                logger.debug("Failed to capture response ID from async stream", exc_info=True)
            try:
                if exc_type:
                    self._self_span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exc_val)))
                else:
                    self._self_span.set_status(trace_api.StatusCode.OK)
                self._self_span.end()
            except Exception:
                logger.debug("Failed to end span for AsyncMessages.stream", exc_info=True)


def _wrap_async_messages_stream() -> None:
    global _original_async_messages_stream
    try:
        from anthropic.resources.messages import AsyncMessages
    except ImportError:
        logger.debug("Could not import AsyncMessages, skipping async stream wrapping")
        return

    _original_async_messages_stream = AsyncMessages.stream  # type: ignore[assignment]

    def _wrapper(wrapped, instance, args, kwargs):  # type: ignore[misc]
        tracer = tracing.paid_tracer_provider.get_tracer("paid.anthropic")
        span = tracer.start_span(name="AsyncMessagesStream", record_exception=False, set_status_on_exception=False)

        try:
            if kwargs.get("model"):
                span.set_attribute("llm.model_name", str(kwargs["model"]))
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", "anthropic")
            span.set_attribute("llm.system", "anthropic")
        except Exception:
            pass

        try:
            response = wrapped(*args, **kwargs)
        except Exception as exc:
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            span.end()
            raise

        return _AsyncMessageStreamManagerProxy(response, span)

    wrap_function_wrapper(module="anthropic.resources.messages", name="AsyncMessages.stream", wrapper=_wrapper)
    logger.debug("Wrapped AsyncMessages.stream for instrumentation")




def _patch_response_accumulator_for_beta() -> None:
    """Extend _MessageResponseAccumulator.process_chunk to handle beta event types and capture response IDs."""
    try:
        from openinference.instrumentation.anthropic._stream import _MessageResponseAccumulator
    except ImportError:
        logger.debug("Could not import _MessageResponseAccumulator, skipping beta accumulator patch")
        return

    try:
        from anthropic.types.beta import (
            BetaRawContentBlockDeltaEvent,
            BetaRawContentBlockStartEvent,
            BetaRawMessageDeltaEvent,
            BetaRawMessageStartEvent,
        )
    except ImportError:
        logger.debug("Could not import beta event types, skipping beta accumulator patch")
        return

    try:
        from anthropic.types.raw_message_start_event import RawMessageStartEvent
    except ImportError:
        RawMessageStartEvent = None  # type: ignore[assignment,misc]

    _original_process_chunk = _MessageResponseAccumulator.process_chunk

    def _process_chunk_with_beta(self, chunk):  # type: ignore[misc]
        """Handles both regular and beta event types, and captures response IDs."""
        if isinstance(chunk, BetaRawMessageStartEvent):
            self._is_null = False
            self._current_message_idx += 1
            # Capture response ID from beta message start
            if hasattr(chunk.message, "id") and chunk.message.id:
                self._values += {"response_id": chunk.message.id}
            value = {
                "messages": {
                    "index": str(self._current_message_idx),
                    "role": chunk.message.role,
                    "input_tokens": str(chunk.message.usage.input_tokens),
                }
            }
            self._values += value
        elif isinstance(chunk, BetaRawContentBlockStartEvent):
            self._is_null = False
            self._current_content_block_type = chunk.content_block
        elif isinstance(chunk, BetaRawContentBlockDeltaEvent):
            self._is_null = False
            # Duck-type check: BetaTextBlock/BetaToolUseBlock have the same .type attribute
            block_type = getattr(self._current_content_block_type, "type", None)
            if block_type == "text":
                value = {
                    "messages": {
                        "index": str(self._current_message_idx),
                        "content": {
                            "index": chunk.index,
                            "type": block_type,
                            "text": chunk.delta.text,  # type: ignore[union-attr]
                        },
                    }
                }
                self._values += value
            elif block_type == "tool_use":
                value = {
                    "messages": {
                        "index": str(self._current_message_idx),
                        "content": {
                            "index": chunk.index,
                            "type": block_type,
                            "tool_name": self._current_content_block_type.name,
                            "tool_input": chunk.delta.partial_json,  # type: ignore[union-attr]
                        },
                    }
                }
                self._values += value
        elif isinstance(chunk, BetaRawMessageDeltaEvent):
            self._is_null = False
            value = {
                "messages": {
                    "index": str(self._current_message_idx),
                    "stop_reason": chunk.delta.stop_reason,
                    "output_tokens": str(chunk.usage.output_tokens),
                }
            }
            self._values += value
        else:
            # Non-beta event — capture response ID from regular RawMessageStartEvent
            # before delegating to the original handler.
            if RawMessageStartEvent is not None and isinstance(chunk, RawMessageStartEvent):
                if hasattr(chunk.message, "id") and chunk.message.id:
                    self._values += {"response_id": chunk.message.id}
            return _original_process_chunk(self, chunk)

    _MessageResponseAccumulator.process_chunk = _process_chunk_with_beta  # type: ignore[method-assign]
    logger.debug("Patched _MessageResponseAccumulator.process_chunk for beta event types and response IDs")


def _patch_response_id_extraction() -> None:
    """Patch openinference to extract response.id from non-streaming Anthropic messages."""
    try:
        from openinference.instrumentation.anthropic import _wrappers
    except ImportError:
        logger.debug("Could not import openinference anthropic _wrappers, skipping response ID patch")
        return

    _original = _wrappers._get_llm_model_name_from_response
    _response_id_originals["_get_llm_model_name_from_response"] = _original

    def _patched(message):  # type: ignore[misc]
        yield from _original(message)
        if response_id := getattr(message, "id", None):
            yield _ATTR_RESPONSE_ID, response_id

    _wrappers._get_llm_model_name_from_response = _patched  # type: ignore[attr-defined]
    logger.debug("Patched _get_llm_model_name_from_response to also yield response ID")


def _patch_streaming_response_id_extraction() -> None:
    """Patch openinference _MessageResponseExtractor to yield response ID from accumulated data."""
    try:
        from openinference.instrumentation.anthropic._stream import _MessageResponseExtractor
    except ImportError:
        logger.debug("Could not import _MessageResponseExtractor, skipping streaming response ID patch")
        return

    _original_get_extra = _MessageResponseExtractor.get_extra_attributes
    _response_id_originals["get_extra_attributes"] = _original_get_extra

    def _get_extra_with_response_id(self):  # type: ignore[misc]
        yield from _original_get_extra(self)
        try:
            result = self._response_accumulator._result()
            if result and (response_id := result.get("response_id")):
                yield _ATTR_RESPONSE_ID, response_id
        except Exception:
            pass

    _MessageResponseExtractor.get_extra_attributes = _get_extra_with_response_id  # type: ignore[method-assign]
    logger.debug("Patched _MessageResponseExtractor.get_extra_attributes for response IDs")


def _uninstrument_response_id_patches() -> None:
    """Restore original methods patched by _patch_response_id_extraction / _patch_streaming_response_id_extraction."""
    if "_get_llm_model_name_from_response" in _response_id_originals:
        try:
            from openinference.instrumentation.anthropic import _wrappers

            _wrappers._get_llm_model_name_from_response = _response_id_originals.pop(  # type: ignore[attr-defined]
                "_get_llm_model_name_from_response"
            )
        except Exception:
            pass

    if "get_extra_attributes" in _response_id_originals:
        try:
            from openinference.instrumentation.anthropic._stream import _MessageResponseExtractor

            _MessageResponseExtractor.get_extra_attributes = _response_id_originals.pop(  # type: ignore[method-assign]
                "get_extra_attributes"
            )
        except Exception:
            pass

    _response_id_originals.clear()


# ---------------------------------------------------------------------------
# We reuse the same openinference wrapper classes (_MessagesWrapper,
# _AsyncMessagesWrapper, _MessagesStreamWrapper) targeting the beta module.
# For AsyncMessages.stream we use the same custom approach as Fix 3 since
# openinference has no async stream wrapper.
# ---------------------------------------------------------------------------


def _wrap_beta_messages() -> None:
    """Wrap the beta Messages/AsyncMessages.create and .stream methods."""
    try:
        from anthropic.resources.beta.messages.messages import (
            AsyncMessages as BetaAsyncMessages,
        )
        from anthropic.resources.beta.messages.messages import (
            Messages as BetaMessages,
        )
    except ImportError:
        logger.debug("Could not import beta Messages classes, skipping beta wrapping")
        return

    try:
        from openinference.instrumentation import OITracer, TraceConfig
        from openinference.instrumentation.anthropic._wrappers import (
            _AsyncMessagesWrapper,
            _MessagesStreamWrapper,
            _MessagesWrapper,
        )
    except ImportError:
        logger.debug("Could not import openinference wrappers, skipping beta wrapping")
        return

    tracer: trace_api.Tracer = OITracer(
        tracing.paid_tracer_provider.get_tracer(__name__),
        config=TraceConfig(),
    )

    # --- beta Messages.create (sync) ---
    _beta_originals["Messages.create"] = BetaMessages.create
    wrap_function_wrapper(
        module=_BETA_MODULE,
        name="Messages.create",
        wrapper=_MessagesWrapper(tracer=tracer),
    )

    # --- beta AsyncMessages.create (async — primary path for pydantic-ai) ---
    _beta_originals["AsyncMessages.create"] = BetaAsyncMessages.create
    wrap_function_wrapper(
        module=_BETA_MODULE,
        name="AsyncMessages.create",
        wrapper=_AsyncMessagesWrapper(tracer=tracer),
    )

    # --- beta Messages.stream (sync) ---
    _beta_originals["Messages.stream"] = BetaMessages.stream
    wrap_function_wrapper(
        module=_BETA_MODULE,
        name="Messages.stream",
        wrapper=_MessagesStreamWrapper(tracer=tracer),
    )

    # --- beta AsyncMessages.stream (async) ---
    _beta_originals["AsyncMessages.stream"] = BetaAsyncMessages.stream  # type: ignore[assignment]

    def _beta_async_stream_wrapper(wrapped, instance, args, kwargs):  # type: ignore[misc]
        beta_tracer = tracing.paid_tracer_provider.get_tracer("paid.anthropic")
        span = beta_tracer.start_span(
            name="AsyncMessagesStream", record_exception=False, set_status_on_exception=False
        )

        try:
            if kwargs.get("model"):
                span.set_attribute("llm.model_name", str(kwargs["model"]))
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", "anthropic")
            span.set_attribute("llm.system", "anthropic")
        except Exception:
            pass

        try:
            response = wrapped(*args, **kwargs)
        except Exception as exc:
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            span.end()
            raise

        return _AsyncMessageStreamManagerProxy(response, span)

    wrap_function_wrapper(
        module=_BETA_MODULE,
        name="AsyncMessages.stream",
        wrapper=_beta_async_stream_wrapper,
    )

    logger.debug("Wrapped beta Messages/AsyncMessages.create and .stream for instrumentation")


def _uninstrument_beta_messages() -> None:
    """Restore original beta Messages/AsyncMessages methods."""
    if not _beta_originals:
        return

    try:
        from anthropic.resources.beta.messages.messages import (
            AsyncMessages as BetaAsyncMessages,
        )
        from anthropic.resources.beta.messages.messages import (
            Messages as BetaMessages,
        )
    except ImportError:
        _beta_originals.clear()
        return

    _restore = {
        "Messages.create": (BetaMessages, "create"),
        "Messages.stream": (BetaMessages, "stream"),
        "AsyncMessages.create": (BetaAsyncMessages, "create"),
        "AsyncMessages.stream": (BetaAsyncMessages, "stream"),
    }

    for key, (cls, attr) in _restore.items():
        original = _beta_originals.pop(key, None)
        if original is not None:
            try:
                setattr(cls, attr, original)
            except Exception:
                logger.debug("Failed to restore beta %s", key, exc_info=True)

    _beta_originals.clear()
    logger.debug("Restored beta Messages/AsyncMessages originals")
