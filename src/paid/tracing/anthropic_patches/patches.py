"""
Concrete monkey-patches for openinference-instrumentation-anthropic.

All patching logic lives here; autoinstrumentation.py simply calls
instrument_anthropic() / uninstrument_anthropic().
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy, wrap_function_wrapper

from paid.logger import logger
from paid.tracing import tracing

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_original_async_messages_stream = None
"""Stored original ``AsyncMessages.stream`` for cleanup during uninstrumentation."""


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------


def instrument_anthropic() -> None:
    """Apply all Anthropic patches.

    Call this **after** ``AnthropicInstrumentor().instrument()`` has run.
    """
    _patch_stream_context_managers()
    _patch_message_stream_manager()
    _wrap_async_messages_stream()


def uninstrument_anthropic() -> None:
    """Undo the ``AsyncMessages.stream`` wrap (other patches are class-level
    and get cleaned up when the base instrumentor uninstruments)."""
    global _original_async_messages_stream

    if _original_async_messages_stream is not None:
        try:
            from anthropic.resources.messages import AsyncMessages

            AsyncMessages.stream = _original_async_messages_stream  # type: ignore[method-assign]
        except Exception:
            pass
        _original_async_messages_stream = None


# ---------------------------------------------------------------------------
# Fix 1: _MessagesStream / _Stream — missing context-manager protocol
# ---------------------------------------------------------------------------


def _patch_stream_context_managers() -> None:
    """Add ``__aenter__``/``__aexit__``/``__enter__``/``__exit__`` to the
    openinference stream proxy classes.

    Python resolves dunder methods on the *class*, not the instance, so
    ``ObjectProxy.__getattr__`` delegation does not help.
    """
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

        # Always override __enter__/__exit__ even if ObjectProxy defines them,
        # because ObjectProxy's defaults return the *wrapped* object (not the
        # proxy), which bypasses the instrumentation on iteration.

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


# ---------------------------------------------------------------------------
# Fix 2: _MessageStreamManager — broken __enter__, missing __exit__
# ---------------------------------------------------------------------------


def _patch_message_stream_manager() -> None:
    """Fix ``_MessageStreamManager`` so that ``messages.stream()`` returns a
    proper ``MessageStream`` (with ``.text_stream``, ``.get_final_message()``,
    etc.) instead of a bare raw-event stream.

    The original ``__enter__`` calls ``self.__api_request()`` which returns the
    raw HTTP response and wraps it in ``_MessagesStream``.  This loses the
    high-level ``MessageStream`` wrapper.

    Our fix delegates to the wrapped ``MessageStreamManager.__enter__()``,
    which returns a proper ``MessageStream``, then wraps *that* in
    ``_MessagesStream``.  Since ``_MessagesStream`` is an ``ObjectProxy``,
    attribute access like ``.text_stream`` passes through to the underlying
    ``MessageStream``, while iteration goes through ``_MessagesStream.__iter__``
    which does the tracing.

    We also add ``__exit__`` to ensure the span is finished when the context
    manager exits (the original class was missing ``__exit__`` entirely).
    """
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
            # finish_tracing is idempotent — safe even if iteration already ended it
            self._self_with_span.finish_tracing()

    _MessageStreamManager.__enter__ = _fixed_enter  # type: ignore[attr-defined]
    _MessageStreamManager.__exit__ = _fixed_exit  # type: ignore[attr-defined]

    logger.debug("Patched _MessageStreamManager with fixed __enter__/__exit__")


# ---------------------------------------------------------------------------
# Fix 3: AsyncMessages.stream() — not instrumented at all by openinference
# ---------------------------------------------------------------------------


class _AsyncMessageStreamManagerProxy(ObjectProxy):  # type: ignore[misc]
    """Wraps ``AsyncMessageStreamManager`` with span lifecycle management.

    ``__aenter__`` delegates to the real manager so the caller gets a proper
    ``AsyncMessageStream`` (with ``.text_stream``, ``.get_final_message()``, …).
    ``__aexit__`` finishes the OTEL span.
    """

    def __init__(self, manager: Any, span: trace_api.Span) -> None:
        super().__init__(manager)
        self._self_span = span

    async def __aenter__(self):  # type: ignore[misc]
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # type: ignore[misc]
        try:
            return await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            try:
                if exc_type:
                    self._self_span.set_status(
                        trace_api.Status(trace_api.StatusCode.ERROR, str(exc_val))
                    )
                else:
                    self._self_span.set_status(trace_api.StatusCode.OK)
                self._self_span.end()
            except Exception:
                logger.debug("Failed to end span for AsyncMessages.stream", exc_info=True)


def _wrap_async_messages_stream() -> None:
    """Add instrumentation for ``AsyncMessages.stream()``.

    The openinference instrumentor only wraps sync ``Messages.stream``, not
    ``AsyncMessages.stream``.  We fill this gap by wrapping it ourselves.
    """
    global _original_async_messages_stream

    try:
        from anthropic.resources.messages import AsyncMessages
    except ImportError:
        logger.debug("Could not import AsyncMessages, skipping async stream wrapping")
        return

    _original_async_messages_stream = AsyncMessages.stream  # type: ignore[assignment]

    def _wrapper(wrapped, instance, args, kwargs):  # type: ignore[misc]
        tracer = tracing.paid_tracer_provider.get_tracer("paid.anthropic")
        span = tracer.start_span(
            name="AsyncMessagesStream",
            record_exception=False,
            set_status_on_exception=False,
        )

        # Set basic span attributes from kwargs
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
        module="anthropic.resources.messages",
        name="AsyncMessages.stream",
        wrapper=_wrapper,
    )

    logger.debug("Wrapped AsyncMessages.stream for instrumentation")
