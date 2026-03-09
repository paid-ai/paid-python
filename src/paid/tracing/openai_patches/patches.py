"""Monkey-patches for openinference-instrumentation-openai.

Adds child TOOL spans for:

* **Tool-call spans** — emitted after the LLM response is received, one per
  ``tool_calls`` entry in the assistant's ``ChatCompletion`` choice message.
* **Tool-execution spans** — emitted when the *next* request contains messages
  with ``role="tool"``, one per such message.

Both patches extend the openinference response/request attribute extractors so
that they run inside the existing LLM span context, keeping tool spans as
proper children of the LLM span.
"""

from __future__ import annotations

from typing import Any, Dict

from paid.logger import logger
from paid.tracing.tool_spans import emit_tool_call_span, emit_tool_execution_span

_originals: Dict[str, Any] = {}

_GEN_AI_SYSTEM = "openai"


def instrument_openai() -> None:
    """Apply all OpenAI tool-span patches. Call after OpenAIInstrumentor().instrument()."""
    _patch_tool_call_spans()
    _patch_tool_execution_spans()


def uninstrument_openai() -> None:
    """Restore original OpenAI methods patched for tool spans."""
    if "_get_attributes_from_chat_completion" in _originals:
        try:
            from openinference.instrumentation.openai import _response_attributes_extractor as mod
            mod._ResponseAttributesExtractor._get_attributes_from_chat_completion = _originals.pop(  # type: ignore[method-assign]
                "_get_attributes_from_chat_completion"
            )
        except Exception:
            pass

    if "_get_attributes_from_chat_completion_create_param" in _originals:
        try:
            from openinference.instrumentation.openai import _request_attributes_extractor as req_mod
            req_mod._RequestAttributesExtractor._get_attributes_from_chat_completion_create_param = _originals.pop(  # type: ignore[method-assign]
                "_get_attributes_from_chat_completion_create_param"
            )
        except Exception:
            pass

    _originals.clear()


def _patch_tool_call_spans() -> None:
    """Emit child TOOL spans for tool_calls in ChatCompletion assistant messages."""
    try:
        from openinference.instrumentation.openai import _response_attributes_extractor as mod
    except ImportError:
        logger.debug(
            "Could not import openai _response_attributes_extractor for tool-call span patch, skipping"
        )
        return

    _original = mod._ResponseAttributesExtractor._get_attributes_from_chat_completion
    _originals["_get_attributes_from_chat_completion"] = _original

    def _patched(self, completion, request_parameters):  # type: ignore[misc]
        yield from _original(self, completion=completion, request_parameters=request_parameters)
        try:
            choices = getattr(completion, "choices", None) or []
            for choice in choices:
                message = getattr(choice, "message", None)
                if message is None:
                    continue
                tool_calls = getattr(message, "tool_calls", None) or []
                for tool_call in tool_calls:
                    tool_id = getattr(tool_call, "id", None) or ""
                    function = getattr(tool_call, "function", None)
                    if function is None:
                        continue
                    tool_name = getattr(function, "name", None) or ""
                    arguments = getattr(function, "arguments", None) or ""
                    emit_tool_call_span(
                        tool_name=tool_name,
                        tool_call_id=tool_id,
                        arguments_json=arguments,
                        gen_ai_system=_GEN_AI_SYSTEM,
                    )
        except Exception:
            logger.debug("Failed to emit OpenAI tool-call spans", exc_info=True)

    mod._ResponseAttributesExtractor._get_attributes_from_chat_completion = _patched  # type: ignore[method-assign]
    logger.debug(
        "Patched _ResponseAttributesExtractor._get_attributes_from_chat_completion "
        "to emit tool-call spans"
    )


def _patch_tool_execution_spans() -> None:
    """Emit child TOOL spans for role='tool' messages in the chat completion request."""
    try:
        from openinference.instrumentation.openai import _request_attributes_extractor as req_mod
    except ImportError:
        logger.debug(
            "Could not import openai _request_attributes_extractor for tool-execution span patch, skipping"
        )
        return

    _original = req_mod._RequestAttributesExtractor._get_attributes_from_chat_completion_create_param
    _originals["_get_attributes_from_chat_completion_create_param"] = _original

    def _patched(self, params):  # type: ignore[misc]
        yield from _original(self, params=params)
        try:
            messages = params.get("messages") if hasattr(params, "get") else None
            if not messages:
                return
            for msg in messages:
                role = msg.get("role") if hasattr(msg, "get") else getattr(msg, "role", None)
                if role != "tool":
                    continue
                tool_call_id = msg.get("tool_call_id") if hasattr(msg, "get") else getattr(msg, "tool_call_id", None) or ""
                content = msg.get("content") if hasattr(msg, "get") else getattr(msg, "content", None) or ""
                output_str = content if isinstance(content, str) else ""
                emit_tool_execution_span(
                    tool_name=str(tool_call_id) or "tool_result",
                    tool_call_id=str(tool_call_id),
                    output_value=output_str,
                    gen_ai_system=_GEN_AI_SYSTEM,
                )
        except Exception:
            logger.debug("Failed to emit OpenAI tool-execution spans", exc_info=True)

    req_mod._RequestAttributesExtractor._get_attributes_from_chat_completion_create_param = _patched  # type: ignore[method-assign]
    logger.debug(
        "Patched _RequestAttributesExtractor._get_attributes_from_chat_completion_create_param "
        "to emit tool-execution spans"
    )
