"""Monkey-patches for openinference-instrumentation-google-genai.

Adds response ID extraction and tool-call/execution child spans to both
non-streaming and streaming spans.
"""

from __future__ import annotations

import json

from paid.logger import logger
from paid.tracing.tool_spans import emit_tool_call_span, emit_tool_execution_span

_ATTR_RESPONSE_ID = "gen_ai.response.id"

_GEN_AI_SYSTEM = "google"

# Store originals for uninstrumentation
_originals: dict = {}


def instrument_google_genai() -> None:
    """Apply all Google GenAI patches. Call after GoogleGenAIInstrumentor().instrument()."""
    _patch_response_id_extraction()
    _patch_streaming_response_id_extraction()
    _patch_tool_call_spans()
    _patch_tool_execution_spans()


def uninstrument_google_genai() -> None:
    """Restore original Google GenAI methods."""
    if "get_attributes_from_generate_content" in _originals:
        try:
            from openinference.instrumentation.google_genai import (
                _response_attributes_extractor as mod,
            )

            mod._ResponseAttributesExtractor._get_attributes_from_generate_content = _originals.pop(  # type: ignore[method-assign]
                "get_attributes_from_generate_content"
            )
        except Exception:
            pass

    if "stream_get_extra_attributes" in _originals:
        try:
            from openinference.instrumentation.google_genai import _stream as stream_mod

            stream_mod._ResponseExtractor.get_extra_attributes = _originals.pop(  # type: ignore[method-assign]
                "stream_get_extra_attributes"
            )
        except Exception:
            pass

    if "_get_attributes_from_content_parts" in _originals:
        try:
            from openinference.instrumentation.google_genai import (
                _response_attributes_extractor as mod,
            )

            mod._ResponseAttributesExtractor._get_attributes_from_content_parts = _originals.pop(  # type: ignore[method-assign]
                "_get_attributes_from_content_parts"
            )
        except Exception:
            pass

    if "get_extra_attributes_from_request" in _originals:
        try:
            from openinference.instrumentation.google_genai import (
                _request_attributes_extractor as req_mod,
            )

            req_mod._RequestAttributesExtractor.get_extra_attributes_from_request = _originals.pop(  # type: ignore[method-assign]
                "get_extra_attributes_from_request"
            )
        except Exception:
            pass

    _originals.clear()


def _patch_response_id_extraction() -> None:
    """Patch openinference to extract response_id from non-streaming Gemini responses."""
    try:
        from openinference.instrumentation.google_genai import (
            _response_attributes_extractor as mod,
        )
    except ImportError:
        logger.debug(
            "Could not import openinference google_genai _response_attributes_extractor, "
            "skipping response ID patch"
        )
        return

    _original = mod._ResponseAttributesExtractor._get_attributes_from_generate_content
    _originals["get_attributes_from_generate_content"] = _original

    def _patched(self, response, request_parameters):  # type: ignore[misc]
        yield from _original(self, response=response, request_parameters=request_parameters)
        if response_id := getattr(response, "response_id", None):
            yield _ATTR_RESPONSE_ID, response_id

    mod._ResponseAttributesExtractor._get_attributes_from_generate_content = _patched  # type: ignore[method-assign]
    logger.debug(
        "Patched _ResponseAttributesExtractor._get_attributes_from_generate_content "
        "to also yield response ID"
    )


def _patch_streaming_response_id_extraction() -> None:
    """Patch openinference _ResponseExtractor to yield response_id from accumulated streaming data."""
    try:
        from openinference.instrumentation.google_genai import _stream as stream_mod
    except ImportError:
        logger.debug(
            "Could not import openinference google_genai _stream, "
            "skipping streaming response ID patch"
        )
        return

    _original_get_extra = stream_mod._ResponseExtractor.get_extra_attributes
    _originals["stream_get_extra_attributes"] = _original_get_extra

    def _get_extra_with_response_id(self):  # type: ignore[misc]
        yield from _original_get_extra(self)
        try:
            result = self._response_accumulator._result()
            if result and (response_id := result.get("response_id")):
                yield _ATTR_RESPONSE_ID, response_id
        except Exception:
            pass

    stream_mod._ResponseExtractor.get_extra_attributes = _get_extra_with_response_id  # type: ignore[method-assign]
    logger.debug("Patched _ResponseExtractor.get_extra_attributes for response IDs")


def _patch_tool_call_spans() -> None:
    """Emit child TOOL spans for function_call parts in Gemini response candidates."""
    try:
        from openinference.instrumentation.google_genai import (
            _response_attributes_extractor as mod,
        )
    except ImportError:
        logger.debug(
            "Could not import google_genai _response_attributes_extractor for tool-call span patch, skipping"
        )
        return

    _original = mod._ResponseAttributesExtractor._get_attributes_from_content_parts
    _originals["_get_attributes_from_content_parts"] = _original

    def _patched_content_parts(self, content_parts):  # type: ignore[misc]
        yield from _original(self, content_parts)
        try:
            for part in content_parts:
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                tool_name = getattr(function_call, "name", None) or ""
                raw_args = getattr(function_call, "args", None)
                args_json = json.dumps(raw_args) if raw_args is not None else ""
                emit_tool_call_span(
                    tool_name=tool_name,
                    tool_call_id="",
                    arguments_json=args_json,
                    gen_ai_system=_GEN_AI_SYSTEM,
                )
        except Exception:
            logger.debug("Failed to emit Gemini tool-call spans", exc_info=True)

    mod._ResponseAttributesExtractor._get_attributes_from_content_parts = _patched_content_parts  # type: ignore[method-assign]
    logger.debug(
        "Patched _ResponseAttributesExtractor._get_attributes_from_content_parts "
        "to emit tool-call spans"
    )


def _patch_tool_execution_spans() -> None:
    """Emit child TOOL spans for function_response parts in Gemini request contents."""
    try:
        from openinference.instrumentation.google_genai import (
            _request_attributes_extractor as req_mod,
        )
    except ImportError:
        logger.debug(
            "Could not import google_genai _request_attributes_extractor for tool-execution span patch, skipping"
        )
        return

    _original = req_mod._RequestAttributesExtractor.get_extra_attributes_from_request
    _originals["get_extra_attributes_from_request"] = _original

    def _patched_get_extra(self, request_parameters):  # type: ignore[misc]
        yield from _original(self, request_parameters=request_parameters)
        try:
            if not hasattr(request_parameters, "get"):
                return
            input_contents = request_parameters.get("contents")
            if not isinstance(input_contents, list):
                return
            for content_entry in input_contents:
                role = getattr(content_entry, "role", None) or (
                    content_entry.get("role") if isinstance(content_entry, dict) else None
                )
                if role != "user":
                    continue
                parts = getattr(content_entry, "parts", None) or (
                    content_entry.get("parts") if isinstance(content_entry, dict) else None
                ) or []
                for part in parts:
                    function_response = getattr(part, "function_response", None)
                    if function_response is None and isinstance(part, dict):
                        function_response = part.get("function_response")
                    if function_response is None:
                        continue
                    tool_name = getattr(function_response, "name", None) or (
                        function_response.get("name") if isinstance(function_response, dict) else None
                    ) or ""
                    fn_id = getattr(function_response, "id", None) or (
                        function_response.get("id") if isinstance(function_response, dict) else None
                    ) or ""
                    raw_response = getattr(function_response, "response", None) or (
                        function_response.get("response") if isinstance(function_response, dict) else None
                    )
                    output_str = json.dumps(raw_response) if raw_response is not None else ""
                    emit_tool_execution_span(
                        tool_name=tool_name or str(fn_id) or "function_response",
                        tool_call_id=str(fn_id),
                        output_value=output_str,
                        gen_ai_system=_GEN_AI_SYSTEM,
                    )
        except Exception:
            logger.debug("Failed to emit Gemini tool-execution spans", exc_info=True)

    req_mod._RequestAttributesExtractor.get_extra_attributes_from_request = _patched_get_extra  # type: ignore[method-assign]
    logger.debug(
        "Patched _RequestAttributesExtractor.get_extra_attributes_from_request "
        "to emit tool-execution spans"
    )
