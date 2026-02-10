"""Monkey-patches for openinference-instrumentation-google-genai.

Adds response ID extraction to both non-streaming and streaming spans.
"""

from __future__ import annotations

from paid.logger import logger

_ATTR_RESPONSE_ID = "gen_ai.response.id"

# Store originals for uninstrumentation
_originals: dict = {}


def instrument_google_genai() -> None:
    """Apply all Google GenAI patches. Call after GoogleGenAIInstrumentor().instrument()."""
    _patch_response_id_extraction()
    _patch_streaming_response_id_extraction()


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
