"""Monkey-patches for openinference-instrumentation-openai.

Upstream emits full embedding vectors as span attributes, which creates large
and low-value telemetry payloads. This patch suppresses those vector attributes
while preserving the rest of the embedding metadata.
"""

from __future__ import annotations

from paid.logger import logger

_originals: dict[str, object] = {}


def instrument_openai() -> None:
    """Apply all OpenAI patches. Call after instrumenting the SDK."""
    _patch_embedding_response_attributes()


def uninstrument_openai() -> None:
    """Restore original OpenAI instrumentation methods."""
    try:
        from openinference.instrumentation.openai import _response_attributes_extractor as mod
    except ImportError:
        _originals.clear()
        return

    if "get_attributes_from_create_embedding_response" in _originals:
        mod._ResponseAttributesExtractor._get_attributes_from_create_embedding_response = _originals.pop(  # type: ignore[method-assign]
            "get_attributes_from_create_embedding_response"
        )

    _originals.clear()


def _patch_embedding_response_attributes() -> None:
    try:
        from openinference.instrumentation.openai import _response_attributes_extractor as mod
    except ImportError:
        logger.debug(
            "Could not import openinference openai _response_attributes_extractor, skipping embedding vector patch"
        )
        return

    extractor = mod._ResponseAttributesExtractor
    if "get_attributes_from_create_embedding_response" in _originals:
        return

    _originals["get_attributes_from_create_embedding_response"] = (
        extractor._get_attributes_from_create_embedding_response
    )

    def _patched(self, response):  # type: ignore[misc]
        if usage := getattr(response, "usage", None):
            yield from self._get_attributes_from_embedding_usage(usage)
        if model := getattr(response, "model", None):
            yield mod.SpanAttributes.EMBEDDING_MODEL_NAME, model

    extractor._get_attributes_from_create_embedding_response = _patched  # type: ignore[method-assign]
    logger.debug(
        "Patched _ResponseAttributesExtractor._get_attributes_from_create_embedding_response to omit embedding vectors"
    )
