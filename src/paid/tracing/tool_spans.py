"""Shared helpers for emitting tool-call child spans.

Each helper creates one child span per tool interaction under the current
active span context (which is the parent LLM span at call time).

Two span types are defined:

* **tool-call span** — model decided to call a tool (emitted when the LLM
  response contains a tool-use / function-call block).
* **tool-execution span** — the tool ran and returned a result (emitted when
  the next request contains a tool-result / function-response block).

Spans are always immediately ended after creation so they appear as
instantaneous events within the LLM span.  Both span types carry
``openinference.span.kind = "TOOL"`` so downstream processors treat them
consistently with other TOOL spans in the OpenInference ecosystem.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.trace import Status, StatusCode

from paid.logger import logger
from paid.tracing import tracing

_SPAN_KIND = "openinference.span.kind"
_TOOL_KIND = "TOOL"
_TOOL_NAME = "tool_call.function.name"
_TOOL_CALL_ID = "tool_call.id"
_TOOL_ARGS = "tool_call.function.arguments"
_OUTPUT_VALUE = "output.value"
_OUTPUT_MIME_TYPE = "output.mime_type"
_GEN_AI_SYSTEM = "gen_ai.system"


def _safe_json(value: Any) -> str:
    """Serialize *value* to a JSON string, falling back to repr on failure."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return repr(value)


def emit_tool_call_span(
    *,
    tool_name: str,
    tool_call_id: str = "",
    arguments_json: str = "",
    gen_ai_system: str = "",
) -> None:
    """Emit a single child span representing a model-initiated tool call.

    Must be called while the parent LLM span is current (i.e. during the
    response-processing code that runs inside the LLM span's context).
    """
    try:
        tracer = tracing.paid_tracer_provider.get_tracer("paid.tool_spans")
        span = tracer.start_span(
            name=tool_name,
            context=context_api.get_current(),
        )
        span.set_attribute(_SPAN_KIND, _TOOL_KIND)
        span.set_attribute(_TOOL_NAME, tool_name)
        if tool_call_id:
            span.set_attribute(_TOOL_CALL_ID, tool_call_id)
        if arguments_json:
            span.set_attribute(_TOOL_ARGS, arguments_json)
        if gen_ai_system:
            span.set_attribute(_GEN_AI_SYSTEM, gen_ai_system)
        span.set_status(Status(StatusCode.OK))
        span.end()
    except Exception:
        logger.debug("Failed to emit tool-call span for %s", tool_name, exc_info=True)


def emit_tool_execution_span(
    *,
    tool_name: str,
    tool_call_id: str = "",
    output_value: str = "",
    gen_ai_system: str = "",
) -> None:
    """Emit a single child span representing a successfully executed tool result.

    Must be called while the parent LLM span is current.
    """
    try:
        tracer = tracing.paid_tracer_provider.get_tracer("paid.tool_spans")
        span = tracer.start_span(
            name=tool_name,
            context=context_api.get_current(),
        )
        span.set_attribute(_SPAN_KIND, _TOOL_KIND)
        span.set_attribute(_TOOL_NAME, tool_name)
        if tool_call_id:
            span.set_attribute(_TOOL_CALL_ID, tool_call_id)
        if output_value:
            span.set_attribute(_OUTPUT_VALUE, output_value)
            if output_value.startswith("{") and output_value.endswith("}"):
                span.set_attribute(_OUTPUT_MIME_TYPE, "application/json")
        if gen_ai_system:
            span.set_attribute(_GEN_AI_SYSTEM, gen_ai_system)
        span.set_status(Status(StatusCode.OK))
        span.end()
    except Exception:
        logger.debug("Failed to emit tool-execution span for %s", tool_name, exc_info=True)
