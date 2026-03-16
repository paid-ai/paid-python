"""Monkey-patches for openinference-instrumentation-openai-agents.

Upstream records full tool schemas on response spans, but function/tool spans only
get `tool.name`. This patch caches function tool metadata per trace when the
response span ends, then applies that metadata to subsequent function spans.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from openai.types.responses import FunctionTool, Response
from openinference.instrumentation import safe_json_dumps
from opentelemetry.context import detach
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import AttributeValue

from paid.logger import logger

_originals: dict[str, Any] = {}
_SEMCONV_FALLBACKS = {
    "TOOL_DESCRIPTION": "tool.description",
    "TOOL_PARAMETERS": "tool.parameters",
    "TOOL_JSON_SCHEMA": "tool.json_schema",
}


def instrument_openai_agents() -> None:
    """Apply all OpenAI Agents patches. Call after instrumenting the SDK."""
    _patch_tracing_processor()


def uninstrument_openai_agents() -> None:
    """Restore original OpenAI Agents methods."""
    try:
        from openinference.instrumentation.openai_agents import _processor as processor_mod
    except ImportError:
        _originals.clear()
        return

    processor = processor_mod.OpenInferenceTracingProcessor
    if "on_span_end" in _originals:
        processor.on_span_end = _originals.pop("on_span_end")  # type: ignore[method-assign]
    if "on_trace_end" in _originals:
        processor.on_trace_end = _originals.pop("on_trace_end")  # type: ignore[method-assign]
    _originals.clear()


def _patch_tracing_processor() -> None:
    try:
        from agents.tracing.span_data import FunctionSpanData, ResponseSpanData
        from openinference.instrumentation.openai_agents import _processor as processor_mod
    except ImportError:
        logger.debug("Could not import openai-agents tracing processor, skipping patch")
        return

    processor = processor_mod.OpenInferenceTracingProcessor
    if "on_span_end" in _originals:
        return

    _originals["on_span_end"] = processor.on_span_end
    _originals["on_trace_end"] = processor.on_trace_end

    def _patched_on_trace_end(self, trace):  # type: ignore[misc]
        try:
            return _originals["on_trace_end"](self, trace)
        finally:
            tool_cache = getattr(self, "_paid_tool_schemas_by_trace", None)
            if isinstance(tool_cache, dict):
                tool_cache.pop(trace.trace_id, None)

    def _patched_on_span_end(self, span):  # type: ignore[misc]
        if token := self._tokens.pop(span.span_id, None):
            detach(token)  # type: ignore[arg-type]
        if not (otel_span := self._otel_spans.pop(span.span_id, None)):
            return

        processing_error: Exception | None = None
        end_time: Optional[int] = None
        try:
            otel_span.update_name(processor_mod._get_span_name(span))
            data = span.span_data
            if isinstance(data, ResponseSpanData):
                if hasattr(data, "response") and isinstance(response := data.response, Response):
                    _cache_tool_schemas(self, span.trace_id, response)
                    otel_span.set_attribute(processor_mod.OUTPUT_MIME_TYPE, processor_mod.JSON)
                    otel_span.set_attribute(processor_mod.OUTPUT_VALUE, response.model_dump_json())
                    for k, v in processor_mod._get_attributes_from_response(response):
                        otel_span.set_attribute(k, v)
                if hasattr(data, "input") and (input := data.input):
                    if isinstance(input, str):
                        otel_span.set_attribute(processor_mod.INPUT_VALUE, input)
                    elif isinstance(input, list):
                        otel_span.set_attribute(processor_mod.INPUT_MIME_TYPE, processor_mod.JSON)
                        otel_span.set_attribute(processor_mod.INPUT_VALUE, safe_json_dumps(input))
                        for k, v in processor_mod._get_attributes_from_input(input):
                            otel_span.set_attribute(k, v)
            elif isinstance(data, FunctionSpanData):
                for k, v in processor_mod._get_attributes_from_function_span_data(data):
                    otel_span.set_attribute(k, v)
                for k, v in _get_enriched_function_attributes(self, span.trace_id, data.name):
                    otel_span.set_attribute(k, v)
            elif isinstance(data, processor_mod.MCPListToolsSpanData):
                for k, v in processor_mod._get_attributes_from_mcp_list_tool_span_data(data):
                    otel_span.set_attribute(k, v)
            elif isinstance(data, processor_mod.HandoffSpanData):
                if data.to_agent and data.from_agent:
                    key = f"{data.to_agent}:{span.trace_id}"
                    self._reverse_handoffs_dict[key] = data.from_agent
                    while len(self._reverse_handoffs_dict) > self._MAX_HANDOFFS_IN_FLIGHT:
                        self._reverse_handoffs_dict.popitem(last=False)
            elif isinstance(data, processor_mod.AgentSpanData):
                otel_span.set_attribute(processor_mod.GRAPH_NODE_ID, data.name)
                key = f"{data.name}:{span.trace_id}"
                if parent_node := self._reverse_handoffs_dict.pop(key, None):
                    otel_span.set_attribute(processor_mod.GRAPH_NODE_PARENT_ID, parent_node)
            elif isinstance(data, processor_mod.GenerationSpanData):
                for k, v in processor_mod._get_attributes_from_generation_span_data(data):
                    otel_span.set_attribute(k, v)
            elif isinstance(data, (processor_mod.CustomSpanData, processor_mod.GuardrailSpanData)):
                for k, v in processor_mod._flatten(data.export()):
                    otel_span.set_attribute(k, v)
        except Exception as exc:
            processing_error = exc
            logger.debug("Failed to enrich openai-agents span", exc_info=True)
            try:
                otel_span.record_exception(exc)
            except Exception:
                logger.debug("Failed to record openai-agents patch exception", exc_info=True)
        finally:
            if span.ended_at:
                try:
                    end_time = processor_mod._as_utc_nano(datetime.fromisoformat(span.ended_at))
                except ValueError:
                    pass
            if processing_error is not None:
                otel_span.set_status(Status(StatusCode.ERROR, str(processing_error)))
            else:
                otel_span.set_status(status=processor_mod._get_span_status(span))
            otel_span.end(end_time)

    processor.on_trace_end = _patched_on_trace_end  # type: ignore[method-assign]
    processor.on_span_end = _patched_on_span_end  # type: ignore[method-assign]
    logger.debug("Patched OpenInferenceTracingProcessor to enrich openai-agents tool spans")


def _cache_tool_schemas(processor: Any, trace_id: str, response: Response) -> None:
    per_trace = getattr(processor, "_paid_tool_schemas_by_trace", None)
    if not isinstance(per_trace, dict):
        per_trace = {}
        setattr(processor, "_paid_tool_schemas_by_trace", per_trace)

    tool_map = per_trace.setdefault(trace_id, {})
    for tool in response.tools or []:
        if not isinstance(tool, FunctionTool):
            continue
        attrs: dict[str, AttributeValue] = {}
        if tool.description:
            attrs[processor_mod_attr("TOOL_DESCRIPTION")] = tool.description
        if tool.parameters is not None:
            attrs[processor_mod_attr("TOOL_PARAMETERS")] = safe_json_dumps(tool.parameters)
        attrs[processor_mod_attr("TOOL_JSON_SCHEMA")] = safe_json_dumps(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                },
            }
        )
        tool_map[tool.name] = attrs


def _get_enriched_function_attributes(
    processor: Any,
    trace_id: str,
    tool_name: str,
):
    tool_map = getattr(processor, "_paid_tool_schemas_by_trace", {}).get(trace_id, {})
    yield from tool_map.get(tool_name, {}).items()


def processor_mod_attr(name: str) -> str:
    from openinference.instrumentation.openai_agents import _processor as processor_mod

    return getattr(processor_mod, name, _SEMCONV_FALLBACKS.get(name, name.lower()))
