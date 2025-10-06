"""Exports OpenTelemetry spans to Paid Events API"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

import httpx
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class PaidSpanProcessor(SpanExporter):
    """OpenTelemetry span processor for Paid Events API."""

    def __init__(
        self,
        api_base_url: str,
        api_key: Optional[str],
        agent_id: str,
        customer_id: str,
        default_deadline_hours: int = 24,
        use_bulk: bool = True,
        timeout: int = 30,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        
        if api_key is None:
            api_key = os.environ.get("PAID_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided either as parameter or via PAID_API_KEY environment variable"
                )
        
        self.api_key = api_key
        self.agent_id = agent_id
        self.customer_id = customer_id
        self.default_deadline_hours = default_deadline_hours
        self.use_bulk = use_bulk
        self.timeout = timeout
        self.active_groups: Dict[str, Dict[str, Any]] = {}

    def _get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _calculate_deadline(self) -> str:
        deadline = datetime.now(timezone.utc) + timedelta(hours=self.default_deadline_hours)
        return deadline.isoformat()

    def _span_to_event_data(self, span: ReadableSpan) -> Dict[str, Any]:
        context = span.get_span_context()

        event_data = {
            "span_id": format(context.span_id, "016x"),
            "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
            "span_name": span.name,
            "span_kind": span.kind.name if span.kind else "INTERNAL",
            "start_time_ns": span.start_time,
            "end_time_ns": span.end_time,
            "duration_ns": (span.end_time or 0) - span.start_time if span.end_time else None,
        }

        if span.status:
            event_data["status"] = {"code": span.status.status_code.name, "description": span.status.description}

        if span.attributes:
            event_data["attributes"] = dict(span.attributes)

        if span.events:
            event_data["events"] = [
                {"name": event.name, "timestamp": event.timestamp, "attributes": dict(event.attributes) if event.attributes else {}}
                for event in span.events
            ]

        if span.resource:
            event_data["resource"] = dict(span.resource.attributes)

        return event_data

    def _start_group(self, trace_id: str) -> Optional[str]:
        try:
            response = httpx.post(
                f"{self.api_base_url}/events/groups/start",
                json={"agentId": self.agent_id, "customerId": self.customer_id, "deadline": self._calculate_deadline()},
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            group_id = result.get("groupId")

            if group_id:
                self.active_groups[trace_id] = {"groupId": group_id, "deadline": result.get("deadline")}
                logger.info(f"Started group {group_id} for trace {trace_id}")
                return group_id

            return None
        except Exception as e:
            logger.error(f"Failed to start group: {e}")
            return None

    def _end_group(self, trace_id: str, group_id: str) -> bool:
        try:
            response = httpx.post(
                f"{self.api_base_url}/events/groups/end",
                json={"groupId": group_id},
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()

            self.active_groups.pop(trace_id, None)
            logger.info(f"Ended group {group_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to end group: {e}")
            return False

    def _send_bulk_events(self, group_id: str, spans: Sequence[ReadableSpan]) -> bool:
        try:
            events = [{"eventType": span.name, "data": self._span_to_event_data(span)} for span in spans]

            response = httpx.post(
                f"{self.api_base_url}/events/groups/send-bulk",
                json={"groupId": group_id, "agentId": self.agent_id, "customerId": self.customer_id, "events": events},
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send bulk events: {e}")
            return False

    def _is_root_span(self, span: ReadableSpan) -> bool:
        return span.parent is None or span.parent.span_id == 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            spans_by_trace: Dict[str, list] = {}
            for span in spans:
                context = span.get_span_context()
                trace_id = format(context.trace_id, "032x")

                if trace_id not in spans_by_trace:
                    spans_by_trace[trace_id] = []
                spans_by_trace[trace_id].append(span)

            for trace_id, trace_spans in spans_by_trace.items():
                # Create group if it doesn't exist yet
                if trace_id not in self.active_groups:
                    # Ensures child spans that arrive first aren't dropped
                    group_id = self._start_group(trace_id)
                    if not group_id:
                        # Skip this batch
                        logger.warning(f"Failed to create group for trace {trace_id}, skipping {len(trace_spans)} spans")
                        continue

                group_info = self.active_groups.get(trace_id)
                if not group_info:
                    continue

                group_id = group_info["groupId"]

                if self.use_bulk and len(trace_spans) > 1:
                    self._send_bulk_events(group_id, trace_spans)
                else:
                    for span in trace_spans:
                        event_data = self._span_to_event_data(span)
                        httpx.post(
                            f"{self.api_base_url}/events/groups/send",
                            json={
                                "groupId": group_id,
                                "agentId": self.agent_id,
                                "customerId": self.customer_id,
                                "eventType": span.name,
                                "data": event_data,
                            },
                            headers=self._get_headers(),
                            timeout=self.timeout,
                        )

                # End group if root span is complete
                for span in trace_spans:
                    if self._is_root_span(span) and span.end_time:
                        self._end_group(trace_id, group_id)
                        break

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        for trace_id, group_info in list(self.active_groups.items()):
            self._end_group(trace_id, group_info["groupId"])
        self.active_groups.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
