import json
import typing

from .tracing import get_paid_tracer, logger, paid_external_agent_id_var, paid_external_customer_id_var
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


def signal_(event_name: str, enable_cost_tracing: bool, data: typing.Optional[dict[str, typing.Any]] = None):
    if not event_name:
        logger.error("Event name is required for signal.")
        return

    # Check if there's an active span (from capture())
    current_span = trace.get_current_span()
    if current_span == trace.INVALID_SPAN:
        logger.error("Cannot send signal: you should call signal() within tracing context")
        return

    external_customer_id = paid_external_customer_id_var.get()
    external_agent_id = paid_external_agent_id_var.get()
    if not (external_customer_id and external_agent_id):
        logger.error(
            f"Missing some of: external_customer_id: {external_customer_id}, "
            f"external_agent_id: {external_agent_id}. "
            f"You should call signal() within a tracing context"
        )
        return

    tracer = get_paid_tracer()
    with tracer.start_as_current_span("signal") as span:
        attributes: dict[str, typing.Union[str, bool, int, float]] = {
            "external_customer_id": external_customer_id,
            "external_agent_id": external_agent_id,
            "event_name": event_name,
        }

        if enable_cost_tracing:
            # let the app know to associate this signal with cost traces
            if data is None:
                data = {"paid": {"enable_cost_tracing": True}}
            else:
                data["paid"] = {"enable_cost_tracing": True}

        # optional data (ex. manual cost tracking)
        if data:
            try:
                attributes["data"] = json.dumps(data)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize data into JSON for signal [{event_name}]: {e}")

        span.set_attributes(attributes)
        # Mark span as successful
        span.set_status(Status(StatusCode.OK))
        logger.info(f"Signal [{event_name}] was sent")
