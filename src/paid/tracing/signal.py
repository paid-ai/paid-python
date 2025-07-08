import typing
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from .tracing import paid_external_customer_id_var, paid_token_var, paid_external_agent_id_var
from .tracing import logger

def _signal(event_name: str, data: typing.Optional[typing.Dict] = None):
    if (not event_name):
        logger.error("Event name is required for signal.")
        return

    # Check if there's an active span (from capture())
    current_span = trace.get_current_span()
    if current_span == trace.INVALID_SPAN:
        logger.error("Cannot send signal: you should call signal() within capture()")
        return

    external_customer_id = paid_external_customer_id_var.get()
    external_agent_id = paid_external_agent_id_var.get()
    token = paid_token_var.get()
    if not (external_customer_id and external_agent_id and token):
        logger.error(f'Missing some of: external_customer_id: {external_customer_id}, external_agent_id: {external_agent_id}, or token')
        return

    tracer = trace.get_tracer("paid.python")
    with tracer.start_as_current_span("trace.signal") as span:
        attributes = {
            "external_customer_id": external_customer_id,
            "external_agent_id": external_agent_id,
            "event_name": event_name,
            "token": token
        }

        # optional data (ex. manual cost tracking)
        if data:
            attributes["data"] = json.dumps(data)

        span.set_attributes(attributes)
        # Mark span as successful
        span.set_status(Status(StatusCode.OK))
