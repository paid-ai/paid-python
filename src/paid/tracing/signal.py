import json
import typing

from .tracing import get_paid_tracer, logger, paid_external_agent_id_var, paid_external_customer_id_var
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


def signal(event_name: str, enable_cost_tracing: bool = False, data: typing.Optional[dict[str, typing.Any]] = None):
    """
    Emit a signal within a tracing context.

    This function must be called within an active @paid_tracing() context (decorator or context manager).

    Parameters
    ----------
    event_name : str
        The name of the signal (e.g., "user_signup", "payment_processed", "task_completed").
    enable_cost_tracing : bool, optional
        If True, associates this signal with cost/usage traces from the same tracing context.
        Should only be called once per tracing context to avoid multiple signals referring to the same costs.
        Default is False.
    data : dict[str, Any], optional
        Additional context data to attach to the signal. Will be JSON-serialized and stored
        as a span attribute. Example: {"user_id": "123", "amount": 99.99}.

    Notes
    -----
    - Signal must be called within a @paid_tracing() context; calling outside will log an error and return.
    - Both external_customer_id and external_agent_id must be set in the tracing context.
    - Use enable_cost_tracing=True when you want to mark the point where costs were incurred
      and link that signal to cost/usage data from the same trace.

    Examples
    --------
    Basic signal within a tracing context:

        from paid.tracing import paid_tracing, signal

        @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
        def process_order(order_id):
            # ... do work ...
            signal("order_processed", data={"order_id": order_id})

    Signal with cost tracking:

        @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
        def call_ai_api():
            # ... call AI provider ...
            signal("ai_api_call_complete", enable_cost_tracing=True)

    Using context manager:

        with paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456"):
            # ... do work ...
            signal("milestone_reached", data={"step": "validation_complete"})
    """
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
