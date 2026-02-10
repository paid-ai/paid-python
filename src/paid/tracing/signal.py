import json
from typing import Any, Optional, Union

from .tracing import get_paid_tracer
from opentelemetry.trace import Status, StatusCode

from paid.logger import logger


def signal(event_name: str, enable_cost_tracing: bool = False, data: Optional[dict[str, Any]] = None):
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
    - Use enable_cost_tracing=True when you want to mark the point where costs were incurred
      and link that signal to cost/usage data from the same trace.

    Examples
    --------
    Basic signal within a tracing context:

        from paid.tracing import paid_tracing, signal

        @paid_tracing(external_customer_id="cust_123", external_product_id="product_456")
        def process_order(order_id):
            # ... do work ...
            signal("order_processed", data={"order_id": order_id})

    Signal with cost tracking:

        @paid_tracing(external_customer_id="cust_123", external_product_id="product_456")
        def call_ai_api():
            # ... call AI provider ...
            signal("ai_api_call_complete", enable_cost_tracing=True)

    Using context manager:

        with paid_tracing(external_customer_id="cust_123", external_product_id="product_456"):
            # ... do work ...
            signal("milestone_reached", data={"step": "validation_complete"})
    """

    tracer = get_paid_tracer()
    with tracer.start_as_current_span("signal") as span:
        attributes: dict[str, Union[str, bool, int, float]] = {
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
                if enable_cost_tracing:
                    attributes["data"] = json.dumps({"paid": {"enable_cost_tracing": True}})

        span.set_attributes(attributes)
        # Mark span as successful
        span.set_status(Status(StatusCode.OK))
        logger.info(f"Signal [{event_name}] was sent")
