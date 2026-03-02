import json
import random
import typing
import warnings

from .context_data import ContextData
from .tracing import get_paid_tracer, get_token
from opentelemetry.trace import Status, StatusCode

from paid.client import Paid
from paid.logger import logger
from paid.types.bulk_signals_response import BulkSignalsResponse
from paid.types.customer_by_external_id import CustomerByExternalId
from paid.types.product_by_external_id import ProductByExternalId
from paid.types.signal import Signal


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

    warnings.warn(
        "paid.tracing.signal() is deprecated and will be removed in a future release. "
        "Use paid.tracing.cost_attributed_signal() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    tracer = get_paid_tracer()
    with tracer.start_as_current_span("signal") as span:
        attributes: dict[str, typing.Union[str, bool, int, float]] = {"event_name": event_name}

        if enable_cost_tracing:
            if data is None:
                data = {"paid": {"enable_cost_tracing": True}}
            else:
                data["paid"] = {"enable_cost_tracing": True}

        if data:
            try:
                attributes["data"] = json.dumps(data)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize data into JSON for signal [{event_name}]: {e}")
                if enable_cost_tracing:
                    attributes["data"] = json.dumps({"paid": {"enable_cost_tracing": True}})

        span.set_attributes(attributes)
        span.set_status(Status(StatusCode.OK))
        logger.info(f"Signal [{event_name}] was sent")


def cost_attributed_signal(
    event_name: str,
    data: typing.Optional[dict[str, typing.Any]] = None,
    idempotency_key: typing.Optional[str] = None,
    base_url_override: typing.Optional[str] = None,
) -> BulkSignalsResponse:
    """
    Emit a cost-attributed signal via the Signals REST API.

    This function is expected to be called inside an active @paid_tracing() context.
    It will raise an exception if the external_customer_id or external_product_id are not set in the context.
    """
    if not event_name:
        raise ValueError("cost_attributed_signal requires a non-empty event_name")

    external_customer_id = ContextData.get_context_key("external_customer_id")
    external_product_id = ContextData.get_context_key("external_agent_id")
    paid_scope_id = ContextData.get_context_key("paid_scope_id")

    token = get_token()
    if token is None:
        raise RuntimeError(
            "Cannot send cost_attributed_signal: missing API token. " + "It should be set via initialize_tracing()"
        )

    if external_customer_id is None or external_product_id is None or paid_scope_id is None:
        raise RuntimeError(
            "cost_attributed_signal must be called inside @paid_tracing with "
            + "external_customer_id, external_product_id, and paid_scope_id set"
        )

    payload_data = dict(data) if data is not None else {}
    paid_data = payload_data.get("paid")
    if isinstance(paid_data, dict):
        paid_data["enable_cost_tracing"] = True
    else:
        payload_data["paid"] = {"enable_cost_tracing": True}

    # costs will be attributed to this signal because their paid_scope_id will match
    payload_data["paid"]["paid_scope_id"] = paid_scope_id
    # reset paid_scope_id so that next signal will be sent with a new scope
    ContextData.set_context_key("paid_scope_id", random.randint(1, 2**31 - 1))

    client_kwargs: dict[str, typing.Any] = {"token": token}
    if base_url_override is not None:
        client_kwargs["base_url"] = base_url_override
    client = Paid(**client_kwargs)

    signal_kwargs: dict[str, typing.Any] = {
        "event_name": event_name,
        "customer": CustomerByExternalId(external_customer_id=external_customer_id),
        "attribution": ProductByExternalId(external_product_id=external_product_id),
        "data": payload_data,
    }
    if idempotency_key is not None:
        signal_kwargs["idempotency_key"] = idempotency_key

    res = client.signals.create_signals(signals=[Signal(**signal_kwargs)])

    logger.info(f"Cost-attributed signal [{event_name}] was sent")
    return res
