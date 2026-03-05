import importlib

import pytest

from paid.tracing.context_data import ContextData
from paid.tracing.signal import cost_attributed_signal
from paid.types.customer_by_external_id import CustomerByExternalId
from paid.types.product_by_external_id import ProductByExternalId

signal_module = importlib.import_module("paid.tracing.signal")


def test_cost_attributed_signal_uses_rest_client(monkeypatch):
    calls: list[dict] = []
    created_tokens: list[str] = []

    class FakeSignalsClient:
        def create_signals(self, *, signals):
            calls.append({"signals": signals})
            return {"ok": True}

    class FakePaid:
        def __init__(self, *, token):
            created_tokens.append(token)
            self.signals = FakeSignalsClient()

    monkeypatch.setattr(signal_module, "Paid", FakePaid)
    monkeypatch.setattr(signal_module, "get_token", lambda: "api-token")

    ContextData.set_context_key("external_customer_id", "cust-ext-123")
    ContextData.set_context_key("external_agent_id", "prod-ext-456")
    ContextData.set_context_key("paid_scope_id", 1001)
    try:
        response = cost_attributed_signal("invoice.created", data={"k": "v"}, idempotency_key="idem-1")
    finally:
        ContextData.reset_context()

    assert response == {"ok": True}
    assert created_tokens == ["api-token"]
    assert len(calls) == 1
    sent_signal = calls[0]["signals"][0]
    assert sent_signal.event_name == "invoice.created"
    assert sent_signal.idempotency_key == "idem-1"
    assert sent_signal.data == {"k": "v", "paid": {"paid_scope_id": 1001}}
    assert isinstance(sent_signal.customer, CustomerByExternalId)
    assert sent_signal.customer.external_customer_id == "cust-ext-123"
    assert isinstance(sent_signal.attribution, ProductByExternalId)
    assert sent_signal.attribution.external_product_id == "prod-ext-456"


def test_cost_attributed_signal_idempotency_key_is_optional(monkeypatch):
    calls: list[dict] = []

    class FakeSignalsClient:
        def create_signals(self, *, signals):
            calls.append({"signals": signals})
            return {"ok": True}

    class FakePaid:
        def __init__(self, *, token):
            self.signals = FakeSignalsClient()

    monkeypatch.setattr(signal_module, "Paid", FakePaid)
    monkeypatch.setattr(signal_module, "get_token", lambda: "api-token")

    ContextData.set_context_key("external_customer_id", "cust-ext-123")
    ContextData.set_context_key("external_agent_id", "prod-ext-456")
    ContextData.set_context_key("paid_scope_id", 1002)
    try:
        response = cost_attributed_signal("invoice.created", data={"k": "v"})
    finally:
        ContextData.reset_context()

    assert response == {"ok": True}
    sent_signal = calls[0]["signals"][0]
    assert sent_signal.idempotency_key is None


def test_cost_attributed_signal_raises_for_missing_context(monkeypatch):
    monkeypatch.setattr(signal_module, "get_token", lambda: "api-token")
    with pytest.raises(RuntimeError, match="must be called inside @paid_tracing"):
        cost_attributed_signal("invoice.created", data={"k": "v"})


def test_cost_attributed_signal_raises_for_invalid_input():
    with pytest.raises(ValueError, match="non-empty event_name"):
        cost_attributed_signal("")
