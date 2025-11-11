import contextvars
from typing import Any, Optional

from paid.logger import logger


# this class is used like a namespace, it's not for instantiation
class ContextData:
    _EXTERNAL_CUSTOMER_ID = contextvars.ContextVar[Optional[str]]("external_customer_id", default=None)
    _EXTERNAL_AGENT_ID = contextvars.ContextVar[Optional[str]]("external_agent_id", default=None)
    _TRACE_ID = contextvars.ContextVar[Optional[int]]("trace_id", default=None)
    _STORE_PROMPT = contextvars.ContextVar[Optional[bool]]("store_prompt", default=False)
    _USER_METADATA = contextvars.ContextVar[Optional[dict[str, Any]]]("user_metadata", default=None)

    _context: dict[str, contextvars.ContextVar] = {
        "external_customer_id": _EXTERNAL_CUSTOMER_ID,
        "external_agent_id": _EXTERNAL_AGENT_ID,
        "trace_id": _TRACE_ID,
        "store_prompt": _STORE_PROMPT,
        "user_metadata": _USER_METADATA,
    }

    _reset_tokens: dict[str, Any] = {}

    @classmethod
    def get_context(cls) -> dict[str, Any]:
        return {key: var.get() for key, var in cls._context.items()}

    @classmethod
    def get_context_key(cls, key: str) -> Any:
        return cls._context[key].get() if key in cls._context else None

    @classmethod
    def set_context_key(cls, key: str, value: Any) -> None:
        if key not in cls._context:
            logger.warning("Invalid context key: {key}")
            return
        reset_token = cls._context[key].set(value)
        cls._reset_tokens[key] = reset_token

    @classmethod
    def reset_context(cls) -> None:
        for key, reset_token in cls._reset_tokens.items():
            cls._context[key].reset(reset_token)
        cls._reset_tokens.clear()
