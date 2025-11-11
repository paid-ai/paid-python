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

    # Use ContextVar for reset tokens to avoid race conditions in async/concurrent scenarios
    _reset_tokens: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
        "reset_tokens", default=None
    )

    @classmethod
    def _get_or_create_reset_tokens(cls) -> dict[str, Any]:
        """Get the reset tokens dict for this context, creating a new one if needed."""
        reset_tokens = cls._reset_tokens.get()
        if reset_tokens is None:
            reset_tokens = {}
            cls._reset_tokens.set(reset_tokens)
        return reset_tokens

    @classmethod
    def get_context(cls) -> dict[str, Any]:
        return {key: var.get() for key, var in cls._context.items()}

    @classmethod
    def get_context_key(cls, key: str) -> Any:
        return cls._context[key].get() if key in cls._context else None

    @classmethod
    def set_context_key(cls, key: str, value: Any) -> None:
        if key not in cls._context:
            logger.warning(f"Invalid context key: {key}")
            return
        reset_token = cls._context[key].set(value)
        reset_tokens = cls._get_or_create_reset_tokens()
        reset_tokens[key] = reset_token

    @classmethod
    def reset_context(cls) -> None:
        reset_tokens = cls._reset_tokens.get()
        if reset_tokens:
            for key, reset_token in reset_tokens.items():
                cls._context[key].reset(reset_token)
            reset_tokens.clear()
