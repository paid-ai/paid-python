import warnings

from .context_data import ContextData
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

otel_id_generator = RandomIdGenerator()


def generate_tracing_token() -> int:
    """
    Generate a unique tracing token without setting it in the context.

    Use this when you want to generate a trace ID to store or pass to another
    process/service without immediately associating it with the current tracing context.
    The token can later be used with set_tracing_token() to link traces across
    different execution contexts.

    Returns:
        int: A unique OpenTelemetry trace ID.

    Notes:
        - This function only generates and returns the token; it does NOT set it in the context.
        - Use this when you need to store the token separately before setting it.

    Examples:
        Generate token to store for later use:

            from paid.tracing import generate_tracing_token, set_tracing_token

            # Process 1: Generate and store
            token = generate_tracing_token()
            save_to_database("task_123", token)

            # Process 2: Retrieve and use
            token = load_from_database("task_123")
            set_tracing_token(token)

            @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
            def process_task():
                # This trace is now linked to the same token
                pass

    See Also:
        set_tracing_token: Set a previously generated token.
    """
    return otel_id_generator.generate_trace_id()


def set_tracing_token(token: int):
    """
    Deprecated: Pass tracing_token directly to @paid_tracing() decorator instead.

    This function is deprecated and will be removed in a future version.
    Use the tracing_token parameter in @paid_tracing() to link traces across processes.

    Instead of:
        token = load_from_storage("workflow_123")
        set_tracing_token(token)
        @paid_tracing(external_customer_id="cust_123", external_agent_id="agent_456")
        def process_workflow():
            ...
        unset_tracing_token()

    Use:
        token = load_from_storage("workflow_123")

        @paid_tracing(
            external_customer_id="cust_123",
            external_agent_id="agent_456",
            tracing_token=token
        )
        def process_workflow():
            ...

    Parameters:
        token (int): A tracing token (for backward compatibility only).

    Old behavior (for reference):
        This function set a token in the context, so all subsequent @paid_tracing() calls
        would use it automatically until unset_tracing_token() was called.
    """
    warnings.warn(
        "set_tracing_token() is deprecated and will be removed in a future version. "
        "Pass tracing_token directly to @paid_tracing(tracing_token=...) decorator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ContextData.set_context_key("trace_id", token)


def unset_tracing_token():
    """
    Deprecated: No longer needed. Use tracing_token parameter in @paid_tracing() instead.

    This function is deprecated and will be removed in a future version.
    Since tracing_token is now passed directly to @paid_tracing(), there's no need
    to manually set/unset tokens in the context.

    Old behavior (for reference):
        This function unset a token previously set by set_tracing_token(), allowing subsequent @paid_tracing() calls
        to have independent traces.

    Migration:
        If you were using set_tracing_token() + unset_tracing_token() pattern,
        simply pass the token directly to @paid_tracing(tracing_token=...) instead.
    """
    warnings.warn(
        "unset_tracing_token() is deprecated and will be removed in a future version. "
        "Use tracing_token parameter in @paid_tracing(tracing_token=...) decorator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ContextData.set_context_key("trace_id", None)
