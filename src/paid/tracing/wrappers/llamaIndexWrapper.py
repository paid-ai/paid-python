from typing import Any, Sequence, cast

from ..tracing import logger, paid_external_agent_id_var, paid_external_customer_id_var, paid_token_var
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.llms.openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class PaidLlamaIndexOpenAI:
    def __init__(self, openai_client: OpenAI, optional_tracing: bool = False):
        self.openai = openai_client
        self.tracer = trace.get_tracer("paid.python")
        self.optional_tracing = optional_tracing

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Check if there's an active span (from trace())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling LlamaIndex OpenAI directly.")
                return self.openai.chat(messages=messages, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(
                    f"{self.__class__.__name__} No external_customer_id or token, calling LlamaIndex OpenAI directly"
                )
                return self.openai.chat(messages=messages, **kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        with self.tracer.start_as_current_span("trace.openai.chat") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
                "external_customer_id": external_customer_id,
                "token": token,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = self.openai.chat(messages=messages, **kwargs)
                cast_response = cast(Any, response.raw)

                # Add usage information if available
                if hasattr(cast_response, "usage") and cast_response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": cast_response.usage.prompt_tokens,
                            "gen_ai.usage.output_tokens": cast_response.usage.completion_tokens,
                            "gen_ai.response.model": cast_response.model,
                        }
                    )

                    # Add cached tokens if available (for newer models)
                    if (
                        hasattr(cast_response.usage, "prompt_tokens_details")
                        and cast_response.usage.prompt_tokens_details
                        and hasattr(cast_response.usage.prompt_tokens_details, "cached_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens", cast_response.usage.prompt_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (
                        hasattr(cast_response.usage, "completion_tokens_details")
                        and cast_response.usage.completion_tokens_details
                        and hasattr(cast_response.usage.completion_tokens_details, "reasoning_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            cast_response.usage.completion_tokens_details.reasoning_tokens,
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
