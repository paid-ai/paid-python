from typing import Any, Sequence, cast

from opentelemetry.trace import Status, StatusCode

from paid.tracing.tracing import get_paid_tracer

try:
    from llama_index.core.llms import ChatMessage, ChatResponse
    from llama_index.llms.openai import OpenAI
except ImportError:
    raise ImportError(
        "llama-index-core and llama-index-llms-openai packages are peer-dependencies. "
        "To use the Paid wrapper around llama-index you're assumed to already have "
        "llama-index-core and llama-index-llms-openai packages installed."
    )


class PaidLlamaIndexOpenAI:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.chat") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
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
