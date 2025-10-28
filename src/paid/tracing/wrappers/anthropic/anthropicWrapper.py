import typing

from opentelemetry.trace import Status, StatusCode

from paid.tracing.tracing import (
    get_paid_tracer,
)

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import ModelParam
    from anthropic.types.message_param import MessageParam
except ImportError:
    raise ImportError(
        "anthropic package is a peer-dependency. To use the Paid wrapper around anthropic "
        "you're assumed to already have anthropic package installed."
    )


class PaidAnthropic:
    def __init__(self, anthropic_client: Anthropic):
        self.anthropic = anthropic_client

    @property
    def messages(self):
        return MessagesWrapper(self.anthropic)


class MessagesWrapper:
    def __init__(self, anthropic_client: Anthropic):
        self.anthropic = anthropic_client

    def create(
        self, *, model: ModelParam, messages: typing.Iterable[MessageParam], max_tokens: int, **kwargs
    ) -> typing.Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("anthropic.messages.create") as span:
            attributes = {
                "gen_ai.system": "anthropic",
                "gen_ai.operation.name": "messages",
            }

            try:
                response = self.anthropic.messages.create(
                    model=model, messages=messages, max_tokens=max_tokens, **kwargs
                )

                # Add usage information
                if hasattr(response, "usage") and response.usage:
                    attributes["gen_ai.usage.input_tokens"] = response.usage.input_tokens
                    attributes["gen_ai.usage.output_tokens"] = response.usage.output_tokens
                    attributes["gen_ai.response.model"] = response.model
                    if (
                        hasattr(response.usage, "cache_creation_input_tokens")
                        and response.usage.cache_creation_input_tokens
                    ):
                        attributes["gen_ai.usage.cache_creation_input_tokens"] = (
                            response.usage.cache_creation_input_tokens
                        )
                    if hasattr(response.usage, "cache_read_input_tokens") and response.usage.cache_read_input_tokens:
                        attributes["gen_ai.usage.cached_input_tokens"] = response.usage.cache_read_input_tokens

                span.set_attributes(attributes)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class PaidAsyncAnthropic:
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client

    @property
    def messages(self):
        return AsyncMessagesWrapper(self.anthropic)


class AsyncMessagesWrapper:
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client

    async def create(
        self, *, model: ModelParam, messages: typing.Iterable[MessageParam], max_tokens: int, **kwargs
    ) -> typing.Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("anthropic.messages.create") as span:
            attributes = {
                "gen_ai.system": "anthropic",
                "gen_ai.operation.name": "messages",
            }

            try:
                response = await self.anthropic.messages.create(
                    model=model, messages=messages, max_tokens=max_tokens, **kwargs
                )

                # Add usage information
                if hasattr(response, "usage") and response.usage:
                    attributes["gen_ai.usage.input_tokens"] = response.usage.input_tokens
                    attributes["gen_ai.usage.output_tokens"] = response.usage.output_tokens
                    attributes["gen_ai.response.model"] = response.model
                    if (
                        hasattr(response.usage, "cache_creation_input_tokens")
                        and response.usage.cache_creation_input_tokens
                    ):
                        attributes["gen_ai.usage.cache_creation_input_tokens"] = (
                            response.usage.cache_creation_input_tokens
                        )
                    if hasattr(response.usage, "cache_read_input_tokens") and response.usage.cache_read_input_tokens:
                        attributes["gen_ai.usage.cached_input_tokens"] = response.usage.cache_read_input_tokens

                span.set_attributes(attributes)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
