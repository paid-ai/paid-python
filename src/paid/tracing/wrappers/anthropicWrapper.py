import typing

from ..tracing import logger, paid_external_agent_id_var, paid_external_customer_id_var, paid_token_var
from anthropic import Anthropic
from anthropic.types import ModelParam
from anthropic.types.message_param import MessageParam
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class PaidAnthropic:
    def __init__(self, anthropic_client: Anthropic, optional_tracing: bool = False):
        self.anthropic = anthropic_client
        self.tracer = trace.get_tracer("paid.python")
        self.optional_tracing = optional_tracing

    @property
    def messages(self):
        return MessagesWrapper(self.anthropic, self.tracer, self.optional_tracing)


class MessagesWrapper:
    def __init__(self, anthropic_client: Anthropic, tracer: trace.Tracer, optional_tracing: bool):
        self.anthropic = anthropic_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    def create(
        self, *, model: ModelParam, messages: typing.Iterable[MessageParam], max_tokens: int, **kwargs
    ) -> typing.Any:
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling Anthropic directly.")
                return self.anthropic.messages.create(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling Anthropic directly")
                return self.anthropic.messages.create(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        with self.tracer.start_as_current_span("trace.anthropic.messages") as span:
            attributes = {
                "gen_ai.system": "anthropic",
                "gen_ai.operation.name": "messages",
                "external_customer_id": external_customer_id,
                "token": token,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id

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
                        attributes["gen_ai.usage.cache_read_input_tokens"] = response.usage.cache_read_input_tokens

                span.set_attributes(attributes)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
