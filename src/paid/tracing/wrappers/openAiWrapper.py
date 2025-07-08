from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from typing import Any
from ..tracing import paid_external_customer_id_var, paid_token_var, paid_external_agent_id_var
from ..tracing import logger

class PaidOpenAI:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client
        self.tracer = trace.get_tracer("paid.python")

    @property
    def chat(self):
        return ChatWrapper(self.openai, self.tracer)

    @property
    def responses(self):
        return ResponsesWrapper(self.openai, self.tracer)

    @property
    def embeddings(self):
        return EmbeddingsWrapper(self.openai, self.tracer)

    @property
    def images(self):
        return ImagesWrapper(self.openai, self.tracer)


class ChatWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer):
        self.openai = openai_client
        self.tracer = tracer

    @property
    def completions(self):
        return ChatCompletionsWrapper(self.openai, self.tracer)


class ChatCompletionsWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer):
        self.openai = openai_client
        self.tracer = tracer

    def create(
        self,
        *,
        model: str,
        messages: list,
        **kwargs
    ) -> Any:
        # Check if there's an active span (from capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            raise RuntimeError(
                "No OTEL span found."
                " Make sure to call this method from Paid.trace()."
            )

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        with self.tracer.start_as_current_span("trace.openai.chat") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
            attributes["external_customer_id"] = external_customer_id
            attributes["token"] = token
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )

                # Add usage information if available
                if hasattr(response, 'usage') and response.usage:
                    span.set_attributes({
                        "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                        "gen_ai.usage.output_tokens": response.usage.completion_tokens,
                        "gen_ai.response.model": response.model,
                    })

                    # Add cached tokens if available (for newer models)
                    if (hasattr(response.usage, 'prompt_tokens_details') and
                        response.usage.prompt_tokens_details and
                        hasattr(response.usage.prompt_tokens_details, 'cached_tokens')):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens",
                            response.usage.prompt_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (hasattr(response.usage, 'completion_tokens_details') and
                        response.usage.completion_tokens_details and
                        hasattr(response.usage.completion_tokens_details, 'reasoning_tokens')):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            response.usage.completion_tokens_details.reasoning_tokens
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class EmbeddingsWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer):
        self.openai = openai_client
        self.tracer = tracer

    def create(
        self,
        **kwargs  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            raise RuntimeError(
                "No OTEL span found."
                " Make sure to call this method from Paid.trace()."
            )

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        with self.tracer.start_as_current_span("trace.openai.embeddings") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "embeddings",
            }
            attributes["external_customer_id"] = external_customer_id
            attributes["token"] = token
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = self.openai.embeddings.create(**kwargs)

                # Add usage information if available
                if hasattr(response, 'usage') and response.usage:
                    span.set_attributes({
                        "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                        "gen_ai.response.model": response.model,
                    })

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class ImagesWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer):
        self.openai = openai_client
        self.tracer = tracer

    def generate(
        self,
        **kwargs  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            raise RuntimeError(
                "No OTEL span found."
                " Make sure to call this method from Paid.trace()."
            )

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        # Extract model for span naming with proper defaults
        model = kwargs.get('model', 'dall-e-3')  # Default to dall-e-3

        with self.tracer.start_as_current_span("trace.openai.images") as span:
            attributes = {
                "gen_ai.request.model": model, # there's no model in response, so extract from request
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "image_generation",
            }
            attributes["external_customer_id"] = external_customer_id
            attributes["token"] = token
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = self.openai.images.generate(**kwargs)

                # Add image generation cost factors with proper defaults
                span.set_attributes({
                    "gen_ai.image.count": kwargs.get('n', 1),  # Default to 1 image
                    "gen_ai.image.size": kwargs.get('size', '1024x1024'),  # Default size
                })

                # Add quality with proper defaults based on model
                if model == 'dall-e-3':
                    quality = kwargs.get('quality', 'standard')  # Default to standard quality
                    span.set_attribute("gen_ai.image.quality", quality)
                elif model == 'gpt-image-1':
                    quality = kwargs.get('quality', 'medium')  # Default to medium quality for GPT Image 1
                    span.set_attribute("gen_ai.image.quality", quality)
                # DALL-E 2 doesn't have quality parameter

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class ResponsesWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer):
        self.openai = openai_client
        self.tracer = tracer

    def create(
        self,
        **kwargs  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            raise RuntimeError(
                "No OTEL span found."
                " Make sure to call this method from Paid.trace()."
            )

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        with self.tracer.start_as_current_span("trace.openai.responses") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
            attributes["external_customer_id"] = external_customer_id
            attributes["token"] = token
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = self.openai.responses.create(**kwargs)

                # Add usage information if available
                if hasattr(response, 'usage') and response.usage:
                    span.set_attributes({
                        "gen_ai.usage.input_tokens": response.usage.input_tokens,
                        "gen_ai.usage.output_tokens": response.usage.output_tokens,
                        "gen_ai.response.model": response.model,
                    })

                    # Add cached tokens if available (for newer models)
                    if (hasattr(response.usage, 'input_tokens_details') and
                        response.usage.input_tokens_details and
                        hasattr(response.usage.input_tokens_details, 'cached_tokens')):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens",
                            response.usage.input_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (hasattr(response.usage, 'output_tokens_details') and
                        response.usage.output_tokens_details and
                        hasattr(response.usage.output_tokens_details, 'reasoning_tokens')):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            response.usage.output_tokens_details.reasoning_tokens
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
