from typing import Any

from ..tracing import logger, paid_external_agent_id_var, paid_external_customer_id_var, paid_token_var
from .utils import get_audio_duration
from openai import AsyncOpenAI, OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class PaidOpenAI:
    def __init__(self, openai_client: OpenAI, optional_tracing: bool = False):
        self.openai = openai_client
        self.tracer = trace.get_tracer("paid.python")
        self.optional_tracing = optional_tracing

    @property
    def chat(self):
        return ChatWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def responses(self):
        return ResponsesWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def embeddings(self):
        return EmbeddingsWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def images(self):
        return ImagesWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def audio(self):
        return AudioWrapper(self.openai, self.tracer, self.optional_tracing)


class ChatWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    @property
    def completions(self):
        return ChatCompletionsWrapper(self.openai, self.tracer, self.optional_tracing)


class ChatCompletionsWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    def create(self, *, model: str, messages: list, **kwargs) -> Any:
        # Check if there's an active span (from capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return self.openai.chat.completions.create(model=model, messages=messages, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return self.openai.chat.completions.create(model=model, messages=messages, **kwargs)
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
                response = self.openai.chat.completions.create(model=model, messages=messages, **kwargs)

                # Add usage information if available
                if hasattr(response, "usage") and response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                            "gen_ai.usage.output_tokens": response.usage.completion_tokens,
                            "gen_ai.response.model": response.model,
                        }
                    )

                    # Add cached tokens if available (for newer models)
                    if (
                        hasattr(response.usage, "prompt_tokens_details")
                        and response.usage.prompt_tokens_details
                        and hasattr(response.usage.prompt_tokens_details, "cached_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens", response.usage.prompt_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (
                        hasattr(response.usage, "completion_tokens_details")
                        and response.usage.completion_tokens_details
                        and hasattr(response.usage.completion_tokens_details, "reasoning_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            response.usage.completion_tokens_details.reasoning_tokens,
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error

    def parse(self, **kwargs):
        return self.openai.chat.completions.parse(**kwargs)


class EmbeddingsWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return self.openai.embeddings.create(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return self.openai.embeddings.create(**kwargs)
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
                if hasattr(response, "usage") and response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                            "gen_ai.response.model": response.model,
                        }
                    )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class ImagesWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    def generate(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return self.openai.images.generate(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return self.openai.images.generate(**kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        # Extract model for span naming with proper defaults
        model = kwargs.get("model", "")

        with self.tracer.start_as_current_span("trace.openai.images") as span:
            attributes = {
                "gen_ai.request.model": model,  # there's no model in response, so extract from request
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
                span.set_attributes(
                    {
                        "gen_ai.image.count": kwargs.get("n", 1),  # Default to 1 image
                        "gen_ai.image.size": kwargs.get("size", ""),
                        "gen_ai.image.quality": kwargs.get("quality", ""),
                    }
                )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class ResponsesWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return self.openai.responses.create(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return self.openai.responses.create(**kwargs)
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
                if hasattr(response, "usage") and response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": response.usage.input_tokens,
                            "gen_ai.usage.output_tokens": response.usage.output_tokens,
                            "gen_ai.response.model": response.model,
                        }
                    )

                    # Add cached tokens if available (for newer models)
                    if (
                        hasattr(response.usage, "input_tokens_details")
                        and response.usage.input_tokens_details
                        and hasattr(response.usage.input_tokens_details, "cached_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens", response.usage.input_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (
                        hasattr(response.usage, "output_tokens_details")
                        and response.usage.output_tokens_details
                        and hasattr(response.usage.output_tokens_details, "reasoning_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            response.usage.output_tokens_details.reasoning_tokens,
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class AudioWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    @property
    def transcriptions(self):
        return AudioTranscriptionsWrapper(self.openai, self.tracer, self.optional_tracing)


class AudioTranscriptionsWrapper:
    def __init__(self, openai_client: OpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return self.openai.audio.transcriptions.create(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return self.openai.audio.transcriptions.create(**kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        # Extract model and file for span attributes
        model = kwargs.get("model", "")
        file_input = kwargs.get("file")

        audio_duration = get_audio_duration(file_input) if file_input else None

        with self.tracer.start_as_current_span("trace.openai.audio.transcriptions") as span:
            attributes = {
                "gen_ai.request.model": model,
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "transcription",
            }
            attributes["external_customer_id"] = external_customer_id
            attributes["token"] = token
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            if audio_duration:
                attributes["gen_ai.audio.input_duration_seconds"] = audio_duration
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = self.openai.audio.transcriptions.create(**kwargs)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class PaidAsyncOpenAI:
    def __init__(self, openai_client: AsyncOpenAI, optional_tracing: bool = False):
        self.openai = openai_client
        self.tracer = trace.get_tracer("paid.python")
        self.optional_tracing = optional_tracing

    @property
    def chat(self):
        return AsyncChatWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def responses(self):
        return AsyncResponsesWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def embeddings(self):
        return AsyncEmbeddingsWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def images(self):
        return AsyncImagesWrapper(self.openai, self.tracer, self.optional_tracing)

    @property
    def audio(self):
        return AsyncAudioWrapper(self.openai, self.tracer, self.optional_tracing)


class AsyncChatWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    @property
    def completions(self):
        return AsyncChatCompletionsWrapper(self.openai, self.tracer, self.optional_tracing)


class AsyncChatCompletionsWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    async def create(self, *, model: str, messages: list, **kwargs) -> Any:
        # Check if there's an active span (from capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return await self.openai.chat.completions.create(model=model, messages=messages, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return await self.openai.chat.completions.create(model=model, messages=messages, **kwargs)
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
                response = await self.openai.chat.completions.create(model=model, messages=messages, **kwargs)

                # Add usage information if available
                if hasattr(response, "usage") and response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                            "gen_ai.usage.output_tokens": response.usage.completion_tokens,
                            "gen_ai.response.model": response.model,
                        }
                    )

                    # Add cached tokens if available (for newer models)
                    if (
                        hasattr(response.usage, "prompt_tokens_details")
                        and response.usage.prompt_tokens_details
                        and hasattr(response.usage.prompt_tokens_details, "cached_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens", response.usage.prompt_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (
                        hasattr(response.usage, "completion_tokens_details")
                        and response.usage.completion_tokens_details
                        and hasattr(response.usage.completion_tokens_details, "reasoning_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            response.usage.completion_tokens_details.reasoning_tokens,
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error

    async def parse(self, **kwargs):
        return await self.openai.chat.completions.parse(**kwargs)


class AsyncEmbeddingsWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    async def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return await self.openai.embeddings.create(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return await self.openai.embeddings.create(**kwargs)
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
                response = await self.openai.embeddings.create(**kwargs)

                # Add usage information if available
                if hasattr(response, "usage") and response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
                            "gen_ai.response.model": response.model,
                        }
                    )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class AsyncImagesWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    async def generate(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return await self.openai.images.generate(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return await self.openai.images.generate(**kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        # Extract model for span naming with proper defaults
        model = kwargs.get("model", "")

        with self.tracer.start_as_current_span("trace.openai.images") as span:
            attributes = {
                "gen_ai.request.model": model,  # there's no model in response, so extract from request
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
                response = await self.openai.images.generate(**kwargs)

                # Add image generation cost factors with proper defaults
                span.set_attributes(
                    {
                        "gen_ai.image.count": kwargs.get("n", 1),  # Default to 1 image
                        "gen_ai.image.size": kwargs.get("size", ""),
                        "gen_ai.image.quality": kwargs.get("quality", ""),
                    }
                )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class AsyncResponsesWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    async def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return await self.openai.responses.create(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return await self.openai.responses.create(**kwargs)
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
                response = await self.openai.responses.create(**kwargs)

                # Add usage information if available
                if hasattr(response, "usage") and response.usage:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": response.usage.input_tokens,
                            "gen_ai.usage.output_tokens": response.usage.output_tokens,
                            "gen_ai.response.model": response.model,
                        }
                    )

                    # Add cached tokens if available (for newer models)
                    if (
                        hasattr(response.usage, "input_tokens_details")
                        and response.usage.input_tokens_details
                        and hasattr(response.usage.input_tokens_details, "cached_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.cached_input_tokens", response.usage.input_tokens_details.cached_tokens
                        )

                    # Add reasoning tokens if available (for o1 models)
                    if (
                        hasattr(response.usage, "output_tokens_details")
                        and response.usage.output_tokens_details
                        and hasattr(response.usage.output_tokens_details, "reasoning_tokens")
                    ):
                        span.set_attribute(
                            "gen_ai.usage.reasoning_output_tokens",
                            response.usage.output_tokens_details.reasoning_tokens,
                        )

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error


class AsyncAudioWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    @property
    def transcriptions(self):
        return AsyncAudioTranscriptionsWrapper(self.openai, self.tracer, self.optional_tracing)


class AsyncAudioTranscriptionsWrapper:
    def __init__(self, openai_client: AsyncOpenAI, tracer: trace.Tracer, optional_tracing: bool):
        self.openai = openai_client
        self.tracer = tracer
        self.optional_tracing = optional_tracing

    async def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        # Check if there's an active span (from paid.capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling OpenAI directly.")
                return await self.openai.audio.transcriptions.create(**kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling OpenAI directly")
                return await self.openai.audio.transcriptions.create(**kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        # Extract model and file for span attributes
        model = kwargs.get("model", "")
        file_input = kwargs.get("file")

        # Get audio duration if possible
        audio_duration = get_audio_duration(file_input) if file_input else None

        with self.tracer.start_as_current_span("trace.openai.audio.transcriptions") as span:
            attributes = {
                "gen_ai.request.model": model,
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "transcription",
            }
            attributes["external_customer_id"] = external_customer_id
            attributes["token"] = token
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id
            if audio_duration:
                attributes["gen_ai.audio.input_duration_seconds"] = audio_duration
            span.set_attributes(attributes)

            try:
                # Make the actual OpenAI API call
                response = await self.openai.audio.transcriptions.create(**kwargs)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
