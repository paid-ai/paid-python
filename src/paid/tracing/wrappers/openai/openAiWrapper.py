from typing import Any

from opentelemetry.trace import Status, StatusCode

from paid.tracing.tracing import (
    get_paid_tracer,
)
from paid.tracing.wrappers.utils import get_audio_duration

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    raise ImportError(
        "openai package is a peer-dependency. To use the Paid wrapper around openai "
        "you're assumed to already have openai package installed."
    )


class PaidOpenAI:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    @property
    def chat(self):
        return ChatWrapper(self.openai)

    @property
    def responses(self):
        return ResponsesWrapper(self.openai)

    @property
    def embeddings(self):
        return EmbeddingsWrapper(self.openai)

    @property
    def images(self):
        return ImagesWrapper(self.openai)

    @property
    def audio(self):
        return AudioWrapper(self.openai)


class ChatWrapper:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    @property
    def completions(self):
        return ChatCompletionsWrapper(self.openai)


class ChatCompletionsWrapper:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def create(self, *, model: str, messages: list, **kwargs) -> Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.chat.completions.create") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
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
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.embeddings.create") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "embeddings",
            }
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
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def generate(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        # Extract model for span naming with proper defaults
        model = kwargs.get("model", "")

        with tracer.start_as_current_span("openai.images.generate") as span:
            attributes = {
                "gen_ai.request.model": model,  # there's no model in response, so extract from request
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "image_generation",
            }
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
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.responses.create") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
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
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    @property
    def transcriptions(self):
        return AudioTranscriptionsWrapper(self.openai)


class AudioTranscriptionsWrapper:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        # Extract model and file for span attributes
        model = kwargs.get("model", "")
        file_input = kwargs.get("file")

        audio_duration = get_audio_duration(file_input) if file_input else None

        with tracer.start_as_current_span("openai.audio.transcriptions.create") as span:
            attributes = {
                "gen_ai.request.model": model,
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "transcription",
            }
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
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    @property
    def chat(self):
        return AsyncChatWrapper(self.openai)

    @property
    def responses(self):
        return AsyncResponsesWrapper(self.openai)

    @property
    def embeddings(self):
        return AsyncEmbeddingsWrapper(self.openai)

    @property
    def images(self):
        return AsyncImagesWrapper(self.openai)

    @property
    def audio(self):
        return AsyncAudioWrapper(self.openai)


class AsyncChatWrapper:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    @property
    def completions(self):
        return AsyncChatCompletionsWrapper(self.openai)


class AsyncChatCompletionsWrapper:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    async def create(self, *, model: str, messages: list, **kwargs) -> Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.chat.completions.create") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
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
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    async def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.embeddings.create") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "embeddings",
            }
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
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    async def generate(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        # Extract model for span naming with proper defaults
        model = kwargs.get("model", "")

        with tracer.start_as_current_span("openai.images.generate") as span:
            attributes = {
                "gen_ai.request.model": model,  # there's no model in response, so extract from request
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "image_generation",
            }
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
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    async def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("openai.responses.create") as span:
            attributes = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            }
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
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    @property
    def transcriptions(self):
        return AsyncAudioTranscriptionsWrapper(self.openai)


class AsyncAudioTranscriptionsWrapper:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client

    async def create(
        self,
        **kwargs,  # Accept all parameters as-is to match the actual API
    ) -> Any:
        tracer = get_paid_tracer()

        # Extract model and file for span attributes
        model = kwargs.get("model", "")
        file_input = kwargs.get("file")

        # Get audio duration if possible
        audio_duration = get_audio_duration(file_input) if file_input else None

        with tracer.start_as_current_span("openai.audio.transcriptions.create") as span:
            attributes = {
                "gen_ai.request.model": model,
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "transcription",
            }
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
