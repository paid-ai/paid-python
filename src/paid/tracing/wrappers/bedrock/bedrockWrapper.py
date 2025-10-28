from typing import Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from paid.tracing.tracing import (
    get_paid_tracer,
)


class PaidBedrock:
    def __init__(self, bedrock_client: Any):
        self.bedrock_client = bedrock_client

    def converse(self, *, modelId: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("bedrock.converse") as span:
            attributes = {
                "gen_ai.system": "bedrock",
                "gen_ai.operation.name": "converse",
            }

            try:
                response = self.bedrock_client.converse(modelId=modelId, messages=messages, **kwargs)

                # Add usage information
                if "usage" in response and response["usage"]:
                    usage = response["usage"]
                    attributes["gen_ai.usage.input_tokens"] = usage.get("inputTokens", 0)
                    attributes["gen_ai.usage.output_tokens"] = usage.get("outputTokens", 0)
                    attributes["gen_ai.request.model"] = modelId

                    # Handle cache tokens (always present in Bedrock responses)
                    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
                    cache_write_tokens = usage.get("cacheWriteInputTokens", 0)

                    if cache_read_tokens > 0:
                        attributes["gen_ai.usage.cached_input_tokens"] = cache_read_tokens
                    if cache_write_tokens > 0:
                        attributes["gen_ai.usage.cache_creation_input_tokens"] = cache_write_tokens

                span.set_attributes(attributes)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
