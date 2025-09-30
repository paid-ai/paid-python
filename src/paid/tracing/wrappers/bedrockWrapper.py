from typing import Any, Dict, List

from ..tracing import get_paid_tracer, logger, paid_external_agent_id_var, paid_external_customer_id_var, paid_token_var
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class PaidBedrock:
    def __init__(self, bedrock_client: Any, optional_tracing: bool = False):
        self.bedrock_client = bedrock_client
        self.tracer = get_paid_tracer()
        self.optional_tracing = optional_tracing

    def converse(self, *, modelId: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling Bedrock directly.")
                return self.bedrock_client.converse(modelId=modelId, messages=messages, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling Bedrock directly")
                return self.bedrock_client.converse(modelId=modelId, messages=messages, **kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        with self.tracer.start_as_current_span("trace.bedrock.converse") as span:
            attributes = {
                "gen_ai.system": "bedrock",
                "gen_ai.operation.name": "converse",
                "external_customer_id": external_customer_id,
                "token": token,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id

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
