from typing import Union

from ..tracing import logger, paid_external_agent_id_var, paid_external_customer_id_var, paid_token_var
from agents import Agent, Runner, RunResult, TContext, TResponseInputItem
from agents.models import get_default_model
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class PaidRunner:
    def __init__(self, runner: Runner, optional_tracing: bool = False):
        self.runner = runner
        self.tracer = trace.get_tracer("paid.python")
        self.optional_tracing = optional_tracing

    def run_sync(self, starting_agent: Agent[TContext], input: Union[str, list[TResponseInputItem]], **kwargs) -> RunResult:
        # Check if there's an active span (from capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling Runner directly.")
                return self.runner.run_sync(starting_agent, input, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling Runner directly")
                return self.runner.run_sync(starting_agent, input, **kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        if starting_agent.model:
            model_name = str(starting_agent.model)
        else:
            model_name = get_default_model()

        with self.tracer.start_as_current_span("trace.openai.agents.run") as span:
            attributes: dict[str, Union[str, int]] = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "agent_run",
                "external_customer_id": external_customer_id,
                "token": token,
                "gen_ai.request.model": model_name,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id

            try:
                # Make the actual Runner API call
                response = self.runner.run_sync(starting_agent, input, **kwargs)
                usage = response.context_wrapper.usage

                attributes["gen_ai.usage.input_tokens"] = usage.input_tokens
                attributes["gen_ai.usage.output_tokens"] = usage.output_tokens
                attributes["gen_ai.usage.cached_input_tokens"] = usage.input_tokens_details.cached_tokens
                attributes["gen_ai.usage.reasoning_output_tokens"] = usage.output_tokens_details.reasoning_tokens

                span.set_attributes(attributes)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(error)
                raise error

    async def run(self, starting_agent: Agent[TContext], input: Union[str, list[TResponseInputItem]], **kwargs) -> RunResult:
        # Check if there's an active span (from capture())
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No tracing, calling Runner directly.")
                return await self.runner.run(starting_agent, input, **kwargs)
            raise RuntimeError("No OTEL span found. Make sure to call this method from Paid.trace().")

        external_customer_id = paid_external_customer_id_var.get()
        external_agent_id = paid_external_agent_id_var.get()
        token = paid_token_var.get()

        if not (external_customer_id and token):
            if self.optional_tracing:
                logger.info(f"{self.__class__.__name__} No external_customer_id or token, calling Runner directly")
                return await self.runner.run(starting_agent, input, **kwargs)
            raise RuntimeError(
                "Missing required tracing information: external_customer_id or token."
                " Make sure to call this method from Paid.trace()."
            )

        if starting_agent.model:
            model_name = str(starting_agent.model)
        else:
            model_name = get_default_model()

        with self.tracer.start_as_current_span("trace.openai.agents.run") as span:
            attributes: dict[str, Union[str, int]] = {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "agent_run",
                "external_customer_id": external_customer_id,
                "token": token,
                "gen_ai.request.model": model_name,
            }
            if external_agent_id:
                attributes["external_agent_id"] = external_agent_id

            try:
                # Make the actual Runner API call
                response = await self.runner.run(starting_agent, input, **kwargs)
                usage = response.context_wrapper.usage

                attributes["gen_ai.usage.input_tokens"] = usage.input_tokens
                attributes["gen_ai.usage.output_tokens"] = usage.output_tokens
                attributes["gen_ai.usage.cached_input_tokens"] = usage.input_tokens_details.cached_tokens
                attributes["gen_ai.usage.reasoning_output_tokens"] = usage.output_tokens_details.reasoning_tokens

                span.set_attributes(attributes)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(error)
                raise error
