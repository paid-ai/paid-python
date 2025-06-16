import logging
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from typing import Any

logger = logging.getLogger(__name__)

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
            logger.warning("No active span found")
            # Call OpenAI directly without tracing
            return self.openai.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        
        # Create child span following OTel GenAI conventions
        span_name = f"trace.chat {model}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attributes({
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            })
            
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
            logger.warning("No active span found")
            # Call OpenAI directly without tracing
            return self.openai.embeddings.create(**kwargs)
        
        # Extract model for span naming
        model = kwargs.get('model', 'unknown')
        
        # Create child span following OTel GenAI conventions
        span_name = f"trace.embeddings {model}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attributes({
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "embeddings",
            })
            
            try:
                # Make the actual OpenAI API call
                response = self.openai.embeddings.create(**kwargs)
                logger.warning(f"ATA-DBG : embeddings Response: {response}")

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
            logger.warning("No active span found")
            # Call OpenAI directly without tracing
            return self.openai.responses.create(**kwargs)
        
        # Extract model for span naming
        model = kwargs.get('model', 'unknown')
        
        # Create child span following OTel GenAI conventions
        span_name = f"trace.responses {model}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attributes({
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
            })
            
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