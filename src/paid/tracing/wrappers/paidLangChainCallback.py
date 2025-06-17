import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, ChatResult
from langchain_core.messages import BaseMessage
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class PaidLangChainCallback(BaseCallbackHandler):
    """
    LangChain callback handler that integrates with Paid tracing system.
    
    This callback creates child spans for LangChain operations when there's an active
    Paid tracing span (created by _capture() function).
    """
    
    def __init__(self, name: str = "langchain"):
        """
        Initialize the callback handler.
        
        Args:
            name: Name to use for the tracer
        """
        super().__init__()
        self.tracer = trace.get_tracer(f"paid.python.{name}")
        self._spans: Dict[Union[UUID, str], Any] = {}
    
    def _get_span_name(self, serialized: Dict[str, Any], operation: str) -> str:
        """Generate a descriptive span name."""
        class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        return f"langchain.{operation} {class_name}"
    
    def _extract_model_info(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from serialized data."""
        attributes = {}
        
        # Try to get model name from various possible locations
        if "model_name" in serialized:
            attributes["gen_ai.request.model"] = serialized["model_name"]
        elif "model" in serialized:
            attributes["gen_ai.request.model"] = serialized["model"]
        elif "kwargs" in serialized and "model" in serialized["kwargs"]:
            attributes["gen_ai.request.model"] = serialized["kwargs"]["model"]
        
        # Add system information
        if "openai" in str(serialized).lower():
            attributes["gen_ai.system"] = "openai"
        elif "anthropic" in str(serialized).lower():
            attributes["gen_ai.system"] = "anthropic"
        elif "google" in str(serialized).lower():
            attributes["gen_ai.system"] = "google"
        
        return attributes
    
    def _extract_usage_from_result(self, result: Union[LLMResult, ChatResult]) -> Dict[str, Any]:
        """Extract token usage information from LLM result."""
        attributes = {}
        
        if hasattr(result, 'llm_output') and result.llm_output:
            usage = result.llm_output.get('token_usage', {})
            if usage:
                if 'prompt_tokens' in usage:
                    attributes["gen_ai.usage.input_tokens"] = usage['prompt_tokens']
                if 'completion_tokens' in usage:
                    attributes["gen_ai.usage.output_tokens"] = usage['completion_tokens']
                if 'total_tokens' in usage:
                    attributes["gen_ai.usage.total_tokens"] = usage['total_tokens']
        
        # Try to extract model from result
        if hasattr(result, 'llm_output') and result.llm_output:
            if 'model_name' in result.llm_output:
                attributes["gen_ai.response.model"] = result.llm_output['model_name']
        
        return attributes
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts running."""
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            logger.debug("No active span found for LLM start")
            return
        
        span_name = self._get_span_name(serialized, "llm")
        span = self.tracer.start_span(span_name)
        
        # Set basic attributes
        attributes = {
            "gen_ai.operation.name": "chat",
            "langchain.run_id": str(run_id),
            "langchain.run_type": "llm",
        }
        
        # Add model information
        attributes.update(self._extract_model_info(serialized))
        
        # Add parent run ID if available
        if parent_run_id:
            attributes["langchain.parent_run_id"] = str(parent_run_id)
        
        # Add tags if available
        if tags:
            attributes["langchain.tags"] = ",".join(tags)
        
        # Add prompt information
        attributes["gen_ai.prompt.count"] = len(prompts)
        if prompts:
            # Store first prompt for debugging (truncated to avoid huge spans)
            first_prompt = prompts[0][:500] + "..." if len(prompts[0]) > 500 else prompts[0]
            attributes["gen_ai.prompt.first"] = first_prompt
        
        span.set_attributes(attributes)
        self._spans[run_id] = span
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chat model starts running."""
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            logger.debug("No active span found for chat model start")
            return
        
        span_name = self._get_span_name(serialized, "chat")
        span = self.tracer.start_span(span_name)
        
        # Set basic attributes
        attributes = {
            "gen_ai.operation.name": "chat",
            "langchain.run_id": str(run_id),
            "langchain.run_type": "chat_model",
        }
        
        # Add model information
        attributes.update(self._extract_model_info(serialized))
        
        # Add parent run ID if available
        if parent_run_id:
            attributes["langchain.parent_run_id"] = str(parent_run_id)
        
        # Add tags if available
        if tags:
            attributes["langchain.tags"] = ",".join(tags)
        
        # Add message information
        total_messages = sum(len(msg_list) for msg_list in messages)
        attributes["gen_ai.message.count"] = total_messages
        
        # Store first message for debugging (truncated)
        if messages and messages[0]:
            first_message = str(messages[0][0])
            first_message = first_message[:500] + "..." if len(first_message) > 500 else first_message
            attributes["gen_ai.message.first"] = first_message
        
        span.set_attributes(attributes)
        self._spans[run_id] = span
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM ends running."""
        span = self._spans.pop(run_id, None)
        if not span:
            return
        
        try:
            # Add usage information from response
            usage_attributes = self._extract_usage_from_result(response)
            span.set_attributes(usage_attributes)
            
            # Add generation count
            if hasattr(response, 'generations'):
                span.set_attribute("gen_ai.response.generation_count", len(response.generations))
            
            # Mark span as successful
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            logger.exception("Error processing LLM end event")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM errors."""
        span = self._spans.pop(run_id, None)
        if not span:
            return
        
        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain starts running."""
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            logger.debug("No active span found for chain start")
            return
        
        span_name = self._get_span_name(serialized, "chain")
        span = self.tracer.start_span(span_name)
        
        # Set basic attributes
        attributes = {
            "langchain.operation.name": "chain",
            "langchain.run_id": str(run_id),
            "langchain.run_type": "chain",
        }
        
        # Add parent run ID if available
        if parent_run_id:
            attributes["langchain.parent_run_id"] = str(parent_run_id)
        
        # Add tags if available
        if tags:
            attributes["langchain.tags"] = ",".join(tags)
        
        # Add input information (truncated)
        if inputs:
            attributes["langchain.chain.input_keys"] = ",".join(inputs.keys())
            # Store first input value for debugging (truncated)
            first_key = next(iter(inputs.keys()))
            first_value = str(inputs[first_key])
            first_value = first_value[:300] + "..." if len(first_value) > 300 else first_value
            attributes["langchain.chain.first_input"] = first_value
        
        span.set_attributes(attributes)
        self._spans[run_id] = span
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain ends running."""
        span = self._spans.pop(run_id, None)
        if not span:
            return
        
        try:
            # Add output information
            if outputs:
                span.set_attribute("langchain.chain.output_keys", ",".join(outputs.keys()))
            
            # Mark span as successful
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            logger.exception("Error processing chain end event")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain errors."""
        span = self._spans.pop(run_id, None)
        if not span:
            return
        
        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts running."""
        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            logger.debug("No active span found for tool start")
            return
        
        span_name = self._get_span_name(serialized, "tool")
        span = self.tracer.start_span(span_name)
        
        # Set basic attributes
        attributes = {
            "langchain.operation.name": "tool",
            "langchain.run_id": str(run_id),
            "langchain.run_type": "tool",
        }
        
        # Add parent run ID if available
        if parent_run_id:
            attributes["langchain.parent_run_id"] = str(parent_run_id)
        
        # Add tags if available
        if tags:
            attributes["langchain.tags"] = ",".join(tags)
        
        # Add tool input (truncated)
        if input_str:
            truncated_input = input_str[:300] + "..." if len(input_str) > 300 else input_str
            attributes["langchain.tool.input"] = truncated_input
        
        span.set_attributes(attributes)
        self._spans[run_id] = span
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool ends running."""
        span = self._spans.pop(run_id, None)
        if not span:
            return
        
        try:
            # Add tool output (truncated)
            if output:
                truncated_output = output[:300] + "..." if len(output) > 300 else output
                span.set_attribute("langchain.tool.output", truncated_output)
            
            # Mark span as successful
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            logger.exception("Error processing tool end event")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool errors."""
        span = self._spans.pop(run_id, None)
        if not span:
            return
        
        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()


# Usage example:
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize your Paid tracing
paid.initialize_tracing("your-api-key")

# Create the callback
callback = PaidLangChainCallback()

# Use with LangChain
def my_langchain_function():
    llm = ChatOpenAI(model="gpt-4", callbacks=[callback])
    response = llm.invoke([HumanMessage(content="Hello!")])
    return response

# Capture the entire operation
result = paid.capture("customer-123", my_langchain_function)
"""