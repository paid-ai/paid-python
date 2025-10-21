<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./assets/paid_light.svg" width=600>
        <source media="(prefers-color-scheme: light)" srcset="./assets/paid_dark.svg" width=600>
        <img alt="Fallback image description" src="./assets/paid_light.svg" width=600>
    </picture>
</div>

#

<div align="center">
    <a href="https://buildwithfern.com?utm_source=github&utm_medium=github&utm_campaign=readme&utm_source=https%3A%2F%2Fgithub.com%2FAgentPaid%2Fpaid-python">
        <img src="https://img.shields.io/badge/%F0%9F%8C%BF-Built%20with%20Fern-brightgreen" alt="fern shield">
    </a>
    <a href="https://pypi.org/project/paid-python">
        <img src="https://img.shields.io/pypi/v/paid-python" alt="pypi shield">
    </a>
</div>

Paid is the all-in-one, drop-in Business Engine for AI Agents that handles your pricing, subscriptions, margins, billing, and renewals with just 5 lines of code.
The Paid Python library provides convenient access to the Paid API from Python applications.

## Documentation

See the full API docs [here](https://paid.docs.buildwithfern.com/api-reference/api-reference/customers/list)

## Installation

You can install the package using pip:

```bash
pip install paid-python
```

## Usage

The client needs to be configured with your account's API key, which is available in the [Paid dashboard](https://app.paid.ai/agent-integration/api-keys).

```python
from paid import Paid

client = Paid(token="API_KEY")

client.customers.create(
    name="name"
)
```

## Request And Response Types

The SDK provides Python classes for all request and response types. These are automatically handled when making API calls.

```python
# Example of creating a customer
response = client.customers.create(
    name="John Doe",
)

# Access response data
print(response.name)
print(response.email)
```

## Exception Handling

When the API returns a non-success status code (4xx or 5xx response), the SDK will raise an appropriate error.

```python
try:
    client.customers.create(...)
except paid.Error as e:
    print(e.status_code)
    print(e.message)
    print(e.body)
    print(e.raw_response)
```

## Logging

Supported log levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

For example, to set the log level to debug, you can set the environment variable:

```bash
export PAID_LOG_LEVEL=DEBUG
```

Defaults to ERROR.

## Cost Tracking via OTEL tracing

### Simple Decorator Method

The easiest way to add cost tracking is using the `@paid_tracing` decorator:

```python
from paid.tracing import paid_tracing

@paid_tracing("<external_customer_id>", "<optional_external_agent_id>")  # add this line
def some_agent_workflow():  # your function
    # Your logic - use any AI providers with Paid wrappers or send signals with Paid.signal().
    # This function is typically an event processor that should lead to AI calls or events emitted as Paid signals
```

- Initializes tracing using your API key you provided to the Paid client, falls back to `PAID_API_KEY` environment variable.
- Handles both sync and async functions
- Gracefully falls back to normal execution if tracing fails

### Using the Paid wrappers

You can track usage costs by using Paid wrappers around your AI provider's SDK.
As of now, the following SDKs' APIs are wrapped:

```
openai
openai-agents (as a hook)
anthropic
langchain (as a hook)
llamaindex
bedrock (boto3)
mistral
gemini (google-genai)
```

Example usage:
```python
from openai import OpenAI
from paid.tracing import paid_tracing
from paid.tracing.wrappers.openai import PaidOpenAI

openAIClient = PaidOpenAI(OpenAI(
    # This is the default and can be omitted
    api_key="<OPENAI_API_KEY>",
))

@paid_tracing("your_external_customer_id", "your_external_agent_id")
def image_generate():
    response = openAIClient.images.generate(
        model="dall-e-3",
        prompt="A sunset over mountains",
        size="1024x1024",
        quality="hd",
        style="vivid",
        n=1
    )
    return response

image_generate()
```

### Auto-Instrumentation (OpenTelemetry Instrumentors)

For maximum convenience, you can use OpenTelemetry auto-instrumentation to automatically track costs without modifying your AI library calls. This approach uses official OpenTelemetry instrumentors for supported AI libraries.

#### Quick Start

```python
from paid import Paid
from paid.tracing import paid_autoinstrument
from openai import OpenAI

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

# Enable auto-instrumentation for all supported libraries
paid_autoinstrument()  # instruments all available: anthropic, gemini, openai, openai-agents, bedrock

# Now all OpenAI calls will be automatically traced
openai_client = OpenAI(api_key="<OPENAI_API_KEY>")

@paid_tracing("your_external_customer_id", "your_external_agent_id")
def chat_with_gpt():
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    return response

chat_with_gpt()  # Costs are automatically tracked!
```

#### Supported Libraries

Auto-instrumentation supports the following AI libraries:

```
anthropic          - Anthropic SDK
gemini             - Google Generative AI (google-generativeai)
openai             - OpenAI Python SDK
openai-agents      - OpenAI Agents SDK
bedrock            - AWS Bedrock (boto3)
```

#### Selective Instrumentation

If you only want to instrument specific libraries, pass them to `paid_autoinstrument()`:

```python
from paid.tracing import paid_autoinstrument

# Instrument only Anthropic and OpenAI
paid_autoinstrument(libraries=["anthropic", "openai"])
```

#### How It Works

- Auto-instrumentation uses official OpenTelemetry instrumentors for each AI library
- It automatically wraps library calls without requiring you to use Paid wrapper classes
- Works seamlessly with `@paid_tracing()` decorator or `Paid.trace()` callback
- Costs are tracked in the same way as when using manual wrappers
- Should be called once during application startup, typically before creating AI client instances


## Signaling via OTEL tracing

A more reliable and user-friendly way to send signals is to send them via OTEL tracing.
This allows you to send signals with less arguments and boilerplate as the information is available in the tracing context `Paid.trace()` or `@paid_tracing()`.
The interface is `Paid.signal()`, which takes in signal name, optional data, and a flag that attaches costs from the same trace.
`Paid.signal()` has to be called within a trace - meaning inside of a callback to `Paid.trace()`.
In contrast to `Paid.usage.record_bulk()`, `Paid.signal()` is using OpenTelemetry to provide reliable delivery.

Here's an example of how to use it:
```python
from paid import Paid
from paid.tracing import paid_tracing

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

@paid_tracing("your_external_customer_id", "your_external_agent_id")  # external_agent_id is necessary for sending signals
def do_work():
    # ...do some work...
    client.signal(
        event_name="<your_signal_name>",
        data={ } # optional data (ex. manual cost tracking data)
    )

do_work()
```

Same, but using callback to specify the function to trace:
```python
from paid import Paid

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

# Initialize tracing, must be after initializing Paid SDK
client.initialize_tracing()

def do_work():
    # ...do some work...
    client.signal(
        event_name="<your_signal_name>",
        data={ } # optional data (ex. manual cost tracking data)
    )

# Finally, capture the traces!
client.trace(external_customer_id = "<your_external_customer_id>",
                external_agent_id = "<your_external_agent_id>",  # external_agent_id is required for signals
                fn = lambda: do_work())
```

### Signal-costs - Attaching cost traces to a signal

If you want a signal to carry information about costs,
then the signal should be sent from the same tracing context
as the wrappers and hooks that recorded those costs.

This will look something like this:

```python
from paid import Paid
from paid.tracing import paid_tracing

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

@paid_tracing("your_external_customer_id", "your_external_agent_id")  # external_agent_id is necessary for sending signals
def do_work():
    # ... your workflow logic
    # ... your AI calls made through Paid wrappers or hooks
    client.signal(
        event_name="<your_signal_name>",
        data={ }, # optional data (ex. manual cost tracking data)
        enable_cost_tracing=True, # set this flag to associate it with costs
    )
    # ... your workflow logic
    # ... your AI calls made through Paid wrappers or hooks (can be sent after the signal too)

do_work()
```

Then, all of the costs traced in @paid_tracing() context are related to that signal.

### Signal-costs - Distributed tracing

Sometimes your agent workflow cannot fit into a single traceable function like above,
because it has to be disjoint for whatever reason. It could even be running across different machines.

For such cases, you can pass a tracing token directly to `@paid_tracing()` or `Paid.trace()` to link distributed traces together.

#### Using `tracing_token` parameter (Recommended)

The simplest way to implement distributed tracing is to pass the token directly to the decorator or trace function:

```python
from paid import Paid
from paid.tracing import paid_tracing, generate_tracing_token
from paid.tracing.wrappers.openai import PaidOpenAI
from openai import OpenAI

# Initialize
client = Paid(token="<PAID_API_KEY>")
openai_client = PaidOpenAI(OpenAI(api_key="<OPENAI_API_KEY>"))

# Process 1: Generate token and do initial work
token = generate_tracing_token()
print(f"Tracing token: {token}")

# Store token for other processes (e.g., in Redis, database, message queue)
save_to_storage("workflow_123", token)

@paid_tracing("customer_123", tracing_token=token, external_agent_id="agent_123")
def process_part_1():
    # AI calls here will be traced
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Analyze data"}]
    )
    # Signal without cost tracing
    client.signal("part_1_complete", enable_cost_tracing=False)

process_part_1()

# Process 2 (different machine/process): Retrieve and use token
token = load_from_storage("workflow_123")

@paid_tracing("customer_123", tracing_token=token, external_agent_id="agent_123")
def process_part_2():
    # AI calls here will be linked to the same trace
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Generate response"}]
    )
    # Signal WITH cost tracing - links all costs from both processes
    client.signal("workflow_complete", enable_cost_tracing=True)

process_part_2()
# No cleanup needed - token is scoped to the decorated function
```

Using `Paid.trace()` instead of decorator:

```python
from paid import Paid
from paid.tracing import generate_tracing_token
from paid.tracing.wrappers.openai import PaidOpenAI
from openai import OpenAI

# Initialize
client = Paid(token="<PAID_API_KEY>")
client.initialize_tracing()
openai_client = PaidOpenAI(OpenAI(api_key="<OPENAI_API_KEY>"))

# Process 1: Generate and use token
token = generate_tracing_token()
save_to_storage("workflow_123", token)

def process_part_1():
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Analyze data"}]
    )
    client.signal("part_1_complete", enable_cost_tracing=False)

client.trace(
    external_customer_id="customer_123",
    external_agent_id="agent_123",
    tracing_token=token,
    fn=lambda: process_part_1()
)

# Process 2: Retrieve and use the same token
token = load_from_storage("workflow_123")

def process_part_2():
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Generate response"}]
    )
    client.signal("workflow_complete", enable_cost_tracing=True)

client.trace(
    external_customer_id="customer_123",
    external_agent_id="agent_123",
    tracing_token=token,
    fn=lambda: process_part_2()
)
```

#### Alternative: Using global context (Advanced)

For more complex scenarios where you need to set the tracing context globally, you can use these functions:

```python
from paid.tracing import (
    generate_tracing_token,
    generate_and_set_tracing_token,
    set_tracing_token,
    unset_tracing_token
)

def generate_tracing_token() -> int:
    """
    Generates and returns a tracing token without setting it in the tracing context.
    Useful when you only want to store or send a tracing token somewhere else
    without immediately activating it.

    Returns:
        int: The tracing token (OpenTelemetry trace ID)
    """

def generate_and_set_tracing_token() -> int:
    """
    This function returns tracing token and attaches it to all consequent
    Paid.trace() or @paid_tracing tracing contexts. So all the costs and signals that share this
    tracing context are associated with each other.

    To stop associating the traces one can either call
    generate_and_set_tracing_token() once again or call unset_tracing_token().
    The former is suitable if you still want to trace but in a fresh
    context, and the latter will go back to unique traces per Paid.trace().

    Returns:
        int: The tracing token (OpenTelemetry trace ID)
    """

def set_tracing_token(token: int):
    """
    Sets tracing token. Provided token should come from generate_and_set_tracing_token()
    or generate_tracing_token(). Once set, the consequent traces Paid.trace() or
    @paid_tracing() will be related to each other.

    Args:
        token (int): A tracing token from generate_and_set_tracing_token() or generate_tracing_token()
    """

def unset_tracing_token():
    """
    Unsets the token previously set by generate_and_set_tracing_token()
    or by set_tracing_token(token). Does nothing if the token was never set.
    """
```

Example using global context:

```python
from paid import Paid
from paid.tracing import paid_tracing, generate_and_set_tracing_token, set_tracing_token, unset_tracing_token
from paid.tracing.wrappers.openai import PaidOpenAI
from openai import OpenAI

# Initialize
client = Paid(token="<PAID_API_KEY>")
openai_client = PaidOpenAI(OpenAI(api_key="<OPENAI_API_KEY>"))

# Process 1: Generate token and do initial work
token = generate_and_set_tracing_token()
print(f"Tracing token: {token}")

# Store token for other processes (e.g., in Redis, database, message queue)
save_to_storage("workflow_123", token)

@paid_tracing("customer_123", external_agent_id="agent_123")
def process_part_1():
    # AI calls here will be traced
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Analyze data"}]
    )
    # Signal without cost tracing
    client.signal("part_1_complete", enable_cost_tracing=False)

process_part_1()

# Process 2 (different machine/process): Retrieve and use token
token = load_from_storage("workflow_123")
set_tracing_token(token)

@paid_tracing("customer_123", external_agent_id="agent_123")
def process_part_2():
    # AI calls here will be linked to the same trace
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Generate response"}]
    )
    # Signal WITH cost tracing - links all costs from both processes
    client.signal("workflow_complete", enable_cost_tracing=True)

process_part_2()

# Clean up
unset_tracing_token()
```


## Manual Cost Tracking

If you would prefer to not use Paid to track your costs automatically but you want to send us the costs yourself,
then you can use manual cost tracking mechanism. Just attach the cost information in the following format to a signal payload:

```python
from paid import Paid, Signal

client = Paid(token="<PAID_API_KEY>")

signal = Signal(
    event_name="<your_signal_name>",
    agent_id="<your_agent_id>",
    customer_id="<your_external_customer_id>",
    data = {
        "costData": {
            "vendor": "<any_vendor_name>", # can be anything, traces are grouped by vendors in the UI
            "cost": {
                "amount": 0.002,
                "currency": "USD"
            },
            "gen_ai.response.model": "<ai_model_name>",
        }
    }
)

client.usage.record_bulk(signals=[signal])
```

Alternatively the same `costData` payload can be passed to OTLP signaling mechanism:

```python
from paid import Paid
from paid.tracing import paid_tracing

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

@paid_tracing("your_external_customer_id", "your_external_agent_id")  # external_agent_id is required for sending signals
def do_work():
    # ...do some work...
    client.signal(
        event_name="<your_signal_name>",
        data={
            "costData": {
                "vendor": "<any_vendor_name>", # can be anything, traces are grouped by vendors in the UI
                "cost": {
                    "amount": 0.002,
                    "currency": "USD"
                },
                "gen_ai.response.model": "<ai_model_name>",
            }
        }
    )

do_work()
```

### Manual Usage Tracking

If you would prefer to send us raw usage manually (without wrappers) and have us compute the cost, you can attach usage data in the following format:

```python
from paid import Paid, Signal

client = Paid(token="<PAID_API_KEY>")

signal = Signal(
    event_name="<your_signal_name>",
    agent_id="<your_agent_id>",
    customer_id="<your_external_customer_id>",
    data = {
        "costData": {
            "vendor": "<any_vendor_name>", # can be anything, traces are grouped by vendors in the UI
            "attributes": {
                "gen_ai.response.model": "gpt-4.1-mini",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 300,
                "gen_ai.usage.cached_input_tokens": 600,
            },
        }
    }
)

client.usage.record_bulk(signals=[signal])
```

Same but via OTEL signaling:

```python
from paid import Paid
from paid.tracing import paid_tracing

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

@paid_tracing("your_external_customer_id", "your_external_agent_id")  # external_agent_id is required for sending signals
def do_work():
    # ...do some work...
    client.signal(
        event_name="<your_signal_name>",
        data={
            "costData": {
                "vendor": "<any_vendor_name>", # can be anything, traces are grouped by vendors in the UI
                "attributes": {
                    "gen_ai.response.model": "gpt-4.1-mini",
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 300,
                    "gen_ai.usage.cached_input_tokens": 600,
                },
            }
        }
    )

do_work()
```

## Async Support

All of the functionality from above is available in async flavor too.

### Async Client

Use `AsyncPaid` instead of `Paid` for async operations:

```python
from paid import AsyncPaid

client = AsyncPaid(token="API_KEY")

# Async API calls
customer = await client.customers.create(name="John Doe")
```

### Async Cost Tracking with Decorator

The `@paid_tracing` decorator automatically handles both sync and async functions:

```python
from openai import AsyncOpenAI
from paid.tracing import paid_tracing
from paid.tracing.wrappers.openai import PaidAsyncOpenAI

# Wrap the async OpenAI client
openai_client = PaidAsyncOpenAI(AsyncOpenAI(api_key="<OPENAI_API_KEY>"))

@paid_tracing("your_external_customer_id", "your_external_agent_id")
async def generate_image():
    response = await openai_client.images.generate(
        model="dall-e-3",
        prompt="A sunset over mountains",
        size="1024x1024",
        quality="hd",
        n=1
    )
    return response

# Call the async function
await generate_image()
```

### Async Signaling

The `signal()` method works seamlessly in async contexts:

```python
from paid import AsyncPaid
from paid.tracing import paid_tracing
from paid.tracing.wrappers.openai import PaidAsyncOpenAI
from openai import AsyncOpenAI

client = AsyncPaid(token="PAID_API_KEY")
openai_client = PaidAsyncOpenAI(AsyncOpenAI(api_key="<OPENAI_API_KEY>"))

@paid_tracing("your_external_customer_id", "your_external_agent_id")
async def do_work():
    # Perform async AI operations
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Send signal (synchronous call within async function)
    client.signal(
        event_name="<your_signal_name>",
        enable_cost_tracing=True  # Associate with traced costs
    )

    return response

# Execute
await do_work()
```

## Contributing

While we value open-source contributions to this SDK, this library is generated programmatically.
Additions made directly to this library would have to be moved over to our generation code,
otherwise they would be overwritten upon the next generated release. Feel free to open a PR as
a proof of concept, but know that we will not be able to merge it as-is. We suggest opening
an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
