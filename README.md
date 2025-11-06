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

### Simple Decorator and Context Manager Methods

The easiest way to add cost tracking is using the `@paid_tracing` decorator or context manager:

#### As a Decorator

```python
from paid.tracing import paid_tracing

@paid_tracing("<external_customer_id>", external_agent_id="<optional_external_agent_id>")
def some_agent_workflow():  # your function
    # Your logic - use any AI providers with Paid wrappers or send signals with signal().
    # This function is typically an event processor that should lead to AI calls or events emitted as Paid signals
```

#### As a Context Manager

You can also use `paid_tracing` as a context manager with `with` statements:

```python
from paid.tracing import paid_tracing

# Synchronous
with paid_tracing("customer_123", external_agent_id="agent_456"):
    result = workflow()

# Asynchronous
async with paid_tracing("customer_123", external_agent_id="agent_456"):
    result = await workflow()
```

Both approaches:

- Initialize tracing using your API key you provided to the Paid client, falls back to `PAID_API_KEY` environment variable.
- Handle both sync and async functions/code blocks
- Gracefully fall back to normal execution if tracing fails
- Support the same parameters: `external_customer_id`, `external_agent_id`, `tracing_token`, `store_prompt`, `metadata`

* Note - if it happens that you're calling `paid_tracing` from non-main thread, then it's advised to call:
```python
from paid.tracing import initialize_tracing
initialize_tracing()
```
* `initialize_tracing` also accepts optional arguments like OTEL collector endpoint and api key if you want to reroute your tracing somewhere else :)

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

### Passing User Metadata

You can attach custom metadata to your traces by passing a `metadata` dictionary to the `paid_tracing()` decorator or context manager. This metadata will be stored with the trace and can be used to filter and query traces later.

<Tabs>
  <Tab title="Python - Decorator">
    ```python
    from paid.tracing import paid_tracing, signal
    from paid.tracing.wrappers import PaidOpenAI
    from openai import OpenAI

    openai_client = PaidOpenAI(OpenAI(api_key="<OPENAI_API_KEY>"))

    @paid_tracing(
        "customer_123",
        "agent_123",
        metadata={
            "campaign_id": "campaign_456",
            "environment": "production",
            "user_tier": "enterprise"
        }
    )
    def process_event(event):
        """Process event with custom metadata"""
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": event.content}]
        )

        signal("event_processed", enable_cost_tracing=True)
        return response

    process_event(incoming_event)
    ```
  </Tab>

  <Tab title="Python - Context Manager">
    ```python
    from paid.tracing import paid_tracing, signal
    from paid.tracing.wrappers import PaidOpenAI
    from openai import OpenAI

    openai_client = PaidOpenAI(OpenAI(api_key="<OPENAI_API_KEY>"))

    def process_event(event):
        """Process event with custom metadata"""
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": event.content}]
        )

        signal("event_processed", enable_cost_tracing=True)
        return response

    # Pass metadata to context manager
    with paid_tracing(
        "customer_123",
        external_agent_id="agent_123",
        metadata={
            "campaign_id": "campaign_456",
            "environment": "production",
            "user_tier": "enterprise"
        }
    ):
        process_event(incoming_event)
    ```
  </Tab>

  <Tab title="Node.js">
    ```typescript
    // Metadata support is not yet available in the Node.js SDK.
    // Please use Python for passing custom metadata to traces.
    ```
  </Tab>
</Tabs>

#### Querying Traces by Metadata

Once you've added metadata to your traces, you can filter traces using the metadata parameter in the traces API endpoint:

```bash
# Filter by single metadata field
curl -G "https://api.paid.ai/api/organizations/{orgId}/traces" \
  --data-urlencode 'metadata={"campaign_id":"campaign_456"}' \
  -H "Authorization: Bearer YOUR_API_KEY"

# Filter by multiple metadata fields (all must match)
curl -G "https://api.paid.ai/api/organizations/{orgId}/traces" \
  --data-urlencode 'metadata={"campaign_id":"campaign_456","environment":"production"}' \
  -H "Authorization: Bearer YOUR_API_KEY"
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
paid_autoinstrument()  # instruments all available: anthropic, gemini, openai, openai-agents, bedrock, langchain

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
langchain          - LangChain framework
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
- Works seamlessly with `@paid_tracing()` decorator or context manager
- Costs are tracked in the same way as when using manual wrappers
- Should be called once during application startup, typically before creating AI client instances

## Signaling via OTEL tracing

Signals allow you to emit events within your tracing context. They have access to all tracing information, so you need fewer arguments compared to manual API calls.
Use the `signal()` function which must be called within an active `@paid_tracing()` context (decorator or context manager).

Here's an example of how to use it:

```python
from paid.tracing import paid_tracing, signal

@paid_tracing("your_external_customer_id", "your_external_agent_id")
def do_work():
    # ...do some work...
    signal(
        event_name="<your_signal_name>",
        data={ } # optional data (ex. manual cost tracking data)
    )

do_work()
```

Same approach with context manager:

```python
from paid.tracing import paid_tracing, signal

def do_work():
    # ...do some work...
    signal(
        event_name="<your_signal_name>",
        data={ } # optional data (ex. manual cost tracking data)
    )

# Use context manager instead
with paid_tracing("your_external_customer_id", "your_external_agent_id"):
    do_work()
```

### Signal-costs - Attaching cost traces to a signal

If you want a signal to carry information about costs,
then the signal should be sent from the same tracing context
as the wrappers and hooks that recorded those costs.

This will look something like this:

```python
from paid.tracing import paid_tracing, signal

@paid_tracing("your_external_customer_id", "your_external_agent_id")
def do_work():
    # ... your workflow logic
    # ... your AI calls made through Paid wrappers or hooks
    signal(
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

For such cases, you can pass a tracing token directly to `@paid_tracing()` or context manager to link distributed traces together.

#### Using `tracing_token` parameter (Recommended)

The simplest way to implement distributed tracing is to pass the token directly to the decorator or context manager:

```python
from paid.tracing import paid_tracing, signal, generate_tracing_token
from paid.tracing.wrappers.openai import PaidOpenAI
from openai import OpenAI

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
    signal("part_1_complete", enable_cost_tracing=False)

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
    signal("workflow_complete", enable_cost_tracing=True)

process_part_2()
# No cleanup needed - token is scoped to the decorated function
```

Using context manager instead of decorator:

```python
from paid.tracing import paid_tracing, signal, generate_tracing_token
from paid.tracing.wrappers.openai import PaidOpenAI
from openai import OpenAI

# Initialize
openai_client = PaidOpenAI(OpenAI(api_key="<OPENAI_API_KEY>"))

# Process 1: Generate token and do initial work
token = generate_tracing_token()
save_to_storage("workflow_123", token)

with paid_tracing("customer_123", external_agent_id="agent_123", tracing_token=token):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Analyze data"}]
    )
    signal("part_1_complete", enable_cost_tracing=False)

# Process 2: Retrieve and use the same token
token = load_from_storage("workflow_123")

with paid_tracing("customer_123", external_agent_id="agent_123", tracing_token=token):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Generate response"}]
    )
    signal("workflow_complete", enable_cost_tracing=True)
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
    external_customer_id="<your_external_customer_id>",
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
from paid.tracing import paid_tracing, signal

@paid_tracing("your_external_customer_id", "your_external_agent_id")
def do_work():
    # ...do some work...
    signal(
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
    external_customer_id="<your_external_customer_id>",
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
from paid.tracing import paid_tracing, signal

@paid_tracing("your_external_customer_id", "your_external_agent_id")
def do_work():
    # ...do some work...
    signal(
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

The `signal()` function works seamlessly in async contexts:

```python
from paid.tracing import paid_tracing, signal
from paid.tracing.wrappers.openai import PaidAsyncOpenAI
from openai import AsyncOpenAI

openai_client = PaidAsyncOpenAI(AsyncOpenAI(api_key="<OPENAI_API_KEY>"))

@paid_tracing("your_external_customer_id", "your_external_agent_id")
async def do_work():
    # Perform async AI operations
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Send signal (works in async context)
    signal(
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
