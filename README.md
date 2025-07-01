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

Falls back to no logging.

## Cost Tracking

It's possible to track usage costs by using Paid wrappers around you AI provider API.
As of now, the following OpenAI python APIs are supported:

```
chat.completions.create()
responses.create()
images.generate()
embeddings.create()
```

Example usage:

```python
from openai import OpenAI
from paid import Paid
from paid.tracing.wrappers import PaidOpenAI

# Initialize Paid SDK
client = Paid(token="PAID_API_KEY")

openAIClient = PaidOpenAI(OpenAI(
    # This is the default and can be omitted
    api_key="<OPENAI_API_KEY>",
))

# Initialize tracing, must be after initializeing Paid SKD
client.initialize_tracing()

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

# Finally, capture the traces!
_ = client.trace(external_customer_id = "<your_external_customer_id>",
                external_agent_id = "<your_external_agent_id>",  # can optionally include external_agent_id to enable agent-level cost tracking
                fn = lambda: image_generate())
```

## Signaling within the traces.

A more reliable and user-friendly way to send signals is to send them from within the traces.
This allows you to send signals with the same customer and agent IDs as the trace, with less arguments and boilerplate.
The interface is `Paid.signal()`, which takes in signal name and optional data.
`Paid.signal()` has to be called within a trace - meaning inside of a callback to `Paid.trace()`.
In contrast to `Paid.usage.record_bulk()`, `Paid.signal()` is using OpenTelemetry to provide reliable delivery.

Here's an example of how to use it:
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
_ = client.trace(external_customer_id = "<your_external_customer_id>",
                external_agent_id = "<your_external_agent_id>",  # external_agent_id is required for signals
                fn = lambda: do_work())
```

## Manual Cost Tracking

Manual cost tracking allow to insert your own costs to the usage data and
cost traces will be created based on that info.

```python
from paid import Paid, Signal

client = Paid(token="<PAID_API_KEY>")

signal = Signal(
    event_name="<your_signal_name>",
    agent_id="<your_agent_id>",
    customer_id="<your_external_customer_id>",
    data = {
        "costData": {
            "vendor": "<any_vendor_name>", # can be anything
            "cost": {
                "amount": 0.002,
                "currency": "USD"
            },
            "gen_ai.response.model": "<ai_model_name>",
        }
    }
)

_ = client.usage.record_bulk(signals=[signal])
```

## Contributing

While we value open-source contributions to this SDK, this library is generated programmatically.
Additions made directly to this library would have to be moved over to our generation code,
otherwise they would be overwritten upon the next generated release. Feel free to open a PR as
a proof of concept, but know that we will not be able to merge it as-is. We suggest opening
an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
