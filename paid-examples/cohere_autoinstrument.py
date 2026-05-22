"""
Quick example to verify Cohere auto-instrumentation works end-to-end.

Requirements:
    poetry add cohere  # already a dev dep
    export PAID_API_KEY=...
    export COHERE_API_KEY=...

Run:
    poetry run python paid-examples/cohere_autoinstrument.py
"""

import os

import cohere

from paid import Paid
from paid.tracing import initialize_tracing, paid_autoinstrument, paid_tracing
from dotenv import load_dotenv

_ = load_dotenv()

PAID_API_KEY = os.environ["PAID_API_KEY"]

def main() -> None:
    initialize_tracing(PAID_API_KEY)
    paid_autoinstrument(libraries=["cohere"])

    client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

    with paid_tracing("example_customer", external_product_id="example_product"):
        response = client.chat(
            model="command-r-plus-08-2024",
            messages=[
                cohere.UserChatMessageV2(content="Say hello in one short sentence."),
            ],
        )
        print(response.message.content)


if __name__ == "__main__":
    main()
