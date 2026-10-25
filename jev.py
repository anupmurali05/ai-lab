"""
Example script demonstrating how to use Jev (TypeSafe AI) via OpenRouter.

Jev is a "System One" decision model designed for fast, calibrated,
and structured evaluations (probabilities, discrete choices, and rubric scores)
in a single forward pass without token-by-token text generation.

OpenRouter integration:
    - Base URL: https://openrouter.ai/api
    - Model: typesafe/jev-1.13 (or ~typesafe/jev-latest)
    - API Key: OPENROUTER_API_KEY from .env
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    from typesafe_sdk import (
        Choice,
        Noul,
        Score,
        SystemOneResponse,
        TypeSafeClient,
    )
except ImportError:
    print(
        "Error: typesafe-sdk or python-dotenv is not installed.\n"
        "Please run:\n"
        "  .venv/bin/pip install typesafe-sdk python-dotenv"
    )
    sys.exit(1)


# OpenRouter Configuration for Jev
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "typesafe/jev-1.13"


def display_results(result: SystemOneResponse):
    """Inspect and format Jev System One response."""
    billing_res = result.nouls["is_billing_issue"]
    sentiment_res = result.choices["sentiment"]
    urgency_res = result.scores["urgency"]

    print("\n--- Jev Evaluation Results ---")
    print(f"Model used:            {result.model}")
    print(f"Token usage:           input={result.usage.input_tokens}, output={result.usage.output_tokens}")
    print(f"Billing Issue (Noul):  probability = {billing_res.noul:.3f}")
    print(
        f"Sentiment (Choice):    choice = '{sentiment_res.choice}' "
        f"(confidence: {sentiment_res.confidence:.2f}, probabilities: {sentiment_res.probabilities})"
    )
    print(
        f"Urgency (Score):       expected score = {urgency_res.score:.2f} "
        f"(confidence: {urgency_res.confidence:.2f}, rubric levels: {urgency_res.legend})"
    )


def main():
    # Load OpenRouter API key from .env
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    # The context or document state you want Jev to evaluate
    state = (
        "Customer message: 'I was double charged on my invoice #98231 yesterday. "
        "I need a refund immediately or I will contact my bank to issue a chargeback!'"
    )

    # Define questions using Jev's System One primitives:
    # 1. Noul: Probability (0 to 1) for boolean / yes-or-no questions
    # 2. Choice: Categorical classification among discrete options
    # 3. Score: Ordinal position on a rubric
    questions = {
        "is_billing_issue": Noul(
            instructions="Does this customer inquiry relate to billing or payment issues?"
        ),
        "sentiment": Choice(
            instructions="What is the primary customer sentiment?",
            criteria={
                "calm": "Customer is neutral, polite, or relaxed",
                "frustrated": "Customer is unhappy, expressing dissatisfaction",
                "hostile": "Customer is threatening legal action or chargebacks aggressively",
            },
        ),
        "urgency": Score(
            instructions="How urgent is this customer's issue?",
            criteria=["low", "medium", "high", "critical"],
        ),
    }

    print(f"State context:\n  \"{state}\"\n")

    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set in your .env file.")
        print("Please check that .env contains:\n  OPENROUTER_API_KEY=sk-or-v1-...")
        sys.exit(1)

    masked_key = f"{api_key[:12]}...{api_key[-4:]}" if len(api_key) > 16 else "***"
    print(f"Connecting via OpenRouter: {base_url}")
    print(f"Using Model: {model}")
    print(f"API Key:     {masked_key}")

    # Initialize the client configured for OpenRouter
    client = TypeSafeClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    print("\nCalling Jev (System One) via OpenRouter...")
    try:
        result = client.system_one(state=state, questions=questions)
        display_results(result)
    except Exception as e:
        print(f"\nError calling Jev through OpenRouter: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
