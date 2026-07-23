"""Lightweight live smoke test against each configured LLM provider.

Sends a minimal one-token prompt to every model in schema.models for providers
whose credentials are present in the environment, and reports pass/fail per model.
This is a maintainer tool for periodic model-catalog refreshes (see the
model-refresh skill) -- it is NOT part of the pytest suite, since it makes real
network calls against provider APIs and costs a small amount of real money.

Usage (run from the repo root; PYTHONPATH=src is required, same as src/run_service.py):
    PYTHONPATH=src uv run python scripts/check_live_models.py
    PYTHONPATH=src uv run python scripts/check_live_models.py --provider anthropic google

If ANTHROPIC_API_KEY itself isn't settable in your environment, put the key under a
different variable name and point --anthropic-api-key-env at it:
    PYTHONPATH=src uv run python scripts/check_live_models.py --anthropic-api-key-env MY_VAR
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Callable


def _remap_anthropic_api_key(argv: list[str]) -> None:
    """Copy the --anthropic-api-key-env variable to ANTHROPIC_API_KEY, if given.

    Runs before core.settings is imported, since Settings reads env vars at
    construction time -- so this can't wait for the main argparse pass below.
    Only acts when the flag is explicitly passed; otherwise ANTHROPIC_API_KEY
    is left to whatever is (or isn't) already set.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_var = None
    for i, arg in enumerate(argv):
        if arg == "--anthropic-api-key-env" and i + 1 < len(argv):
            env_var = argv[i + 1]
            break
        if arg.startswith("--anthropic-api-key-env="):
            env_var = arg.split("=", 1)[1]
            break
    if env_var and os.environ.get(env_var):
        os.environ["ANTHROPIC_API_KEY"] = os.environ[env_var]


_remap_anthropic_api_key(sys.argv[1:])

from core.llm import get_model  # noqa: E402
from core.settings import settings  # noqa: E402
from schema.models import (  # noqa: E402
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
    VertexAIModelName,
)

PROMPT = "Reply with exactly one word: OK"

# Providers that only need an API key/flag to smoke test. Ollama, the OpenAI-compatible
# slot, and the fake model are excluded: they need local infra or bespoke config rather
# than a simple "is a key set" check, so they aren't a fit for this generic sweep.
PROVIDER_MODELS: dict[Provider, tuple[type[AllModelEnum], Callable[[], bool]]] = {
    Provider.OPENAI: (OpenAIModelName, lambda: bool(settings.OPENAI_API_KEY)),
    Provider.ANTHROPIC: (AnthropicModelName, lambda: bool(settings.ANTHROPIC_API_KEY)),
    Provider.GOOGLE: (GoogleModelName, lambda: bool(settings.GOOGLE_API_KEY)),
    Provider.GROQ: (GroqModelName, lambda: bool(settings.GROQ_API_KEY)),
    Provider.DEEPSEEK: (DeepseekModelName, lambda: bool(settings.DEEPSEEK_API_KEY)),
    Provider.OPENROUTER: (OpenRouterModelName, lambda: bool(settings.OPENROUTER_API_KEY)),
    Provider.AWS: (AWSModelName, lambda: settings.USE_AWS_BEDROCK),
    Provider.AZURE_OPENAI: (AzureOpenAIModelName, lambda: bool(settings.AZURE_OPENAI_API_KEY)),
    Provider.VERTEXAI: (VertexAIModelName, lambda: bool(settings.GOOGLE_APPLICATION_CREDENTIALS)),
}


async def check_model(model_name: AllModelEnum) -> tuple[bool, str]:
    try:
        model = get_model(model_name)
        result = await model.ainvoke(PROMPT)
        return True, str(result.content).strip()[:60]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:120]


async def main(provider_filter: set[str]) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for provider, (model_enum, has_credentials) in PROVIDER_MODELS.items():
        if provider_filter and provider.value not in provider_filter:
            continue
        if not has_credentials():
            rows.append((provider.value, "-", "SKIP", "no credentials configured"))
            continue
        for model_name in model_enum:
            ok, detail = await check_model(model_name)
            rows.append((provider.value, model_name.value, "PASS" if ok else "FAIL", detail))

    name_width = max((len(r[1]) for r in rows), default=10)
    for provider, model, status, detail in rows:
        print(f"{status:5} {provider:12} {model:<{name_width}} {detail}")

    if any(r[2] == "FAIL" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        nargs="*",
        default=[],
        metavar="PROVIDER",
        help="Limit to these provider values (e.g. anthropic google). Default: all configured.",
    )
    parser.add_argument(
        "--anthropic-api-key-env",
        default=None,
        metavar="VAR",
        help=(
            "Env var to read the Anthropic key from if ANTHROPIC_API_KEY itself isn't set. "
            "No fallback is checked unless this is passed. Applied before this script's own "
            "imports run, so the value here is informational -- set the env var itself before "
            "invoking the script."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(set(args.provider)))
