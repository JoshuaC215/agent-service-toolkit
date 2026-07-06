# Model Catalog Refresh

A maintainer guide for keeping the LLM model catalog (`src/schema/models.py`) current:
adding models providers have shipped since the last pass, and removing ones they've
deprecated or are steering users away from. Paired with the `model-refresh` Claude Code
skill, which runs the workflow below.

## Where model config lives

| What | File |
|---|---|
| The enum of every supported model, per provider | `src/schema/models.py` (`AllModelEnum`) |
| Per-provider default model + which models are "available" when a key is set | `src/core/settings.py` (`Settings.model_post_init`) |
| Enum → API model-string mapping, and provider-specific construction quirks (temperature, streaming, tool-binding) | `src/core/llm.py` (`_MODEL_TABLE`, `get_model`) |
| Unit tests asserting `get_model` builds the right LangChain class per model | `tests/core/test_llm.py` |
| Settings/default-model tests | `tests/core/test_settings.py` |
| Live smoke test (real API calls) | `scripts/check_live_models.py` |

## Provider docs to check each refresh

Each `StrEnum` class in `models.py` already carries the provider's docs URL in its
docstring — these are the same links:

- **OpenAI**: https://platform.openai.com/docs/models
- **Azure OpenAI**: deployment-based; check https://learn.microsoft.com/azure/ai-services/openai/concepts/models for which base models Azure currently supports, since Azure lags OpenAI's own releases.
- **DeepSeek**: https://api-docs.deepseek.com/quick_start/pricing
- **Anthropic**: https://docs.anthropic.com/en/docs/about-claude/models#model-names — also check https://docs.anthropic.com/en/docs/resources/model-deprecations for retirement dates.
- **Google (Gemini API)**: https://ai.google.dev/gemini-api/docs/models/gemini
- **Vertex AI**: https://cloud.google.com/vertex-ai/generative-ai/docs/models — note Vertex model names/paths sometimes differ from the Gemini API's own names (e.g. `models/gemini-2.5-flash` vs `gemini-2.5-flash`); check both columns.
- **Groq**: https://console.groq.com/docs/models
- **AWS Bedrock**: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
- **Ollama**: https://ollama.com/search (user-supplied model name at runtime; nothing to add here, just verify the generic pass-through in `llm.py` still works)
- **OpenRouter**: https://openrouter.ai/models

## Naming conventions already in use

Match the existing style when adding a member so the catalog stays consistent:

- Enum member names are `SCREAMING_SNAKE_CASE`, usually family+size, e.g. `SONNET_45`,
  `GEMINI_25_PRO`, `LLAMA_33_70B`. Version numbers drop dots (`4.5` → `45`).
- Enum values are the exact string the provider's API expects (`claude-sonnet-4-5`,
  `gemini-2.5-pro`, `gpt-5.1`) — copy verbatim from the provider docs, don't guess.
- Keep provider families grouped and roughly ordered by size/generation within a class,
  matching how they read today.

## Live end-to-end test (needs real API keys)

Unlike `docs/Dependency_Upgrades.md`'s live test (which uses `USE_FAKE_MODEL` and needs
no credentials), validating that a *model name* is actually accepted by a provider
requires a real call to that provider with a real key. `scripts/check_live_models.py`
does this cheaply: one trivial one-word-reply prompt per model, only for providers whose
credentials are present in the environment.

```sh
uv run python scripts/check_live_models.py
uv run python scripts/check_live_models.py --provider anthropic google   # narrow it down
```

Cost is negligible (a handful of few-token completions per provider) but it is real
spend against real keys — don't wire it into CI or run it on every commit.

## Running this unattended in Claude Code on the web

You can have a cloud session run the survey + live test on a schedule, but be
deliberate about API keys given how cloud environments currently work:

- **There is no dedicated secrets store yet.** Environment variables and setup
  scripts you attach to a cloud environment are stored in that environment's config
  and are visible to anyone who can edit it — treat them like config, not like a
  vault.
- **Recommended setup:** create a separate cloud environment (Claude Code on the
  web → environment selector → "Add environment") used *only* for this refresh
  workflow. Give it its own API keys, ideally ones with tight spend/rate limits or
  provider-side budget alerts, rather than reusing production keys. Add them as
  plain `KEY=value` env vars in that environment's settings.
- **The research/diff step (survey + edit `models.py`/`settings.py`) needs no
  keys at all** — only the live-test step does. If you'd rather not put any
  provider keys in a cloud environment, skip the "Live testing" step in the skill
  during the scheduled run and do it locally by hand afterward with your own
  `.env`.
- **Scheduling:** a monthly cadence is enough to stay ahead of model churn without
  being noisy — most providers ship new models/deprecation notices at most every
  few weeks. Point the trigger at a fresh session running the `model-refresh`
  skill; have it report findings (or open a PR) rather than push directly, so a
  human reviews catalog changes before they land.

## Triage principles

- **Prefer GA over preview/experimental** when both exist for the same capability
  tier, but this repo has taken preview-only models before (`gemini-3-pro-preview`)
  when there was no GA alternative — use judgment, don't block on GA-only purism.
- **Don't remove a provider's last remaining model** even if it's deprecated;
  flag it in the summary instead so the maintainer can decide (removing it would
  break that provider entirely with no fallback).
- **Repoint `DEFAULT_MODEL` fallbacks conservatively** — the default should be a
  cheap/fast model still recommended by the provider for general use, matching the
  intent of the current defaults (e.g. Haiku/Flash/Nano-tier, not the flagship).
- **A model rename is a breaking change for existing deployments** pinning the old
  enum value in `DEFAULT_MODEL`/`AVAILABLE_MODELS` env config — call this out
  explicitly in the summary/PR description rather than silently swapping the enum
  value out from under users.
