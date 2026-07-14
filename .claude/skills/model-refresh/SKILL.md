---
name: model-refresh
description: >-
  Periodically audit the LLM model catalog in src/schema/models.py against what
  each provider currently ships: add newly released models, remove/flag ones the
  provider has deprecated or discourages, and re-point DEFAULT_MODEL fallbacks at
  a current model. Use when asked to "check for new models", "update the model
  list", "refresh the model catalog", or on the scheduled model-refresh trigger.
---

# Model Catalog Refresh

Keep `src/schema/models.py` (the `AllModelEnum` catalog) current: add models a
provider shipped since the last refresh, remove ones the provider has deprecated
or is actively steering users away from, and make sure `src/core/settings.py`'s
per-provider `DEFAULT_MODEL` fallbacks point at a model that still exists.

## Where model config lives

| What | File |
|---|---|
| The enum of every supported model, per provider, with the provider's docs URL in each class docstring | `src/schema/models.py` (`AllModelEnum`) |
| Per-provider default model + which models are "available" when a key is set | `src/core/settings.py` (`Settings.model_post_init`) |
| Enum → API model-string mapping, and provider-specific construction quirks (temperature, streaming, tool-binding) | `src/core/llm.py` (`_MODEL_TABLE`, `get_model`) |
| Unit tests asserting `get_model` builds the right LangChain class per model | `tests/core/test_llm.py` |
| Settings/default-model tests | `tests/core/test_settings.py` |
| Live smoke test (real API calls) | `scripts/check_live_models.py` |

## Workflow

1. **Survey.** For each provider block in `src/schema/models.py`, fetch the docs
   URL already in that class's docstring and diff its current model list against
   the enum values present. These docstring URLs are the **single canonical
   source** for where to look — don't hardcode a second copy of them elsewhere,
   since provider doc URLs shift over time and duplicated links silently rot.
   - **If a docstring URL 404s, redirects to a generic landing page, or otherwise
     no longer points at a model listing:** find the current canonical URL for
     that provider's model docs and update the docstring to it as part of this
     same change, then continue the survey from the corrected page. Don't skip
     the provider just because the old link broke.
   - A few providers need a second check beyond the docstring URL: Azure OpenAI
     lags OpenAI's own releases (it's deployment-based — check what base models
     Azure currently supports), Anthropic has a separate deprecations page worth
     checking for retirement dates, and Vertex AI model paths sometimes differ
     from the Gemini API's own names for the same model (e.g. `models/gemini-2.5-flash`
     vs `gemini-2.5-flash`) — check both columns. Ollama takes a user-supplied
     model name at runtime, so there's nothing to add there; just confirm the
     generic pass-through in `llm.py` still works.
   - **`docs.aws.amazon.com` returns HTTP 403 to `WebFetch`** (it blocks the
     fetcher's user agent — this is not an egress-policy denial). Use `WebSearch`
     for Bedrock model IDs instead; it reads the same doc pages server-side and
     surfaces the model-card and inference-profile-support pages. The per-model
     card pages (`.../model-card-anthropic-claude-<model>.html`) list the exact
     `modelId` and profile IDs.
2. **Classify each gap:**
   - *New model, generally available* → add it.
   - *New model, preview/experimental* → use judgment; this repo has taken preview
     models before (e.g. `gemini-3-pro-preview`) when there's no GA alternative yet,
     but prefer GA when one exists.
   - *Existing model deprecated or sunset by the provider* → remove it, unless it's
     the only model left for that provider (flag that case instead of leaving the
     catalog empty).
   - *Existing model merely superseded but still served* → leave it unless the
     provider's docs explicitly say to migrate off it.
3. **Match existing naming conventions** when adding a member:
   - Enum member names are `SCREAMING_SNAKE_CASE`, usually family+size, e.g.
     `SONNET_45`, `GEMINI_25_PRO`, `LLAMA_33_70B`. Version numbers drop dots
     (`4.5` → `45`).
   - Enum values are the exact string the provider's API expects
     (`claude-sonnet-4-5`, `gemini-2.5-pro`, `gpt-5.1`) — copy verbatim from the
     provider docs, don't guess.
   - Keep provider families grouped and roughly ordered by size/generation within
     a class, matching how they read today.
4. **Apply changes across every coupled location** — do not edit only the enum:
   - `src/schema/models.py`: add/remove the `StrEnum` member. Keep the docstring
     URL current (see step 1).
   - `src/core/settings.py` `model_post_init`: if the removed model was a
     `DEFAULT_MODEL` fallback, repoint it at a remaining (ideally cheap/fast) model
     for that provider — match the intent of the current defaults (e.g.
     Haiku/Flash/Nano-tier, not the flagship).
   - `src/core/llm.py`: nothing to do for a plain rename/add (the `_MODEL_TABLE`
     and `if model_name in ...Name` dispatch are enum-driven), but check for any
     provider-specific special-casing (see the Groq safeguard-model branch) that
     might need to apply to the new model too.
   - `tests/core/test_llm.py`, `tests/core/test_settings.py`: update any test that
     hardcodes a model value being removed; add a case for a notable new model if
     it has special handling (temperature override, tool-binding quirk, etc).
   - `.env.example`, other docs: update only if they name a specific model that
     changed (most don't).
5. **Do not touch provider credentials or add live network calls during the
   survey step.** This step is pure research + code edit, no API keys needed.
6. **Live-test what you changed** (needs API keys — see "Live testing" below).
7. **Summarize** what was added/removed/repointed and why, citing the provider
   doc for each change. Flag any model rename explicitly — it's a breaking change
   for existing deployments pinning the old enum value in `DEFAULT_MODEL`/
   `AVAILABLE_MODELS` env config, not something to swap silently. Commit and push
   per the repo's normal git workflow; open a PR only if asked.

## Providers you can't live-verify (Bedrock, Azure, Vertex service-account, DeepSeek, OpenRouter)

Not having a key for a provider is **not** a reason to skip it — a stale or
broken catalog entry is worse than a doc-sourced one. Update these from docs
just like the rest, and in the PR mark them explicitly as **doc-only /
unverified**, citing the provider page and calling out any caveat below so the
next person with a key knows exactly what to spot-check. Only leave a provider
untouched when the docs themselves are ambiguous *and* the change would be a
product decision (e.g. adding a whole new pricing tier), not a freshness update.

- **AWS Bedrock** — the enum *value* is passed straight to `ChatBedrock(model_id=...)`,
  so it must be a real Bedrock ID, not a friendly label. Two gotchas:
  1. The latest Claude models are **not invocable on-demand by their base model
     ID** — a bare `anthropic.claude-...` call 400s with "on-demand throughput
     isn't supported." They must go through a cross-region **inference profile**:
     the base ID prefixed with a geo (`us.`/`eu.`/`apac.`) or `global.`. Prefer
     `global.` (routes dynamically, region-agnostic — the best fit for a catalog
     value with no region context) and note in the PR that single-region
     deployments not enrolled in Global CRIS should swap the prefix for their geo.
  2. Bedrock inherits the same **sampling-parameter restrictions** as the direct
     Anthropic API — e.g. a Sonnet-5-class model rejects `temperature`. If you
     point a Bedrock entry at such a model, mirror the no-`temperature` branch
     that already exists for it in `llm.py`'s Anthropic and Bedrock dispatch.
  If AWS credentials happen to be present, `boto3.client("bedrock").list_foundation_models()`
  is the fastest way to confirm real IDs — but note that Bedrock access is a
  separate enablement from plain AWS creds, so this can fail with an auth error
  even when other AWS calls would work.
- **Azure OpenAI** is the one genuinely heavier lift, because it's deployment-based
  and the catalog is coupled in more places than the enum:
  - `settings.py` `model_post_init` hardcodes a `required_models` set (currently
    `{"gpt-4o", "gpt-4o-mini"}`) that it validates the `AZURE_OPENAI_DEPLOYMENT_MAP`
    against — bumping the enum means bumping that set and the `.env.example`
    deployment-map sample and the ~7 Azure cases in `tests/core/test_settings.py`.
  - `llm.py` hardcodes `temperature=0.5` for the Azure path, but Azure's GPT-5-era
    **reasoning** variants reject `temperature` (400). If you move Azure onto one,
    add a no-`temperature` branch like the Anthropic/Bedrock Sonnet-5 handling.
  - Changing an Azure enum value is a **breaking change** to every user's
    deployment map (they name deployments after these keys). Treat an Azure
    generation bump as its own reviewed change, and flag the deployment-map break
    loudly — don't fold it silently into a routine refresh.

## Live testing

`scripts/check_live_models.py` sends a trivial one-word prompt to every model of
every provider that has credentials configured in the environment, and reports
PASS/FAIL/SKIP per model. It is deliberately outside the pytest suite (real
network calls, tiny real cost) — run it by hand or from a scheduled trigger with
keys populated:

```sh
PYTHONPATH=src uv run python scripts/check_live_models.py                    # all configured providers
PYTHONPATH=src uv run python scripts/check_live_models.py --provider anthropic google
```

A `SKIP` line means no credentials were present for that provider — that's
expected and not a failure. Only treat `FAIL` rows as build-blocking. Cost is
negligible (a handful of few-token completions per provider) but it's real spend
against real keys — don't wire it into CI or run it on every commit.

If no provider credentials are configured in the current environment, skip this
step entirely rather than failing — the survey/edit step is still fully useful on
its own.
