---
name: model-refresh
description: >-
  Periodically audit the LLM model catalog in src/schema/models.py against what
  each provider currently ships: add newly released models, remove/flag ones the
  provider has deprecated or discourages, and re-point DEFAULT_MODEL fallbacks at
  a current model. Use when asked to "check for new models", "update the model
  list", "refresh the model catalog", or on the scheduled model-refresh trigger.
  Full background and provider-by-provider notes live in docs/Model_Refresh.md.
---

# Model Catalog Refresh

Keep `src/schema/models.py` (the `AllModelEnum` catalog) current: add models a
provider shipped since the last refresh, remove ones the provider has deprecated
or is actively steering users away from, and make sure `src/core/settings.py`'s
per-provider `DEFAULT_MODEL` fallbacks point at a model that still exists.

Read `docs/Model_Refresh.md` first — it has the per-provider docs URLs to check,
naming conventions already used in this repo, and the coupling points (settings.py
defaults, llm.py's `_MODEL_TABLE` and dispatch, tests, `.env.example`) that every
addition/removal touches.

## Workflow

1. **Survey.** For each provider block in `src/schema/models.py`, fetch that
   provider's current model-listing page (URLs are in the class docstrings and in
   `docs/Model_Refresh.md`) and diff it against the enum values already there.
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
3. **Apply changes across every coupled location** — do not edit only the enum:
   - `src/schema/models.py`: add/remove the `StrEnum` member. Keep the docstring
     URL current.
   - `src/core/settings.py` `model_post_init`: if the removed model was a
     `DEFAULT_MODEL` fallback, repoint it at a remaining (ideally cheap/fast) model
     for that provider.
   - `src/core/llm.py`: nothing to do for a plain rename/add (the `_MODEL_TABLE`
     and `if model_name in ...Name` dispatch are enum-driven), but check for any
     provider-specific special-casing (see the Groq safeguard-model branch) that
     might need to apply to the new model too.
   - `tests/core/test_llm.py`, `tests/core/test_settings.py`: update any test that
     hardcodes a model value being removed; add a case for a notable new model if
     it has special handling (temperature override, tool-binding quirk, etc).
   - `.env.example`, `docs/*.md`: update only if they name a specific model that
     changed (most don't).
4. **Do not touch provider credentials or add live network calls during the
   survey step.** This step is pure research + code edit, no API keys needed.
5. **Live-test what you changed** (needs API keys — see "Live testing" below).
6. **Summarize** what was added/removed/repointed and why, citing the provider
   doc for each change. Commit and push per the repo's normal git workflow; open a
   PR only if asked.

## Live testing

`scripts/check_live_models.py` sends a trivial one-word prompt to every model of
every provider that has credentials configured in the environment, and reports
PASS/FAIL per model. It is deliberately outside the pytest suite (real network
calls, tiny real cost) — run it by hand or from a scheduled trigger with keys
populated:

```sh
uv run python scripts/check_live_models.py                    # all configured providers
uv run python scripts/check_live_models.py --provider anthropic google
```

A `SKIP` line means no credentials were present for that provider — that's
expected and not a failure. Only treat `FAIL` rows as build-blocking.

See `docs/Model_Refresh.md` for how to wire API keys into a cloud environment for
this to run unattended, and the tradeoffs of doing so.
