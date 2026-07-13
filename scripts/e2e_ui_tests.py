"""Browser end-to-end scenarios for the Streamlit app UI.

Extends ``scripts/smoke_live_app.py`` (a single chat round-trip) with a small
suite covering the key user journeys that Streamlit version bumps or client /
schema changes have quietly broken before. It drives a real browser through the
app the same way a user would, so it catches breakage that the pytest suite
(which mocks the transport) and the docker CI health checks cannot.

Usage:
    uv run --with playwright python scripts/e2e_ui_tests.py [URL] [scenario ...]

Run everything (the default), or only named scenarios:
    uv run --with playwright python scripts/e2e_ui_tests.py http://localhost:8501
    uv run --with playwright python scripts/e2e_ui_tests.py http://localhost:8501 chat feedback
    uv run --with playwright python scripts/e2e_ui_tests.py --list

The URL defaults to the deployed app, or set ``LIVE_APP_URL``. Like the smoke
script, every scenario is URL-parameterized, so the same suite gates PRs locally
(point it at a ``USE_FAKE_MODEL=true`` service + ``streamlit run``) and monitors
production (point it at the deployed URL).

Design notes for running against the live app:
  - Scenarios send only short prompts, and never switch the model away from its
    default, so a production run stays to a handful of cheap LLM calls.
  - The feedback scenario verifies the widget renders and is interactive (the part
    a Streamlit bump breaks) but does not submit a rating - clicking a star writes
    to LangSmith through the backend, which would pollute the production project on
    every monitoring run.

Requires a Chromium Playwright can find: either ``playwright install chromium``,
or a pre-provisioned browser via PLAYWRIGHT_BROWSERS_PATH (as in Claude Code
cloud environments). Exits 0 if every selected scenario passes, 1 otherwise, and
writes ``e2e_<scenario>_failure.png`` next to the CWD for any scenario that fails.
"""

import os
import sys
import time
import urllib.parse

from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

DEFAULT_URL = "https://agent-service-toolkit.streamlit.app/"
# Stable symlink to the pre-provisioned browser in Claude Code cloud environments,
# used as a fallback when the installed playwright's own browser build is absent.
CLOUD_CHROMIUM = "/opt/pw-browsers/chromium"

# A simple, no-tool agent keeps the message thread deterministic (exactly one
# assistant message per turn) so count-based waits are reliable regardless of the
# model behind it. Scenarios that specifically test agent selection override this.
CHAT_AGENT = "chatbot"

WAKE_TIMEOUT_S = 180
RESPONSE_TIMEOUT_S = 120
STREAM_SETTLE_S = 4

CHAT_INPUT = '[data-testid="stChatInput"] textarea'
CHAT_MESSAGE = '[data-testid="stChatMessage"]'


class E2EError(Exception):
    """A scenario assertion failed."""


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Browser / app helpers
# --------------------------------------------------------------------------- #
def launch_browser(p) -> Browser:
    try:
        return p.chromium.launch()
    except Exception:
        executable = os.environ.get("CHROMIUM_EXECUTABLE", CLOUD_CHROMIUM)
        if not os.path.exists(executable):
            raise
        log(f"bundled browser missing - falling back to {executable}")
        return p.chromium.launch(executable_path=executable)


def build_url(base_url: str, **params: str) -> str:
    """Merge query params into base_url, preserving any it already carries."""
    parts = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    query.update({k: v for k, v in params.items() if v is not None})
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def wake_if_sleeping(page: Page) -> None:
    """Streamlit Community Cloud shows a wake-up screen for slept apps."""
    wake_button = page.get_by_text("get this app back up", exact=False)
    try:
        wake_button.first.wait_for(state="visible", timeout=5_000)
    except PlaywrightTimeoutError:
        return  # not sleeping
    log("app is asleep - clicking wake-up button")
    wake_button.first.click()


def open_app(browser: Browser, url: str, agent: str | None = None) -> Page:
    """Open a fresh browser context on the app and wait until it is interactive."""
    if agent:
        url = build_url(url, agent=agent)
    page = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    wake_if_sleeping(page)
    # The chat input appearing means Streamlit booted, the websocket is up, and
    # the app script ran (it renders after agent/model init).
    page.locator(CHAT_INPUT).wait_for(state="visible", timeout=WAKE_TIMEOUT_S * 1_000)
    return page


def query_param(page: Page, key: str) -> str | None:
    query = urllib.parse.urlsplit(page.url).query
    return dict(urllib.parse.parse_qsl(query)).get(key)


def message_texts(page: Page) -> list[str]:
    return [(t or "").strip() for t in page.locator(CHAT_MESSAGE).all_inner_texts()]


def send_message(page: Page, text: str) -> None:
    chat_input = page.locator(CHAT_INPUT)
    chat_input.fill(text)
    chat_input.press("Enter")


def wait_for_response(
    page: Page,
    prompt: str,
    min_count: int,
    timeout_s: int = RESPONSE_TIMEOUT_S,
) -> str:
    """Wait for a new assistant reply to `prompt` to appear and stop streaming.

    Uses the message count (not text) to detect a new turn, since the fake model
    replies with identical text every turn. The reply is considered done once the
    last message is non-empty, is not the prompt itself, and stops changing for
    STREAM_SETTLE_S.
    """
    deadline = time.monotonic() + timeout_s
    last_text, stable_since = "", None
    while time.monotonic() < deadline:
        texts = message_texts(page)
        # The prompt must have landed as an earlier message, with the assistant
        # reply after it, before we start trusting the last message.
        if len(texts) >= min_count and any(prompt in t for t in texts[:-1]):
            text = texts[-1]
            if text and text != prompt:
                if text == last_text:
                    if stable_since and time.monotonic() - stable_since >= STREAM_SETTLE_S:
                        return text
                else:
                    last_text, stable_since = text, time.monotonic()
        time.sleep(1)
    raise E2EError(f"no stable assistant reply to {prompt!r} (>= {min_count} msgs) in {timeout_s}s")


def open_settings(page: Page) -> None:
    """Open the Settings popover. It stays open across reruns, so open it once and
    do all settings interactions before dismissing it - re-clicking closes it."""
    page.get_by_role("button", name="Settings").first.click()
    page.locator('[data-testid="stSelectbox"]').first.wait_for(state="visible", timeout=15_000)


def selectbox_value(page: Page, label: str) -> str | None:
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label)
    return box.get_by_role("combobox").get_attribute("value")


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def scenario_chat(browser: Browser, base_url: str) -> None:
    """Baseline: one message in, one assistant reply streams back and settles."""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    log(f"assistant replied ({len(reply)} chars): {reply[:80]!r}")


def scenario_multi_turn_resume(browser: Browser, base_url: str) -> None:
    """Two-turn conversation, then resume it from the Share link in a fresh session.

    Exercises thread persistence, the agent-aware /history fetch on resume, and the
    Share/resume dialog - the dialog regression that #330 fixed would fail here
    because building the share URL would error instead of rendering a link.
    """
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    thread_id = query_param(page, "thread_id")
    if not thread_id:
        raise E2EError("thread_id was not published to the URL on load")

    send_message(page, "First turn: remember the number 7")
    wait_for_response(page, "First turn: remember the number 7", min_count=2)
    send_message(page, "Second turn: what number did I mention?")
    wait_for_response(page, "Second turn: what number did I mention?", min_count=4)

    # Open the Share/resume dialog and read the shareable URL it builds. The
    # dialog frame appears before Streamlit streams in its markdown, so wait for
    # the code block's text rather than reading it the instant the dialog opens.
    page.get_by_role("button", name="Share/resume chat").first.click()
    dialog = page.locator('[role="dialog"]')
    dialog.wait_for(state="visible", timeout=15_000)
    code = dialog.locator('[data-testid="stCode"] code')
    share_url = ""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if code.count() and (share_url := code.first.inner_text().strip()):
            break
        page.wait_for_timeout(500)
    if not share_url:
        raise E2EError("Share/resume dialog rendered no chat URL (share_chat_dialog broken?)")
    if thread_id not in share_url or "agent=" not in share_url:
        raise E2EError(f"share URL missing thread_id/agent: {share_url!r}")
    log(f"share URL: {share_url}")

    # Resume in a brand-new session (no shared state) and confirm the thread is
    # rehydrated from history rather than starting fresh. A fresh thread renders
    # the agent's "...Ask me anything!" welcome and nothing else; a resumed one
    # replays the persisted conversation instead. (We don't assert an exact count:
    # the fake model used for local runs persists only its reply, while a real
    # backend persists every human+assistant turn.)
    resumed = open_app(browser, share_url)
    if query_param(resumed, "thread_id") != thread_id:
        raise E2EError("resumed session did not carry the original thread_id from the share URL")
    texts = message_texts(resumed)
    joined = "\n".join(texts)
    if not texts or "Ask me anything" in joined:
        raise E2EError(f"resume did not replay the thread - it looks empty/fresh: {texts}")
    log(f"resumed thread {thread_id} replayed {len(texts)} message(s) from history")


def scenario_settings_selectors(browser: Browser, base_url: str) -> None:
    """Settings popover: the model + agent selectboxes render, and switching the
    agent to a non-default one syncs it into the ?agent= URL param."""
    page = open_app(browser, base_url)  # default agent, so ?agent= starts absent
    open_settings(page)

    model = selectbox_value(page, "LLM to use")
    if not model:
        raise E2EError("LLM selectbox rendered without a selected model")
    log(f"model selectbox shows: {model!r}")

    default_agent = selectbox_value(page, "Agent to use")
    agents = page.locator('[data-testid="stSelectbox"]').filter(has_text="Agent to use")
    agents.get_by_role("combobox").click()
    options = page.locator('[role="option"]')
    options.first.wait_for(state="visible", timeout=10_000)
    all_agents = [options.nth(i).inner_text().strip() for i in range(options.count())]
    if len(all_agents) < 2:
        raise E2EError(f"expected multiple agents to choose from, saw {all_agents}")
    # Pick any agent other than the default; the default is dropped from the URL,
    # so switching to a non-default is what proves the query-param binding works.
    target = next(a for a in all_agents if a != default_agent)
    options.filter(has_text=target).first.click()
    page.wait_for_timeout(1_500)

    if selectbox_value(page, "Agent to use") != target:
        raise E2EError(f"agent selectbox did not switch to {target!r}")
    if query_param(page, "agent") != target:
        raise E2EError(f"?agent= URL param is {query_param(page, 'agent')!r}, expected {target!r}")
    log(f"agent switched {default_agent!r} -> {target!r} and synced to the URL")


def scenario_feedback(browser: Browser, base_url: str) -> None:
    """After a reply, the star feedback widget renders and is interactive.

    Asserts the widget's structure - the part a Streamlit bump breaks - without
    submitting a rating: clicking a star writes to LangSmith via the backend, and
    doing that on every run would pollute the production LangSmith project during
    monitoring (and hang against a backend that can't reach LangSmith). We verify
    the stars render, carry their expected aria-labels, and are enabled/clickable.
    """
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    wait_for_response(page, prompt, min_count=2)

    widget = page.locator('[data-testid="stFeedback"]').last
    widget.wait_for(state="visible", timeout=15_000)
    stars = widget.locator('[data-testid="stFeedbackButton"]')
    if stars.count() != 5:
        raise E2EError(f"expected a 5-star feedback widget, found {stars.count()} stars")
    labels = [stars.nth(i).get_attribute("aria-label") for i in range(5)]
    if labels != [f"{i} out of 5 stars" for i in range(1, 6)]:
        raise E2EError(f"feedback stars have unexpected aria-labels: {labels}")
    if not stars.first.is_enabled() or not stars.last.is_enabled():
        raise E2EError("feedback stars rendered but are not interactive")
    log("5-star feedback widget rendered with expected labels and is interactive")


def scenario_streaming_toggle(browser: Browser, base_url: str) -> None:
    """Turning off 'Stream results' still produces a reply via the non-streaming
    (ainvoke) path - a code path the default streaming run never exercises."""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    open_settings(page)
    toggle = page.locator('[data-testid="stCheckbox"]').filter(has_text="Stream results")
    toggle.wait_for(state="visible", timeout=10_000)
    toggle.click()  # default is on -> turn it off
    page.keyboard.press("Escape")  # dismiss the popover so the chat input is reachable
    page.wait_for_timeout(500)

    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    log(f"non-streaming reply rendered ({len(reply)} chars)")


def scenario_new_chat(browser: Browser, base_url: str) -> None:
    """'New Chat' starts a fresh thread: a new thread_id in the URL and a cleared
    conversation."""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    wait_for_response(page, prompt, min_count=2)
    old_thread = query_param(page, "thread_id")

    page.get_by_role("button", name="New Chat").first.click()
    page.locator(CHAT_INPUT).wait_for(state="visible", timeout=30_000)

    deadline = time.monotonic() + 15
    while query_param(page, "thread_id") == old_thread and time.monotonic() < deadline:
        page.wait_for_timeout(500)
    new_thread = query_param(page, "thread_id")
    if not new_thread or new_thread == old_thread:
        raise E2EError(f"thread_id did not change on New Chat (still {old_thread!r})")
    if any(prompt in t for t in message_texts(page)):
        raise E2EError("previous conversation was not cleared after New Chat")
    log(f"New Chat reset thread {old_thread} -> {new_thread} and cleared the conversation")


SCENARIOS = {
    "chat": scenario_chat,
    "multi_turn_resume": scenario_multi_turn_resume,
    "settings_selectors": scenario_settings_selectors,
    "feedback": scenario_feedback,
    "streaming_toggle": scenario_streaming_toggle,
    "new_chat": scenario_new_chat,
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        print("\n".join(SCENARIOS))
        return

    url = os.environ.get("LIVE_APP_URL", DEFAULT_URL)
    names = []
    for arg in args:
        if arg in SCENARIOS:
            names.append(arg)
        elif arg.startswith(("http://", "https://")):
            url = arg
        else:
            print(f"unknown argument: {arg!r} (scenarios: {', '.join(SCENARIOS)})")
            sys.exit(2)
    names = names or list(SCENARIOS)

    log(f"target: {url}")
    log(f"scenarios: {', '.join(names)}")

    results: dict[str, str] = {}
    with sync_playwright() as p:
        browser = launch_browser(p)
        for name in names:
            log(f"--- {name} ---")
            start = time.monotonic()
            try:
                SCENARIOS[name](browser, url)
                results[name] = "PASS"
                log(f"PASS: {name} ({time.monotonic() - start:.0f}s)")
            except Exception as e:
                results[name] = f"FAIL: {e}"
                log(f"FAIL: {name}: {e}")
                _screenshot_failure(browser, name)
        browser.close()

    log("=" * 60)
    for name in names:
        log(f"{results[name].split(':')[0]:<4} {name}: {results[name]}")
    failed = [n for n, r in results.items() if not r.startswith("PASS")]
    if failed:
        log(f"{len(failed)} of {len(names)} scenario(s) failed: {', '.join(failed)}")
        sys.exit(1)
    log(f"all {len(names)} scenario(s) passed")


def _screenshot_failure(browser: Browser, name: str) -> None:
    """Save a screenshot of the last open page for a failed scenario."""
    path = f"e2e_{name}_failure.png"
    try:
        contexts = browser.contexts
        if contexts and contexts[-1].pages:
            contexts[-1].pages[-1].screenshot(path=path, full_page=True)
            log(f"screenshot saved to {path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
