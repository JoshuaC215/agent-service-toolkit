"""Smoke test the deployed Streamlit app end-to-end with a real browser.

Loads the app, sends one chat message, and verifies an assistant response
streams back. Streamlit is websocket-driven, so a plain HTTP check only proves
the page shell loads — this drives the actual chat round-trip (browser ->
Streamlit -> agent service -> LLM -> back).

Usage:
    uv run --with playwright python scripts/smoke_live_app.py [URL]

The URL defaults to the deployed app, or set LIVE_APP_URL. For local testing:
    uv run --with playwright python scripts/smoke_live_app.py http://localhost:8501

Requires a Chromium Playwright can find: either `playwright install chromium`,
or a pre-provisioned browser via PLAYWRIGHT_BROWSERS_PATH (as in Claude Code
cloud environments). Exits 0 on pass, 1 on fail, and writes
smoke_live_app_failure.png next to the CWD on failure for diagnosis.

Note: against the deployed app this sends one real message, which costs one
(cheap) LLM call. Streamlit Community Cloud apps that went to sleep are woken
by clicking through the wake-up screen; allow a couple of minutes for that.
"""

import os
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://agent-service-toolkit.streamlit.app/"
# Stable symlink to the pre-provisioned browser in Claude Code cloud environments,
# used as a fallback when the installed playwright's own browser build is absent.
CLOUD_CHROMIUM = "/opt/pw-browsers/chromium"
TEST_MESSAGE = "Reply with the single word: pong"
WAKE_TIMEOUT_S = 180
RESPONSE_TIMEOUT_S = 120
STREAM_SETTLE_S = 4


def log(msg: str) -> None:
    print(f"[smoke_live_app] {msg}", flush=True)


def fail(page, reason: str) -> None:
    log(f"FAIL: {reason}")
    try:
        page.screenshot(path="smoke_live_app_failure.png", full_page=True)
        log("screenshot saved to smoke_live_app_failure.png")
    except Exception:
        pass
    sys.exit(1)


def wake_if_sleeping(page) -> None:
    """Streamlit Community Cloud shows a wake-up screen for slept apps."""
    wake_button = page.get_by_text("get this app back up", exact=False)
    try:
        wake_button.first.wait_for(state="visible", timeout=5_000)
    except PlaywrightTimeoutError:
        return  # not sleeping
    log("app is asleep - clicking wake-up button")
    wake_button.first.click()


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LIVE_APP_URL", DEFAULT_URL)
    log(f"target: {url}")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            executable = os.environ.get("CHROMIUM_EXECUTABLE", CLOUD_CHROMIUM)
            if not os.path.exists(executable):
                raise
            log(f"bundled browser missing - falling back to {executable}")
            browser = p.chromium.launch(executable_path=executable)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        wake_if_sleeping(page)

        # The chat input appearing means Streamlit booted, the websocket is up,
        # and the app script ran (it renders after agent/model init).
        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        try:
            chat_input.wait_for(state="visible", timeout=WAKE_TIMEOUT_S * 1_000)
        except PlaywrightTimeoutError:
            fail(page, f"chat input never appeared within {WAKE_TIMEOUT_S}s")
        log("app loaded, chat input visible")

        messages = page.locator('[data-testid="stChatMessage"]')
        # The welcome message only renders on an empty thread and disappears on
        # the rerun after sending, so detection is text-based, not count-based.
        pre_send_last = (messages.last.inner_text() or "").strip() if messages.count() else ""

        chat_input.fill(TEST_MESSAGE)
        chat_input.press("Enter")
        log("message sent, waiting for assistant response")

        # Expect our message to appear in the thread, followed by a final
        # assistant message that is non-empty, new, and stable (streaming done).
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        last_text, stable_since = "", None
        while time.monotonic() < deadline:
            texts = [(t or "").strip() for t in messages.all_inner_texts()]
            if texts and any(TEST_MESSAGE in t for t in texts[:-1]):
                text = texts[-1]
                if text and text != TEST_MESSAGE and text != pre_send_last:
                    if text == last_text:
                        if stable_since and time.monotonic() - stable_since >= STREAM_SETTLE_S:
                            log(f"PASS: assistant responded ({len(text)} chars): {text[:120]!r}")
                            browser.close()
                            return
                    else:
                        last_text, stable_since = text, time.monotonic()
            time.sleep(1)

        fail(page, f"no stable assistant response within {RESPONSE_TIMEOUT_S}s")


if __name__ == "__main__":
    main()
