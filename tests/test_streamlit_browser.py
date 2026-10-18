"""Headless browser smoke test for every Streamlit tab and responsive modes.

CI installs Streamlit and Chromium. Environments that intentionally omit the UI
runtime skip with an explicit reason; they must not report the browser run as passed.
"""
from __future__ import annotations
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Install requirements-dev.txt to run the Streamlit browser test")
playwright = pytest.importorskip("playwright.sync_api", reason="Install requirements-dev.txt and Chromium to run the browser test")
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(url: str, process: subprocess.Popen, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    last_error = "not attempted"
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"Streamlit exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # server takes a few seconds to import/load data
            last_error = str(exc)
        time.sleep(.25)
    raise AssertionError(f"Streamlit did not become ready within {timeout}s: {last_error}")


def test_every_streamlit_screen_has_no_console_errors_and_fits_mobile():
    port = _port()
    db_path = ROOT / ".pytest_streamlit_budget.sqlite3"
    env = dict(os.environ)
    env.update({
        "AI_ENABLED": "false",
        "APP_DB_PATH": str(db_path),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
        "STREAMLIT_SERVER_HEADLESS": "true",
    })
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", str(port), "--server.headless", "true", "--server.fileWatcherType", "none"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    console_errors: list[str] = []
    page_errors: list[str] = []
    try:
        url = f"http://127.0.0.1:{port}"
        _wait_ready(url, process)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.get_by_text("Is it still alive?", exact=True).wait_for(timeout=60000)
            tabs = [
                ("Live chase", "Is it still alive?"),
                ("Best replay", "Replay the evidence"),
                ("More analysis", "Additional research panels"),
            ]
            for tab_name, heading in tabs:
                page.get_by_role("tab", name=tab_name).click()
                page.get_by_text(heading, exact=False).wait_for(timeout=15000)
                assert page.locator('[data-testid="stException"]').count() == 0, f"Streamlit exception on {tab_name}"
                if tab_name == "Best replay":
                    slider = page.locator('[data-testid="stSlider"]')
                    thumb = slider.locator('[data-testid="stSliderThumbValue"]')
                    thumb.wait_for(timeout=15000)
                    # Streamlit's React Aria slider sometimes swallows the first
                    # keyboard commit. Retry the End → Enter → blur sequence up to
                    # three times before giving up, with a generous wait for the
                    # full 130-event replay to re-render after each attempt.
                    result_locator = page.get_by_text("FINAL RESULT REVEALED", exact=False)
                    for _attempt in range(3):
                        thumb.click(force=True)
                        page.keyboard.press("End")
                        page.keyboard.press("Enter")
                        page.get_by_text(heading, exact=False).click()
                        try:
                            result_locator.wait_for(timeout=30000)
                            break
                        except playwright.Error:
                            if _attempt == 2:
                                raise

                    # Move slider back to zero (Home) and commit with Enter + blur
                    thumb.click(force=True)
                    page.keyboard.press("Home")
                    page.keyboard.press("Enter")
                    page.get_by_text(heading, exact=False).click()
                    result_locator.wait_for(state="detached", timeout=30000)
                    page.get_by_text("Final scores, winner and result margin are withheld until all source records have been reached.", exact=True).wait_for(timeout=30000)
                    assert result_locator.count() == 0, "Replay result leaked after moving the cursor to zero"
            # Lower-priority research lives behind one parent tab, not in the main nav.
            page.get_by_role("tab", name="More analysis").click()
            secondary = [
                ("Matchups", "Batter vs bowler"),
                ("Pre-match & pitch", "Team form and ground history"),
                ("How accurate is this?", "How accurate is this?"),
            ]
            for tab_name, heading in secondary:
                page.get_by_role("tab", name=tab_name).click()
                page.get_by_text(heading, exact=True).first.wait_for(timeout=15000)
                assert page.locator('[data-testid="stException"]').count() == 0, f"Streamlit exception on secondary tab {tab_name}"
            # Dark mode is a real toggle; reduced-motion preference is used for the run.
            page.get_by_text("Dark editorial mode", exact=True).click()
            page.wait_for_timeout(250)
            page.set_viewport_size({"width": 360, "height": 800})
            page.wait_for_timeout(250)
            widths = page.evaluate("({viewport: window.innerWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth})")
            assert widths["document"] <= widths["viewport"] + 1, f"Horizontal document overflow at 360px: {widths}"
            assert widths["body"] <= widths["viewport"] + 1, f"Horizontal body overflow at 360px: {widths}"
            browser.close()
        assert not page_errors, "Browser page errors: " + " | ".join(page_errors)
        assert not console_errors, "Browser console errors: " + " | ".join(console_errors)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)
        try: db_path.unlink(missing_ok=True)
        except OSError: pass
