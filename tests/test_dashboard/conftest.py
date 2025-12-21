"""
Pytest fixtures for Playwright dashboard tests.

Provides Streamlit server management and browser automation fixtures.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Generator

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DASHBOARD_APP = PROJECT_ROOT / "src" / "dashboard" / "app.py"
DEFAULT_TIMEOUT = 30000  # 30 seconds
PAGE_LOAD_TIMEOUT = 60000  # 60 seconds for slow pages


def find_free_port() -> int:
    """Find an available port for the Streamlit server."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Wait for the Streamlit server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(1)
                if s.connect_ex(("localhost", port)) == 0:
                    # Additional wait for Streamlit to fully initialize
                    time.sleep(2)
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ============================================================================
# Streamlit Server Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def streamlit_port() -> int:
    """Get a free port for the Streamlit server."""
    return find_free_port()


@pytest.fixture(scope="session")
def streamlit_server(streamlit_port: int) -> Generator[str, None, None]:
    """
    Start and manage Streamlit server for the test session.

    Yields the base URL of the running server.
    """
    # Set environment variables
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(streamlit_port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_THEME_BASE"] = "light"

    # Start Streamlit server
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(DASHBOARD_APP),
        f"--server.port={streamlit_port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.runOnSave=false",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    try:
        # Wait for server to be ready
        if not wait_for_server(streamlit_port, timeout=60):
            stdout, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"Streamlit server failed to start on port {streamlit_port}.\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

        base_url = f"http://localhost:{streamlit_port}"
        yield base_url

    finally:
        # Gracefully terminate the server
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


# ============================================================================
# Playwright Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def playwright_instance() -> Generator[Playwright, None, None]:
    """Create a Playwright instance for the test session."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright) -> Generator[Browser, None, None]:
    """Launch a browser for the test session."""
    browser = playwright_instance.chromium.launch(
        headless=True,
        args=[
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    yield browser
    browser.close()


@pytest.fixture(scope="function")
def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """Create a browser context for each test."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True,
    )
    # Capture console messages
    context.on("console", lambda msg: None)  # Suppress console noise
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(context: BrowserContext, streamlit_server: str) -> Generator[Page, None, None]:
    """Create a page for each test with error tracking."""
    page = context.new_page()
    page.set_default_timeout(DEFAULT_TIMEOUT)

    # Store errors for assertion
    page._console_errors = []
    page._page_errors = []

    # Capture console errors
    def handle_console(msg):
        if msg.type == "error":
            page._console_errors.append(msg.text)

    # Capture page errors (uncaught exceptions)
    def handle_page_error(error):
        page._page_errors.append(str(error))

    page.on("console", handle_console)
    page.on("pageerror", handle_page_error)

    yield page
    page.close()


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture
def base_url(streamlit_server: str) -> str:
    """Get the base URL of the Streamlit server."""
    return streamlit_server


@pytest.fixture
def assert_no_errors():
    """Fixture that provides error assertion helper."""
    def _assert_no_errors(page: Page, allow_warnings: bool = True):
        """
        Assert that no critical errors occurred on the page.

        Args:
            page: The Playwright page object
            allow_warnings: If True, allow console warnings
        """
        # Filter out non-critical errors
        critical_errors = [
            err for err in page._console_errors
            if not any(ignore in err.lower() for ignore in [
                "favicon",
                "websocket",
                "manifest.json",
                "deprecated",
                "warning",
            ])
        ]

        # Check for Streamlit error elements
        error_elements = page.locator('[data-testid="stException"]').count()
        error_alerts = page.locator('.stAlert.error, .element-container .stException').count()

        # Build error message
        errors = []
        if critical_errors:
            errors.append(f"Console errors: {critical_errors}")
        if page._page_errors:
            errors.append(f"Page errors: {page._page_errors}")
        if error_elements > 0:
            errors.append(f"Found {error_elements} Streamlit exception elements")
        if error_alerts > 0:
            errors.append(f"Found {error_alerts} error alerts")

        assert not errors, "\n".join(errors)

    return _assert_no_errors


@pytest.fixture
def wait_for_streamlit():
    """Fixture that provides Streamlit page wait helper."""
    def _wait_for_streamlit(page: Page, timeout: int = PAGE_LOAD_TIMEOUT):
        """
        Wait for Streamlit page to fully load.

        Args:
            page: The Playwright page object
            timeout: Maximum wait time in milliseconds
        """
        # Wait for main app container
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout)

        # Wait for initial loading to complete
        try:
            page.wait_for_selector(
                '[data-testid="stStatusWidget"]',
                state="hidden",
                timeout=timeout
            )
        except Exception:
            pass  # Status widget might not appear

        # Additional wait for dynamic content
        page.wait_for_load_state("networkidle", timeout=timeout)

        # Small buffer for JavaScript execution
        time.sleep(0.5)

    return _wait_for_streamlit


# ============================================================================
# Page Registry for Test Parametrization
# ============================================================================

# All dashboard pages with their expected elements
DASHBOARD_PAGES = [
    {
        "name": "Home",
        "path": "/",
        "expected_title": "Jira AI Co-pilot",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Executive Cockpit",
        "path": "/Executive_Cockpit",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Overview",
        "path": "/Overview",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Board",
        "path": "/Board",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Sprint Health",
        "path": "/Sprint_Health",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Team Workload",
        "path": "/Team_Workload",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Predictions",
        "path": "/Predictions",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Reports",
        "path": "/Reports",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Strategic Gap",
        "path": "/Strategic_Gap",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Burnout Risk",
        "path": "/Burnout_Risk",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Delivery Forecast",
        "path": "/Delivery_Forecast",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Good Morning",
        "path": "/Good_Morning",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Scrum Master HQ",
        "path": "/Scrum_Master_HQ",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "CEO Command Center",
        "path": "/CEO_Command_Center",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Daily Action Intelligence",
        "path": "/Daily_Action_Intelligence",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "One on One Hub",
        "path": "/One_on_One_Hub",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Scope Negotiator",
        "path": "/Scope_Negotiator",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Blocker Assassin",
        "path": "/Blocker_Assassin",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Relationship Pulse",
        "path": "/Relationship_Pulse",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Sixth Sense",
        "path": "/Sixth_Sense",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "What Breaks If",
        "path": "/What_Breaks_If",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Resource Shock Absorber",
        "path": "/Resource_Shock_Absorber",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "The Oracle",
        "path": "/The_Oracle",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Autonomous Agents",
        "path": "/Autonomous_Agents",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
    {
        "name": "Project Weather",
        "path": "/Project_Weather",
        "expected_elements": ['[data-testid="stAppViewContainer"]'],
    },
]


@pytest.fixture(scope="session")
def dashboard_pages() -> list[dict]:
    """Return the list of all dashboard pages for testing."""
    return DASHBOARD_PAGES
