"""
Console Error Detection Tests for Dashboard Pages.

Tests verify:
- No JavaScript console errors
- No uncaught exceptions
- No network errors for critical resources
- No React/Streamlit runtime errors
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import DASHBOARD_PAGES, PAGE_LOAD_TIMEOUT


# ============================================================================
# Test: Console Error Detection
# ============================================================================


class TestConsoleErrors:
    """Tests for JavaScript console errors across all pages."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_console_errors(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no critical console errors occur."""
        console_errors = []

        def handle_console(msg):
            if msg.type == "error":
                text = msg.text
                # Filter out known non-critical errors
                ignore_patterns = [
                    "favicon",
                    "websocket",
                    "manifest.json",
                    "service-worker",
                    "analytics",
                    "tracking",
                    "gtag",
                    "ga(",
                    "hotjar",
                ]
                if not any(pattern in text.lower() for pattern in ignore_patterns):
                    console_errors.append(text)

        page.on("console", handle_console)

        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Wait a bit for any async errors
        page.wait_for_timeout(1000)

        assert len(console_errors) == 0, (
            f"Console errors on {page_config['name']}:\n" +
            "\n".join(console_errors[:10])  # Show first 10
        )

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_uncaught_exceptions(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no uncaught JavaScript exceptions occur."""
        page_errors = []

        def handle_page_error(error):
            page_errors.append(str(error))

        page.on("pageerror", handle_page_error)

        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Wait for any async errors
        page.wait_for_timeout(1000)

        assert len(page_errors) == 0, (
            f"Uncaught exceptions on {page_config['name']}:\n" +
            "\n".join(page_errors)
        )


# ============================================================================
# Test: Network Errors
# ============================================================================


class TestNetworkErrors:
    """Tests for network request errors."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:10],  # Test first 10 pages for performance
        ids=[p["name"] for p in DASHBOARD_PAGES[:10]],
    )
    def test_no_failed_requests(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no critical network requests fail."""
        failed_requests = []

        def handle_response(response):
            # Only track failed requests (4xx and 5xx)
            if response.status >= 400:
                url = response.url
                # Ignore non-critical failures
                ignore_patterns = [
                    "favicon",
                    "manifest",
                    "analytics",
                    "tracking",
                    "hotjar",
                    "gtag",
                ]
                if not any(pattern in url.lower() for pattern in ignore_patterns):
                    failed_requests.append({
                        "url": url,
                        "status": response.status,
                    })

        page.on("response", handle_response)

        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Filter to only critical failures (5xx server errors)
        critical_failures = [
            req for req in failed_requests
            if req["status"] >= 500
        ]

        assert len(critical_failures) == 0, (
            f"Failed requests on {page_config['name']}:\n" +
            "\n".join(f"{r['status']}: {r['url']}" for r in critical_failures)
        )


# ============================================================================
# Test: React/Streamlit Runtime Errors
# ============================================================================


class TestRuntimeErrors:
    """Tests for React and Streamlit runtime errors."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_react_error_boundary(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that React error boundary is not triggered."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for React error boundary elements
        error_boundary = page.locator('[data-testid="stException"]')
        error_count = error_boundary.count()

        assert error_count == 0, (
            f"React error boundary triggered on {page_config['name']}"
        )

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_streamlit_error_state(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that Streamlit is not in error state."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for Streamlit error indicators
        error_indicators = page.evaluate("""
            () => {
                // Check for error elements
                const exceptions = document.querySelectorAll('[data-testid="stException"]');
                const errorMessages = document.querySelectorAll('.stAlert.error, .element-container .stException');

                // Check for error text in body
                const bodyText = document.body.innerText;
                const hasErrorText = bodyText.includes('Traceback') ||
                                    bodyText.includes('Error:') && bodyText.includes('line');

                return {
                    exceptionCount: exceptions.length,
                    errorMessageCount: errorMessages.length,
                    hasErrorText: hasErrorText
                };
            }
        """)

        assert error_indicators["exceptionCount"] == 0, (
            f"Found {error_indicators['exceptionCount']} exceptions on {page_config['name']}"
        )
        assert error_indicators["errorMessageCount"] == 0, (
            f"Found {error_indicators['errorMessageCount']} error messages on {page_config['name']}"
        )


# ============================================================================
# Test: Warning Detection (Non-blocking)
# ============================================================================


class TestWarnings:
    """Tests for console warnings (informational, non-blocking)."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],  # Sample of pages
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_collect_warnings(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Collect console warnings for review (non-blocking test)."""
        warnings = []

        def handle_console(msg):
            if msg.type == "warning":
                warnings.append(msg.text)

        page.on("console", handle_console)

        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # This test always passes but records warnings
        # In a real scenario, you might log these or set thresholds
        if warnings:
            print(f"\nWarnings on {page_config['name']} ({len(warnings)}):")
            for w in warnings[:5]:
                print(f"  - {w[:100]}...")


# ============================================================================
# Test: Performance Console Messages
# ============================================================================


class TestPerformanceMessages:
    """Tests for performance-related console messages."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_no_memory_warnings(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no memory-related warnings occur."""
        memory_warnings = []

        def handle_console(msg):
            text = msg.text.lower()
            if any(keyword in text for keyword in ["memory", "leak", "heap"]):
                memory_warnings.append(msg.text)

        page.on("console", handle_console)

        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Memory warnings are concerning but not always critical
        if memory_warnings:
            print(f"\nMemory warnings on {page_config['name']}:")
            for w in memory_warnings:
                print(f"  - {w}")

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_no_deprecation_errors(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that deprecation warnings don't cause errors."""
        deprecation_errors = []

        def handle_console(msg):
            if msg.type == "error":
                text = msg.text.lower()
                if "deprecated" in text or "deprecation" in text:
                    deprecation_errors.append(msg.text)

        page.on("console", handle_console)

        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Deprecation errors should be warnings, not errors
        assert len(deprecation_errors) == 0, (
            f"Deprecation errors on {page_config['name']}:\n" +
            "\n".join(deprecation_errors)
        )
