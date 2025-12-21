"""
Smoke Tests for Dashboard Pages.

Quick tests to verify basic functionality across all pages.
These tests are designed to run fast and catch obvious issues.
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import DASHBOARD_PAGES, PAGE_LOAD_TIMEOUT


# ============================================================================
# Smoke Test: All Pages Load
# ============================================================================


class TestSmoke:
    """Quick smoke tests for all pages."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_smoke(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """
        Smoke test: verify page loads and displays content.

        This is a quick test that verifies:
        1. Page navigates successfully
        2. Main container is visible
        3. No Streamlit exceptions
        4. No critical console errors
        """
        # Navigate
        url = f"{base_url}{page_config['path']}"
        response = page.goto(url, timeout=PAGE_LOAD_TIMEOUT)

        # Verify navigation succeeded
        assert response is not None, f"Navigation to {page_config['name']} returned None"
        assert response.ok, f"Navigation to {page_config['name']} failed: {response.status}"

        # Wait for Streamlit
        wait_for_streamlit(page)

        # Verify main container
        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()

        # Verify no exceptions
        exceptions = page.locator('[data-testid="stException"]')
        assert exceptions.count() == 0, f"Exception found on {page_config['name']}"

        # Verify content exists
        content = page.locator('[data-testid="stAppViewContainer"]').text_content()
        assert content and len(content.strip()) > 10, (
            f"Page {page_config['name']} appears empty"
        )


# ============================================================================
# Critical Path Tests
# ============================================================================


class TestCriticalPaths:
    """Tests for critical user paths."""

    def test_home_to_overview_navigation(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test navigation from home to overview page."""
        # Start at home
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)
        assert_no_errors(page)

        # Navigate to overview
        page.goto(f"{base_url}/Overview", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)
        assert_no_errors(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()

    def test_executive_pages_load(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that executive-level pages load successfully."""
        executive_pages = [
            "/Executive_Cockpit",
            "/CEO_Command_Center",
            "/Scrum_Master_HQ",
        ]

        for path in executive_pages:
            page.goto(f"{base_url}{path}", timeout=PAGE_LOAD_TIMEOUT)
            wait_for_streamlit(page)
            assert_no_errors(page)
            expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()

    def test_analytics_pages_load(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that analytics pages load successfully."""
        analytics_pages = [
            "/Predictions",
            "/Delivery_Forecast",
            "/Strategic_Gap",
            "/Burnout_Risk",
        ]

        for path in analytics_pages:
            page.goto(f"{base_url}{path}", timeout=PAGE_LOAD_TIMEOUT)
            wait_for_streamlit(page)
            assert_no_errors(page)
            expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()


# ============================================================================
# Quick Health Check
# ============================================================================


class TestHealthCheck:
    """Quick health check tests."""

    def test_app_is_running(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Test that the Streamlit app is running and responsive."""
        response = page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)

        assert response is not None
        assert response.ok
        assert response.status == 200

        wait_for_streamlit(page)

    def test_all_pages_return_200(
        self,
        page: Page,
        base_url: str,
    ):
        """Quick test that all page URLs return 200."""
        failed_pages = []

        for page_config in DASHBOARD_PAGES:
            url = f"{base_url}{page_config['path']}"
            try:
                response = page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until="commit")
                if response is None or not response.ok:
                    status = response.status if response else "None"
                    failed_pages.append(f"{page_config['name']}: {status}")
            except Exception as e:
                failed_pages.append(f"{page_config['name']}: {str(e)[:50]}")

        assert len(failed_pages) == 0, (
            f"Pages with errors:\n" + "\n".join(failed_pages)
        )

    def test_streamlit_version_compatible(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Test that Streamlit version is compatible."""
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for Streamlit-specific elements
        has_streamlit = page.evaluate("""
            () => {
                // Check for Streamlit-specific data attributes
                const hasAppView = !!document.querySelector('[data-testid="stAppViewContainer"]');
                const hasRoot = !!document.querySelector('#root');
                return hasAppView && hasRoot;
            }
        """)

        assert has_streamlit, "Streamlit app structure not detected"


# ============================================================================
# Quick Summary Report
# ============================================================================


class TestSummaryReport:
    """Generate a quick summary of page status."""

    def test_generate_status_report(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Generate a summary report of all page statuses."""
        results = []

        for page_config in DASHBOARD_PAGES:
            url = f"{base_url}{page_config['path']}"
            status = "OK"
            error_msg = ""

            try:
                response = page.goto(url, timeout=PAGE_LOAD_TIMEOUT)

                if response is None or not response.ok:
                    status = "FAILED"
                    error_msg = f"HTTP {response.status if response else 'None'}"
                else:
                    wait_for_streamlit(page)

                    # Check for exceptions
                    exceptions = page.locator('[data-testid="stException"]')
                    if exceptions.count() > 0:
                        status = "ERROR"
                        error_msg = "Streamlit exception"

            except Exception as e:
                status = "TIMEOUT"
                error_msg = str(e)[:50]

            results.append({
                "name": page_config["name"],
                "path": page_config["path"],
                "status": status,
                "error": error_msg,
            })

        # Print summary
        print("\n" + "=" * 60)
        print("DASHBOARD PAGE STATUS REPORT")
        print("=" * 60)

        ok_count = len([r for r in results if r["status"] == "OK"])
        failed_count = len(results) - ok_count

        for result in results:
            status_icon = {
                "OK": "[OK]    ",
                "FAILED": "[FAIL]  ",
                "ERROR": "[ERROR] ",
                "TIMEOUT": "[TIMEOUT]",
            }.get(result["status"], "[???]")

            print(f"{status_icon} {result['name']}")
            if result["error"]:
                print(f"         -> {result['error']}")

        print("=" * 60)
        print(f"Total: {len(results)} | OK: {ok_count} | Failed: {failed_count}")
        print("=" * 60)

        # Assert all pages are OK
        assert failed_count == 0, f"{failed_count} pages failed"
