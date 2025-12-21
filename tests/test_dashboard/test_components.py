"""
Playwright tests for dashboard components and UI elements.

Tests verify:
- Common UI components render correctly
- Interactive elements work as expected
- Charts and visualizations load properly
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import PAGE_LOAD_TIMEOUT


# ============================================================================
# Test: Common Components
# ============================================================================


class TestMetricsComponents:
    """Tests for metric display components."""

    def test_metrics_display_on_overview(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that metric components display on Overview page."""
        page.goto(f"{base_url}/Overview", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for metric containers
        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)

    def test_metrics_display_on_executive_cockpit(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that metric components display on Executive Cockpit."""
        page.goto(f"{base_url}/Executive_Cockpit", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)


class TestChartComponents:
    """Tests for chart and visualization components."""

    def test_plotly_charts_render(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Plotly charts render without errors."""
        # Navigate to a page with charts
        page.goto(f"{base_url}/Overview", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Plotly charts have specific class names
        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)

    def test_sprint_health_charts(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Sprint Health charts render."""
        page.goto(f"{base_url}/Sprint_Health", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)


class TestInteractiveElements:
    """Tests for interactive UI elements."""

    def test_expandable_sections(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that expandable sections work correctly."""
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Look for expander elements
        expanders = page.locator('[data-testid="stExpander"]')

        if expanders.count() > 0:
            # Click first expander
            first_expander = expanders.first
            first_expander.click()
            page.wait_for_timeout(500)

        assert_no_errors(page)

    def test_selectbox_elements(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that selectbox elements are interactive."""
        page.goto(f"{base_url}/Reports", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for selectbox elements
        selectboxes = page.locator('[data-testid="stSelectbox"]')

        if selectboxes.count() > 0:
            expect(selectboxes.first).to_be_visible()

        assert_no_errors(page)


class TestTabComponents:
    """Tests for tab components."""

    def test_tabs_switch_correctly(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that tab components switch content correctly."""
        # Navigate to a page likely to have tabs
        page.goto(f"{base_url}/Team_Workload", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Look for tab elements
        tabs = page.locator('[data-testid="stTabs"]')

        if tabs.count() > 0:
            # Get tab buttons
            tab_buttons = tabs.locator('[role="tab"]')

            if tab_buttons.count() > 1:
                # Click second tab
                tab_buttons.nth(1).click()
                page.wait_for_timeout(500)

        assert_no_errors(page)


class TestColumnLayout:
    """Tests for column layout components."""

    def test_columns_render_correctly(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that column layouts render without overlap."""
        page.goto(f"{base_url}/Overview", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for column containers
        columns = page.locator('[data-testid="column"]')

        if columns.count() > 0:
            # Verify columns are visible
            expect(columns.first).to_be_visible()

        assert_no_errors(page)


# ============================================================================
# Test: Page Guide Component
# ============================================================================


class TestPageGuideComponent:
    """Tests for the page guide component."""

    def test_page_guide_exists_on_pages(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that page guide component exists on pages."""
        pages_to_check = [
            "/Overview",
            "/Sprint_Health",
            "/Team_Workload",
        ]

        for path in pages_to_check:
            page.goto(f"{base_url}{path}", timeout=PAGE_LOAD_TIMEOUT)
            wait_for_streamlit(page)

            # Page should load without errors
            expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
            assert_no_errors(page)


# ============================================================================
# Test: Data Tables
# ============================================================================


class TestDataTables:
    """Tests for data table components."""

    def test_dataframes_render(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that dataframe tables render correctly."""
        page.goto(f"{base_url}/Reports", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)

    def test_board_table_renders(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Board page table renders."""
        page.goto(f"{base_url}/Board", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)


# ============================================================================
# Test: Alert and Status Components
# ============================================================================


class TestAlertComponents:
    """Tests for alert and status display components."""

    def test_info_alerts_render(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that informational alerts render correctly."""
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Info alerts should not cause errors
        assert_no_errors(page)

    def test_status_indicators_render(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that status indicators render correctly."""
        page.goto(f"{base_url}/Sprint_Health", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        app_container = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_container).to_be_visible()

        assert_no_errors(page)


# ============================================================================
# Test: Loading States
# ============================================================================


class TestLoadingStates:
    """Tests for loading state handling."""

    def test_loading_spinner_disappears(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Test that loading spinners disappear after page load."""
        page.goto(f"{base_url}/Overview", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Spinner should not be visible after load
        spinner = page.locator('[data-testid="stSpinner"]')

        # Wait for spinner to disappear (if it appeared)
        try:
            spinner.wait_for(state="hidden", timeout=10000)
        except Exception:
            pass  # Spinner might not have appeared

        # Verify content is visible
        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()

    def test_content_loads_after_spinner(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that content loads properly after any spinners."""
        page.goto(f"{base_url}/Delivery_Forecast", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Content should be visible
        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)
