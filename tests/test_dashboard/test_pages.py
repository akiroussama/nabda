"""
Comprehensive Playwright E2E tests for all Jira AI Co-pilot Dashboard pages.

Tests verify:
- Each page loads successfully without errors
- No JavaScript/console errors
- No Streamlit exception elements
- Key UI elements are present
- Pages are responsive and interactive
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import DASHBOARD_PAGES, PAGE_LOAD_TIMEOUT


# ============================================================================
# Test: All Pages Load Successfully
# ============================================================================


class TestAllPagesLoad:
    """Test that all dashboard pages load without errors."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_loads_successfully(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """
        Test that a dashboard page loads successfully.

        Verifies:
        - Page navigates without errors
        - Streamlit app container is present
        - No exception elements are displayed
        - No critical console errors
        """
        # Navigate to the page
        url = f"{base_url}{page_config['path']}"
        page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)

        # Wait for Streamlit to fully load
        wait_for_streamlit(page)

        # Verify expected elements are present
        for selector in page_config.get("expected_elements", []):
            expect(page.locator(selector).first).to_be_visible(timeout=PAGE_LOAD_TIMEOUT)

        # Assert no errors occurred
        assert_no_errors(page)


# ============================================================================
# Test: Individual Page Tests
# ============================================================================


class TestHomePage:
    """Tests for the main dashboard home page."""

    def test_home_page_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that the home page loads correctly."""
        page.goto(base_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Verify main app container
        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()

        # Check for header/title elements
        header = page.locator("h1, h2, .stTitle, [data-testid='stHeader']").first
        expect(header).to_be_visible(timeout=10000)

        assert_no_errors(page)

    def test_sidebar_navigation_exists(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Test that sidebar navigation is accessible."""
        page.goto(base_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check sidebar exists (may be collapsed)
        sidebar = page.locator('[data-testid="stSidebar"], .stSidebar')
        expect(sidebar).to_be_attached()


class TestExecutiveCockpit:
    """Tests for the Executive Cockpit page."""

    def test_executive_cockpit_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Executive Cockpit page loads correctly."""
        page.goto(f"{base_url}/Executive_Cockpit", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestOverviewPage:
    """Tests for the Overview page."""

    def test_overview_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Overview page loads correctly."""
        page.goto(f"{base_url}/Overview", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestBoardPage:
    """Tests for the Board page."""

    def test_board_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Board page loads correctly."""
        page.goto(f"{base_url}/Board", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestSprintHealthPage:
    """Tests for the Sprint Health page."""

    def test_sprint_health_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Sprint Health page loads correctly."""
        page.goto(f"{base_url}/Sprint_Health", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestTeamWorkloadPage:
    """Tests for the Team Workload page."""

    def test_team_workload_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Team Workload page loads correctly."""
        page.goto(f"{base_url}/Team_Workload", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestPredictionsPage:
    """Tests for the Predictions page."""

    def test_predictions_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Predictions page loads correctly."""
        page.goto(f"{base_url}/Predictions", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestReportsPage:
    """Tests for the Reports page."""

    def test_reports_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Reports page loads correctly."""
        page.goto(f"{base_url}/Reports", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestStrategicGapPage:
    """Tests for the Strategic Gap page."""

    def test_strategic_gap_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Strategic Gap page loads correctly."""
        page.goto(f"{base_url}/Strategic_Gap", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestBurnoutRiskPage:
    """Tests for the Burnout Risk page."""

    def test_burnout_risk_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Burnout Risk page loads correctly."""
        page.goto(f"{base_url}/Burnout_Risk", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestDeliveryForecastPage:
    """Tests for the Delivery Forecast page."""

    def test_delivery_forecast_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Delivery Forecast page loads correctly."""
        page.goto(f"{base_url}/Delivery_Forecast", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestGoodMorningPage:
    """Tests for the Good Morning page."""

    def test_good_morning_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Good Morning page loads correctly."""
        page.goto(f"{base_url}/Good_Morning", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestScrumMasterHQPage:
    """Tests for the Scrum Master HQ page."""

    def test_scrum_master_hq_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Scrum Master HQ page loads correctly."""
        page.goto(f"{base_url}/Scrum_Master_HQ", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestCEOCommandCenterPage:
    """Tests for the CEO Command Center page."""

    def test_ceo_command_center_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that CEO Command Center page loads correctly."""
        page.goto(f"{base_url}/CEO_Command_Center", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestDailyActionIntelligencePage:
    """Tests for the Daily Action Intelligence page."""

    def test_daily_action_intelligence_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Daily Action Intelligence page loads correctly."""
        page.goto(f"{base_url}/Daily_Action_Intelligence", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestOneOnOneHubPage:
    """Tests for the One on One Hub page."""

    def test_one_on_one_hub_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that One on One Hub page loads correctly."""
        page.goto(f"{base_url}/One_on_One_Hub", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestScopeNegotiatorPage:
    """Tests for the Scope Negotiator page."""

    def test_scope_negotiator_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Scope Negotiator page loads correctly."""
        page.goto(f"{base_url}/Scope_Negotiator", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestBlockerAssassinPage:
    """Tests for the Blocker Assassin page."""

    def test_blocker_assassin_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Blocker Assassin page loads correctly."""
        page.goto(f"{base_url}/Blocker_Assassin", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestRelationshipPulsePage:
    """Tests for the Relationship Pulse page."""

    def test_relationship_pulse_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Relationship Pulse page loads correctly."""
        page.goto(f"{base_url}/Relationship_Pulse", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestSixthSensePage:
    """Tests for the Sixth Sense page."""

    def test_sixth_sense_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Sixth Sense page loads correctly."""
        page.goto(f"{base_url}/Sixth_Sense", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestWhatBreaksIfPage:
    """Tests for the What Breaks If page."""

    def test_what_breaks_if_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that What Breaks If page loads correctly."""
        page.goto(f"{base_url}/What_Breaks_If", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestResourceShockAbsorberPage:
    """Tests for the Resource Shock Absorber page."""

    def test_resource_shock_absorber_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Resource Shock Absorber page loads correctly."""
        page.goto(f"{base_url}/Resource_Shock_Absorber", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestTheOraclePage:
    """Tests for The Oracle page."""

    def test_the_oracle_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that The Oracle page loads correctly."""
        page.goto(f"{base_url}/The_Oracle", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestAutonomousAgentsPage:
    """Tests for the Autonomous Agents page."""

    def test_autonomous_agents_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Autonomous Agents page loads correctly."""
        page.goto(f"{base_url}/Autonomous_Agents", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


class TestProjectWeatherPage:
    """Tests for the Project Weather page."""

    def test_project_weather_loads(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that Project Weather page loads correctly."""
        page.goto(f"{base_url}/Project_Weather", timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


# ============================================================================
# Test: Error Detection and UI Validation
# ============================================================================


class TestErrorDetection:
    """Tests for error detection across all pages."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_streamlit_exceptions(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no Streamlit exception elements are displayed."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for Streamlit exception containers
        exception_count = page.locator('[data-testid="stException"]').count()
        assert exception_count == 0, f"Found {exception_count} exception(s) on {page_config['name']}"

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_error_alerts(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no error alert elements are displayed."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for error alert containers
        error_alerts = page.locator('.stAlert[data-baseweb="notification"]').filter(
            has_text="Error"
        ).count()

        assert error_alerts == 0, f"Found {error_alerts} error alert(s) on {page_config['name']}"


# ============================================================================
# Test: Navigation
# ============================================================================


class TestNavigation:
    """Tests for page navigation functionality."""

    def test_navigate_between_pages(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test navigating between multiple pages sequentially."""
        pages_to_visit = [
            "/",
            "/Overview",
            "/Board",
            "/Sprint_Health",
            "/Team_Workload",
        ]

        for path in pages_to_visit:
            page.goto(f"{base_url}{path}", timeout=PAGE_LOAD_TIMEOUT)
            wait_for_streamlit(page)

            expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
            assert_no_errors(page)

    def test_sidebar_navigation_works(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Test that sidebar navigation links work correctly."""
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Expand sidebar if collapsed
        sidebar_button = page.locator('[data-testid="collapsedControl"]')
        if sidebar_button.is_visible():
            sidebar_button.click()
            page.wait_for_timeout(500)

        # Check sidebar is visible
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible(timeout=5000)


# ============================================================================
# Test: Responsive Design
# ============================================================================


class TestResponsiveDesign:
    """Tests for responsive design across different viewport sizes."""

    @pytest.mark.parametrize(
        "viewport",
        [
            {"width": 1920, "height": 1080, "name": "Desktop Full HD"},
            {"width": 1366, "height": 768, "name": "Laptop"},
            {"width": 768, "height": 1024, "name": "Tablet Portrait"},
            {"width": 1024, "height": 768, "name": "Tablet Landscape"},
        ],
        ids=lambda v: v.get("name", "Unknown"),
    )
    def test_home_page_at_viewport(
        self,
        page: Page,
        base_url: str,
        viewport: dict,
        wait_for_streamlit,
        assert_no_errors,
    ):
        """Test that home page renders correctly at different viewport sizes."""
        page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
        assert_no_errors(page)


# ============================================================================
# Test: Performance
# ============================================================================


class TestPerformance:
    """Tests for page load performance."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],  # Test first 5 pages for performance
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_page_loads_within_timeout(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that pages load within acceptable time limits."""
        import time

        url = f"{base_url}{page_config['path']}"

        start_time = time.time()
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)
        load_time = time.time() - start_time

        # Assert page loads in less than 30 seconds
        assert load_time < 30, f"{page_config['name']} took {load_time:.2f}s to load"

        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()
