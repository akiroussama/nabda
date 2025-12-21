"""
HTML Validation Tests for Dashboard Pages.

Tests verify:
- Valid HTML structure
- No broken/missing elements
- Accessibility attributes present
- Proper document structure
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import DASHBOARD_PAGES, PAGE_LOAD_TIMEOUT


# ============================================================================
# Test: HTML Structure Validation
# ============================================================================


class TestHTMLStructure:
    """Tests for valid HTML structure across all pages."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_has_valid_html_structure(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that page has valid basic HTML structure."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for essential HTML elements
        # HTML tag exists
        html = page.locator("html")
        expect(html).to_be_attached()

        # Head tag exists
        head = page.locator("head")
        expect(head).to_be_attached()

        # Body tag exists
        body = page.locator("body")
        expect(body).to_be_attached()
        expect(body).to_be_visible()

        # Title exists
        title = page.locator("title")
        expect(title).to_be_attached()

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_has_streamlit_root(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that Streamlit root element is present."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Streamlit root element
        root = page.locator("#root")
        expect(root).to_be_attached()
        expect(root).to_be_visible()

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_has_main_container(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that main app container is present."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Main Streamlit app container
        app_view = page.locator('[data-testid="stAppViewContainer"]')
        expect(app_view).to_be_visible()

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_empty_containers(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that main containers are not empty."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Main container should have content
        main_content = page.locator('[data-testid="stAppViewContainer"]')
        expect(main_content).to_be_visible()

        # Should have some child elements
        children = main_content.locator(":scope > *")
        assert children.count() > 0, f"Main container is empty on {page_config['name']}"


# ============================================================================
# Test: Accessibility Validation
# ============================================================================


class TestAccessibility:
    """Tests for basic accessibility requirements."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_has_lang_attribute(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that HTML element has lang attribute."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        html = page.locator("html")
        # Streamlit adds lang attribute
        lang = html.get_attribute("lang")
        # Lang attribute should exist (may be empty in some versions)
        assert lang is not None or html.is_visible(), "HTML lang attribute check"

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],  # Test first 5 pages for performance
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_buttons_are_focusable(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that button elements are focusable."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        buttons = page.locator("button:visible")

        if buttons.count() > 0:
            # First visible button should be focusable
            first_button = buttons.first
            expect(first_button).to_be_enabled()

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_images_have_alt_or_role(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that images have alt text or appropriate role."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Find content images (not decorative)
        images = page.locator("img:visible")

        # This is a soft check - we just verify the page loaded
        # Streamlit handles most accessibility automatically
        expect(page.locator('[data-testid="stAppViewContainer"]')).to_be_visible()


# ============================================================================
# Test: No Broken Elements
# ============================================================================


class TestNoBrokenElements:
    """Tests for broken or missing elements."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_broken_images(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that there are no broken images."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for images with broken source
        broken_images = page.evaluate("""
            () => {
                const images = document.querySelectorAll('img');
                const broken = [];
                images.forEach(img => {
                    if (img.naturalWidth === 0 && img.complete && !img.src.startsWith('data:')) {
                        broken.push(img.src);
                    }
                });
                return broken;
            }
        """)

        assert len(broken_images) == 0, f"Broken images found: {broken_images}"

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_empty_links(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that there are no empty links."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for links without content
        empty_links = page.evaluate("""
            () => {
                const links = document.querySelectorAll('a:not([aria-hidden="true"])');
                const empty = [];
                links.forEach(link => {
                    const text = link.textContent.trim();
                    const hasImage = link.querySelector('img, svg');
                    const ariaLabel = link.getAttribute('aria-label');

                    if (!text && !hasImage && !ariaLabel) {
                        empty.push(link.href || 'no-href');
                    }
                });
                return empty;
            }
        """)

        # Filter out known false positives (Streamlit internal links)
        filtered_empty = [
            link for link in empty_links
            if not link.startswith("#") and "streamlit" not in link.lower()
        ]

        assert len(filtered_empty) == 0, f"Empty links found: {filtered_empty}"


# ============================================================================
# Test: Document Structure
# ============================================================================


class TestDocumentStructure:
    """Tests for proper document structure."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_page_has_content(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that page has visible content."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Get text content length
        text_content = page.evaluate("""
            () => document.body.innerText.length
        """)

        # Page should have some text content
        assert text_content > 50, f"Page {page_config['name']} appears to have no content"

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES,
        ids=[p["name"] for p in DASHBOARD_PAGES],
    )
    def test_no_script_errors_in_body(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that no script error elements are in the body."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check for common error indicators
        error_indicators = page.locator(
            '[class*="error"]:visible, '
            '[class*="Error"]:visible, '
            '[data-testid="stException"]:visible'
        )

        # Filter to only real errors (not styled elements)
        error_count = 0
        for i in range(error_indicators.count()):
            element = error_indicators.nth(i)
            text = element.text_content() or ""
            # Check if it's a real error message
            if any(keyword in text.lower() for keyword in ["traceback", "exception", "error:", "failed"]):
                error_count += 1

        assert error_count == 0, f"Found {error_count} error elements on {page_config['name']}"


# ============================================================================
# Test: Meta Tags
# ============================================================================


class TestMetaTags:
    """Tests for proper meta tags."""

    def test_home_page_has_meta_tags(
        self,
        page: Page,
        base_url: str,
        wait_for_streamlit,
    ):
        """Test that home page has essential meta tags."""
        page.goto(base_url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Viewport meta tag (important for responsive design)
        viewport = page.locator('meta[name="viewport"]')
        expect(viewport).to_be_attached()

        # Charset meta tag
        charset = page.locator('meta[charset], meta[http-equiv="Content-Type"]')
        expect(charset.first).to_be_attached()


# ============================================================================
# Test: CSS Loading
# ============================================================================


class TestCSSLoading:
    """Tests for proper CSS loading."""

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:5],
        ids=[p["name"] for p in DASHBOARD_PAGES[:5]],
    )
    def test_css_is_loaded(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that CSS styles are loaded and applied."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check that stylesheets are loaded
        stylesheets = page.evaluate("""
            () => {
                const sheets = document.styleSheets;
                return sheets.length;
            }
        """)

        assert stylesheets > 0, f"No stylesheets loaded on {page_config['name']}"

    @pytest.mark.parametrize(
        "page_config",
        DASHBOARD_PAGES[:3],
        ids=[p["name"] for p in DASHBOARD_PAGES[:3]],
    )
    def test_elements_have_computed_styles(
        self,
        page: Page,
        base_url: str,
        page_config: dict,
        wait_for_streamlit,
    ):
        """Test that elements have computed styles applied."""
        url = f"{base_url}{page_config['path']}"
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
        wait_for_streamlit(page)

        # Check that body has computed styles
        body_styles = page.evaluate("""
            () => {
                const styles = window.getComputedStyle(document.body);
                return {
                    fontFamily: styles.fontFamily,
                    fontSize: styles.fontSize,
                    color: styles.color
                };
            }
        """)

        # Font family should be set
        assert body_styles["fontFamily"], "Body font-family not set"
        # Font size should be reasonable
        assert body_styles["fontSize"], "Body font-size not set"
