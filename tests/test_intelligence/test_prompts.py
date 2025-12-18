"""Tests for prompt template management."""

import pytest
from pathlib import Path

from src.intelligence.prompts import PromptTemplateManager, BUILTIN_TEMPLATES, create_default_templates


class TestPromptTemplateManager:
    """Tests for PromptTemplateManager."""

    @pytest.fixture
    def temp_templates_dir(self, tmp_path):
        """Create temporary templates directory."""
        templates_dir = tmp_path / "prompts"
        templates_dir.mkdir()
        return templates_dir

    @pytest.fixture
    def manager_with_templates(self, temp_templates_dir):
        """Create manager with default templates."""
        create_default_templates(temp_templates_dir)
        return PromptTemplateManager(templates_dir=temp_templates_dir)

    def test_init_creates_directories(self, temp_templates_dir):
        """Test that init creates required directories."""
        manager = PromptTemplateManager(templates_dir=temp_templates_dir)

        assert (temp_templates_dir / "system").exists()
        assert (temp_templates_dir / "ticket").exists()
        assert (temp_templates_dir / "sprint").exists()
        assert (temp_templates_dir / "developer").exists()

    def test_render_inline_basic(self, temp_templates_dir):
        """Test basic inline template rendering."""
        manager = PromptTemplateManager(templates_dir=temp_templates_dir)

        result = manager.render_inline("Hello {{ name }}!", name="World")

        assert result == "Hello World!"

    def test_render_inline_with_filters(self, temp_templates_dir):
        """Test inline rendering with custom filters."""
        manager = PromptTemplateManager(templates_dir=temp_templates_dir)

        result = manager.render_inline(
            "Duration: {{ hours | format_duration }}",
            hours=48.5
        )

        assert "2.0 days" in result

    def test_truncate_smart_filter(self, temp_templates_dir):
        """Test smart truncation filter."""
        manager = PromptTemplateManager(templates_dir=temp_templates_dir)

        long_text = "This is a sentence. This is another sentence. And more."
        result = manager.render_inline(
            "{{ text | truncate_smart(30) }}",
            text=long_text
        )

        # Should truncate at sentence boundary
        assert len(result) <= 35  # Allow for "..."
        assert "This is a sentence." in result

    def test_risk_emoji_filter(self, temp_templates_dir):
        """Test risk emoji filter."""
        manager = PromptTemplateManager(templates_dir=temp_templates_dir)

        result_low = manager.render_inline("{{ 'low' | risk_emoji }}")
        result_high = manager.render_inline("{{ 'high' | risk_emoji }}")

        assert result_low == "🟢"
        assert result_high == "🔴"

    def test_render_file_template(self, manager_with_templates, temp_templates_dir):
        """Test rendering file-based template."""
        # Create a simple template
        template_path = temp_templates_dir / "system" / "test.j2"
        template_path.write_text("Hello {{ name }}!")

        result = manager_with_templates.render("system/test", name="Test")

        assert result == "Hello Test!"

    def test_template_exists(self, manager_with_templates, temp_templates_dir):
        """Test template existence check."""
        # Create a template
        template_path = temp_templates_dir / "ticket" / "test.j2"
        template_path.write_text("test")

        assert manager_with_templates.template_exists("ticket/test")
        assert not manager_with_templates.template_exists("ticket/nonexistent")

    def test_list_templates(self, manager_with_templates, temp_templates_dir):
        """Test listing templates."""
        # Create some templates
        (temp_templates_dir / "ticket" / "test1.j2").write_text("test1")
        (temp_templates_dir / "ticket" / "test2.j2").write_text("test2")
        (temp_templates_dir / "sprint" / "test3.j2").write_text("test3")

        all_templates = manager_with_templates.list_templates()
        ticket_templates = manager_with_templates.list_templates(category="ticket")

        assert "ticket/test1" in all_templates
        assert "ticket/test2" in all_templates
        assert "sprint/test3" in all_templates

        assert "ticket/test1" in ticket_templates
        assert "ticket/test2" in ticket_templates
        assert "sprint/test3" not in ticket_templates

    def test_get_template_hash(self, manager_with_templates, temp_templates_dir):
        """Test template hash generation."""
        template_path = temp_templates_dir / "ticket" / "hash_test.j2"
        template_path.write_text("{{ value }}")

        hash1 = manager_with_templates.get_template_hash("ticket/hash_test", value="a")
        hash2 = manager_with_templates.get_template_hash("ticket/hash_test", value="b")
        hash3 = manager_with_templates.get_template_hash("ticket/hash_test", value="a")

        # Same inputs should produce same hash
        assert hash1 == hash3
        # Different inputs should produce different hash
        assert hash1 != hash2


class TestBuiltinTemplates:
    """Tests for builtin templates."""

    def test_builtin_templates_exist(self):
        """Test that all expected builtin templates exist."""
        expected = [
            "system/base",
            "system/analyst",
            "ticket/summarize",
            "sprint/explain_risk",
            "sprint/suggest_priorities",
            "developer/workload_summary",
        ]

        for template_name in expected:
            assert template_name in BUILTIN_TEMPLATES

    def test_builtin_templates_are_valid_jinja(self, tmp_path):
        """Test that all builtin templates are valid Jinja2."""
        from jinja2 import Environment

        env = Environment()

        for name, content in BUILTIN_TEMPLATES.items():
            try:
                env.from_string(content)
            except Exception as e:
                pytest.fail(f"Template {name} is not valid Jinja2: {e}")

    def test_create_default_templates(self, tmp_path):
        """Test creating default templates."""
        templates_dir = tmp_path / "prompts"

        create_default_templates(templates_dir)

        # Check that files were created
        assert (templates_dir / "system" / "base.j2").exists()
        assert (templates_dir / "ticket" / "summarize.j2").exists()
        assert (templates_dir / "sprint" / "explain_risk.j2").exists()

    def test_create_default_templates_idempotent(self, tmp_path):
        """Test that creating templates twice doesn't overwrite."""
        templates_dir = tmp_path / "prompts"

        create_default_templates(templates_dir)

        # Modify a template
        test_file = templates_dir / "system" / "base.j2"
        original_content = test_file.read_text()
        test_file.write_text("modified")

        # Create again - should not overwrite
        create_default_templates(templates_dir)

        assert test_file.read_text() == "modified"
