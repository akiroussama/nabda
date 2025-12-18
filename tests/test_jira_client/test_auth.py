"""Tests for Jira authentication module."""

from unittest.mock import MagicMock, patch

import pytest
from jira.exceptions import JIRAError

from src.jira_client.auth import (
    JiraAuthenticationError,
    JiraAuthenticator,
    JiraConnectionError,
    JiraUser,
)


class TestJiraAuthenticator:
    """Tests for JiraAuthenticator class."""

    @pytest.fixture
    def mock_jira(self):
        """Create a mock JIRA client."""
        with patch("src.jira_client.auth.JIRA") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.myself.return_value = {
                "accountId": "test-user-id",
                "displayName": "Test User",
                "emailAddress": "test@example.com",
                "active": True,
                "timeZone": "UTC",
            }
            yield mock, mock_instance

    def test_init_with_validation(self, mock_jira):
        """Test initialization with connection validation."""
        mock_class, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=True,
        )

        assert auth.is_connected
        assert auth.url == "https://test.atlassian.net"
        mock_class.assert_called_once()
        mock_instance.myself.assert_called_once()

    def test_init_without_validation(self, mock_jira):
        """Test initialization without connection validation."""
        mock_class, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net/",  # With trailing slash
            email="test@example.com",
            api_token="test-token",
            validate_on_init=False,
        )

        assert not auth.is_connected
        assert auth.url == "https://test.atlassian.net"  # Trailing slash removed
        mock_class.assert_not_called()

    def test_connect_success(self, mock_jira):
        """Test successful connection."""
        mock_class, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=False,
        )

        jira = auth.connect()

        assert jira is mock_instance
        assert auth.is_connected
        mock_class.assert_called_once()

    def test_connect_auth_failure(self, mock_jira):
        """Test connection with authentication failure."""
        mock_class, _ = mock_jira
        error = JIRAError(status_code=401, text="Unauthorized")
        mock_class.side_effect = error

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="wrong-token",
            validate_on_init=False,
        )

        with pytest.raises(JiraAuthenticationError) as exc_info:
            auth.connect()

        assert "Invalid credentials" in str(exc_info.value)

    def test_connect_forbidden(self, mock_jira):
        """Test connection with forbidden access."""
        mock_class, _ = mock_jira
        error = JIRAError(status_code=403, text="Forbidden")
        mock_class.side_effect = error

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=False,
        )

        with pytest.raises(JiraAuthenticationError) as exc_info:
            auth.connect()

        assert "Access forbidden" in str(exc_info.value)

    def test_validate_connection(self, mock_jira):
        """Test connection validation."""
        mock_class, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=False,
        )
        auth.connect()

        user = auth.validate_connection()

        assert isinstance(user, JiraUser)
        assert user.account_id == "test-user-id"
        assert user.display_name == "Test User"
        assert user.email == "test@example.com"
        assert user.active is True

    def test_validate_connection_not_connected(self):
        """Test validation when not connected."""
        with patch("src.jira_client.auth.JIRA"):
            auth = JiraAuthenticator(
                url="https://test.atlassian.net",
                email="test@example.com",
                api_token="test-token",
                validate_on_init=False,
            )

            with pytest.raises(JiraConnectionError) as exc_info:
                auth.validate_connection()

            assert "Not connected" in str(exc_info.value)

    def test_get_current_user_cached(self, mock_jira):
        """Test that current user is cached."""
        _, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=True,
        )

        # First call - already made during init
        user1 = auth.get_current_user()
        user2 = auth.get_current_user()

        assert user1 is user2
        # myself() called once during init validation
        assert mock_instance.myself.call_count == 1

    def test_test_project_access_success(self, mock_jira):
        """Test successful project access check."""
        _, mock_instance = mock_jira
        mock_instance.project.return_value = MagicMock(name="Test Project")

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=True,
        )

        result = auth.test_project_access("PROJ")

        assert result is True
        mock_instance.project.assert_called_with("PROJ")

    def test_test_project_access_not_found(self, mock_jira):
        """Test project access check when project not found."""
        _, mock_instance = mock_jira
        mock_instance.project.side_effect = JIRAError(status_code=404, text="Not found")

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=True,
        )

        result = auth.test_project_access("UNKNOWN")

        assert result is False

    def test_context_manager(self, mock_jira):
        """Test context manager usage."""
        _, mock_instance = mock_jira

        with JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=False,
        ) as auth:
            assert auth.is_connected

        mock_instance.close.assert_called_once()

    def test_disconnect(self, mock_jira):
        """Test disconnection."""
        _, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=True,
        )

        assert auth.is_connected
        auth.disconnect()

        assert not auth.is_connected
        mock_instance.close.assert_called_once()

    def test_client_property_when_connected(self, mock_jira):
        """Test client property returns JIRA instance when connected."""
        _, mock_instance = mock_jira

        auth = JiraAuthenticator(
            url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            validate_on_init=True,
        )

        assert auth.client is mock_instance

    def test_client_property_when_not_connected(self):
        """Test client property raises when not connected."""
        with patch("src.jira_client.auth.JIRA"):
            auth = JiraAuthenticator(
                url="https://test.atlassian.net",
                email="test@example.com",
                api_token="test-token",
                validate_on_init=False,
            )

            with pytest.raises(JiraConnectionError):
                _ = auth.client


class TestJiraUser:
    """Tests for JiraUser dataclass."""

    def test_jira_user_creation(self):
        """Test JiraUser creation with all fields."""
        user = JiraUser(
            account_id="123",
            display_name="Test User",
            email="test@example.com",
            active=True,
            timezone="UTC",
        )

        assert user.account_id == "123"
        assert user.display_name == "Test User"
        assert user.email == "test@example.com"
        assert user.active is True
        assert user.timezone == "UTC"

    def test_jira_user_optional_fields(self):
        """Test JiraUser with optional fields."""
        user = JiraUser(
            account_id="123",
            display_name="Test User",
            email=None,
            active=False,
        )

        assert user.email is None
        assert user.timezone is None
        assert user.active is False
