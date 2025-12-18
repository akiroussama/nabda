"""
Jira authentication module.

Provides secure authentication to Jira Cloud using Basic Auth (email + API token).
"""

from dataclasses import dataclass
from typing import Any

from jira import JIRA
from jira.exceptions import JIRAError
from loguru import logger


@dataclass
class JiraUser:
    """Represents an authenticated Jira user."""

    account_id: str
    display_name: str
    email: str | None
    active: bool
    timezone: str | None = None


class JiraAuthenticationError(Exception):
    """Raised when Jira authentication fails."""

    pass


class JiraConnectionError(Exception):
    """Raised when connection to Jira fails."""

    pass


class JiraAuthenticator:
    """
    Handles authentication to Jira Cloud.

    Uses Basic Auth with email and API token.
    Validates connection by calling the /myself endpoint.

    Example:
        >>> auth = JiraAuthenticator(
        ...     url="https://your-instance.atlassian.net",
        ...     email="user@example.com",
        ...     api_token="your-api-token"
        ... )
        >>> jira = auth.connect()
        >>> user = auth.get_current_user()
    """

    def __init__(
        self,
        url: str,
        email: str,
        api_token: str,
        *,
        validate_on_init: bool = True,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the Jira authenticator.

        Args:
            url: Jira instance URL (e.g., https://your-instance.atlassian.net)
            email: User email address
            api_token: Jira API token (generate at https://id.atlassian.com/manage-profile/security/api-tokens)
            validate_on_init: If True, validate connection immediately
            timeout: Request timeout in seconds
        """
        self._url = url.rstrip("/")
        self._email = email
        self._api_token = api_token
        self._timeout = timeout
        self._jira: JIRA | None = None
        self._current_user: JiraUser | None = None

        if validate_on_init:
            self.connect()
            self.validate_connection()

    @property
    def url(self) -> str:
        """Get the Jira instance URL."""
        return self._url

    @property
    def is_connected(self) -> bool:
        """Check if connected to Jira."""
        return self._jira is not None

    @property
    def client(self) -> JIRA:
        """Get the JIRA client instance."""
        if self._jira is None:
            raise JiraConnectionError("Not connected to Jira. Call connect() first.")
        return self._jira

    def connect(self) -> JIRA:
        """
        Establish connection to Jira.

        Returns:
            JIRA client instance

        Raises:
            JiraAuthenticationError: If authentication fails
            JiraConnectionError: If connection fails
        """
        try:
            logger.info(f"Connecting to Jira at {self._url}")

            self._jira = JIRA(
                server=self._url,
                basic_auth=(self._email, self._api_token),
                options={
                    "verify": True,
                    "headers": {
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                },
                timeout=self._timeout,
            )

            logger.info("Successfully connected to Jira")
            return self._jira

        except JIRAError as e:
            if e.status_code == 401:
                logger.error("Authentication failed: Invalid credentials")
                raise JiraAuthenticationError(
                    "Invalid credentials. Check your email and API token."
                ) from e
            elif e.status_code == 403:
                logger.error("Authentication failed: Access forbidden")
                raise JiraAuthenticationError(
                    "Access forbidden. Check your permissions."
                ) from e
            else:
                logger.error(f"Jira error: {e.text}")
                raise JiraConnectionError(f"Jira error: {e.text}") from e

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise JiraConnectionError(f"Failed to connect to Jira: {e}") from e

    def validate_connection(self) -> JiraUser:
        """
        Validate the connection by fetching current user info.

        Returns:
            JiraUser with current user details

        Raises:
            JiraConnectionError: If validation fails
        """
        if self._jira is None:
            raise JiraConnectionError("Not connected. Call connect() first.")

        try:
            logger.debug("Validating connection via /myself endpoint")
            myself = self._jira.myself()

            self._current_user = JiraUser(
                account_id=myself.get("accountId", ""),
                display_name=myself.get("displayName", ""),
                email=myself.get("emailAddress"),
                active=myself.get("active", True),
                timezone=myself.get("timeZone"),
            )

            logger.info(
                f"Authenticated as: {self._current_user.display_name} "
                f"({self._current_user.email})"
            )

            return self._current_user

        except JIRAError as e:
            logger.error(f"Connection validation failed: {e.text}")
            raise JiraConnectionError(f"Connection validation failed: {e.text}") from e

    def get_current_user(self) -> JiraUser:
        """
        Get the currently authenticated user.

        Returns:
            JiraUser with current user details
        """
        if self._current_user is None:
            return self.validate_connection()
        return self._current_user

    def get_server_info(self) -> dict[str, Any]:
        """
        Get Jira server information.

        Returns:
            Dictionary with server info (version, deployment type, etc.)
        """
        if self._jira is None:
            raise JiraConnectionError("Not connected. Call connect() first.")

        try:
            info = self._jira.server_info()
            logger.debug(f"Server info: {info}")
            return info
        except JIRAError as e:
            logger.error(f"Failed to get server info: {e.text}")
            raise JiraConnectionError(f"Failed to get server info: {e.text}") from e

    def test_project_access(self, project_key: str) -> bool:
        """
        Test if the user has access to a specific project.

        Args:
            project_key: The project key to test (e.g., "PROJ")

        Returns:
            True if access is granted, False otherwise
        """
        if self._jira is None:
            raise JiraConnectionError("Not connected. Call connect() first.")

        try:
            project = self._jira.project(project_key)
            logger.info(f"Access confirmed for project: {project.name} ({project_key})")
            return True
        except JIRAError as e:
            if e.status_code == 404:
                logger.warning(f"Project {project_key} not found or no access")
                return False
            logger.error(f"Error checking project access: {e.text}")
            return False

    def disconnect(self) -> None:
        """Close the Jira connection."""
        if self._jira is not None:
            logger.info("Disconnecting from Jira")
            self._jira.close()
            self._jira = None
            self._current_user = None

    def __enter__(self) -> "JiraAuthenticator":
        """Context manager entry."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def create_jira_client_from_settings() -> JiraAuthenticator:
    """
    Create a JiraAuthenticator from application settings.

    Returns:
        Configured JiraAuthenticator instance
    """
    from config.settings import get_settings

    settings = get_settings()

    return JiraAuthenticator(
        url=settings.jira.url,
        email=settings.jira.email,
        api_token=settings.jira.api_token,
        validate_on_init=True,
    )
