#!/usr/bin/env python3
"""
Dashboard E2E Test Runner

This script runs Playwright tests for the Jira AI Co-pilot Dashboard.
It handles:
- Installing Playwright browsers if needed
- Running tests with appropriate options
- Generating reports

Usage:
    python scripts/run_dashboard_tests.py [--install] [--headed] [--slow] [--report]

Options:
    --install   Install Playwright browsers before running tests
    --headed    Run tests in headed mode (show browser)
    --slow      Run in slow motion for debugging
    --report    Generate HTML report
    --parallel  Run tests in parallel (faster)
    --page NAME Run tests for specific page only
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def install_playwright_browsers() -> int:
    """Install Playwright browser binaries."""
    print("Installing Playwright browsers...")
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        cwd=PROJECT_ROOT,
    )
    return result.returncode


def run_tests(
    headed: bool = False,
    slow_mo: int = 0,
    html_report: bool = False,
    parallel: bool = False,
    page_filter: str | None = None,
    verbose: bool = True,
) -> int:
    """
    Run Playwright dashboard tests.

    Args:
        headed: Run in headed mode (visible browser)
        slow_mo: Slow motion delay in ms
        html_report: Generate HTML report
        parallel: Run tests in parallel
        page_filter: Filter tests by page name
        verbose: Verbose output

    Returns:
        Exit code (0 for success)
    """
    cmd = [sys.executable, "-m", "pytest"]

    # Test directory
    cmd.append("tests/test_dashboard/")

    # Verbose output
    if verbose:
        cmd.append("-v")

    # Short traceback
    cmd.append("--tb=short")

    # Filter by page name
    if page_filter:
        cmd.extend(["-k", page_filter])

    # Parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])

    # HTML report
    if html_report:
        cmd.extend(["--html=reports/dashboard_tests.html", "--self-contained-html"])

    # Environment variables for Playwright
    env = {
        "PWDEBUG": "0",
    }

    if headed:
        env["HEADED"] = "1"

    if slow_mo > 0:
        env["SLOW_MO"] = str(slow_mo)

    print(f"Running: {' '.join(cmd)}")
    print(f"Environment: {env}")

    import os
    full_env = os.environ.copy()
    full_env.update(env)

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=full_env,
    )

    return result.returncode


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Playwright E2E tests for the dashboard"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install Playwright browsers before running tests",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run tests in headed mode (show browser)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run in slow motion for debugging",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel",
    )
    parser.add_argument(
        "--page",
        type=str,
        default=None,
        help="Run tests for specific page only (e.g., 'Overview')",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet output",
    )

    args = parser.parse_args()

    # Install browsers if requested
    if args.install:
        exit_code = install_playwright_browsers()
        if exit_code != 0:
            print("Failed to install Playwright browsers")
            return exit_code
        print("Playwright browsers installed successfully")

    # Create reports directory if generating report
    if args.report:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)

    # Run tests
    exit_code = run_tests(
        headed=args.headed,
        slow_mo=500 if args.slow else 0,
        html_report=args.report,
        parallel=args.parallel,
        page_filter=args.page,
        verbose=not args.quiet,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
