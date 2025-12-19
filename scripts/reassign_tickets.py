#!/usr/bin/env python3
"""
Reassign 80% of tickets to 6 famous developers with unequal distribution.
"""

import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.jira_client import JiraAuthenticator, JiraFetcher


# Famous developers with their account IDs and weighted distribution
FAMOUS_DEVELOPERS = [
    {
        "name": "Elon.musk",
        "account_id": "63c7cfbcf6d42a7a4632a742",
        "weight": 25,  # 25% of assigned tickets
    },
    {
        "name": "Steve Jobs",
        "account_id": "557058:43ed1027-d233-4703-a7f1-822db64452c8",
        "weight": 20,  # 20% of assigned tickets
    },
    {
        "name": "Sam Altman",
        "account_id": "70121:7c987c0d-1c90-4895-8bc4-09fd45391a92",
        "weight": 18,  # 18% of assigned tickets
    },
    {
        "name": "mark.zuckerberg",
        "account_id": "712020:67efcd7d-e70f-4df7-bea1-d286bbc8673a",
        "weight": 15,  # 15% of assigned tickets
    },
    {
        "name": "satya.nadella",
        "account_id": "712020:fed1a934-d9aa-4fb0-b916-69b526f9f462",
        "weight": 12,  # 12% of assigned tickets
    },
    {
        "name": "sundar.pichai",
        "account_id": "712020:fcace4a9-ffd5-4a88-ba38-cdb17d1aa1cd",
        "weight": 10,  # 10% of assigned tickets
    },
]


def weighted_random_choice(developers):
    """Select a developer based on their weight."""
    total_weight = sum(d["weight"] for d in developers)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for dev in developers:
        cumulative += dev["weight"]
        if r <= cumulative:
            return dev
    return developers[-1]


def main():
    print("=" * 60)
    print("Reassign Tickets to Famous Developers")
    print("=" * 60)

    # Connect to Jira
    settings = get_settings()
    auth = JiraAuthenticator(
        url=settings.jira.url,
        email=settings.jira.email,
        api_token=settings.jira.api_token
    )
    auth.connect()
    fetcher = JiraFetcher(auth)
    jira = fetcher.jira

    # Fetch all issues using enhanced search (required for Jira Cloud)
    print("\n=== Fetching all issues ===")
    all_issues = []
    next_page_token = None
    batch_size = 100

    while True:
        result = jira.enhanced_search_issues(
            "project = DEV",
            maxResults=batch_size,
            nextPageToken=next_page_token,
            fields="key,summary,assignee"
        )

        if not result:
            break

        all_issues.extend(result)
        print(f"  Fetched {len(all_issues)} issues...")

        next_page_token = getattr(result, 'nextPageToken', None)
        if not next_page_token or getattr(result, 'isLast', True):
            break

    print(f"  Total issues: {len(all_issues)}")

    # Calculate how many to reassign (80%)
    num_to_reassign = int(len(all_issues) * 0.8)
    print(f"\n=== Reassigning {num_to_reassign} issues (80%) ===")

    # Shuffle and select 80%
    random.shuffle(all_issues)
    issues_to_reassign = all_issues[:num_to_reassign]

    # Track assignments
    assignments = {dev["name"]: 0 for dev in FAMOUS_DEVELOPERS}

    # Reassign issues
    success = 0
    errors = 0

    for i, issue in enumerate(issues_to_reassign):
        dev = weighted_random_choice(FAMOUS_DEVELOPERS)
        issue_key = issue.key if hasattr(issue, 'key') else issue.get("key")

        try:
            jira.assign_issue(issue_key, dev["account_id"])
            assignments[dev["name"]] += 1
            success += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{num_to_reassign} issues reassigned")

        except Exception as e:
            errors += 1
            print(f"  Error assigning {issue_key}: {e}")

    # Summary
    print(f"\n=== Reassignment Complete ===")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"\n=== Distribution ===")

    for dev in FAMOUS_DEVELOPERS:
        count = assignments[dev["name"]]
        pct = (count / success * 100) if success > 0 else 0
        print(f"  {dev['name']}: {count} issues ({pct:.1f}%)")

    print(f"\n=== Update Database ===")
    print("Run 'uv run jira-copilot sync full' to update local database")


if __name__ == "__main__":
    main()
