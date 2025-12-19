#!/usr/bin/env python3
"""
Generate realistic Jira test data with:
- Famous tech personalities as developers
- Frontend/Backend labels
- Story points
- Multiple sprints (current + 2 future)
- ~650 tickets distributed realistically
"""

import random
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.jira_client.auth import JiraAuthenticator

# Famous tech personalities for developers
FAMOUS_DEVELOPERS = [
    {"name": "Elon Musk", "label": "dev-elon-musk", "specialty": "backend"},
    {"name": "Steve Jobs", "label": "dev-steve-jobs", "specialty": "frontend"},
    {"name": "Bill Gates", "label": "dev-bill-gates", "specialty": "backend"},
    {"name": "Mark Zuckerberg", "label": "dev-mark-zuckerberg", "specialty": "frontend"},
    {"name": "Jeff Bezos", "label": "dev-jeff-bezos", "specialty": "backend"},
    {"name": "Sundar Pichai", "label": "dev-sundar-pichai", "specialty": "backend"},
    {"name": "Tim Cook", "label": "dev-tim-cook", "specialty": "frontend"},
    {"name": "Satya Nadella", "label": "dev-satya-nadella", "specialty": "backend"},
    {"name": "Jensen Huang", "label": "dev-jensen-huang", "specialty": "backend"},
    {"name": "Sam Altman", "label": "dev-sam-altman", "specialty": "backend"},
    {"name": "Linus Torvalds", "label": "dev-linus-torvalds", "specialty": "backend"},
    {"name": "Ada Lovelace", "label": "dev-ada-lovelace", "specialty": "backend"},
    {"name": "Grace Hopper", "label": "dev-grace-hopper", "specialty": "backend"},
    {"name": "Margaret Hamilton", "label": "dev-margaret-hamilton", "specialty": "backend"},
    {"name": "Sheryl Sandberg", "label": "dev-sheryl-sandberg", "specialty": "frontend"},
]

# Realistic issue templates by category
ISSUE_TEMPLATES = {
    "frontend": {
        "Story": [
            "Implement responsive {component} component",
            "Create {component} UI with animations",
            "Design and build {component} page layout",
            "Add dark mode support for {component}",
            "Implement accessibility features for {component}",
            "Create mobile-first {component} design",
            "Build interactive {component} with drag-and-drop",
            "Implement {component} with lazy loading",
            "Add skeleton loading for {component}",
            "Create {component} with micro-interactions",
        ],
        "Bug": [
            "Fix {component} layout breaking on mobile",
            "Resolve {component} flickering on scroll",
            "Fix CSS overflow issue in {component}",
            "Correct {component} z-index stacking",
            "Fix {component} not rendering in Safari",
            "Resolve {component} accessibility violations",
            "Fix keyboard navigation in {component}",
            "Correct color contrast in {component}",
        ],
        "Task": [
            "Update {component} to use design tokens",
            "Migrate {component} to TypeScript",
            "Add unit tests for {component}",
            "Document {component} props and usage",
            "Optimize {component} bundle size",
            "Add Storybook stories for {component}",
        ],
    },
    "backend": {
        "Story": [
            "Implement {service} REST API endpoint",
            "Create {service} service with caching",
            "Build {service} data processing pipeline",
            "Implement {service} authentication flow",
            "Create {service} webhook integration",
            "Build {service} batch processing job",
            "Implement {service} real-time notifications",
            "Create {service} GraphQL resolver",
            "Build {service} message queue consumer",
            "Implement {service} rate limiting",
        ],
        "Bug": [
            "Fix {service} memory leak under load",
            "Resolve {service} race condition",
            "Fix {service} timeout handling",
            "Correct {service} error response format",
            "Fix {service} database connection pooling",
            "Resolve {service} cache invalidation issue",
            "Fix {service} authentication bypass",
            "Correct {service} pagination logic",
        ],
        "Task": [
            "Add integration tests for {service}",
            "Optimize {service} database queries",
            "Add monitoring for {service}",
            "Document {service} API endpoints",
            "Migrate {service} to async handlers",
            "Add health check for {service}",
        ],
    },
}

# Components and services for templates
FRONTEND_COMPONENTS = [
    "Dashboard", "UserProfile", "Settings", "Navigation", "Sidebar",
    "Modal", "DataTable", "Chart", "Form", "Card", "Header", "Footer",
    "Dropdown", "Tooltip", "Toast", "Calendar", "FileUploader", "Search",
    "Pagination", "Tabs", "Accordion", "Carousel", "Avatar", "Badge",
    "Button", "Input", "Select", "Checkbox", "Radio", "Switch", "Slider",
    "Progress", "Spinner", "Alert", "Breadcrumb", "Menu", "Tree",
]

BACKEND_SERVICES = [
    "User", "Auth", "Payment", "Notification", "Analytics", "Report",
    "Email", "SMS", "Push", "Webhook", "Scheduler", "Queue", "Cache",
    "Search", "Upload", "Export", "Import", "Audit", "Logging", "Metrics",
    "Config", "Feature", "Permission", "Role", "Session", "Token",
    "Order", "Invoice", "Subscription", "Billing", "Inventory", "Shipping",
]

# Epic themes
EPIC_THEMES = [
    ("User Experience Overhaul", "Modernize the entire user interface", "frontend"),
    ("API v2 Migration", "Migrate all endpoints to new API version", "backend"),
    ("Performance Optimization", "Improve system performance by 50%", "backend"),
    ("Mobile App Launch", "Launch native mobile application", "frontend"),
    ("Security Hardening", "Implement advanced security measures", "backend"),
    ("Analytics Dashboard", "Build comprehensive analytics platform", "frontend"),
    ("Payment Integration", "Integrate multiple payment providers", "backend"),
    ("Accessibility Compliance", "Achieve WCAG 2.1 AA compliance", "frontend"),
    ("Microservices Migration", "Break monolith into microservices", "backend"),
    ("Design System v2", "Create unified design system", "frontend"),
    ("AI/ML Integration", "Add machine learning capabilities", "backend"),
    ("Real-time Features", "Implement WebSocket-based features", "backend"),
]

# Story point distribution (fibonacci-like, weighted towards smaller)
STORY_POINTS = [1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 8, 8, 13]

# Priorities with weights
PRIORITIES = ["Highest", "High", "Medium", "Medium", "Medium", "Low", "Low", "Lowest"]


class RealisticDataGenerator:
    """Generate realistic Jira data."""

    def __init__(self):
        settings = get_settings()
        self.project_key = settings.jira.project_key
        self.board_id = settings.jira.board_id
        self.story_points_field = settings.jira.story_points_field
        self.sprint_field = settings.jira.sprint_field

        # Connect to Jira
        auth = JiraAuthenticator(
            url=settings.jira.url,
            email=settings.jira.email,
            api_token=settings.jira.api_token,
        )
        self.jira = auth.client

        # Get existing users for assignment
        self.assignable_users = self._get_assignable_users()
        print(f"Found {len(self.assignable_users)} assignable users")

        # Track created items
        self.created_sprints = []
        self.created_epics = []
        self.created_issues = []

    def _get_assignable_users(self) -> list:
        """Get users that can be assigned issues."""
        try:
            users = self.jira.search_assignable_users_for_projects(
                username="",
                project=self.project_key,
                maxResults=50
            )
            return [{"accountId": u.accountId, "name": u.displayName} for u in users]
        except Exception as e:
            print(f"Warning: Could not fetch assignable users: {e}")
            return []

    def create_sprints(self) -> list:
        """Create 3 sprints: current active + 2 future."""
        print("\n=== Creating Sprints ===")

        today = datetime.now()
        sprints_to_create = [
            {
                "name": f"Sprint {today.strftime('%Y-W%W')} - Current",
                "startDate": (today - timedelta(days=7)).isoformat(),
                "endDate": (today + timedelta(days=7)).isoformat(),
                "state": "active",
                "goal": "Complete core features and fix critical bugs"
            },
            {
                "name": f"Sprint {(today + timedelta(weeks=2)).strftime('%Y-W%W')} - Next",
                "startDate": (today + timedelta(days=8)).isoformat(),
                "endDate": (today + timedelta(days=21)).isoformat(),
                "state": "future",
                "goal": "Performance improvements and new integrations"
            },
            {
                "name": f"Sprint {(today + timedelta(weeks=4)).strftime('%Y-W%W')} - Future",
                "startDate": (today + timedelta(days=22)).isoformat(),
                "endDate": (today + timedelta(days=35)).isoformat(),
                "state": "future",
                "goal": "Mobile app features and API v2"
            },
        ]

        for sprint_data in sprints_to_create:
            try:
                sprint = self.jira.create_sprint(
                    name=sprint_data["name"],
                    board_id=self.board_id,
                    startDate=sprint_data.get("startDate"),
                    endDate=sprint_data.get("endDate"),
                    goal=sprint_data.get("goal")
                )
                self.created_sprints.append(sprint)
                print(f"  Created sprint: {sprint_data['name']} (ID: {sprint.id})")
            except Exception as e:
                print(f"  Warning: Could not create sprint {sprint_data['name']}: {e}")
                # Try to find existing sprint
                try:
                    existing = self.jira.sprints(self.board_id, state="active,future")
                    for s in existing:
                        if sprint_data["name"] in s.name or len(self.created_sprints) < 3:
                            self.created_sprints.append(s)
                            print(f"  Using existing sprint: {s.name} (ID: {s.id})")
                            break
                except:
                    pass

        return self.created_sprints

    def create_epics(self) -> list:
        """Create epics for organizing work."""
        print("\n=== Creating Epics ===")

        for theme, description, category in EPIC_THEMES:
            try:
                epic = self.jira.create_issue(
                    project=self.project_key,
                    summary=theme,
                    description=description,
                    issuetype={"name": "Epic"},
                    labels=[category, f"epic-{category}"],
                )
                self.created_epics.append({
                    "key": epic.key,
                    "name": theme,
                    "category": category
                })
                print(f"  Created epic: {epic.key} - {theme}")
            except Exception as e:
                print(f"  Warning: Could not create epic '{theme}': {e}")

        return self.created_epics

    def generate_issue_data(self, category: str, issue_type: str) -> dict:
        """Generate a single issue's data."""
        templates = ISSUE_TEMPLATES[category][issue_type]
        template = random.choice(templates)

        if category == "frontend":
            component = random.choice(FRONTEND_COMPONENTS)
            summary = template.format(component=component)
        else:
            service = random.choice(BACKEND_SERVICES)
            summary = template.format(service=service)

        # Select developer (prefer matching specialty)
        matching_devs = [d for d in FAMOUS_DEVELOPERS if d["specialty"] == category]
        dev = random.choice(matching_devs if matching_devs else FAMOUS_DEVELOPERS)

        # Build labels
        labels = [category, dev["label"]]

        # Add complexity labels
        points = random.choice(STORY_POINTS)
        labels.append(f"sp-{points}")

        # Add random tags
        if random.random() > 0.7:
            labels.append("priority")
        if random.random() > 0.8:
            labels.append("tech-debt")
        if random.random() > 0.9:
            labels.append("needs-review")

        # Select epic
        matching_epics = [e for e in self.created_epics if e["category"] == category]
        epic = random.choice(matching_epics) if matching_epics else None

        return {
            "summary": summary,
            "issue_type": issue_type,
            "category": category,
            "labels": labels,
            "story_points": points,
            "priority": random.choice(PRIORITIES),
            "developer": dev,
            "epic_key": epic["key"] if epic else None,
        }

    def create_issues(self, count: int = 650) -> list:
        """Create issues distributed across sprints."""
        print(f"\n=== Creating {count} Issues ===")

        # Distribution: 60% current sprint, 25% next, 15% future/backlog
        sprint_distribution = [0.60, 0.25, 0.15]

        # Issue type distribution: 50% Story, 30% Bug, 20% Task
        type_weights = [("Story", 0.50), ("Bug", 0.30), ("Task", 0.20)]

        # Category distribution: 55% frontend, 45% backend
        category_weights = [("frontend", 0.55), ("backend", 0.45)]

        # Status distribution for current sprint
        current_sprint_statuses = [
            ("Done", 0.35),
            ("In Progress", 0.25),
            ("In Review", 0.15),
            ("To Do", 0.25),
        ]

        # Status for future sprints
        future_sprint_statuses = [("To Do", 1.0)]

        created_count = 0
        batch_size = 50

        for i in range(count):
            # Select category
            category = random.choices(
                [c[0] for c in category_weights],
                weights=[c[1] for c in category_weights]
            )[0]

            # Select issue type
            issue_type = random.choices(
                [t[0] for t in type_weights],
                weights=[t[1] for t in type_weights]
            )[0]

            # Generate issue data
            issue_data = self.generate_issue_data(category, issue_type)

            # Select sprint
            if self.created_sprints:
                sprint_idx = random.choices(
                    range(len(self.created_sprints)),
                    weights=sprint_distribution[:len(self.created_sprints)]
                )[0]
                sprint = self.created_sprints[sprint_idx]
            else:
                sprint = None
                sprint_idx = -1

            # Determine status based on sprint
            if sprint_idx == 0:  # Current sprint
                status = random.choices(
                    [s[0] for s in current_sprint_statuses],
                    weights=[s[1] for s in current_sprint_statuses]
                )[0]
            else:
                status = "To Do"

            try:
                # Build fields
                fields = {
                    "project": {"key": self.project_key},
                    "summary": issue_data["summary"],
                    "issuetype": {"name": issue_data["issue_type"]},
                    "labels": issue_data["labels"],
                    "priority": {"name": issue_data["priority"]},
                    self.story_points_field: issue_data["story_points"],
                }

                # Add assignee if available
                if self.assignable_users:
                    assignee = random.choice(self.assignable_users)
                    fields["assignee"] = {"accountId": assignee["accountId"]}

                # Add description
                fields["description"] = self._generate_description(issue_data)

                # Create issue
                issue = self.jira.create_issue(fields=fields)

                # Add to sprint if available
                if sprint:
                    try:
                        self.jira.add_issues_to_sprint(sprint.id, [issue.key])
                    except Exception as e:
                        print(f"  Warning: Could not add {issue.key} to sprint: {e}")

                # Transition to target status if not "To Do"
                if status != "To Do":
                    self._transition_issue(issue.key, status)

                self.created_issues.append(issue.key)
                created_count += 1

                if created_count % batch_size == 0:
                    print(f"  Progress: {created_count}/{count} issues created")

            except Exception as e:
                print(f"  Error creating issue: {e}")

        print(f"  Completed: {created_count} issues created")
        return self.created_issues

    def _generate_description(self, issue_data: dict) -> str:
        """Generate a realistic description."""
        category = issue_data["category"]
        issue_type = issue_data["issue_type"]

        if issue_type == "Story":
            return f"""## User Story

As a user, I want to {issue_data['summary'].lower()} so that I can improve my workflow.

## Acceptance Criteria
- [ ] Feature is implemented according to design specs
- [ ] Unit tests cover main scenarios
- [ ] Documentation is updated
- [ ] Code review approved

## Technical Notes
- Category: {category.upper()}
- Estimated: {issue_data['story_points']} story points
- Developer: {issue_data['developer']['name']}

## Design
[Link to Figma/design if applicable]
"""
        elif issue_type == "Bug":
            return f"""## Bug Description

{issue_data['summary']}

## Steps to Reproduce
1. Navigate to the affected area
2. Perform the action
3. Observe the unexpected behavior

## Expected Behavior
The feature should work as designed.

## Actual Behavior
[Describe what actually happens]

## Environment
- Browser: Chrome/Firefox/Safari
- OS: Windows/Mac/Linux
- Category: {category.upper()}

## Priority
{issue_data['priority']}
"""
        else:  # Task
            return f"""## Task Description

{issue_data['summary']}

## Checklist
- [ ] Research and planning
- [ ] Implementation
- [ ] Testing
- [ ] Documentation

## Notes
- Category: {category.upper()}
- Estimated: {issue_data['story_points']} story points
"""

    def _transition_issue(self, issue_key: str, target_status: str):
        """Transition issue to target status."""
        try:
            transitions = self.jira.transitions(issue_key)
            for t in transitions:
                if t["name"].lower() == target_status.lower() or \
                   target_status.lower() in t["name"].lower():
                    self.jira.transition_issue(issue_key, t["id"])
                    return
        except Exception as e:
            pass  # Silently fail transitions

    def run(self, issue_count: int = 650):
        """Run the full data generation."""
        print("=" * 60)
        print("Realistic Jira Data Generator")
        print("=" * 60)
        print(f"Project: {self.project_key}")
        print(f"Board: {self.board_id}")
        print(f"Target issues: {issue_count}")
        print(f"Developers: {len(FAMOUS_DEVELOPERS)} famous tech personalities")

        # Create sprints
        self.create_sprints()

        # Create epics
        self.create_epics()

        # Create issues
        self.create_issues(issue_count)

        # Summary
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"Sprints created: {len(self.created_sprints)}")
        print(f"Epics created: {len(self.created_epics)}")
        print(f"Issues created: {len(self.created_issues)}")
        print("\nRun 'jira-copilot sync full' to sync data to local database.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate realistic Jira test data")
    parser.add_argument(
        "--count", "-c", type=int, default=650,
        help="Number of issues to create (default: 650)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be created without actually creating"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - Would create:")
        print(f"  - 3 sprints (current + 2 future)")
        print(f"  - {len(EPIC_THEMES)} epics")
        print(f"  - {args.count} issues")
        print(f"  - {len(FAMOUS_DEVELOPERS)} developer labels")
        print("\nDeveloper labels:")
        for dev in FAMOUS_DEVELOPERS:
            print(f"  - {dev['label']} ({dev['name']}, {dev['specialty']})")
    else:
        generator = RealisticDataGenerator()
        generator.run(issue_count=args.count)
