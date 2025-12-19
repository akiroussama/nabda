#!/usr/bin/env python3
"""
Jira Test Data Generator

Generates realistic test data for AI Co-pilot MVP validation.
Creates 6 months of historical sprint data with various patterns.
"""

import random
import sys
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

# Add project root to path
sys.path.insert(0, '/mnt/d/workspace/3chiri/lab/pocJira/nabda')

from src.jira_client.auth import create_jira_client_from_settings

# Configuration
PROJECT_KEY = 'DEV'
BOARD_ID = 5  # Scrum board created earlier
START_DATE = datetime(2025, 6, 18)  # 6 months ago
TODAY = datetime(2025, 12, 19)
SPRINT_DURATION_DAYS = 14

# Custom field IDs
STORY_POINTS_FIELD = 'customfield_10016'
SPRINT_FIELD = 'customfield_10020'
EPIC_LINK_FIELD = 'customfield_10014'

# Developer profiles with performance characteristics
DEVELOPERS = {
    'dev-alice': {'skill': 'senior', 'speed': 0.8, 'quality': 0.95, 'wip_limit': 3},
    'dev-bob': {'skill': 'senior', 'speed': 0.7, 'quality': 0.9, 'wip_limit': 7},  # Overloaded
    'dev-carol': {'skill': 'mid', 'speed': 0.6, 'quality': 0.85, 'wip_limit': 4},  # Improving
    'dev-david': {'skill': 'mid', 'speed': 0.4, 'quality': 0.7, 'wip_limit': 4},  # Struggling
    'dev-emma': {'skill': 'specialist', 'focus': 'Frontend', 'speed': 0.75, 'quality': 0.9, 'wip_limit': 3},
    'dev-frank': {'skill': 'specialist', 'focus': 'Backend', 'speed': 0.75, 'quality': 0.9, 'wip_limit': 3},
    'dev-grace': {'skill': 'junior', 'speed': 0.35, 'quality': 0.75, 'wip_limit': 2},  # New, improving
    'dev-henry': {'skill': 'mid', 'speed': 0.6, 'quality': 0.85, 'wip_limit': 2},  # Underutilized
}

# Epic definitions
EPICS = [
    {'name': 'User Authentication', 'sprints': (1, 3), 'tickets': 15, 'status': 'done'},
    {'name': 'Dashboard MVP', 'sprints': (2, 5), 'tickets': 20, 'status': 'done'},
    {'name': 'API v1', 'sprints': (4, 7), 'tickets': 25, 'status': 'done'},
    {'name': 'Reporting Module', 'sprints': (6, 9), 'tickets': 18, 'status': 'done'},
    {'name': 'Performance Optimization', 'sprints': (8, 10), 'tickets': 12, 'status': 'done'},
    {'name': 'Mobile Responsive', 'sprints': (10, 13), 'tickets': 20, 'status': 'in_progress'},
    {'name': 'API v2', 'sprints': (12, 15), 'tickets': 15, 'status': 'in_progress'},
    {'name': 'Analytics Integration', 'sprints': (14, 17), 'tickets': 20, 'status': 'backlog'},
]

# Sprint performance patterns
SPRINT_PATTERNS = {
    1: {'completion': 0.85, 'scope_change': 0},
    2: {'completion': 0.90, 'scope_change': 0},
    3: {'completion': 0.75, 'scope_change': 5},  # Partial
    4: {'completion': 0.88, 'scope_change': 0},
    5: {'completion': 0.55, 'scope_change': 8},  # Failed
    6: {'completion': 0.92, 'scope_change': 0},
    7: {'completion': 0.87, 'scope_change': 0},
    8: {'completion': 0.65, 'scope_change': 6},  # Partial
    9: {'completion': 0.95, 'scope_change': 0},
    10: {'completion': 0.50, 'scope_change': 10},  # Failed
    11: {'completion': 0.88, 'scope_change': 0},
    12: {'completion': 0.70, 'scope_change': 3},  # Partial
    13: {'completion': 0.35, 'scope_change': 5},  # Current sprint - at risk
}

# Realistic ticket templates by type
TICKET_TEMPLATES = {
    'story': [
        "Implement {feature} for {module}",
        "Add {feature} to {page} page",
        "Create {component} component for {module}",
        "Build {feature} functionality",
        "Design and implement {feature}",
        "Develop {module} {feature}",
        "Enable {feature} in {context}",
        "Support {feature} for {user_type}",
    ],
    'bug': [
        "Fix: {component} {issue} on {context}",
        "Bug: {feature} not working in {context}",
        "Resolve {issue} in {component}",
        "Fix {issue} when {action}",
        "Address {component} {issue}",
        "Correct {issue} in {module}",
    ],
    'task': [
        "Update {component} configuration",
        "Configure {feature} settings",
        "Set up {component} for {context}",
        "Migrate {component} to {target}",
        "Document {feature} usage",
        "Review and update {component}",
    ],
    'spike': [
        "Spike: Evaluate {technology} for {purpose}",
        "Research: {technology} integration options",
        "POC: {feature} implementation approach",
        "Investigate {issue} root cause",
        "Explore {technology} alternatives",
    ],
}

# Template variables
TEMPLATE_VARS = {
    'feature': ['authentication', 'search', 'filtering', 'sorting', 'pagination',
                'caching', 'validation', 'notifications', 'export', 'import',
                'bulk actions', 'user preferences', 'dark mode', 'localization'],
    'module': ['user', 'dashboard', 'reports', 'settings', 'admin', 'api', 'auth'],
    'page': ['home', 'profile', 'settings', 'dashboard', 'reports', 'admin'],
    'component': ['button', 'form', 'modal', 'table', 'chart', 'sidebar', 'header',
                  'dropdown', 'datepicker', 'autocomplete', 'file uploader'],
    'issue': ['crash', 'error', 'slow response', 'incorrect display', 'missing data',
              'memory leak', 'timeout', 'race condition', 'null reference'],
    'context': ['mobile Safari', 'Firefox', 'IE11', 'slow network', 'large dataset',
                'production', 'edge cases', 'concurrent users'],
    'action': ['submitting form', 'loading page', 'clicking button', 'refreshing',
               'navigating back', 'switching tabs'],
    'technology': ['Redis', 'Elasticsearch', 'GraphQL', 'WebSocket', 'PWA', 'SSR'],
    'purpose': ['caching', 'search', 'real-time updates', 'offline support'],
    'target': ['new architecture', 'cloud', 'microservices', 'v2'],
    'user_type': ['admin users', 'guests', 'mobile users', 'API consumers'],
}

# Priority distribution
PRIORITIES = [
    ('Highest', 0.05),
    ('High', 0.20),
    ('Medium', 0.50),
    ('Low', 0.20),
    ('Lowest', 0.05),
]

# Story points distribution (Fibonacci)
STORY_POINTS = [
    (1, 0.15),
    (2, 0.25),
    (3, 0.30),
    (5, 0.20),
    (8, 0.08),
    (13, 0.02),
]

# Type distribution
TICKET_TYPES = [
    ('story', 0.50),
    ('bug', 0.25),
    ('task', 0.15),
    ('spike', 0.10),
]


def weighted_choice(choices: list[tuple[Any, float]]) -> Any:
    """Select item based on weighted probability."""
    total = sum(w for _, w in choices)
    r = random.uniform(0, total)
    cumulative = 0
    for item, weight in choices:
        cumulative += weight
        if r <= cumulative:
            return item
    return choices[-1][0]


def generate_ticket_title(ticket_type: str) -> str:
    """Generate a realistic ticket title."""
    templates = TICKET_TEMPLATES[ticket_type]
    template = random.choice(templates)

    result = template
    for var in TEMPLATE_VARS:
        placeholder = '{' + var + '}'
        if placeholder in result:
            result = result.replace(placeholder, random.choice(TEMPLATE_VARS[var]))

    return result


def generate_description(ticket_type: str, title: str) -> str:
    """Generate a realistic ticket description."""
    if ticket_type == 'story':
        return f"""## User Story
As a user, I want to {title.lower()} so that I can improve my workflow.

## Acceptance Criteria
- [ ] Feature is implemented as described
- [ ] Unit tests are written
- [ ] Documentation is updated
- [ ] Code review is completed

## Technical Notes
Consider existing patterns in the codebase.
"""
    elif ticket_type == 'bug':
        return f"""## Bug Description
{title}

## Steps to Reproduce
1. Navigate to the affected area
2. Perform the triggering action
3. Observe the issue

## Expected Behavior
The feature should work correctly without errors.

## Actual Behavior
The described issue occurs.

## Environment
- Browser: Chrome latest
- OS: Windows/Mac
"""
    elif ticket_type == 'task':
        return f"""## Task Description
{title}

## Steps
1. Review current implementation
2. Make necessary changes
3. Test the changes
4. Update documentation if needed

## Definition of Done
- [ ] Changes implemented
- [ ] Tests pass
- [ ] Reviewed
"""
    else:  # spike
        return f"""## Spike Objective
{title}

## Research Questions
1. What are the options available?
2. What are the trade-offs?
3. What is the recommended approach?

## Deliverables
- Research document with findings
- Recommendation for implementation
- Time estimate for implementation

## Time Box
Max 2 days
"""


def calculate_cycle_time(ticket_type: str, story_points: int, developer: str) -> float:
    """Calculate realistic cycle time in days."""
    dev_profile = DEVELOPERS[developer]

    # Base cycle time based on points
    base_times = {1: 1.5, 2: 2, 3: 3, 5: 5, 8: 8, 13: 12}
    base_time = base_times.get(story_points, story_points)

    # Adjust for type
    type_multiplier = {
        'bug': 0.6,  # Bugs usually faster
        'story': 1.0,
        'task': 0.8,
        'spike': 1.5,  # Variable
    }

    # Adjust for developer speed
    speed = dev_profile['speed']

    # Add randomness
    variance = random.uniform(0.7, 1.3)

    cycle_time = base_time * type_multiplier.get(ticket_type, 1.0) / speed * variance

    # Junior devs improve over time (for dev-grace)
    # Carol also improves

    return max(0.5, cycle_time)


class TestDataGenerator:
    """Generates realistic Jira test data."""

    def __init__(self):
        """Initialize the generator."""
        logger.info("Initializing test data generator...")
        self.auth = create_jira_client_from_settings()
        self.jira = self.auth.client

        self.epics = {}  # name -> issue key
        self.sprints = {}  # sprint_number -> sprint object
        self.created_issues = []

        # Get available users for assignment
        users = self.jira.search_assignable_users_for_projects(
            username="", projectKeys=PROJECT_KEY, maxResults=10
        )
        self.user_ids = [u.accountId for u in users]
        logger.info(f"Found {len(self.user_ids)} assignable users")

    def create_epics(self):
        """Create all epics (or load existing ones)."""
        logger.info("Creating/loading epics...")

        # Check for existing epics
        existing = self.jira.search_issues(f'project = {PROJECT_KEY} AND issuetype = Epic')
        existing_map = {e.fields.summary: e.key for e in existing}

        for epic_def in EPICS:
            if epic_def['name'] in existing_map:
                self.epics[epic_def['name']] = existing_map[epic_def['name']]
                logger.info(f"  Found existing epic: {existing_map[epic_def['name']]} - {epic_def['name']}")
            else:
                epic = self.jira.create_issue(
                    project=PROJECT_KEY,
                    summary=epic_def['name'],
                    description=f"Epic for {epic_def['name']} feature development.",
                    issuetype={'name': 'Epic'},
                )
                self.epics[epic_def['name']] = epic.key
                logger.info(f"  Created epic: {epic.key} - {epic_def['name']}")

        return self.epics

    def create_sprints(self):
        """Create all sprints (or load existing ones)."""
        logger.info("Creating/loading sprints...")

        # Check for existing sprints
        try:
            existing_sprints = self.jira.sprints(BOARD_ID)
            existing_map = {s.name: s for s in existing_sprints}
        except Exception:
            existing_map = {}

        for sprint_num in range(1, 14):
            start_date = START_DATE + timedelta(days=(sprint_num - 1) * SPRINT_DURATION_DAYS)
            end_date = start_date + timedelta(days=SPRINT_DURATION_DAYS)

            # Determine state
            if sprint_num < 13:
                state = 'closed'
            else:
                state = 'active'

            sprint_name = f"Sprint {sprint_num}"
            if sprint_name in existing_map:
                sprint = existing_map[sprint_name]
                self.sprints[sprint_num] = {
                    'id': sprint.id,
                    'name': sprint.name,
                    'start': start_date,
                    'end': end_date,
                    'state': state,
                }
                logger.info(f"  Found existing sprint: {sprint.name} (id: {sprint.id})")
            else:
                sprint = self.jira.create_sprint(
                    name=sprint_name,
                    board_id=BOARD_ID,
                    startDate=start_date.isoformat(),
                    endDate=end_date.isoformat(),
                )

                self.sprints[sprint_num] = {
                    'id': sprint.id,
                    'name': sprint.name,
                    'start': start_date,
                    'end': end_date,
                    'state': state,
                }
                logger.info(f"  Created sprint: {sprint.name} (id: {sprint.id})")

        return self.sprints

    def create_ticket(self, epic_name: str, sprint_num: int | None,
                     ticket_type: str, for_backlog: bool = False) -> dict:
        """Create a single ticket with all attributes."""

        # Generate title and description
        title = generate_ticket_title(ticket_type)
        description = generate_description(ticket_type, title)

        # Select attributes
        priority = weighted_choice(PRIORITIES)
        story_points = weighted_choice(STORY_POINTS)

        # Select developer based on type and component patterns
        developer = self._select_developer(ticket_type, epic_name)

        # Select component
        component = self._select_component(epic_name, developer)

        # Determine labels (including story points as label since field not on screen)
        labels = [ticket_type, developer, f'sp-{story_points}']
        if for_backlog:
            labels.append('backlog')

        # Create the issue
        issue_dict = {
            'project': PROJECT_KEY,
            'summary': title,
            'description': description,
            'issuetype': {'name': 'Tâche'},
            'priority': {'name': priority},
            'labels': labels,
            'components': [{'name': component}],
        }

        # Note: Story points field not on screen, using labels instead
        issue = self.jira.create_issue(fields=issue_dict)

        # Add to sprint if specified
        if sprint_num and sprint_num in self.sprints:
            self.jira.add_issues_to_sprint(
                self.sprints[sprint_num]['id'],
                [issue.key]
            )

        # Assign to a real user (rotating between available users)
        user_idx = hash(developer) % len(self.user_ids)
        self.jira.assign_issue(issue.key, self.user_ids[user_idx])

        ticket_data = {
            'key': issue.key,
            'type': ticket_type,
            'developer': developer,
            'story_points': story_points,
            'sprint': sprint_num,
            'epic': epic_name,
            'component': component,
            'priority': priority,
        }

        self.created_issues.append(ticket_data)
        return ticket_data

    def _select_developer(self, ticket_type: str, epic_name: str) -> str:
        """Select appropriate developer based on patterns."""
        devs = list(DEVELOPERS.keys())

        # Specialists get their domain tickets
        if 'Frontend' in epic_name or 'Dashboard' in epic_name or 'Mobile' in epic_name:
            if random.random() < 0.4:
                return 'dev-emma'  # Frontend specialist

        if 'API' in epic_name or 'Backend' in epic_name:
            if random.random() < 0.4:
                return 'dev-frank'  # Backend specialist

        # Seniors get complex work
        if ticket_type == 'spike':
            return random.choice(['dev-alice', 'dev-bob', 'dev-frank'])

        # Junior gets simpler tasks
        if random.random() < 0.1:
            return 'dev-grace'

        # Rest distributed with bias
        weights = {
            'dev-alice': 0.15,
            'dev-bob': 0.20,  # Overloaded
            'dev-carol': 0.15,
            'dev-david': 0.12,
            'dev-emma': 0.10,
            'dev-frank': 0.10,
            'dev-grace': 0.08,
            'dev-henry': 0.05,  # Underutilized
        }

        return weighted_choice([(d, w) for d, w in weights.items()])

    def _select_component(self, epic_name: str, developer: str) -> str:
        """Select component based on epic and developer."""
        components = ['Frontend', 'Backend', 'Database', 'DevOps', 'Documentation']

        # Epic-based selection
        if 'Dashboard' in epic_name or 'Mobile' in epic_name:
            return random.choice(['Frontend', 'Frontend', 'Backend'])
        if 'API' in epic_name:
            return random.choice(['Backend', 'Backend', 'Database'])
        if 'Authentication' in epic_name:
            return random.choice(['Backend', 'Frontend', 'Database'])

        # Developer specialty
        if developer == 'dev-emma':
            return 'Frontend'
        if developer == 'dev-frank':
            return random.choice(['Backend', 'Database'])

        return random.choice(components)

    def transition_ticket(self, issue_key: str, to_status: str,
                         transition_date: datetime = None):
        """Transition a ticket to a new status."""
        transitions = self.jira.transitions(issue_key)

        # Find matching transition
        transition_id = None
        for t in transitions:
            if to_status.lower() in t['name'].lower():
                transition_id = t['id']
                break

        if transition_id:
            self.jira.transition_issue(issue_key, transition_id)
            return True
        return False

    def simulate_ticket_lifecycle(self, ticket: dict, sprint_data: dict,
                                 should_complete: bool = True):
        """Simulate a ticket going through the workflow."""
        issue_key = ticket['key']

        if not should_complete:
            # Leave in To Do or In Progress
            if random.random() < 0.5:
                self.transition_ticket(issue_key, 'En cours')
            return

        # Transition to In Progress
        self.transition_ticket(issue_key, 'En cours')

        # Potentially add blockers (15% chance)
        if random.random() < 0.15:
            self.jira.add_comment(issue_key, "Blocked: Waiting for external dependency")
            # Then unblock
            self.jira.add_comment(issue_key, "Unblocked: Dependency resolved")

        # Complete the ticket
        self.transition_ticket(issue_key, 'Terminé')

        # Add worklog
        cycle_time = calculate_cycle_time(
            ticket['type'],
            ticket['story_points'],
            ticket['developer']
        )

        # Log time (60-80% of cycle time as logged work)
        logged_hours = int(cycle_time * 8 * random.uniform(0.6, 0.8))
        if logged_hours > 0:
            try:
                self.jira.add_worklog(
                    issue_key,
                    timeSpentSeconds=logged_hours * 3600,
                    comment=f"Development work"
                )
            except Exception as e:
                logger.debug(f"Could not add worklog: {e}")

    def generate_sprint_tickets(self, sprint_num: int):
        """Generate tickets for a specific sprint."""
        sprint_data = self.sprints[sprint_num]
        pattern = SPRINT_PATTERNS.get(sprint_num, {'completion': 0.85, 'scope_change': 0})

        logger.info(f"Generating tickets for Sprint {sprint_num} "
                   f"(completion: {pattern['completion']:.0%})")

        # Find relevant epics for this sprint
        relevant_epics = [
            e for e in EPICS
            if e['sprints'][0] <= sprint_num <= e['sprints'][1]
        ]

        if not relevant_epics:
            relevant_epics = EPICS[:3]  # Default to first epics

        # Calculate tickets for sprint
        target_points = random.randint(28, 38)  # Velocity range
        current_points = 0
        sprint_tickets = []

        while current_points < target_points:
            epic = random.choice(relevant_epics)
            ticket_type = weighted_choice(TICKET_TYPES)

            ticket = self.create_ticket(
                epic_name=epic['name'],
                sprint_num=sprint_num,
                ticket_type=ticket_type,
            )

            sprint_tickets.append(ticket)
            current_points += ticket['story_points']

        # Determine which tickets complete based on pattern
        completion_rate = pattern['completion']
        tickets_to_complete = int(len(sprint_tickets) * completion_rate)

        # Simulate lifecycle
        for i, ticket in enumerate(sprint_tickets):
            should_complete = i < tickets_to_complete

            if sprint_num < 13:  # Past sprints
                self.simulate_ticket_lifecycle(ticket, sprint_data, should_complete)
            else:  # Current sprint
                # Sprint 13: 35% complete at day 8
                if i < len(sprint_tickets) * 0.35:
                    self.simulate_ticket_lifecycle(ticket, sprint_data, True)
                elif i < len(sprint_tickets) * 0.50:
                    # In progress
                    self.transition_ticket(ticket['key'], 'En cours')
                    # Some are blocked
                    if random.random() < 0.15:
                        self.jira.add_comment(ticket['key'],
                            "🚫 BLOCKED: Waiting for external API review")
                # Rest stay in To Do

        # Add scope creep tickets
        for _ in range(pattern['scope_change']):
            epic = random.choice(relevant_epics)
            ticket = self.create_ticket(
                epic_name=epic['name'],
                sprint_num=sprint_num,
                ticket_type='bug' if random.random() < 0.6 else 'task',
            )
            # Label as scope creep
            issue = self.jira.issue(ticket['key'])
            labels = list(issue.fields.labels) + ['scope-creep']
            issue.update(fields={'labels': labels})
            sprint_tickets.append(ticket)

        logger.info(f"  Created {len(sprint_tickets)} tickets, "
                   f"{tickets_to_complete} completed")

        return sprint_tickets

    def generate_backlog(self):
        """Generate backlog tickets."""
        logger.info("Generating backlog tickets...")

        backlog_tickets = []

        # Analytics Integration epic (future work)
        for _ in range(20):
            ticket = self.create_ticket(
                epic_name='Analytics Integration',
                sprint_num=None,
                ticket_type=weighted_choice([('story', 0.6), ('task', 0.3), ('spike', 0.1)]),
                for_backlog=True,
            )
            backlog_tickets.append(ticket)

        # Tech debt
        for _ in range(10):
            ticket = self.create_ticket(
                epic_name=random.choice(['API v1', 'Dashboard MVP']),
                sprint_num=None,
                ticket_type='task',
                for_backlog=True,
            )
            # Update labels
            issue = self.jira.issue(ticket['key'])
            labels = list(issue.fields.labels) + ['tech-debt']
            issue.update(fields={'labels': labels})
            backlog_tickets.append(ticket)

        # Reported bugs (non-critical)
        for _ in range(10):
            ticket = self.create_ticket(
                epic_name=random.choice([e['name'] for e in EPICS[:5]]),
                sprint_num=None,
                ticket_type='bug',
                for_backlog=True,
            )
            # Set to low priority
            issue = self.jira.issue(ticket['key'])
            issue.update(fields={'priority': {'name': 'Low'}})
            backlog_tickets.append(ticket)

        # New feature requests
        for _ in range(10):
            ticket = self.create_ticket(
                epic_name=random.choice(['API v2', 'Mobile Responsive']),
                sprint_num=None,
                ticket_type='story',
                for_backlog=True,
            )
            backlog_tickets.append(ticket)

        logger.info(f"  Created {len(backlog_tickets)} backlog tickets")
        return backlog_tickets

    def create_dependencies(self):
        """Create issue links (dependencies)."""
        logger.info("Creating issue dependencies...")

        # Get all issues
        done_issues = [t for t in self.created_issues
                      if self.jira.issue(t['key']).fields.status.name == 'Terminé(e)']

        dependency_count = 0

        # Create ~10 dependencies
        for _ in range(10):
            if len(done_issues) < 2:
                break

            # Select blocker (earlier issue) and blocked (later issue)
            blocker_idx = random.randint(0, len(done_issues) // 2)
            blocked_idx = random.randint(len(done_issues) // 2, len(done_issues) - 1)

            blocker = done_issues[blocker_idx]
            blocked = done_issues[blocked_idx]

            try:
                self.jira.create_issue_link(
                    type='Blocks',
                    inwardIssue=blocked['key'],
                    outwardIssue=blocker['key'],
                    comment={'body': 'Dependency for implementation'}
                )
                dependency_count += 1
                logger.debug(f"  {blocker['key']} blocks {blocked['key']}")
            except Exception as e:
                logger.debug(f"  Could not create link: {e}")

        logger.info(f"  Created {dependency_count} dependencies")

    def close_past_sprints(self):
        """Close all past sprints."""
        logger.info("Closing past sprints...")

        for sprint_num in range(1, 13):
            sprint_id = self.sprints[sprint_num]['id']
            try:
                # Move incomplete issues to backlog before closing
                url = f"{self.jira._options['server']}/rest/agile/1.0/sprint/{sprint_id}"
                self.jira._session.post(
                    url,
                    json={'state': 'closed'}
                )
                logger.info(f"  Closed Sprint {sprint_num}")
            except Exception as e:
                logger.debug(f"  Could not close sprint {sprint_num}: {e}")

    def run(self):
        """Run the full data generation."""
        logger.info("=" * 60)
        logger.info("Starting Jira Test Data Generation")
        logger.info("=" * 60)

        # Phase 1: Create epics
        self.create_epics()

        # Phase 2: Create sprints
        self.create_sprints()

        # Phase 3: Generate tickets for each sprint
        all_sprint_tickets = []
        for sprint_num in range(1, 14):
            tickets = self.generate_sprint_tickets(sprint_num)
            all_sprint_tickets.extend(tickets)

        # Phase 4: Generate backlog
        backlog_tickets = self.generate_backlog()

        # Phase 5: Create dependencies
        self.create_dependencies()

        # Phase 6: Close past sprints
        self.close_past_sprints()

        # Summary
        logger.info("=" * 60)
        logger.info("Generation Complete!")
        logger.info("=" * 60)

        total = len(self.created_issues)
        done = len([t for t in self.created_issues
                   if self.jira.issue(t['key']).fields.status.name == 'Terminé(e)'])
        in_progress = len([t for t in self.created_issues
                         if 'En cours' in self.jira.issue(t['key']).fields.status.name])

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    GENERATION SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Total Tickets Created: {total:>5}                              ║
║  ├─ Done:              {done:>5}                              ║
║  ├─ In Progress:       {in_progress:>5}                              ║
║  └─ To Do/Backlog:     {total - done - in_progress:>5}                              ║
║                                                              ║
║  Sprints: 13 (12 closed + 1 active)                          ║
║  Epics: 8                                                    ║
║  Developers: 8 (simulated via labels)                        ║
╚══════════════════════════════════════════════════════════════╝
""")

        return {
            'total': total,
            'done': done,
            'in_progress': in_progress,
            'backlog': total - done - in_progress,
            'sprints': 13,
            'epics': 8,
        }


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    generator = TestDataGenerator()
    results = generator.run()
