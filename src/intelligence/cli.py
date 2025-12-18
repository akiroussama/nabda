"""
CLI interface for intelligence module.

Provides command-line access to LLM-powered Jira insights.
"""

import json
import sys
from typing import Any

from loguru import logger


def get_ticket_data(conn, issue_key: str) -> dict[str, Any] | None:
    """Fetch ticket data for intelligence processing."""
    query = """
    SELECT
        i.key,
        i.summary,
        i.description,
        i.issue_type,
        i.status,
        i.priority,
        i.assignee_id,
        u.pseudonym as assignee_name,
        i.created,
        i.updated,
        i.resolved,
        i.story_points,
        i.sprint_id,
        i.sprint_name
    FROM issues i
    LEFT JOIN users u ON i.assignee_id = u.account_id
    WHERE i.key = ?
    """

    result = conn.execute(query, [issue_key]).fetchone()
    if not result:
        return None

    columns = [
        "key", "summary", "description", "issue_type", "status", "priority",
        "assignee_id", "assignee_name", "created", "updated", "resolved",
        "story_points", "sprint_id", "sprint_name"
    ]

    ticket = dict(zip(columns, result))

    # Get recent changelog
    changelog_query = """
    SELECT field, from_value, to_value, changed_at
    FROM issue_changelog
    WHERE issue_key = ?
    ORDER BY changed_at DESC
    LIMIT 10
    """

    changelog_results = conn.execute(changelog_query, [issue_key]).fetchall()
    ticket["changelog"] = [
        {
            "field": row[0],
            "from_value": row[1],
            "to_value": row[2],
            "changed_at": str(row[3]),
        }
        for row in changelog_results
    ]

    return ticket


def get_sprint_tickets(conn, sprint_id: int) -> list[dict[str, Any]]:
    """Fetch tickets in a sprint."""
    query = """
    SELECT
        key,
        summary,
        issue_type,
        status,
        priority,
        story_points,
        CASE WHEN status IN ('Blocked', 'On Hold') THEN true ELSE false END as is_blocked
    FROM issues
    WHERE sprint_id = ?
    ORDER BY
        CASE priority
            WHEN 'Highest' THEN 1
            WHEN 'High' THEN 2
            WHEN 'Medium' THEN 3
            WHEN 'Low' THEN 4
            ELSE 5
        END,
        key
    """

    results = conn.execute(query, [sprint_id]).fetchall()
    columns = ["key", "summary", "issue_type", "status", "priority", "story_points", "is_blocked"]

    return [dict(zip(columns, row)) for row in results]


def main():
    """CLI entry point for intelligence commands."""
    import argparse

    parser = argparse.ArgumentParser(description="Jira AI Co-pilot Intelligence")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Summarize ticket
    summarize_parser = subparsers.add_parser("summarize", help="Summarize a ticket")
    summarize_parser.add_argument("issue_key", help="Jira issue key")
    summarize_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Explain sprint risk
    risk_parser = subparsers.add_parser("risk", help="Explain sprint risk")
    risk_parser.add_argument("sprint_id", type=int, nargs="?", help="Sprint ID (or active)")
    risk_parser.add_argument("--board", "-b", type=int, help="Board ID for active sprint")
    risk_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Suggest priorities
    priority_parser = subparsers.add_parser("priorities", help="Suggest sprint priorities")
    priority_parser.add_argument("sprint_id", type=int, nargs="?", help="Sprint ID")
    priority_parser.add_argument("--board", "-b", type=int, help="Board ID")
    priority_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Developer workload
    workload_parser = subparsers.add_parser("workload", help="Assess developer workload")
    workload_parser.add_argument("developer_id", nargs="?", help="Developer ID")
    workload_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Standup summary
    standup_parser = subparsers.add_parser("standup", help="Generate standup summary")
    standup_parser.add_argument("sprint_id", type=int, nargs="?", help="Sprint ID")
    standup_parser.add_argument("--board", "-b", type=int, help="Board ID")

    args = parser.parse_args()

    if args.command == "summarize":
        return cmd_summarize(args)
    elif args.command == "risk":
        return cmd_risk(args)
    elif args.command == "priorities":
        return cmd_priorities(args)
    elif args.command == "workload":
        return cmd_workload(args)
    elif args.command == "standup":
        return cmd_standup(args)
    else:
        parser.print_help()
        return 1


def cmd_summarize(args) -> int:
    """Handle summarize command."""
    from src.data.schema import get_connection
    from src.intelligence.orchestrator import create_intelligence_from_settings

    conn = get_connection()

    # Get ticket data
    ticket = get_ticket_data(conn, args.issue_key)
    if not ticket:
        print(f"Ticket {args.issue_key} not found", file=sys.stderr)
        return 1

    # Create intelligence
    try:
        intel = create_intelligence_from_settings()
    except Exception as e:
        print(f"Failed to initialize intelligence: {e}", file=sys.stderr)
        return 1

    # Generate summary
    summary = intel.summarize_ticket(ticket)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        status_emoji = {
            "on_track": "🟢",
            "at_risk": "🟡",
            "blocked": "🔴",
        }.get(summary.status_assessment, "⚪")

        print(f"\n📋 {summary.issue_key} Summary")
        print(f"   Status: {status_emoji} {summary.status_assessment.upper()}")
        print(f"\n   {summary.summary}")
        print(f"\n   Next: {summary.next_action}")
        if summary.key_blocker:
            print(f"   ⚠️  Blocker: {summary.key_blocker}")

    return 0


def cmd_risk(args) -> int:
    """Handle risk explanation command."""
    from src.data.schema import get_connection
    from src.features.sprint_features import SprintFeatureExtractor
    from src.models.sprint_risk import SprintRiskScorer
    from src.intelligence.orchestrator import create_intelligence_from_settings

    conn = get_connection()
    extractor = SprintFeatureExtractor(conn)

    # Get sprint features
    if args.sprint_id:
        sprint_features = extractor.extract_features(args.sprint_id)
    else:
        sprint_features = extractor.get_active_sprint_features(args.board)

    if not sprint_features:
        print("Sprint not found", file=sys.stderr)
        return 1

    # Calculate risk score
    scorer = SprintRiskScorer(mode="rule_based")
    risk_score = scorer.score(sprint_features)

    # Create intelligence
    try:
        intel = create_intelligence_from_settings()
    except Exception as e:
        print(f"Failed to initialize intelligence: {e}", file=sys.stderr)
        return 1

    # Generate explanation
    explanation = intel.explain_sprint_risk(sprint_features, risk_score)

    if args.json:
        print(json.dumps(explanation.to_dict(), indent=2))
    else:
        level_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
            explanation.risk_level, "⚪"
        )

        print(f"\n🎯 Sprint Risk Analysis: {explanation.sprint_name}")
        print(f"   Risk Score: {explanation.risk_score}/100 {level_emoji} {explanation.risk_level.upper()}")
        print(f"\n   {explanation.risk_summary}")

        if explanation.main_concerns:
            print(f"\n   Main Concerns:")
            for concern in explanation.main_concerns:
                print(f"   • {concern}")

        if explanation.recommended_actions:
            print(f"\n   Recommended Actions:")
            for action in explanation.recommended_actions:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    action.get("priority", ""), "⚪"
                )
                print(f"   {priority_emoji} {action.get('action', '')}")

        print(f"\n   Outlook: {explanation.outlook.upper()}")

    return 0


def cmd_priorities(args) -> int:
    """Handle priorities command."""
    from src.data.schema import get_connection
    from src.features.sprint_features import SprintFeatureExtractor
    from src.intelligence.orchestrator import create_intelligence_from_settings

    conn = get_connection()
    extractor = SprintFeatureExtractor(conn)

    # Get sprint features
    if args.sprint_id:
        sprint_features = extractor.extract_features(args.sprint_id)
        sprint_id = args.sprint_id
    else:
        sprint_features = extractor.get_active_sprint_features(args.board)
        sprint_id = sprint_features.get("sprint_id") if sprint_features else None

    if not sprint_features:
        print("Sprint not found", file=sys.stderr)
        return 1

    # Get sprint tickets
    tickets = get_sprint_tickets(conn, sprint_id)

    # Create intelligence
    try:
        intel = create_intelligence_from_settings()
    except Exception as e:
        print(f"Failed to initialize intelligence: {e}", file=sys.stderr)
        return 1

    # Generate suggestions
    suggestions = intel.suggest_priorities(sprint_features, tickets)

    if args.json:
        print(json.dumps(suggestions.to_dict(), indent=2))
    else:
        print(f"\n📊 Priority Suggestions: {suggestions.sprint_name}")

        if suggestions.must_complete:
            print(f"\n   Must Complete:")
            for key in suggestions.must_complete:
                print(f"   ✅ {key}")

        if suggestions.consider_deferring:
            print(f"\n   Consider Deferring:")
            for key in suggestions.consider_deferring:
                print(f"   ⏳ {key}")

        print(f"\n   Focus: {suggestions.focus_recommendation}")
        print(f"   Risk if unchanged: {suggestions.risk_if_unchanged}")

    return 0


def cmd_workload(args) -> int:
    """Handle workload command."""
    from src.features.pipeline import create_pipeline_from_settings
    from src.models.workload_scorer import WorkloadScorer
    from src.intelligence.orchestrator import create_intelligence_from_settings

    pipeline = create_pipeline_from_settings()
    developer_df = pipeline.build_developer_features()

    if developer_df.empty:
        print("No developer data available", file=sys.stderr)
        return 1

    # Get specific developer or first one
    if args.developer_id:
        dev_data = developer_df[developer_df["assignee_id"] == args.developer_id]
        if dev_data.empty:
            print(f"Developer {args.developer_id} not found", file=sys.stderr)
            return 1
    else:
        # Use first developer with highest workload
        dev_data = developer_df.head(1)

    developer = dev_data.iloc[0].to_dict()

    # Calculate workload score
    scorer = WorkloadScorer()
    scorer.set_team_baselines(developer_df)
    score_result = scorer.score(developer)
    developer.update(score_result)

    # Create intelligence
    try:
        intel = create_intelligence_from_settings()
    except Exception as e:
        print(f"Failed to initialize intelligence: {e}", file=sys.stderr)
        return 1

    # Generate assessment
    assessment = intel.assess_developer_workload(developer)

    if args.json:
        print(json.dumps(assessment.to_dict(), indent=2))
    else:
        status_emoji = {
            "sustainable": "🟢",
            "concerning": "🟡",
            "critical": "🔴",
        }.get(assessment.assessment, "⚪")

        print(f"\n👤 Workload Assessment: {assessment.developer_name}")
        print(f"   Status: {status_emoji} {assessment.assessment.upper()}")
        print(f"\n   {assessment.summary}")

        if assessment.immediate_concerns:
            print(f"\n   Immediate Concerns:")
            for concern in assessment.immediate_concerns:
                print(f"   ⚠️  {concern}")

        if assessment.recommendations:
            print(f"\n   Recommendations:")
            for rec in assessment.recommendations:
                print(f"   • {rec}")

    return 0


def cmd_standup(args) -> int:
    """Handle standup command."""
    from src.data.schema import get_connection
    from src.features.sprint_features import SprintFeatureExtractor
    from src.intelligence.orchestrator import create_intelligence_from_settings

    conn = get_connection()
    extractor = SprintFeatureExtractor(conn)

    # Get sprint features
    if args.sprint_id:
        sprint_features = extractor.extract_features(args.sprint_id)
        sprint_id = args.sprint_id
    else:
        sprint_features = extractor.get_active_sprint_features(args.board)
        sprint_id = sprint_features.get("sprint_id") if sprint_features else None

    if not sprint_features:
        print("Sprint not found", file=sys.stderr)
        return 1

    # Get sprint tickets
    tickets = get_sprint_tickets(conn, sprint_id)

    # Create intelligence
    try:
        intel = create_intelligence_from_settings()
    except Exception as e:
        print(f"Failed to initialize intelligence: {e}", file=sys.stderr)
        return 1

    # Generate standup summary
    summary = intel.generate_standup_summary(sprint_features, tickets)

    print(f"\n📣 Daily Standup - {sprint_features.get('sprint_name', 'Sprint')}")
    print("-" * 50)
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
