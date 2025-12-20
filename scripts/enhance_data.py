"""
Data Enhancement Script.

Adds sample data to make the dashboard more realistic:
- Story points to issues
- Sprint assignments
- Blocked/On Hold statuses
"""

import duckdb
import random
from pathlib import Path


def enhance_data():
    """Enhance database with sample data."""
    db_path = Path("data/jira.duckdb")

    if not db_path.exists():
        print("ERROR: Database not found!")
        return False

    conn = duckdb.connect(str(db_path))

    print("=== DATA ENHANCEMENT SCRIPT ===\n")

    # 1. Add story points
    points_count = conn.execute(
        "SELECT COUNT(*) FROM issues WHERE story_points IS NOT NULL AND story_points > 0"
    ).fetchone()[0]

    if points_count > 0:
        print(f"1. Story points already added ({points_count} issues) - SKIPPED")
    else:
        print("1. Adding story points...")
        issues = conn.execute("SELECT key, issue_type FROM issues").fetchall()

        point_values = [1, 2, 3, 5, 8, 13]
        weights_by_type = {
            'Bug': [0.3, 0.3, 0.2, 0.15, 0.05, 0.0],
            'Story': [0.1, 0.2, 0.3, 0.25, 0.1, 0.05],
            'Task': [0.2, 0.3, 0.25, 0.15, 0.08, 0.02],
            'Epic': [0.0, 0.05, 0.1, 0.2, 0.35, 0.3],
            'Sub-task': [0.4, 0.35, 0.2, 0.05, 0.0, 0.0],
        }
        default_weights = [0.15, 0.25, 0.3, 0.2, 0.08, 0.02]

        for key, issue_type in issues:
            weights = weights_by_type.get(issue_type, default_weights)
            points = random.choices(point_values, weights=weights)[0]
            conn.execute("UPDATE issues SET story_points = ? WHERE key = ?", [points, key])

        print(f"   Updated {len(issues)} issues with story points")

    # 2. Normalize status
    print("\n2. Normalizing status names...")
    status_mapping = {
        'À faire': 'To Do',
        'Terminé(e)': 'Done',
        'En cours': 'In Progress',
    }

    for old_status, new_status in status_mapping.items():
        count = conn.execute(
            "SELECT COUNT(*) FROM issues WHERE status = ?", [old_status]
        ).fetchone()[0]
        if count > 0:
            conn.execute(
                "UPDATE issues SET status = ? WHERE status = ?",
                [new_status, old_status]
            )
            print(f"   Mapped '{old_status}' -> '{new_status}' ({count} issues)")

    # 3. Find/activate sprint
    print("\n3. Finding active sprint...")
    active_sprint = conn.execute(
        "SELECT id, name FROM sprints WHERE state = 'active' LIMIT 1"
    ).fetchone()

    if not active_sprint:
        any_sprint = conn.execute(
            "SELECT id, name FROM sprints ORDER BY id LIMIT 1"
        ).fetchone()
        if any_sprint:
            conn.execute(
                "UPDATE sprints SET state = 'active' WHERE id = ?",
                [any_sprint[0]]
            )
            active_sprint = any_sprint
            print(f"   Activated sprint: {any_sprint[1]}")
        else:
            print("   No sprints found!")
    else:
        print(f"   Found active sprint: {active_sprint[1]}")

    # 4. Assign issues to sprints
    if active_sprint:
        sprint_id, sprint_name = active_sprint

        current_count = conn.execute(
            "SELECT COUNT(*) FROM issues WHERE sprint_id IS NOT NULL"
        ).fetchone()[0]

        if current_count > 0:
            print(f"\n4. Sprint assignments exist ({current_count} issues) - SKIPPED")
        else:
            print("\n4. Assigning issues to sprints...")
            # Get keys to assign
            to_assign = conn.execute("""
                SELECT key FROM issues
                WHERE sprint_id IS NULL AND status != 'Done'
                LIMIT 50
            """).fetchall()

            for (key,) in to_assign:
                conn.execute(
                    "UPDATE issues SET sprint_id = ?, sprint_name = ? WHERE key = ?",
                    [sprint_id, sprint_name, key]
                )

            # Also assign completed
            completed = conn.execute("""
                SELECT key FROM issues
                WHERE sprint_id IS NULL AND status = 'Done'
                LIMIT 30
            """).fetchall()

            for (key,) in completed:
                conn.execute(
                    "UPDATE issues SET sprint_id = ?, sprint_name = ? WHERE key = ?",
                    [sprint_id, sprint_name, key]
                )

            total = len(to_assign) + len(completed)
            print(f"   Assigned {total} issues to sprint '{sprint_name}'")

    # 5. Add blockers
    print("\n5. Adding blocked issues...")
    blocked_count = conn.execute(
        "SELECT COUNT(*) FROM issues WHERE status = 'Blocked'"
    ).fetchone()[0]

    if blocked_count > 0:
        print(f"   Blocked issues exist ({blocked_count}) - SKIPPED")
    else:
        in_progress = conn.execute("""
            SELECT key FROM issues WHERE status = 'In Progress' LIMIT 5
        """).fetchall()

        for (key,) in in_progress:
            conn.execute(
                "UPDATE issues SET status = 'Blocked' WHERE key = ?",
                [key]
            )

        print(f"   Created {len(in_progress)} blocked issues")

    # 6. Update sprint metrics
    print("\n6. Updating sprint metrics...")
    if active_sprint:
        sprint_id = active_sprint[0]
        metrics = conn.execute("""
            SELECT
                COALESCE(SUM(story_points), 0) as total,
                COALESCE(SUM(CASE WHEN status IN ('Done', 'Resolved', 'Closed')
                    THEN story_points ELSE 0 END), 0) as completed
            FROM issues WHERE sprint_id = ?
        """, [sprint_id]).fetchone()

        total_points, completed_points = metrics
        conn.execute("""
            UPDATE sprints
            SET committed_points = ?,
                completed_points = ?,
                completion_rate = CASE WHEN ? > 0 THEN CAST(? AS FLOAT) / ? ELSE 0 END
            WHERE id = ?
        """, [total_points, completed_points, total_points, completed_points, total_points, sprint_id])
        print(f"   Sprint: {total_points} committed, {completed_points} completed")

    # 7. Verify
    print("\n=== VERIFICATION ===\n")
    checks = [
        ("Issues with story_points", "SELECT COUNT(*) FROM issues WHERE story_points > 0"),
        ("Issues with sprint", "SELECT COUNT(*) FROM issues WHERE sprint_id IS NOT NULL"),
        ("Blocked issues", "SELECT COUNT(*) FROM issues WHERE status = 'Blocked'"),
        ("In Progress", "SELECT COUNT(*) FROM issues WHERE status = 'In Progress'"),
        ("To Do", "SELECT COUNT(*) FROM issues WHERE status = 'To Do'"),
        ("Done", "SELECT COUNT(*) FROM issues WHERE status IN ('Done', 'Resolved', 'Closed')"),
        ("Worklogs", "SELECT COUNT(*) FROM worklogs"),
        ("Changelog", "SELECT COUNT(*) FROM issue_changelog"),
    ]

    all_ok = True
    for label, query in checks:
        count = conn.execute(query).fetchone()[0]
        status = "OK" if count > 0 else "NEEDS DATA"
        if count == 0:
            all_ok = False
        print(f"  {label}: {count} [{status}]")

    print("\n=== STATUS DISTRIBUTION ===")
    status_dist = conn.execute("""
        SELECT status, COUNT(*) as cnt FROM issues
        GROUP BY status ORDER BY cnt DESC
    """).fetchall()
    for status, cnt in status_dist:
        print(f"  {status}: {cnt}")

    conn.close()
    print("\n=== DATA ENHANCEMENT COMPLETE ===")
    return all_ok


if __name__ == "__main__":
    enhance_data()
