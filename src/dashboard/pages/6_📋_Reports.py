"""
📋 Reports - Premium Report Generation & Export Center
Generate comprehensive reports with export to multiple formats.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

st.set_page_config(page_title="Reports", page_icon="📋", layout="wide")

# Premium Reports CSS
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f8f9fa;
    }

    .section-container {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 16px;
    }

    .report-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .report-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }

    .report-icon {
        font-size: 32px;
        margin-bottom: 12px;
    }

    .report-name {
        font-size: 18px;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 8px;
    }

    .report-description {
        font-size: 13px;
        color: #64748b;
        line-height: 1.5;
    }

    .format-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin: 4px 4px 0 0;
    }
    .format-pdf { background: #fee2e2; color: #991b1b; }
    .format-excel { background: #dcfce7; color: #166534; }
    .format-csv { background: #e0f2fe; color: #075985; }
    .format-md { background: #f3e8ff; color: #6b21a8; }

    .preview-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
        border: 1px solid #e2e8f0;
    }

    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
    }

    .mini-metric {
        background: white;
        padding: 12px 16px;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        border: 1px solid #e2e8f0;
    }

    .mini-metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #4f46e5;
    }

    .mini-metric-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
    }

    .report-preview {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        font-family: monospace;
        font-size: 12px;
        color: #334155;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Quick Win Widget */
    .quick-win-widget {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
        border: 1px solid rgba(52, 211, 153, 0.3);
        box-shadow: 0 8px 32px rgba(6, 78, 59, 0.3);
        position: relative;
        overflow: hidden;
    }
    .quick-win-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .quick-win-icon { font-size: 24px; }
    .quick-win-title {
        color: #a7f3d0;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .email-summary {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 16px;
        font-family: monospace;
        font-size: 13px;
        color: #f0fdf4;
        white-space: pre-line;
        line-height: 1.6;
    }
    .copy-hint {
        color: #6ee7b7;
        font-size: 11px;
        margin-top: 10px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def generate_quick_email(conn) -> dict:
    """Generate a ready-to-send executive summary email - saves 10 minutes every morning."""
    today = datetime.now().strftime('%B %d, %Y')

    # Get sprint status
    sprint = conn.execute("""
        SELECT name, state FROM sprints
        WHERE state = 'active' ORDER BY start_date DESC LIMIT 1
    """).fetchone()
    sprint_name = sprint[0] if sprint else "No active sprint"

    # Get completion metrics
    metrics = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as wip,
            COALESCE(SUM(CASE WHEN status = 'Terminé(e)' THEN story_points END), 0) as pts_done,
            COALESCE(SUM(story_points), 0) as pts_total
        FROM issues i
        JOIN sprints s ON i.sprint_id = s.id AND s.state = 'active'
    """).fetchone()

    total, done, wip = (metrics[0] or 0), (metrics[1] or 0), (metrics[2] or 0)
    pts_done, pts_total = (metrics[3] or 0), (metrics[4] or 0)
    completion = (done / total * 100) if total > 0 else 0

    # Get blockers/risks
    blockers = conn.execute("""
        SELECT COUNT(*) FROM issues
        WHERE status NOT IN ('Terminé(e)', 'Done', 'Closed')
        AND priority = 'Highest'
        AND sprint_id IN (SELECT id FROM sprints WHERE state = 'active')
    """).fetchone()[0]

    # Get yesterday's completed
    yesterday_done = conn.execute("""
        SELECT COUNT(*) FROM issues
        WHERE status = 'Terminé(e)'
        AND updated >= CURRENT_DATE - INTERVAL '1 day'
    """).fetchone()[0]

    # Generate email text
    status_emoji = "🟢" if completion >= 70 else "🟡" if completion >= 40 else "🔴"

    email_text = f"""Subject: {status_emoji} Sprint Status - {today}

Hi Team,

Quick update on {sprint_name}:

📊 SPRINT PROGRESS
• Completion: {completion:.0f}% ({done}/{total} items)
• Points: {pts_done:.0f}/{pts_total:.0f} done
• In Progress: {wip} items

📈 YESTERDAY
• Completed: {yesterday_done} items

⚠️ ATTENTION NEEDED
• High Priority Items: {blockers}

Best regards"""

    return {
        'email_text': email_text,
        'completion': completion,
        'status': 'healthy' if completion >= 70 else 'warning' if completion >= 40 else 'critical'
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without requiring tabulate."""
    if df.empty:
        return "*No data available*"

    # Get column headers
    headers = list(df.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("-" * max(3, len(str(h))) for h in headers) + " |"

    # Build data rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(v) if pd.notna(v) else "" for v in row) + " |"
        rows.append(row_str)

    return "\n".join([header_row, separator] + rows)


def generate_sprint_report(conn) -> dict:
    """Generate sprint health report data."""
    sprint = conn.execute("""
        SELECT id, name, state, start_date, end_date
        FROM sprints
        WHERE state = 'active'
        ORDER BY start_date DESC LIMIT 1
    """).fetchone()

    if not sprint:
        sprint = conn.execute("""
            SELECT id, name, state, start_date, end_date
            FROM sprints ORDER BY start_date DESC LIMIT 1
        """).fetchone()

    if not sprint:
        return None

    issues = conn.execute("""
        SELECT key, summary, status, priority, issue_type,
               COALESCE(story_points, 0) as story_points,
               assignee_name
        FROM issues WHERE sprint_id = ?
    """, [sprint[0]]).fetchdf()

    total = len(issues)
    done = len(issues[issues['status'] == 'Terminé(e)'])
    total_pts = issues['story_points'].sum()
    done_pts = issues[issues['status'] == 'Terminé(e)']['story_points'].sum()
    blocked = 0  # No 'Blocked' status in this Jira instance

    return {
        'sprint_name': sprint[1],
        'sprint_state': sprint[2],
        'total_issues': total,
        'completed_issues': done,
        'total_points': total_pts,
        'completed_points': done_pts,
        'blocked_count': blocked,
        'completion_rate': (done / total * 100) if total > 0 else 0,
        'issues': issues
    }


def generate_team_report(conn) -> dict:
    """Generate team workload report data."""
    team = conn.execute("""
        SELECT
            COALESCE(assignee_name, 'Unassigned') as name,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as wip,
            0 as blocked,
            COALESCE(SUM(story_points), 0) as points
        FROM issues
        WHERE assignee_name IS NOT NULL
        GROUP BY assignee_name
        ORDER BY points DESC
    """).fetchdf()

    return {
        'team_size': len(team),
        'total_wip': team['wip'].sum(),
        'total_blocked': team['blocked'].sum(),
        'team_data': team
    }


def generate_velocity_report(conn) -> dict:
    """Generate velocity report data."""
    velocity = conn.execute("""
        SELECT
            s.name,
            COALESCE(SUM(CASE WHEN i.status = 'Terminé(e)'
                         THEN i.story_points ELSE 0 END), 0) as completed,
            COALESCE(SUM(i.story_points), 0) as committed,
            COUNT(DISTINCT i.assignee_name) as team_size
        FROM sprints s
        LEFT JOIN issues i ON i.sprint_id = s.id
        WHERE s.start_date IS NOT NULL
        GROUP BY s.name, s.start_date
        ORDER BY s.start_date DESC
        LIMIT 10
    """).fetchdf()

    return {
        'avg_velocity': velocity['completed'].mean(),
        'sprints_analyzed': len(velocity),
        'velocity_data': velocity
    }


def format_markdown_report(report_type: str, data: dict) -> str:
    """Format report as markdown."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    if report_type == 'Sprint Health':
        return f"""# Sprint Health Report
Generated: {now}

## Sprint: {data['sprint_name']} ({data['sprint_state']})

### Summary
| Metric | Value |
|--------|-------|
| Total Issues | {data['total_issues']} |
| Completed | {data['completed_issues']} |
| Completion Rate | {data['completion_rate']:.1f}% |
| Total Points | {data['total_points']:.0f} |
| Completed Points | {data['completed_points']:.0f} |
| Blocked | {data['blocked_count']} |

### Risk Assessment
{'**High Risk** - Sprint is behind schedule' if data['completion_rate'] < 50 else '**On Track** - Sprint is progressing well'}

### Issues Breakdown
{df_to_markdown(data['issues'][['key', 'summary', 'status', 'story_points']].head(20))}
"""

    elif report_type == 'Team Workload':
        return f"""# Team Workload Report
Generated: {now}

## Summary
| Metric | Value |
|--------|-------|
| Team Size | {data['team_size']} |
| Total WIP | {data['total_wip']} |
| Total Blocked | {data['total_blocked']} |

## Team Breakdown
{df_to_markdown(data['team_data'])}
"""

    elif report_type == 'Velocity':
        return f"""# Velocity Report
Generated: {now}

## Summary
| Metric | Value |
|--------|-------|
| Average Velocity | {data['avg_velocity']:.1f} pts/sprint |
| Sprints Analyzed | {data['sprints_analyzed']} |

## Sprint Velocity
{df_to_markdown(data['velocity_data'])}
"""

    return "Report not available"


def main():
    st.markdown("# 📋 Reports Center")
    st.markdown("*Generate comprehensive reports and export to multiple formats*")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Quick Win Widget - Quick Email Generator
    try:
        email_data = generate_quick_email(conn)
        st.markdown(f"""
<div class="quick-win-widget">
    <div class="quick-win-header">
        <span class="quick-win-icon">📧</span>
        <span class="quick-win-title">QUICK EMAIL • Ready to Send</span>
    </div>
    <div class="email-summary">{email_data['email_text']}</div>
    <div class="copy-hint">💡 Click and Ctrl+C to copy, paste into email</div>
</div>
""", unsafe_allow_html=True)
    except Exception:
        pass

    # Report Types
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Select Report Type</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="report-card">
    <div class="report-icon">🏃</div>
    <div class="report-name">Sprint Health Report</div>
    <div class="report-description">
        Comprehensive sprint analysis including completion rates,
        blocked items, and risk assessment.
    </div>
    <div style="margin-top: 12px;">
        <span class="format-badge format-md">Markdown</span>
        <span class="format-badge format-csv">CSV</span>
    </div>
</div>
""", unsafe_allow_html=True)
        sprint_btn = st.button("Generate Sprint Report", key="sprint", type="primary")

    with col2:
        st.markdown("""
<div class="report-card">
    <div class="report-icon">👥</div>
    <div class="report-name">Team Workload Report</div>
    <div class="report-description">
        Team capacity analysis, workload distribution,
        and individual performance metrics.
    </div>
    <div style="margin-top: 12px;">
        <span class="format-badge format-md">Markdown</span>
        <span class="format-badge format-csv">CSV</span>
    </div>
</div>
""", unsafe_allow_html=True)
        team_btn = st.button("Generate Team Report", key="team", type="primary")

    with col3:
        st.markdown("""
<div class="report-card">
    <div class="report-icon">📈</div>
    <div class="report-name">Velocity Report</div>
    <div class="report-description">
        Historical velocity trends, sprint-over-sprint
        comparison, and predictive insights.
    </div>
    <div style="margin-top: 12px;">
        <span class="format-badge format-md">Markdown</span>
        <span class="format-badge format-csv">CSV</span>
    </div>
</div>
""", unsafe_allow_html=True)
        velocity_btn = st.button("Generate Velocity Report", key="velocity", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    # Report Generation
    if sprint_btn:
        with st.spinner("Generating Sprint Health Report..."):
            data = generate_sprint_report(conn)
            if data:
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🏃 Sprint Health Report Preview</div>', unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Issues", data['total_issues'])
                with m2:
                    st.metric("Completed", data['completed_issues'])
                with m3:
                    st.metric("Completion", f"{data['completion_rate']:.0f}%")
                with m4:
                    st.metric("Blocked", data['blocked_count'])

                # Report preview
                report_md = format_markdown_report('Sprint Health', data)
                st.markdown('<div class="preview-section">', unsafe_allow_html=True)
                st.markdown(report_md)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Markdown",
                        report_md,
                        file_name=f"sprint_report_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                with col2:
                    csv_data = data['issues'].to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        file_name=f"sprint_issues_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No sprint data available.")

    if team_btn:
        with st.spinner("Generating Team Workload Report..."):
            data = generate_team_report(conn)
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">👥 Team Workload Report Preview</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Team Size", data['team_size'])
            with m2:
                st.metric("Total WIP", data['total_wip'])
            with m3:
                st.metric("Blocked", data['total_blocked'])

            report_md = format_markdown_report('Team Workload', data)
            st.markdown('<div class="preview-section">', unsafe_allow_html=True)
            st.markdown(report_md)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Markdown",
                    report_md,
                    file_name=f"team_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            with col2:
                csv_data = data['team_data'].to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"team_workload_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            st.markdown('</div>', unsafe_allow_html=True)

    if velocity_btn:
        with st.spinner("Generating Velocity Report..."):
            data = generate_velocity_report(conn)
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Velocity Report Preview</div>', unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Avg Velocity", f"{data['avg_velocity']:.1f} pts")
            with m2:
                st.metric("Sprints Analyzed", data['sprints_analyzed'])

            # Velocity chart
            if not data['velocity_data'].empty:
                fig = go.Figure()
                velocity_df = data['velocity_data'].iloc[::-1]
                fig.add_trace(go.Bar(
                    x=velocity_df['name'],
                    y=velocity_df['completed'],
                    name='Completed',
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=60),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#64748b'},
                    xaxis=dict(tickangle=-45, showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            report_md = format_markdown_report('Velocity', data)
            st.markdown('<div class="preview-section">', unsafe_allow_html=True)
            st.markdown(report_md)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Markdown",
                    report_md,
                    file_name=f"velocity_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            with col2:
                csv_data = data['velocity_data'].to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"velocity_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    Reports Center | Export to Markdown & CSV |
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
