"""Main Streamlit dashboard application."""

import streamlit as st
from pathlib import Path
import altair as alt
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Jira AI Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa726; font-weight: bold; }
    .risk-low { color: #66bb6a; font-weight: bold; }
    .stMetric > div { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


def get_connection():
    """Get database connection with caching."""
    import duckdb
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run 'jira-copilot init' first.")
    return duckdb.connect(str(db_path))


def get_predictor():
    """Get unified predictor with caching."""
    from src.models.predictor import Predictor
    return Predictor.from_model_dir("models")


def get_intelligence():
    """Get intelligence orchestrator."""
    from src.intelligence.orchestrator import JiraIntelligence
    return JiraIntelligence()


def main():
    """Main dashboard entry point."""
    # Sidebar navigation
    st.sidebar.title("🤖 Jira AI Co-pilot")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "📋 Board", "🏃 Sprint Health", "👥 Team Workload", "🎯 Predictions", "📋 Reports"],
        label_visibility="collapsed",
    )

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    try:
        conn = get_connection()
        issue_count = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        sprint_count = conn.execute("SELECT COUNT(*) FROM sprints").fetchone()[0]

        st.sidebar.metric("Total Issues", issue_count)
        st.sidebar.metric("Total Sprints", sprint_count)
    except Exception:
        st.sidebar.warning("Database not connected")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with [Streamlit](https://streamlit.io) • "
        "[GitHub](https://github.com)"
    )

    # Page routing
    if page == "📊 Overview":
        show_overview()
    elif page == "📋 Board":
        show_board()
    elif page == "🏃 Sprint Health":
        show_sprint_health()
    elif page == "👥 Team Workload":
        show_team_workload()
    elif page == "🎯 Predictions":
        show_predictions()
    elif page == "📋 Reports":
        show_reports()


def show_overview():
    """Show dashboard overview."""
    st.markdown('<h1 class="main-header">📊 Dashboard Overview</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()
        
        # --- Row 1 ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Vue d'ensemble de l'état")
            st.caption("Obtenez un instantané de l'état de vos tickets. Afficher tous les tickets")
            
            status_df = conn.execute("SELECT status, COUNT(*) as count FROM issues GROUP BY status").fetchdf()
            if not status_df.empty:
                total_issues = status_df['count'].sum()
                
                # Doughnut chart
                base = alt.Chart(status_df).encode(
                    theta=alt.Theta("count", stack=True)
                )
                
                pie = base.mark_arc(innerRadius=60).encode(
                    color=alt.Color("status", legend=alt.Legend(title="Status")),
                    order=alt.Order("count", sort="descending"),
                    tooltip=["status", "count"]
                )
                
                text = base.mark_text(radius=0, fontSize=20, fontWeight="bold").encode(
                    text=alt.value(f"{total_issues}")
                )
                
                st.altair_chart(pie + text, use_container_width=True)
            else:
                st.info("No data available")

        with col2:
            st.markdown("### Activité récente")
            st.caption("Tenez-vous au courant de ce qui se passe tout au long de l'espace.")
            
            recent_activity = conn.execute("""
                SELECT COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee_name,
                       i.summary, i.status, i.updated, i.issue_type
                FROM issues i
                LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
                ORDER BY i.updated DESC
                LIMIT 5
            """).fetchdf()
            
            if not recent_activity.empty:
                for _, row in recent_activity.iterrows():
                    with st.container():
                        c1, c2 = st.columns([1, 10])
                        with c1:
                            # Initials circle
                            name = row['assignee_name']
                            initials = "".join([n[0] for n in name.split()[:2]]) if name else "??"
                            st.markdown(f"<div style='background-color:#ff4b4b;color:white;border-radius:50%;width:30px;height:30px;text-align:center;line-height:30px;font-size:12px;margin-top:5px;'>{initials}</div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"<p style='margin-bottom:0px;font-size:14px;'><b>{name}</b> a mis à jour « {row['summary'][:40]}... »</p>", unsafe_allow_html=True)
                            st.caption(f"{row['issue_type']} • {row['status']} • {row['updated']}")

        st.markdown("---")

        # --- Row 2 ---
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Répartition des priorités")
            st.caption("Obtenez une vue globale montrant comment le travail est priorisé.")
            
            priority_df = conn.execute("SELECT priority, COUNT(*) as count FROM issues GROUP BY priority").fetchdf()
            if not priority_df.empty:
                chart = alt.Chart(priority_df).mark_bar().encode(
                    x=alt.X('priority', sort=None, title="Priority"),
                    y=alt.Y('count', title="Issue Count"),
                    color='priority',
                    tooltip=['priority', 'count']
                )
                st.altair_chart(chart, use_container_width=True)

        with col4:
            st.markdown("### Types de ticket")
            st.caption("Obtenez une répartition des tickets par type.")
            
            type_df = conn.execute("SELECT issue_type, COUNT(*) as count FROM issues GROUP BY issue_type").fetchdf()
            if not type_df.empty:
                total = type_df['count'].sum()
                for _, row in type_df.iterrows():
                    pct = (row['count'] / total) * 100
                    st.write(f"**{row['issue_type']}** ({pct:.0f}%)")
                    st.progress(row['count'] / total)

        st.markdown("---")

        # --- Row 3 ---
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("### Charge de travail de l'équipe")
            st.caption("Surveillez la capacité de votre équipe.")
            
            workload_df = conn.execute("""
                SELECT COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee_name,
                       COUNT(*) as count
                FROM issues i
                LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
                WHERE i.status != 'Done'
                GROUP BY COALESCE(un.display_name, i.assignee_id, 'Unassigned')
                ORDER BY count DESC
                LIMIT 5
            """).fetchdf()
            
            if not workload_df.empty:
                max_load = workload_df['count'].max()
                for _, row in workload_df.iterrows():
                    name = row['assignee_name']
                    count = row['count']
                    st.write(f"**{name}**")
                    val = count / max_load if max_load > 0 else 0
                    st.progress(val, text=f"{count} tickets")

        with col6:
            st.markdown("### Epic : avancement")
            st.caption("Utilisez des epics pour suivre les initiatives les plus importantes.")
            
            # Simple placeholder for now as epic logic can be complex
            epics_df = conn.execute("""
                SELECT epic_name, 
                       COUNT(*) as total,
                       SUM(CASE WHEN status = 'Done' THEN 1 ELSE 0 END) as completed
                FROM issues 
                WHERE epic_name IS NOT NULL 
                GROUP BY epic_name
                LIMIT 5
            """).fetchdf()
            
            if not epics_df.empty:
                 for _, row in epics_df.iterrows():
                     epic = row['epic_name']
                     total = row['total']
                     completed = row['completed']
                     pct = completed / total if total > 0 else 0
                     
                     st.write(f"**{epic}**")
                     st.progress(pct, text=f"{completed}/{total} Done")
            else:
                 st.info("No epics found used in active issues.")
                 # Show a placeholder image or text to match the "no data" look if needed, 
                 # but for now this is good. The image shows a placeholder "Epic : avancement ... Une epic, qu'est-ce que c'est ?" 
                 # if empty, maybe I should match that?
                 if epics_df.empty:
                     st.markdown("""
                     <div style="text-align: center; color: gray; padding: 20px;">
                        <h3>🧩</h3>
                        <p><b>Epic : avancement</b></p>
                        <p>Utilisez des epics pour suivre les initiatives les plus importantes.</p>
                     </div>
                     """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading overview: {e}")


def show_sprint_health():
    """Show sprint health page."""
    st.markdown('<h1 class="main-header">🏃 Sprint Health</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()

        # Sprint selector
        sprints = conn.execute("""
            SELECT id, name, state
            FROM sprints
            ORDER BY start_date DESC
            LIMIT 20
        """).fetchall()

        if not sprints:
            st.warning("No sprints found")
            return

        sprint_options = {f"{s[1]} ({s[2]})": s[0] for s in sprints}
        selected = st.selectbox("Select Sprint", list(sprint_options.keys()))
        sprint_id = sprint_options[selected]

        # Get sprint data
        from src.features.sprint_features import SprintFeatureExtractor

        extractor = SprintFeatureExtractor(conn)
        features = extractor.extract_features(sprint_id)

        predictor = get_predictor()
        risk = predictor.predict_sprint_risk(features)

        # Risk gauge
        col1, col2 = st.columns([1, 2])

        with col1:
            risk_score = risk.get("score", 0)
            risk_level = risk.get("level", "unknown")

            st.markdown("### Risk Score")

            # Color based on risk
            color = {"low": "green", "medium": "orange", "high": "red"}.get(risk_level, "gray")
            st.markdown(
                f'<h1 style="color: {color}; text-align: center;">{risk_score:.0f}</h1>',
                unsafe_allow_html=True
            )
            st.markdown(f'<p style="text-align: center;">{risk_level.upper()}</p>', unsafe_allow_html=True)

        with col2:
            st.markdown("### Key Metrics")

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                st.metric("Total Points", features.get("total_points", 0))
                st.metric("Completed", features.get("completed_points", 0))

            with metrics_col2:
                st.metric("Days Elapsed", features.get("days_elapsed", 0))
                st.metric("Days Remaining", features.get("days_remaining", 0))

            with metrics_col3:
                st.metric("Completion Rate", f"{features.get('completion_rate', 0):.1f}%")
                st.metric("Blocked Items", features.get("blocked_count", 0))

        # Risk factors
        st.markdown("### Risk Factors")

        if risk.get("factors"):
            import pandas as pd

            factors_data = [
                {"Factor": k.replace("_", " ").title(), "Contribution": v.get("contribution", 0) * 100}
                for k, v in risk["factors"].items()
            ]
            df = pd.DataFrame(factors_data)
            df = df.sort_values("Contribution", ascending=True)

            st.bar_chart(df.set_index("Factor"))

        # AI explanation
        with st.expander("🤖 AI Risk Explanation", expanded=False):
            if st.button("Generate Explanation"):
                with st.spinner("Analyzing..."):
                    intel = get_intelligence()
                    explanation = intel.explain_sprint_risk(features, risk)

                    st.markdown(f"**Summary:** {explanation.risk_summary}")

                    if explanation.main_concerns:
                        st.markdown("**Main Concerns:**")
                        for concern in explanation.main_concerns:
                            st.markdown(f"- {concern}")

                    if explanation.recommended_actions:
                        st.markdown("**Recommendations:**")
                        for action in explanation.recommended_actions:
                            priority = action.get("priority", "medium")
                            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                            st.markdown(f"{emoji} **{action.get('action', '')}**")
                            st.markdown(f"   _{action.get('rationale', '')}_")

        # Sprint issues
        st.markdown("### Sprint Issues")

        issues = conn.execute("""
            SELECT i.key, i.summary, i.status, i.priority, i.story_points,
                   COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee_name,
                   CASE WHEN i.status IN ('Blocked', 'On Hold') THEN true ELSE false END as is_blocked
            FROM issues i
            LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
            WHERE i.sprint_id = ?
            ORDER BY
                CASE i.status
                    WHEN 'Blocked' THEN 1
                    WHEN 'In Progress' THEN 2
                    WHEN 'To Do' THEN 3
                    ELSE 4
                END
        """, [sprint_id]).fetchall()

        if issues:
            import pandas as pd

            df = pd.DataFrame(
                issues,
                columns=["Key", "Summary", "Status", "Priority", "Points", "Assignee", "Blocked"]
            )
            df["Summary"] = df["Summary"].str[:40]
            df["Blocked"] = df["Blocked"].apply(lambda x: "🚫" if x else "")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Blocked": st.column_config.TextColumn("🚫", width="small"),
                }
            )

    except Exception as e:
        st.error(f"Error loading sprint health: {e}")


def show_team_workload():
    """Show team workload page."""
    st.markdown('<h1 class="main-header">👥 Team Workload</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()
        from src.features.developer_features import DeveloperFeatureExtractor

        # Get project key from existing issues
        project_result = conn.execute("""
            SELECT DISTINCT project_key FROM issues LIMIT 1
        """).fetchone()

        if not project_result:
            st.warning("No project data found. Run 'jira-copilot sync full' first.")
            return

        project_key = project_result[0]

        extractor = DeveloperFeatureExtractor(conn)

        # Get all developers with their features
        dev_df = extractor.extract_all_developers(project_key)

        if dev_df.empty:
            st.warning("No active developers found")
            return

        import pandas as pd

        # Build workload data from the extracted features
        workload_data = []
        for _, row in dev_df.iterrows():
            workload_data.append({
                "Developer": row.get("assignee_name", "Unknown")[:20],
                "WIP Count": int(row.get("wip_count", 0)),
                "WIP Points": float(row.get("wip_points", 0)),
                "Blocked": int(row.get("blocked_count", 0)),
                "7-day Velocity": float(row.get("completed_7d", 0)),
                "Status": "overloaded" if row.get("is_overloaded", False) else "optimal",
                "Score": float(row.get("workload_ratio", 1.0)),
            })

        df = pd.DataFrame(workload_data)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Team Size", len(developers))

        with col2:
            overloaded = len([d for d in workload_data if d["Status"] == "overloaded"])
            st.metric("Overloaded", overloaded, delta_color="inverse" if overloaded > 0 else "normal")

        with col3:
            total_wip = sum(d["WIP Points"] for d in workload_data)
            st.metric("Total WIP Points", total_wip)

        with col4:
            total_blocked = sum(d["Blocked"] for d in workload_data)
            st.metric("Total Blocked", total_blocked)

        # Workload chart
        st.markdown("### Workload Distribution")

        chart_df = df[["Developer", "WIP Points", "7-day Velocity"]].set_index("Developer")
        st.bar_chart(chart_df)

        # Team table
        st.markdown("### Team Status")

        def status_color(status):
            colors = {
                "overloaded": "🔴",
                "high": "🟡",
                "optimal": "🟢",
                "underloaded": "⚪"
            }
            return f"{colors.get(status, '⚪')} {status}"

        df["Status"] = df["Status"].apply(status_color)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Recommendations
        overloaded_devs = [d for d in workload_data if d["Status"].startswith("🔴")]
        if overloaded_devs:
            st.markdown("### ⚠️ Attention Required")
            for dev in overloaded_devs:
                st.warning(f"**{dev['Developer']}** is overloaded with {dev['WIP Points']} points in progress")

        # AI Assessment
        with st.expander("🤖 AI Workload Assessment", expanded=False):
            dev_select = st.selectbox(
                "Select Developer",
                [d["Developer"] for d in workload_data]
            )

            if st.button("Generate Assessment"):
                with st.spinner("Analyzing..."):
                    dev_data = next(d for d in workload_data if d["Developer"] == dev_select)
                    intel = get_intelligence()

                    # Prepare data for assessment
                    assessment_data = {
                        "assignee_id": dev_select,
                        "pseudonym": dev_select,
                        "wip_count": dev_data["WIP Count"],
                        "wip_points": dev_data["WIP Points"],
                        "blocked_count": dev_data["Blocked"],
                        "status": dev_data["Status"].split()[-1],
                        "score": dev_data["Score"],
                        "relative_to_team": 1.0,
                        "team_avg_wip_points": sum(d["WIP Points"] for d in workload_data) / len(workload_data),
                        "team_avg_completed_points": sum(d["7-day Velocity"] for d in workload_data) / len(workload_data),
                    }

                    assessment = intel.assess_developer_workload(assessment_data)

                    st.markdown(f"**Assessment:** {assessment.assessment}")
                    st.markdown(assessment.summary)

                    if assessment.recommendations:
                        st.markdown("**Recommendations:**")
                        for rec in assessment.recommendations:
                            st.markdown(f"- {rec}")

    except Exception as e:
        st.error(f"Error loading team workload: {e}")


def show_predictions():
    """Show predictions page."""
    st.markdown('<h1 class="main-header">🎯 Predictions</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Ticket Duration", "Sprint Risk"])

    with tab1:
        st.markdown("### Predict Ticket Duration")

        try:
            conn = get_connection()

            # Get open issues
            issues = conn.execute("""
                SELECT key, summary FROM issues
                WHERE status NOT IN ('Done', 'Closed')
                ORDER BY updated DESC
                LIMIT 100
            """).fetchall()

            if not issues:
                st.warning("No open issues found")
            else:
                issue_options = {f"{i[0]}: {i[1][:50]}...": i[0] for i in issues}
                selected = st.selectbox("Select Issue", list(issue_options.keys()))
                issue_key = issue_options[selected]

                if st.button("Predict Duration", key="predict_ticket"):
                    with st.spinner("Predicting..."):
                        from src.features.ticket_features import TicketFeatureExtractor

                        extractor = TicketFeatureExtractor(conn)
                        features = extractor.extract_features(issue_key)

                        predictor = get_predictor()
                        prediction = predictor.predict_ticket(features)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            hours = prediction.get("predicted_hours", 0)
                            st.metric("Predicted Duration", f"{hours:.1f} hours")

                        with col2:
                            days = prediction.get("predicted_days", 0)
                            st.metric("In Days", f"{days:.1f} days")

                        with col3:
                            if prediction.get("confidence_interval"):
                                ci = prediction["confidence_interval"]
                                st.metric(
                                    "Confidence Range",
                                    f"{ci.get('lower_hours', 0):.1f} - {ci.get('upper_hours', 0):.1f}h"
                                )

                        st.info(f"Model: {prediction.get('model_type', 'Unknown')}")

        except Exception as e:
            st.error(f"Error: {e}")

    with tab2:
        st.markdown("### Predict Sprint Risk")

        try:
            conn = get_connection()

            sprints = conn.execute("""
                SELECT id, name, state
                FROM sprints
                ORDER BY start_date DESC
                LIMIT 10
            """).fetchall()

            if not sprints:
                st.warning("No sprints found")
            else:
                sprint_options = {f"{s[1]} ({s[2]})": s[0] for s in sprints}
                selected = st.selectbox("Select Sprint", list(sprint_options.keys()), key="sprint_risk")
                sprint_id = sprint_options[selected]

                if st.button("Predict Risk", key="predict_risk"):
                    with st.spinner("Analyzing..."):
                        from src.features.sprint_features import SprintFeatureExtractor

                        extractor = SprintFeatureExtractor(conn)
                        features = extractor.extract_features(sprint_id)

                        predictor = get_predictor()
                        risk = predictor.predict_sprint_risk(features)

                        col1, col2 = st.columns(2)

                        with col1:
                            score = risk.get("score", 0)
                            level = risk.get("level", "unknown")
                            color = {"low": "green", "medium": "orange", "high": "red"}.get(level, "gray")

                            st.markdown(
                                f'<h1 style="color: {color}; text-align: center;">{score:.0f}/100</h1>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f'<p style="text-align: center;"><b>{level.upper()}</b></p>', unsafe_allow_html=True)

                        with col2:
                            st.markdown("**Top Risk Factors:**")
                            if risk.get("factors"):
                                factors = sorted(
                                    risk["factors"].items(),
                                    key=lambda x: x[1].get("contribution", 0),
                                    reverse=True
                                )
                                for factor, details in factors[:5]:
                                    contrib = details.get("contribution", 0) * 100
                                    st.markdown(f"- {factor.replace('_', ' ').title()}: {contrib:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")


def show_reports():
    """Show reports page."""
    st.markdown('<h1 class="main-header">📋 Reports</h1>', unsafe_allow_html=True)

    report_type = st.selectbox(
        "Report Type",
        ["Sprint Health Report", "Team Workload Report", "Velocity Report"]
    )

    col1, col2 = st.columns(2)

    with col1:
        format_option = st.selectbox("Format", ["Markdown", "HTML"])

    with col2:
        if st.button("Generate Report"):
            with st.spinner("Generating..."):
                try:
                    conn = get_connection()

                    if report_type == "Sprint Health Report":
                        from src.features.sprint_features import SprintFeatureExtractor
                        from src.models.predictor import Predictor
                        from src.intelligence.orchestrator import JiraIntelligence

                        sprint = conn.execute("""
                            SELECT id, name FROM sprints
                            WHERE state = 'active' LIMIT 1
                        """).fetchone()

                        if sprint:
                            extractor = SprintFeatureExtractor(conn)
                            features = extractor.extract_features(sprint[0])

                            predictor = Predictor.from_model_dir("models")
                            risk = predictor.predict_sprint_risk(features)

                            intel = JiraIntelligence()
                            explanation = intel.explain_sprint_risk(features, risk)

                            report = f"""# Sprint Health Report

**Sprint:** {features.get('sprint_name', 'Unknown')}

## Summary
{explanation.risk_summary}

## Metrics
- Completion Rate: {features.get('completion_rate', 0):.1f}%
- Risk Score: {risk.get('score', 0):.0f}/100 ({risk.get('level', 'unknown')})
- Days Remaining: {features.get('days_remaining', 0)}

## Recommendations
"""
                            for action in explanation.recommended_actions:
                                report += f"- {action.get('action', '')}\n"

                            st.markdown(report)

                            if format_option == "HTML":
                                import markdown
                                html = markdown.markdown(report)
                                st.download_button(
                                    "Download HTML",
                                    html,
                                    file_name="sprint_report.html",
                                    mime="text/html"
                                )
                            else:
                                st.download_button(
                                    "Download Markdown",
                                    report,
                                    file_name="sprint_report.md",
                                    mime="text/markdown"
                                )
                        else:
                            st.warning("No active sprint found")

                    elif report_type == "Team Workload Report":
                        st.info("Use CLI: jira-copilot report workload")

                    elif report_type == "Velocity Report":
                        st.info("Use CLI: jira-copilot analyze velocity")

                except Exception as e:
                    st.error(f"Error generating report: {e}")



def show_board():
    """Show Kanban board."""
    st.markdown('<h1 class="main-header">📋 Project Board</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()
        
        # Filter options
        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("🔍 Search issues...", "")
        with col2:
            my_issues = st.checkbox("Only my issues")

        # Base query
        query = """
            SELECT i.key, i.summary, i.status, i.priority,
                   COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee_name,
                   i.issue_type
            FROM issues i
            LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
        """
        params = []
        where_clauses = []

        if search:
            where_clauses.append("(summary ILIKE ? OR key ILIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        
        if my_issues:
            # For demo purposes, we might not have 'current user' auth, 
            # so we'll just mock it or skip if not strictly required.
            pass

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        issues = conn.execute(query, params).fetchdf()

        if issues.empty:
            st.info("No issues found matching your criteria.")
            return

        # kanban columns
        cols = st.columns(3)
        statuses = [
            ("To Do", ["To Do", "Open", "Reopened", "Backlog", "Blocked"]),
            ("In Progress", ["In Progress", "Review", "QA", "On Hold"]),
            ("Done", ["Done", "Closed", "Resolved", "Deployed"])
        ]

        for i, (col_name, status_list) in enumerate(statuses):
            with cols[i]:
                st.markdown(f"### {col_name}")
                st.markdown("---")
                
                # Filter issues for this column
                col_issues = issues[issues['status'].isin(status_list)]
                
                count = len(col_issues)
                st.caption(f"{count} issues")

                for _, issue in col_issues.iterrows():
                    with st.container():
                        # Card styling
                        priority_color = {
                            "High": "#ff4b4b", 
                            "Medium": "#ffa726", 
                            "Low": "#66bb6a"
                        }.get(issue['priority'], "gray")
                        
                        summary = issue['summary']
                        if len(summary) > 60:
                            summary = summary[:60] + "..."
                            
                        assignee = issue['assignee_name'] if issue['assignee_name'] else "Unassigned"
                        initial = assignee[0] if assignee != "Unassigned" else "?"
                        
                        st.markdown(f"""
                        <div style="
                            background-color: white;
                            padding: 10px;
                            border-radius: 5px;
                            border-left: 5px solid {priority_color};
                            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                            margin-bottom: 10px;
                            color: black;
                        ">
                            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                                <span style="font-weight:bold;font-size:0.9em;color:#333;">{issue['key']}</span>
                                <span style="font-size:0.7em;background:#f0f2f6;padding:2px 6px;border-radius:4px;color:#555;">{issue['issue_type']}</span>
                            </div>
                            <div style="margin: 8px 0; font-weight:500; font-size:0.95em; line-height:1.2;">{summary}</div>
                            <div style="display:flex;align-items:center;font-size:0.8em;color:#666;margin-top:8px;">
                                <div style="width:20px;height:20px;background:#e0e0e0;border-radius:50%;text-align:center;line-height:20px;margin-right:8px;font-size:10px;color:#555;">
                                    {initial}
                                </div>
                                {assignee}
                            </div>
                            <div style="margin-top:5px;text-align:right;">
                                <span style="font-size:0.7em;color:{priority_color};font-weight:bold;">{issue['priority']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading board: {e}")


if __name__ == "__main__":
    main()
