"""
Feature #2: The Burnout Barometer™
Dashboard for Early Warning Burnout Detection.
"""

import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
import plotly.express as px

from src.features.burnout_models import BurnoutAnalyzer
from config.settings import get_settings

st.set_page_config(page_title="Burnout Barometer", page_icon="🕯️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .risk-card-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .risk-card-med {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .risk-score {
        font-size: 24px;
        font-weight: 800;
    }
    .risk-high { color: #DC2626; }
    .risk-med { color: #D97706; }
    .risk-low { color: #059669; }
</style>
""", unsafe_allow_html=True)

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)

def main():
    st.title("🕯️ Burnout Barometer™")
    st.markdown("### Early Warning System for Behavioral Anomalies")
    
    conn = get_connection()
    if not conn:
        st.error("Database not found. Sync data first.")
        st.stop()

    with st.spinner("Analyzing behavioral patterns across the team..."):
        # 1. Fetch Data
        # Issues
        df_issues = conn.execute("SELECT * FROM issues").fetchdf()
        
        # Worklogs
        try:
            df_worklogs = conn.execute("SELECT * FROM worklogs").fetchdf()
        except:
            df_worklogs = pd.DataFrame()
            
        # Users
        df_users = conn.execute("SELECT * FROM users WHERE active = true").fetchdf()
        
        if df_issues.empty or df_users.empty:
            st.warning("Not enough data to analyze.")
            st.stop()
            
        # 2. Analyze
        analyzer = BurnoutAnalyzer()
        risk_profiles = analyzer.analyze_team_risks(df_issues, df_worklogs, df_users)
    
    # 3. Aggregates
    high_risk = [p for p in risk_profiles if p.risk_level == 'High Risk']
    med_risk = [p for p in risk_profiles if p.risk_level == 'Elevated']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Highest Risk Score", f"{risk_profiles[0].risk_score:.0f}/100" if risk_profiles else "0", 
                 delta="Critical" if risk_profiles and risk_profiles[0].risk_score > 75 else "Normal", delta_color="inverse")
    col2.metric("At-Risk Engineers", f"{len(high_risk)} High / {len(med_risk)} Elevated")
    col3.info("Analysis based on 6-month behavioral baseline vs last 30 days.")

    st.divider()

    # 4. Detailed Risk Cards
    st.header("🔴 High Priority Attention Needed")
    
    if not high_risk:
        st.success("No high-risk individuals detected across the organization.")
    
    for profile in high_risk:
        with st.container():
            st.markdown(f"""
            <div class="risk-card-high">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3>👤 {profile.user_name}</h3>
                    <div class="risk-score risk-high">Risk: {profile.risk_score:.0f}/100</div>
                </div>
                <b>⚠️ Primary Signals:</b>
                <ul>
                    {''.join([f'<li>{f}</li>' for f in profile.top_risk_factors])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Tickets/Week", f"{profile.current_metrics.get('ticket_volume',0):.1f}", 
                      f"{profile.deviations.get('volume_change',0)*100:+.0f}% vs baseline")
            m2.metric("Weekend Work Ratio", f"{profile.current_metrics.get('weekend_ratio',0)*100:.1f}%", 
                      f"{profile.deviations.get('weekend_change',0)*100:+.1f}% pts")
            m3.metric("After-Hours Ratio", f"{profile.current_metrics.get('after_hours_ratio',0)*100:.1f}%", 
                      f"{profile.deviations.get('after_hours_change',0)*100:+.1f}% pts")
            
            with st.expander("Full Behavioral Profile"):
                # Mock Sparkline Visualization
                dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='W')
                import numpy as np
                mock_activity = np.random.randint(5, 15, size=12) 
                if 'Workload' in str(profile.top_risk_factors):
                    mock_activity[-4:] = mock_activity[-4:] * 2 # surge at end
                    
                df_spark = pd.DataFrame({'Week': dates, 'Activity': mock_activity})
                st.bar_chart(df_spark.set_index('Week'))
                
    st.divider()
    
    st.header("🟡 Elevated Risk Watchlist")
    if not med_risk:
        st.write("No elevated risks detected.")
        
    for profile in med_risk:
        st.markdown(f"""
        <div class="risk-card-med">
            <b>{profile.user_name}</b> (Score: {profile.risk_score:.0f}) - {', '.join(profile.top_risk_factors)}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    st.header("🟢 Healthy Team Members")
    healthy = [p for p in risk_profiles if p.risk_level == 'Healthy']
    if healthy:
        st.dataframe(
            pd.DataFrame([{'Name': p.user_name, 'Risk Score': p.risk_score} for p in healthy]),
            width="stretch",
            hide_index=True
        )

if __name__ == "__main__":
    main()
