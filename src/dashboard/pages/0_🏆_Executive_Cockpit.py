"""
The Holy Trinity™ - Executive Cockpit.
Aggregates Strategic Gap, Burnout Risk, and Delivery Forecast.
"""

import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, date, timedelta

from src.intelligence.classifier import WorkClassifier
from src.features.strategic_alignment import StrategicAlignmentAnalyzer
from src.features.burnout_models import BurnoutAnalyzer
from src.features.delivery_forecast import DeliveryForecaster

st.set_page_config(page_title="Executive Cockpit", page_icon="🏆", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 5px solid #3B82F6;
    }
    .kpi-title { font-size: 14px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
    .kpi-value { font-size: 36px; font-weight: 800; color: #1E293B; margin: 10px 0; }
    .kpi-context { font-size: 14px; color: #64748B; }
    
    .status-red { color: #DC2626; }
    .status-amber { color: #D97706; }
    .status-green { color: #059669; }
</style>
""", unsafe_allow_html=True)

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)

@st.cache_resource
def get_classifier():
    return WorkClassifier()

def main():
    st.title("🏆 Executive Cockpit")
    st.markdown("### Engineering Organization Health")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Sync data first.")
        st.stop()

    with st.spinner("Aggregating Intelligence from all modules..."):
        # 1. Fetch Data
        df_tickets = conn.execute("SELECT * FROM issues").fetchdf()
        df_status = conn.execute("SELECT * FROM issues WHERE status IN ('Done', 'Closed', 'Resolved')").fetchdf()
        df_users = conn.execute("SELECT * FROM users WHERE active = true").fetchdf()
        try:
            df_worklogs = conn.execute("SELECT * FROM worklogs").fetchdf()
        except:
            df_worklogs = pd.DataFrame()
        
        # 2. Compute Metrics
        
        # A. Strategy
        classifier = get_classifier()
        strat_analyzer = StrategicAlignmentAnalyzer(classifier)
        try:
            recent_tickets = df_tickets[pd.to_datetime(df_tickets['created'], utc=True) >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90))]
        except:
             recent_tickets = df_tickets # Fallback if timezone issues
             
        strat_res = strat_analyzer.calculate_alignment(
            recent_tickets, 
            {"New Value": 0.7, "Maintenance": 0.3}, # Simplified default
            team_size=len(df_users)
        )
        
        # B. Burnout
        burn_analyzer = BurnoutAnalyzer()
        burn_res = burn_analyzer.analyze_team_risks(df_tickets, df_worklogs, df_users)
        high_risk_count = len([p for p in burn_res if p.risk_level == 'High Risk'])
        
        # C. Delivery
        forecaster = DeliveryForecaster()
        del_params = forecaster.analyze_historical_performance(df_status, pd.DataFrame())
        # Simulate generic 100-item project due in 3 months
        del_res = forecaster.run_simulation(
            remaining_backlog_items=100, 
            target_date=datetime.now() + timedelta(days=90),
            historical_params=del_params
        )

    # 3. Dashboard
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #3B82F6">
            <div class="kpi-title">Strategic Drift</div>
            <div class="kpi-value status-{'green' if strat_res.drift_velocity < 0.1 else 'amber'}">
                ${strat_res.total_drift_cost/1000:.0f}k
            </div>
            <div class="kpi-context">Wasted Spend (Q4)</div>
            <hr>
            <div>Hidden Work: <b>{strat_res.shadow_work_percentage*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        color = "red" if high_risk_count > 0 else "green"
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #DC2626">
            <div class="kpi-title">Attrition Risk</div>
            <div class="kpi-value status-{color}">
                {high_risk_count}
            </div>
            <div class="kpi-context">High Risk Engineers</div>
            <hr>
            <div>Team Health Score: <b>{100 - (high_risk_count*10):.0f}/100</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        prob = del_res.target_date_prob * 100
        color = "green" if prob > 80 else "amber" if prob > 50 else "red"
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #10B981">
            <div class="kpi-title">Delivery Confidence</div>
            <div class="kpi-value status-{color}">
                {prob:.0f}%
            </div>
            <div class="kpi-context">On-Time Probability (90d)</div>
            <hr>
            <div>Bias Factor: <b>{del_params['estimation_bias_mean']:.1f}x</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Questions Answered
    st.markdown("### 💡 Strategic Insights")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    
    with col_q1:
        st.info(f"**Are we building the right things?**\n\nNo. We are spending ${strat_res.total_drift_cost:,.0f} on unplanned maintenance/firefighting, identifying {len(strat_res.top_shadow_tickets)} shadow tickets.")
    
    with col_q2:
        st.warning(f"**Are we burning out?**\n\nPossibly. {high_risk_count} key engineers are showing behavioral anomalies (surge in weekend work/hours).")

    with col_q3:
        p_safe = del_res.target_date_prob
        msg = "Unlikely." if p_safe < 0.5 else "Likely."
        st.success(f"**Will we ship on time?**\n\n{msg} Historical bias ({del_params['estimation_bias_mean']:.1f}x) suggests we need to pad estimates significantly.")

if __name__ == "__main__":
    main()
