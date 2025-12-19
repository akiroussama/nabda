"""
Feature #1: The Strategic Execution Gap™
Dashboard to compare Stated Strategy vs Actual Execution.
"""

import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.intelligence.classifier import WorkClassifier, ALL_CATEGORIES
from src.features.strategic_alignment import StrategicAlignmentAnalyzer
from config.settings import get_settings

st.set_page_config(page_title="Strategic Execution Gap", page_icon="🎯", layout="wide")

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: 600;
        color: #1E293B;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748B;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 28px;
        font-weight: 700;
    }
    .alert-warning {
        color: #D97706;
        font-weight: 600;
    }
    .alert-danger {
        color: #DC2626;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_classifier():
    return WorkClassifier() # It loads the model, which is cached by the class/library usually, but good to cache here too

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)

def main():
    st.title("🎯 Strategic Execution Gap™")
    st.markdown("### The Blind Spot: Stated Strategy vs. Actual Execution")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please run 'jira-copilot sync full' first.")
        st.stop()

    # 1. Controls: Stated Strategy
    with st.expander("🛠️ Configure Stated Strategy", expanded=True):
        st.info("Define your target investment allocation across categories.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_new_value = st.slider("New Features / Value", 0, 100, 70, 5)
        with col2:
            target_platform = st.slider("Platform / Maintenance", 0, 100, 20, 5)
        with col3:
            target_tech_debt = st.slider("Tech Debt", 0, 100, 10, 5)
            
        # Normalize if necessary or just alert
        total = target_new_value + target_platform + target_tech_debt
        if total != 100:
            st.warning(f"Total allocation is {total}%. It is recommended to sum to 100%.")

        stated_strategy = {
            "New Value": target_new_value / 100.0,
            "Maintenance": target_platform / 100.0, # We map "Platform" to "Maintenance" broadly
            "Tech Debt": target_tech_debt / 100.0,
            "Firefighting": 0.0, # Usually you don't validly "plan" for firefighting unless purely reactive team
            "Dependency/Blocked": 0.0,
            "Rework": 0.0
        }

    # 2. Data Loading & Analysis
    with st.spinner("Analyzing Work Patterns... (Extracting Embeddings & Classifying)"):
        # Fetch tickets from last 90 days (Quarterly view)
        query = """
            SELECT * 
            FROM issues 
            WHERE created >= CURRENT_DATE - INTERVAL 90 DAY
        """
        df_tickets = conn.execute(query).fetchdf()
        
        if df_tickets.empty:
            st.warning("No tickets found in the last 90 days.")
            st.stop()

        classifier = get_classifier()
        analyzer = StrategicAlignmentAnalyzer(classifier)
        
        # Run Analysis
        result = analyzer.calculate_alignment(
            df_tickets, 
            stated_strategy,
            avg_cost_per_engineer_year=185000, # Mock or Config
            team_size=20 # Mock or Count from DB
        )

    # 3. Dashboard Layout
    
    # -- Top Metrics --
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">STRATEGIC DRIFT COST (Q4)</div>
            <div class="metric-value">${result.total_drift_cost:,.0f}</div>
            <div class="metric-delta">💸 Unintended Spend</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">HIDDEN WORK</div>
            <div class="metric-value">{result.shadow_work_percentage*100:.1f}%</div>
            <div class="metric-delta">🚨 Mislabeled / Unplanned</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        velocity_color = "red" if result.drift_velocity > 0 else "green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DRIFT VELOCITY</div>
            <div class="metric-value" style="color: {velocity_color}">+{result.drift_velocity*100:.1f}%</div>
            <div class="metric-delta">Month-over-Month</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # -- Visualization: Gap Chart --
    st.subheader("📊 Stated vs. Actual Allocation")
    
    col_chart, col_details = st.columns([2, 1])
    
    with col_chart:
        # Prepare data for plotting
        categories = list(stated_strategy.keys())
        # Add others found
        for cat in result.allocation_actual.keys():
            if cat not in categories:
                categories.append(cat)
        
        data = []
        for cat in categories:
            stated = result.allocation_stated.get(cat, 0.0) * 100
            actual = result.allocation_actual.get(cat, 0.0) * 100
            data.append({"Category": cat, "Type": "Stated Strategy", "Percentage": stated})
            data.append({"Category": cat, "Type": "Actual Execution", "Percentage": actual})
            
        df_chart = pd.DataFrame(data)
        
        fig = px.bar(
            df_chart, 
            x="Percentage", 
            y="Category", 
            color="Type", 
            barmode="group",
            orientation='h',
            color_discrete_map={"Stated Strategy": "#94A3B8", "Actual Execution": "#3B82F6"},
            height=400
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.markdown("#### Gap Analysis")
        for cat in ["New Value", "Maintenance", "Tech Debt", "Firefighting", "Dependency/Blocked"]:
            formatted_cat = cat
            gap_data = result.gap_breakdown.get(cat, {})
            delta = gap_data.get('delta', 0.0)
            cost = gap_data.get('cost', 0.0)
            
            if abs(delta) < 0.01:
                continue
                
            icon = "⚠️" if (cat != "New Value" and delta > 0.05) or (cat == "New Value" and delta < -0.05) else ""
            color = "red" if (cat != "New Value" and delta > 0) or (cat == "New Value" and delta < 0) else "green"
            
            st.markdown(f"""
            **{cat}** {icon}  
            Gap: <span style='color:{color}'>{delta*100:+.1f}%</span>  
            Cost: <span style='color:{color}'>${cost:,.0f}</span>
            """, unsafe_allow_html=True)
            st.divider()

    # -- Shadow Work --
    st.subheader("🕵️ Shadow Work Detection")
    st.markdown("These tickets were labeled as **Features/Stories** but semantically match **Maintenance/Firefighting**.")
    
    if not result.top_shadow_tickets.empty:
        st.dataframe(
            result.top_shadow_tickets[[
                'key', 'summary', 'status', 'predicted_category', 'classification_confidence'
            ]].rename(columns={
                'predicted_category': 'Actual Work Type',
                'classification_confidence': 'Confidence'
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No significant shadow work detected! (Or strict filtering applied)")

if __name__ == "__main__":
    main()
