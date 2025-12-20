"""
Delivery Forecast - Monte Carlo Simulation.

Probabilistic forecasting for project delivery dates.
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

from src.features.delivery_forecast import DeliveryForecaster

st.set_page_config(page_title="Delivery Forecast", page_icon="game_die", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .forecast-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .prob-high { color: #059669; }
    .prob-medium { color: #D97706; }
    .prob-low { color: #DC2626; }
</style>
""", unsafe_allow_html=True)


def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def main():
    st.title("Delivery Forecast")
    st.markdown("### Monte Carlo Simulation for Project Delivery")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Sync data first.")
        st.stop()

    # Get completed tickets for historical analysis
    try:
        df_completed = conn.execute("""
            SELECT * FROM issues
            WHERE resolved IS NOT NULL
        """).fetchdf()

        df_sprints = conn.execute("SELECT * FROM sprints").fetchdf()

        # Get all issues for backlog estimation
        df_all = conn.execute("""
            SELECT * FROM issues
        """).fetchdf()

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")

    # Calculate remaining backlog
    done_statuses = ['Done', 'Resolved', 'Closed', 'Terminé(e)', 'Complete']
    df_remaining = df_all[~df_all['status'].isin(done_statuses)]
    remaining_count = len(df_remaining)

    backlog_size = st.sidebar.slider(
        "Remaining Backlog Items",
        min_value=10,
        max_value=max(500, remaining_count * 2),
        value=max(remaining_count, 50),
        step=10,
    )

    target_weeks = st.sidebar.slider(
        "Target Weeks from Now",
        min_value=4,
        max_value=52,
        value=12,
        step=2,
    )

    target_date = datetime.now() + timedelta(weeks=target_weeks)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### What-If Scenarios")

    team_multiplier = st.sidebar.slider(
        "Team Size Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="1.0 = current team, 1.5 = 50% more capacity",
    )

    scope_cut = st.sidebar.slider(
        "Scope Cut (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Percentage of backlog to cut",
    )

    bias_fix = st.sidebar.slider(
        "Estimation Improvement",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = no change, 1 = perfect estimation",
    )

    n_simulations = st.sidebar.number_input(
        "Number of Simulations",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000,
    )

    # Run simulation
    forecaster = DeliveryForecaster()

    with st.spinner("Running Monte Carlo simulations..."):
        # Analyze historical data
        params = forecaster.analyze_historical_performance(df_completed, df_sprints)

        # Run baseline simulation
        baseline_result = forecaster.run_simulation(
            remaining_backlog_items=backlog_size,
            target_date=target_date,
            historical_params=params,
            n_simulations=n_simulations,
        )

        # Run what-if simulation
        whatif_result = forecaster.run_simulation(
            remaining_backlog_items=backlog_size,
            target_date=target_date,
            historical_params=params,
            n_simulations=n_simulations,
            team_size_multiplier=team_multiplier,
            scope_cut_percentage=scope_cut / 100,
            estimation_fix_factor=bias_fix,
        )

    # Display Results
    st.markdown("---")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        prob = baseline_result.target_date_prob * 100
        prob_class = "prob-high" if prob > 70 else "prob-medium" if prob > 40 else "prob-low"
        st.markdown(f"""
        <div class="forecast-card">
            <h4>On-Time Probability</h4>
            <h1 class="{prob_class}">{prob:.0f}%</h1>
            <small>Target: {target_date.strftime('%b %d, %Y')}</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="forecast-card">
            <h4>50% Likely (Optimistic)</h4>
            <h2>{baseline_result.p50_date.strftime('%b %d, %Y')}</h2>
            <small>Half of simulations complete by</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="forecast-card">
            <h4>85% Likely (Realistic)</h4>
            <h2>{baseline_result.p85_date.strftime('%b %d, %Y')}</h2>
            <small>Conservative estimate</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="forecast-card">
            <h4>95% Likely (Safe)</h4>
            <h2>{baseline_result.p95_date.strftime('%b %d, %Y')}</h2>
            <small>Near-certain completion</small>
        </div>
        """, unsafe_allow_html=True)

    # What-If Comparison
    if team_multiplier != 1.0 or scope_cut > 0 or bias_fix > 0:
        st.markdown("### What-If Scenario Impact")

        col1, col2 = st.columns(2)

        with col1:
            delta_prob = (whatif_result.target_date_prob - baseline_result.target_date_prob) * 100
            st.metric(
                "On-Time Probability Change",
                f"{whatif_result.target_date_prob * 100:.0f}%",
                f"{delta_prob:+.0f}%",
            )

        with col2:
            days_saved = (baseline_result.p85_date - whatif_result.p85_date).days
            st.metric(
                "Days Saved (85% estimate)",
                f"{days_saved:+d} days",
            )

    # Distribution Histogram
    st.markdown("### Simulation Distribution")

    # Convert to DataFrame for plotting
    sim_dates = pd.DataFrame({
        'Completion Date': baseline_result.simulation_dates,
        'Scenario': 'Baseline'
    })

    if team_multiplier != 1.0 or scope_cut > 0 or bias_fix > 0:
        whatif_dates = pd.DataFrame({
            'Completion Date': whatif_result.simulation_dates,
            'Scenario': 'What-If'
        })
        sim_dates = pd.concat([sim_dates, whatif_dates])

    fig = px.histogram(
        sim_dates,
        x='Completion Date',
        color='Scenario',
        nbins=50,
        barmode='overlay',
        opacity=0.7,
        title="Distribution of Simulated Completion Dates",
    )

    # Add target date line
    fig.add_vline(
        x=target_date.timestamp() * 1000,  # Plotly uses ms
        line_dash="dash",
        line_color="red",
        annotation_text="Target",
    )

    fig.update_layout(
        xaxis_title="Completion Date",
        yaxis_title="Number of Simulations",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

    # Risk Factors
    st.markdown("### Risk Factors (from Historical Data)")

    col1, col2, col3 = st.columns(3)

    with col1:
        bias = params.get('estimation_bias_mean', 1.0)
        bias_color = "green" if bias < 1.2 else "orange" if bias < 1.5 else "red"
        st.markdown(f"**Estimation Bias:** :{bias_color}[{bias:.1f}x]")
        st.caption("Actual vs Estimated effort ratio")

    with col2:
        vel_var = params.get('velocity_std', 0) / max(params.get('velocity_mean', 1), 0.1) * 100
        var_color = "green" if vel_var < 20 else "orange" if vel_var < 40 else "red"
        st.markdown(f"**Velocity Variance:** :{var_color}[±{vel_var:.0f}%]")
        st.caption("Throughput consistency")

    with col3:
        creep = params.get('scope_creep_mean', 0) * 100
        creep_color = "green" if creep < 10 else "orange" if creep < 20 else "red"
        st.markdown(f"**Scope Creep Rate:** :{creep_color}[{creep:.0f}%]")
        st.caption("Work added mid-project")

    # Model Assumptions
    with st.expander("Model Assumptions"):
        st.markdown("""
        **Monte Carlo Simulation Parameters:**

        - **Velocity Distribution**: Normal distribution based on historical weekly throughput
        - **Estimation Bias**: Ratio of actual vs estimated effort from past tickets
        - **Scope Creep**: Expected percentage of work added during project

        **Limitations:**
        - Assumes team composition remains constant
        - Does not account for holidays or PTO
        - Past performance may not predict future results

        **Recommendations:**
        - Use 85% probability date for commitments
        - Re-run simulation weekly as data changes
        - Consider what-if scenarios for risk mitigation
        """)

    # Summary Stats
    st.markdown("### Analysis Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backlog Analysis:**")
        st.write(f"- Total Issues: {len(df_all)}")
        st.write(f"- Completed: {len(df_completed)}")
        st.write(f"- Remaining: {remaining_count}")
        st.write(f"- Simulated Backlog: {backlog_size}")

    with col2:
        st.markdown("**Historical Performance:**")
        st.write(f"- Velocity: {params['velocity_mean']:.1f} items/week (±{params['velocity_std']:.1f})")
        st.write(f"- Estimation Bias: {params['estimation_bias_mean']:.1f}x")
        st.write(f"- Scope Creep: {params['scope_creep_mean']*100:.0f}%")


if __name__ == "__main__":
    main()
