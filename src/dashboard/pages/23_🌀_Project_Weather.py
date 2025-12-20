"""
Project Weather System - Climate Intelligence for Work

The Paradigm Shift:
- Traditional dashboards show the STATE of work
- This shows the CLIMATE of work

You don't read data. You FEEL the weather.
You see storms coming. You prepare. You protect.
And when the sun comes out, you know it's real.

This widget doesn't help you manage work.
It gives you INTUITION at a glance.
It makes you FEEL your project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import sys

# Path setup
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

# Import weather engine
from src.features.weather_engine import (
    WeatherEngine, WeatherZone, Storm, Front, Forecast,
    WeatherCondition, StormSeverity, PressureLevel, FrontType,
    create_demo_weather_engine
)

# Import page guide component
from src.dashboard.components import render_page_guide

# Page configuration
st.set_page_config(
    page_title="Project Weather",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS - Weather System Theme (Light Mode)
st.markdown("""
<style>
    /* Base theme - Light atmospheric */
    .stApp {
        background: #f8f9fa;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Weather Header */
    .weather-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #8b5cf6 100%);
        border: none;
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .weather-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, rgba(255,255,255,0.3), rgba(255,255,255,0.6), rgba(255,255,255,0.3));
    }

    .weather-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .weather-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.85);
    }

    /* Overall Weather Display */
    .overall-weather {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-top: 1.5rem;
    }

    .weather-icon-large {
        font-size: 4rem;
        filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.5));
    }

    .weather-stats {
        display: flex;
        gap: 2rem;
    }

    .weather-stat {
        text-align: center;
    }

    .weather-stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }

    .weather-stat-value.alert { color: #fecaca; }
    .weather-stat-value.warning { color: #fde68a; }
    .weather-stat-value.good { color: #bbf7d0; }

    .weather-stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255, 255, 255, 0.7);
    }

    /* Zone Weather Cards */
    .zone-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .zone-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }

    .zone-card.storm {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        animation: storm-pulse 2s infinite;
    }

    @keyframes storm-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 20px 5px rgba(239, 68, 68, 0.15); }
    }

    .zone-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }

    .zone-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
    }

    .zone-weather-icon {
        font-size: 1.75rem;
    }

    .zone-condition {
        font-size: 0.85rem;
        color: #64748b;
    }

    .zone-momentum {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.75rem;
        font-size: 0.8rem;
    }

    .momentum-label {
        color: #64748b;
    }

    .momentum-value {
        font-weight: 600;
        color: #1e293b;
    }

    .momentum-value.high { color: #16a34a; }
    .momentum-value.low { color: #dc2626; }

    /* Storm Panel */
    .storm-panel {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.1);
    }

    .storm-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444, #f97316, #ef4444);
        animation: storm-bar 2s linear infinite;
    }

    @keyframes storm-bar {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    .storm-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }

    .storm-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .storm-icon {
        font-size: 1.5rem;
        animation: spin 3s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .storm-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #991b1b;
    }

    .storm-severity {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
        border: 1px solid rgba(220, 38, 38, 0.3);
    }

    .storm-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .storm-metric {
        text-align: center;
        padding: 0.75rem;
        background: rgba(220, 38, 38, 0.08);
        border-radius: 8px;
    }

    .storm-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #991b1b;
    }

    .storm-metric-label {
        font-size: 0.7rem;
        color: #b91c1c;
        text-transform: uppercase;
    }

    .storm-forecast {
        background: rgba(220, 38, 38, 0.08);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .storm-forecast-title {
        font-size: 0.75rem;
        color: #d97706;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .storm-forecast-text {
        font-size: 0.9rem;
        color: #b91c1c;
    }

    .storm-actions {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .storm-action {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #fecaca;
        background: white;
        color: #991b1b;
    }

    .storm-action:hover {
        background: #fef2f2;
        border-color: #f87171;
    }

    .storm-action.primary {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-color: transparent;
        color: white;
    }

    /* Front Warning */
    .front-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(217, 119, 6, 0.08);
    }

    .front-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .front-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #b45309;
    }

    .front-countdown {
        font-size: 0.8rem;
        color: #d97706;
        font-weight: 600;
    }

    .front-description {
        font-size: 0.85rem;
        color: #92400e;
    }

    /* Pressure Map */
    .pressure-map-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .pressure-map-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .pressure-legend {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
        font-size: 0.75rem;
        color: #64748b;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 2px;
    }

    /* Forecast Panel */
    .forecast-panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .forecast-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }

    .forecast-day {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid #e5e7eb;
    }

    .forecast-day:last-child {
        border-bottom: none;
    }

    .forecast-day-label {
        width: 80px;
        font-size: 0.85rem;
        color: #64748b;
    }

    .forecast-day-weather {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .forecast-day-icon {
        font-size: 1.5rem;
    }

    .forecast-day-condition {
        font-size: 0.85rem;
        color: #1e293b;
    }

    .forecast-risk {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .forecast-risk-title {
        font-size: 0.75rem;
        color: #dc2626;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .forecast-risk-text {
        font-size: 0.9rem;
        color: #b91c1c;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .section-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        background: rgba(59, 130, 246, 0.1);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* Alerts */
    .alert-badge {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        z-index: 1000;
        animation: alert-bounce 1s infinite;
    }

    @keyframes alert-bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 0.25rem;
        gap: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: white;
    }

    .stTabs [aria-selected="true"] {
        color: #1e293b !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    /* Impact if unresolved box */
    .impact-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .impact-title {
        font-size: 0.75rem;
        color: #dc2626;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .impact-list {
        font-size: 0.85rem;
        color: #b91c1c;
    }

    .impact-list li {
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE & INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state with weather engine."""
    if 'weather_engine' not in st.session_state:
        st.session_state.weather_engine = create_demo_weather_engine()

    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()


def get_engine() -> WeatherEngine:
    """Get weather engine from session state."""
    return st.session_state.weather_engine


# =============================================================================
# HEADER COMPONENT
# =============================================================================

def render_weather_header():
    """Render the main weather system header."""
    engine = get_engine()
    summary = engine.get_system_summary()

    st.markdown(f"""<div class="weather-header">
<div class="weather-title">
    <span>🌀</span>
    <span>Project Weather System</span>
</div>
<div class="weather-subtitle">
    You don't read data. You feel the weather. You see the storms coming.
</div>
<div class="overall-weather">
    <div class="weather-icon-large">{summary['overall_emoji']}</div>
    <div class="weather-stats">
        <div class="weather-stat">
            <div class="weather-stat-value {'alert' if summary['active_storms'] > 0 else 'good'}">{summary['active_storms']}</div>
            <div class="weather-stat-label">Active Storms</div>
        </div>
        <div class="weather-stat">
            <div class="weather-stat-value {'warning' if summary['approaching_fronts'] > 0 else 'good'}">{summary['approaching_fronts']}</div>
            <div class="weather-stat-label">Fronts Approaching</div>
        </div>
        <div class="weather-stat">
            <div class="weather-stat-value {'alert' if summary['total_blocked_people'] > 0 else 'good'}">{summary['total_blocked_people']}</div>
            <div class="weather-stat-label">People Blocked</div>
        </div>
        <div class="weather-stat">
            <div class="weather-stat-value good">{summary['zones_clear']}</div>
            <div class="weather-stat-label">Clear Zones</div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# ZONE WEATHER DISPLAY
# =============================================================================

def render_zone_weather():
    """Render weather for each zone/team."""
    engine = get_engine()

    st.markdown("""<div class="section-header">
<div class="section-icon">🗺️</div>
<div class="section-title">Zone Weather</div>
</div>""", unsafe_allow_html=True)

    # Create columns for zones
    zones = list(engine.zones.values())
    cols = st.columns(3)

    for idx, zone in enumerate(zones):
        with cols[idx % 3]:
            storm_class = "storm" if zone.condition == WeatherCondition.STORM else ""
            momentum_class = "high" if zone.velocity_ratio >= 1.0 else "low" if zone.velocity_ratio < 0.7 else ""

            factors_html = ""
            if zone.factors:
                factors_html = "<br>".join([f"• {f}" for f in zone.factors[:2]])

            st.markdown(f"""<div class="zone-card {storm_class}">
<div class="zone-header">
    <div class="zone-name">{zone.name}</div>
    <div class="zone-weather-icon">{zone.get_emoji()}</div>
</div>
<div class="zone-condition">{zone.get_description()}</div>
<div class="zone-momentum">
    <span class="momentum-label">Momentum:</span>
    <span class="momentum-value {momentum_class}">{'HIGH' if zone.velocity_ratio >= 1.0 else 'LOW' if zone.velocity_ratio < 0.7 else 'NORMAL'}</span>
</div>
{f'<div style="margin-top: 0.75rem; font-size: 0.8rem; color: #64748b;">{factors_html}</div>' if factors_html else ''}
</div>""", unsafe_allow_html=True)


# =============================================================================
# ACTIVE STORMS
# =============================================================================

def render_active_storms():
    """Render active storm panel."""
    engine = get_engine()
    storms = list(engine.storms.values())

    if not storms:
        st.markdown("""<div style="text-align: center; padding: 2rem; color: #16a34a;">
<div style="font-size: 3rem; margin-bottom: 0.5rem;">☀️</div>
<div style="font-size: 1.1rem; font-weight: 600;">No Active Storms</div>
<div style="font-size: 0.9rem; color: #64748b;">All clear across all zones</div>
</div>""", unsafe_allow_html=True)
        return

    for storm in storms:
        actions_html = "".join([
            f'<div class="storm-action {"primary" if i == 0 else ""}">'
            f'{action["emoji"]} {action["label"]}'
            f'</div>'
            for i, action in enumerate(storm.recommended_actions[:4])
        ])

        spread_text = ""
        if storm.spreading:
            spread_zones = ", ".join([engine.zones[z].name for z in storm.spread_forecast if z in engine.zones])
            spread_text = f"Spreads to {spread_zones} in ~{storm.time_to_spread_hours:.0f} hours"

        st.markdown(f"""<div class="storm-panel">
<div class="storm-header">
    <div class="storm-title">
        <span class="storm-icon">🌀</span>
        <span class="storm-name">{storm.name}</span>
    </div>
    <span class="storm-severity">{storm.severity.name}</span>
</div>
<div class="storm-metrics">
    <div class="storm-metric">
        <div class="storm-metric-value">{len(storm.affected_people)}</div>
        <div class="storm-metric-label">People Blocked</div>
    </div>
    <div class="storm-metric">
        <div class="storm-metric-value">{storm.duration_hours:.1f}h</div>
        <div class="storm-metric-label">Duration</div>
    </div>
    <div class="storm-metric">
        <div class="storm-metric-value">{len(storm.affected_tasks)}</div>
        <div class="storm-metric-label">Tasks Affected</div>
    </div>
</div>
<div style="margin-bottom: 1rem;">
    <div style="font-size: 0.75rem; color: #d97706; text-transform: uppercase; margin-bottom: 0.25rem;">ROOT CAUSE</div>
    <div style="font-size: 0.9rem; color: #991b1b;">{storm.root_cause}</div>
</div>
{f'<div class="storm-forecast"><div class="storm-forecast-title">⚠️ FORECAST</div><div class="storm-forecast-text">{spread_text}</div></div>' if spread_text else ''}
<div style="font-size: 0.75rem; color: #b91c1c; margin-bottom: 0.5rem;">ACTIONS</div>
<div class="storm-actions">{actions_html}</div>
<div class="impact-box">
    <div class="impact-title">Impact if unresolved by 4PM</div>
    <ul class="impact-list">
        <li>Frontend sprint at risk</li>
        <li>Demo to client Thursday in jeopardy</li>
        <li>Team morale impact (3rd storm this week)</li>
    </ul>
</div>
</div>""", unsafe_allow_html=True)

        # Interactive buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🔥 Escalate", key=f"escalate_{storm.id}", use_container_width=True):
                engine.resolve_storm(storm.id, "escalated")
                st.success("Escalated to engineering lead")
                st.rerun()
        with col2:
            if st.button("🔄 Reroute", key=f"reroute_{storm.id}", use_container_width=True):
                st.info("Rerouting blocked resources...")
        with col3:
            if st.button("⏸️ Shelter", key=f"shelter_{storm.id}", use_container_width=True):
                st.info("Sheltering affected work...")
        with col4:
            if st.button("✅ Resolved", key=f"resolve_{storm.id}", use_container_width=True):
                engine.resolve_storm(storm.id, "resolved")
                st.success("Storm resolved!")
                st.rerun()


# =============================================================================
# PRESSURE MAP
# =============================================================================

def render_pressure_map():
    """Render the pressure map visualization."""
    engine = get_engine()
    pressure_data = engine.get_pressure_map_data()

    st.markdown("""<div class="pressure-map-container">
<div class="pressure-map-title">🗺️ PRESSURE MAP <span style="font-size: 0.75rem; color: #64748b; margin-left: 0.5rem;">Hot zones = overcommitment</span></div>
</div>""", unsafe_allow_html=True)

    # Create heatmap
    z_values = [[cell['pressure'] for cell in row] for row in pressure_data]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        colorscale=[
            [0, '#22c55e'],      # Green - low pressure
            [0.3, '#84cc16'],    # Lime
            [0.5, '#f59e0b'],    # Amber
            [0.7, '#f97316'],    # Orange
            [1.0, '#ef4444'],    # Red - high pressure
        ],
        showscale=False,
        hovertemplate="Pressure: %{z:.0%}<extra></extra>"
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""<div class="pressure-legend">
<div class="legend-item"><div class="legend-color" style="background: #22c55e;"></div><span>Normal</span></div>
<div class="legend-item"><div class="legend-color" style="background: #f59e0b;"></div><span>High</span></div>
<div class="legend-item"><div class="legend-color" style="background: #ef4444;"></div><span>Critical</span></div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# APPROACHING FRONTS
# =============================================================================

def render_approaching_fronts():
    """Render approaching front warnings."""
    engine = get_engine()
    fronts = list(engine.fronts.values())

    if not fronts:
        st.info("No approaching fronts detected")
        return

    st.markdown("""<div class="section-header">
<div class="section-icon">⚠️</div>
<div class="section-title">Approaching Fronts (Collision Warnings)</div>
</div>""", unsafe_allow_html=True)

    for front in fronts:
        urgency_color = front.get_urgency_color()

        st.markdown(f"""<div class="front-warning">
<div class="front-header">
    <div class="front-title"><span>⚠️</span><span>{" + ".join(front.colliding_elements)}</span></div>
    <div class="front-countdown" style="color: {urgency_color};">{front.days_until_collision} days away</div>
</div>
<div class="front-description">{front.impact_description}</div>
</div>""", unsafe_allow_html=True)

        # Prevention actions
        with st.expander(f"Prevention Actions for {front.description[:30]}..."):
            for action in front.prevention_actions:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{action['label']}**")
                    st.caption(action['description'])
                with col2:
                    if st.button("Take Action", key=f"front_action_{front.id}_{action['id']}"):
                        st.success(f"Action '{action['label']}' initiated")


# =============================================================================
# 48-HOUR FORECAST
# =============================================================================

def render_forecast():
    """Render the 48-hour forecast panel."""
    engine = get_engine()
    forecast = engine.get_48hour_forecast()

    st.markdown(f"""<div class="forecast-panel">
<div class="forecast-title">📅 48-HOUR FORECAST</div>
<div class="forecast-day">
    <div class="forecast-day-label">Today</div>
    <div class="forecast-day-weather">
        <span class="forecast-day-icon">{forecast['today']['emoji']}</span>
        <span class="forecast-day-condition">{forecast['today']['summary']}</span>
    </div>
</div>
<div class="forecast-day">
    <div class="forecast-day-label">Tomorrow</div>
    <div class="forecast-day-weather">
        <span class="forecast-day-icon">{forecast['tomorrow']['emoji']}</span>
        <span class="forecast-day-condition">{forecast['tomorrow']['summary']}</span>
    </div>
</div>
<div class="forecast-risk">
    <div class="forecast-risk-title">⚠️ RISK ASSESSMENT</div>
    <div class="forecast-risk-text">{int(forecast['delay_probability'] * 100)}% chance of delay if {forecast['delay_trigger']}</div>
</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# WEATHER TIMELINE CHART
# =============================================================================

def render_weather_timeline():
    """Render the weather conditions timeline chart."""
    engine = get_engine()

    # Create timeline data for all zones
    zones = list(engine.zones.values())

    fig = go.Figure()

    # Add a trace for each zone
    colors = {
        "clear": "#22c55e",
        "partly_cloudy": "#84cc16",
        "cloudy": "#f59e0b",
        "rain": "#f97316",
        "storm": "#ef4444",
        "tornado": "#991b1b",
    }

    for idx, zone in enumerate(zones):
        # Generate fake historical data
        hours = list(range(-24, 1))
        conditions = []
        current_level = ["clear", "partly_cloudy", "cloudy", "rain", "storm"].index(zone.condition.value) if zone.condition.value != "tornado" else 4

        for h in hours:
            if h == 0:
                conditions.append(current_level)
            else:
                # Random walk toward current
                prev = conditions[-1] if conditions else current_level
                delta = 1 if prev < current_level else -1 if prev > current_level else 0
                new_level = prev + delta + np.random.randint(-1, 2)
                conditions.append(max(0, min(4, new_level)))

        conditions.reverse()

        fig.add_trace(go.Scatter(
            x=hours,
            y=[c + idx * 0.1 for c in conditions],
            mode='lines',
            name=zone.name,
            line=dict(width=2),
            hovertemplate=f"{zone.name}: %{{y:.0f}}<extra></extra>"
        ))

    fig.update_layout(
        height=250,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Hours",
            color='#64748b',
            gridcolor='rgba(148, 163, 184, 0.1)',
            zerolinecolor='rgba(148, 163, 184, 0.3)'
        ),
        yaxis=dict(
            title="Severity",
            color='#64748b',
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["☀️", "🌤️", "⛅", "🌧️", "⛈️"]
        ),
        legend=dict(
            orientation='h',
            y=1.15,
            font=dict(color='#94a3b8')
        ),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()

    # Render page guide in sidebar
    render_page_guide()

    # Simulate weather tick for real-time feel
    engine = get_engine()
    engine.simulate_tick()

    # Header
    render_weather_header()

    # Main content
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Active Storms (if any)
        if engine.storms:
            st.markdown("### 🔴 Active Storm Alert")
            render_active_storms()

        # Zone Weather
        render_zone_weather()

        # Weather Timeline
        st.markdown("### 📈 Weather History (24h)")
        render_weather_timeline()

    with col_right:
        # Pressure Map
        render_pressure_map()

        # 48-Hour Forecast
        render_forecast()

        # Approaching Fronts
        render_approaching_fronts()

    # Sidebar
    with st.sidebar:
        st.markdown("## Weather Controls")

        # Auto-refresh
        auto_refresh = st.checkbox("⚡ Live Updates", value=False)
        if auto_refresh:
            import time
            time.sleep(2)
            st.rerun()

        if st.button("🔄 Refresh Weather", use_container_width=True):
            st.rerun()

        st.divider()

        # Alerts
        st.markdown("### 🔔 Alerts")
        for alert in engine.alerts:
            if not alert.acknowledged:
                severity_color = "#dc2626" if alert.severity == "critical" else "#d97706"
                bg_color = "#fef2f2" if alert.severity == "critical" else "#fffbeb"
                st.markdown(f"""<div style="background: {bg_color}; border-left: 3px solid {severity_color}; padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 4px;">
<div style="font-weight: 600; color: #1e293b; font-size: 0.85rem;">{alert.title}</div>
<div style="font-size: 0.75rem; color: #64748b;">{alert.message}</div>
</div>""", unsafe_allow_html=True)
                if st.button("Acknowledge", key=f"ack_{alert.id}"):
                    engine.acknowledge_alert(alert.id)
                    st.rerun()

        st.divider()

        st.markdown("### The Vision")
        st.caption(
            '"You don\'t manage projects. You read the weather. '
            'You see the storms coming. You prepare. You protect. '
            'And when the sun comes out, you know it\'s real."'
        )


if __name__ == "__main__":
    main()
