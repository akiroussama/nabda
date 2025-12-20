
import streamlit as st
import duckdb
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Jira AI Co-pilot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 🎨 PREMIUM CSS & ANIMATIONS (The "Wow" Factor)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Light Theme */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* PULSE ANIMATION */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); transform: scale(1); }
        70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); transform: scale(1.02); }
        100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); transform: scale(1); }
    }
    
    .pulse-container {
        animation: pulse-glow 2s infinite;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        border-radius: 16px;
        padding: 2px;
        margin-bottom: 2rem;
    }
    
    .pulse-content {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    
    /* HERO TEXT */
    .hero-text {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .hero-sub {
        font-size: 1.1rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* GLASSMORPHISM CARDS - LIGHT MODE */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, border-color 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .stat-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .stat-delta {
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }
    
    .visible-green { color: #16a34a; }
    .visible-red { color: #dc2626; }
    
    /* GAMIFICATION BAR */
    .xp-bar-bg {
        background-color: #e2e8f0;
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .xp-bar-fill {
        background: linear-gradient(90deg, #f59e0b 0%, #ea580c 100%);
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-out;
    }
    
    /* SPOTLIGHT AVATAR */
    .avatar-circle {
        width: 64px;
        height: 64px;
        background: linear-gradient(45deg, #f472b6, #db2777);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 0 auto 1rem auto;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* SPRINT STATS GRID */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 1.5rem;
    }
    
    .stat-mini-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .stat-mini-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-mini-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Oracle Widget */
    .oracle-widget {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }
    
    .oracle-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
    }
    
    .oracle-icon { font-size: 24px; }
    
    .oracle-title {
        color: #c7d2fe;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .oracle-insight {
        color: #fff;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.5;
        padding: 12px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 8px;
    }
    
    .oracle-meta {
        color: #a5b4fc;
        font-size: 11px;
        text-align: right;
    }
    
    /* Quick Actions */
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: all 0.2s;
    }
    
    .nav-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🔋 BACKEND LOGIC
# -----------------------------------------------------------------------------
def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)

def calculate_team_xp(conn):
    """Calculate 'Team Level' based on total story points completed."""
    try:
        res = conn.execute("SELECT COALESCE(SUM(story_points), 0) FROM issues WHERE status = 'Terminé(e)'").fetchone()
        total_points = res[0] if res and res[0] else 0
        
        level = max(1, int((total_points / 50) ** 0.5)) if total_points > 0 else 1
        current_level_base = (level ** 2) * 50
        next_level_base = ((level + 1) ** 2) * 50
        
        progress_points = total_points - current_level_base
        points_needed = next_level_base - current_level_base
        progress_pct = min(100, max(0, (progress_points / points_needed) * 100))
        
        titles = ["Rookies", "Git Guardians", "Code Ninjas", "Agile Warriors", "Product Visionaries", "Legends"]
        title = titles[min(len(titles)-1, (level-1)//3)]
        
        return {
            "level": level,
            "title": title,
            "progress": progress_pct,
            "total_points": total_points,
            "next_level_points": int(max(0, points_needed - progress_points))
        }
    except Exception:
        return {"level": 1, "title": "Rookies", "progress": 0, "total_points": 0, "next_level_points": 50}

def get_mvp_spotlight(conn):
    """Highlight the top contributor based on completed story points."""
    try:
        query = """
            SELECT assignee_name, COALESCE(SUM(story_points), 0) as score
            FROM issues 
            WHERE status = 'Terminé(e)'
            AND assignee_name IS NOT NULL AND assignee_name != ''
            GROUP BY assignee_name
            ORDER BY score DESC
            LIMIT 1
        """
        res = conn.execute(query).fetchone()
        if res and res[1] > 0:
            return {"name": res[0], "score": res[1]}
        return None
    except Exception:
        return None

def get_sprint_stats(conn):
    """Get current sprint statistics."""
    try:
        stats = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
                SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN priority IN ('Highest', 'High') AND status != 'Terminé(e)' THEN 1 ELSE 0 END) as high_priority,
                COALESCE(SUM(CASE WHEN status = 'Terminé(e)' THEN story_points END), 0) as pts_done,
                COALESCE(SUM(story_points), 0) as pts_total
            FROM issues i
            JOIN sprints s ON i.sprint_id = s.id AND s.state = 'active'
        """).fetchone()
        
        if stats and stats[0] > 0:
            return {
                'total': stats[0],
                'done': stats[1] or 0,
                'in_progress': stats[2] or 0,
                'high_priority': stats[3] or 0,
                'pts_done': stats[4] or 0,
                'pts_total': stats[5] or 0,
                'completion': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
            }
        return None
    except Exception:
        return None

def get_waiting_on_summary(conn):
    """Get waiting-on inbox summary for secondary surface widget."""
    try:
        result = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN expected_by < CURRENT_DATE THEN 1 ELSE 0 END) as overdue
            FROM waiting_on
            WHERE status IN ('active', 'acknowledged')
        """).fetchone()

        if result:
            return {
                'total': result[0] or 0,
                'overdue': result[1] or 0
            }
    except:
        pass
    return {'total': 0, 'overdue': 0}


def get_oracle_insights(conn):
    """Generate real data-driven insights."""
    insights = []
    
    try:
        # Insight 1: Sprint progress
        sprint_stats = get_sprint_stats(conn)
        if sprint_stats:
            completion = sprint_stats['completion']
            if completion >= 75:
                insights.append(f"🎯 Sprint is {completion:.0f}% complete. Strong momentum!")
            elif completion >= 50:
                insights.append(f"📊 Sprint at {completion:.0f}%. Keep pushing to hit goals.")
            else:
                insights.append(f"⚠️ Sprint only {completion:.0f}% done. Consider scope adjustment.")
        
        # Insight 2: Blockers
        blockers = conn.execute("""
            SELECT COUNT(*) FROM issues 
            WHERE priority = 'Highest' AND status != 'Terminé(e)'
        """).fetchone()[0]
        if blockers > 0:
            insights.append(f"🚨 {blockers} critical item(s) need immediate attention.")
        
        # Insight 3: Workload balance
        workload = conn.execute("""
            SELECT assignee_name, COUNT(*) as cnt 
            FROM issues 
            WHERE status = 'En cours' AND assignee_name IS NOT NULL
            GROUP BY assignee_name
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()
        if workload and workload[1] >= 5:
            insights.append(f"👤 {workload[0].split()[0]} has {workload[1]} items in progress.")
        
        # Insight 4: Velocity trend (simplified)
        completed_recently = conn.execute("""
            SELECT COUNT(*) FROM issues
            WHERE status = 'Terminé(e)'
            AND updated >= CURRENT_DATE - INTERVAL '7 days'
        """).fetchone()[0]
        if completed_recently > 0:
            insights.append(f"✅ {completed_recently} items completed in the last 7 days.")

        # Insight 5: Waiting-On overdue items
        try:
            waiting_overdue = conn.execute("""
                SELECT COUNT(*) FROM waiting_on
                WHERE status = 'active' AND expected_by < CURRENT_DATE
            """).fetchone()[0]
            if waiting_overdue > 0:
                insights.insert(0, f"📬 {waiting_overdue} waiting-on item(s) are overdue. Check your inbox!")
        except:
            pass
            
    except Exception:
        pass
    
    if not insights:
        insights.append("📈 All systems nominal. Keep up the great work!")
    
    return insights[:3]  # Return top 3 insights

def main():
    # 1. HEADER
    st.markdown('<div class="hero-text">Jira AI Co-pilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Mission Control Center</div>', unsafe_allow_html=True)
    
    conn = get_connection()
    
    if not conn:
        st.error("❌ Database not found. Please sync data first.")
        st.stop()
        
    # --- TOP BAR: GAMIFICATION ---
    xp_data = calculate_team_xp(conn)
    col_xp1, col_xp2 = st.columns([1, 4])
    with col_xp1:
        st.markdown(f"**Level {xp_data['level']}** — {xp_data['title']}")
    with col_xp2:
        st.markdown(f"""
        <div class="xp-bar-bg">
            <div class="xp-bar-fill" style="width: {xp_data['progress']}%;"></div>
        </div>
        <div style="text-align: right; font-size: 0.75rem; color: #64748b; margin-top: 4px;">
            {xp_data['next_level_points']} XP to next level • {xp_data['total_points']:.0f} total points
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- SPRINT STATS CARDS ---
    sprint_stats = get_sprint_stats(conn)
    if sprint_stats:
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-mini-card">
                <div class="stat-mini-value">{sprint_stats['completion']:.0f}%</div>
                <div class="stat-mini-label">Sprint Progress</div>
            </div>
            <div class="stat-mini-card">
                <div class="stat-mini-value">{sprint_stats['done']}/{sprint_stats['total']}</div>
                <div class="stat-mini-label">Items Done</div>
            </div>
            <div class="stat-mini-card">
                <div class="stat-mini-value">{sprint_stats['pts_done']:.0f}</div>
                <div class="stat-mini-label">Points Delivered</div>
            </div>
            <div class="stat-mini-card">
                <div class="stat-mini-value" style="background: linear-gradient(135deg, #dc2626 0%, #f87171 100%); -webkit-background-clip: text;">{sprint_stats['high_priority']}</div>
                <div class="stat-mini-label">High Priority</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- MAIN GRID ---
    col_main_l, col_main_r = st.columns([2, 1])
    
    with col_main_l:
        # NAVIGATION GRID
        st.markdown("### 🚀 Quick Access")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.page_link("pages/10_🌅_Good_Morning.py", label="Good Morning", icon="🌅", help="Daily Briefing")
            st.page_link("pages/0_🏆_Executive_Cockpit.py", label="Executive Cockpit", icon="🏆", help="Strategy Overview")
            
        with c2:
            st.page_link("pages/1_📊_Overview.py", label="Overview", icon="📊", help="Project Stats")
            st.page_link("pages/3_🏃_Sprint_Health.py", label="Sprint Health", icon="🏃", help="Live Risks")
            
        with c3:
            st.page_link("pages/4_👥_Team_Workload.py", label="Team Workload", icon="👥", help="Capacity Check")
            st.page_link("pages/9_🎲_Delivery_Forecast.py", label="Forecast", icon="🎲", help="Monte Carlo")

        # Waiting-On Inbox Widget
        waiting_data = get_waiting_on_summary(conn)
        if waiting_data['total'] > 0 or True:  # Always show for discoverability
            overdue_class = "overdue" if waiting_data['overdue'] > 0 else ""
            overdue_badge = f'<span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-left: 8px;">{waiting_data["overdue"]} overdue</span>' if waiting_data['overdue'] > 0 else ''

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0c4a6e 0%, #0284c7 100%); border-radius: 12px; padding: 16px 20px; margin: 16px 0; border: 1px solid rgba(56, 189, 248, 0.3);">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">📬</span>
                        <div>
                            <div style="color: white; font-weight: 600; font-size: 14px;">Waiting-On Inbox {overdue_badge}</div>
                            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">{waiting_data['total']} active items</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/16_📬_Waiting_On_Inbox.py", label="Open Inbox →", icon="📬")

        st.markdown("")
        
        # THE ORACLE - Real Insights
        st.markdown("### 🔮 The Oracle")
        insights = get_oracle_insights(conn)
        
        oracle_html = ""
        for insight in insights:
            oracle_html += f'<div class="oracle-insight">{insight}</div>'
        
        st.markdown(f"""
        <div class="oracle-widget">
            <div class="oracle-header">
                <span class="oracle-icon">🧠</span>
                <span class="oracle-title">AI-Powered Insights</span>
            </div>
            {oracle_html}
            <div class="oracle-meta">Updated just now</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_main_r:
        # MVP SPOTLIGHT
        st.markdown("### 🌟 Top Contributor")
        
        mvp = get_mvp_spotlight(conn)
        if mvp:
            initials = "".join([n[0] for n in mvp['name'].split()[:2]]).upper()
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="avatar-circle">{initials}</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #1e293b;">{mvp['name']}</div>
                <div style="color: #16a34a; font-weight: 600; font-size: 1.5rem;">{mvp['score']:.0f}</div>
                <div style="font-size: 0.8rem; color: #64748b;">Story Points Delivered</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div class="avatar-circle">🏆</div>
                <div style="font-size: 1.1rem; font-weight: bold; color: #1e293b;">Start Completing Tasks!</div>
                <div style="font-size: 0.8rem; color: #64748b;">MVP will appear here</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Team Overview
        st.markdown("### 👥 Team Activity")
        try:
            team_activity = conn.execute("""
                SELECT 
                    COALESCE(assignee_name, 'Unassigned') as name,
                    COUNT(CASE WHEN status = 'En cours' THEN 1 END) as wip,
                    COUNT(CASE WHEN status = 'Terminé(e)' THEN 1 END) as done
                FROM issues
                WHERE assignee_name IS NOT NULL AND assignee_name != ''
                GROUP BY assignee_name
                ORDER BY done DESC
                LIMIT 5
            """).fetchdf()
            
            if not team_activity.empty:
                for _, row in team_activity.iterrows():
                    initials = "".join([n[0].upper() for n in row['name'].split()[:2]])
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: white; border-radius: 8px; margin-bottom: 6px; border: 1px solid #e2e8f0;">
                        <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: 600;">{initials}</div>
                        <div style="flex: 1;">
                            <div style="font-size: 13px; font-weight: 600; color: #1e293b;">{row['name'].split()[0]}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #16a34a; font-size: 12px; font-weight: 600;">{row['done']} done</span>
                            <span style="color: #64748b; font-size: 11px;"> • {row['wip']} wip</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass

    # Footer
    st.markdown("---")
    st.markdown("<center style='color: #94a3b8; font-size: 0.8rem;'>Jira AI Co-pilot v1.0 • AI-Powered Project Intelligence</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
