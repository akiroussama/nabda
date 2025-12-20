"""
Page Guide Component - Universal Documentation Panel

A collapsible right-side panel that displays contextual documentation
for each page, including goals, added value, and how it helps team leaders.

Usage:
    from src.dashboard.components import render_page_guide
    render_page_guide()  # Auto-detects current page

The component auto-refreshes when navigating between pages.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import re


# =============================================================================
# PAGE METADATA STRUCTURE
# =============================================================================

@dataclass
class PageMetadata:
    """Metadata for a dashboard page."""
    id: str                          # Unique identifier (filename without number/emoji)
    title: str                       # Display title
    emoji: str                       # Page emoji
    tagline: str                     # One-line description
    goal: str                        # What this page aims to achieve
    added_value: List[str]           # Key value propositions
    helps_with: List[str]            # How it helps team leaders
    key_features: List[str]          # Main features
    pro_tips: List[str]              # Tips for getting the most out of it
    metrics_tracked: List[str] = field(default_factory=list)  # Metrics shown
    version: str = "1.0"             # Feature version
    new_features: List[str] = field(default_factory=list)     # Recent additions


# =============================================================================
# PAGE REGISTRY - All Page Metadata
# =============================================================================

PAGE_REGISTRY: Dict[str, PageMetadata] = {

    # -------------------------------------------------------------------------
    # EXECUTIVE & OVERVIEW PAGES
    # -------------------------------------------------------------------------

    "Executive_Cockpit": PageMetadata(
        id="Executive_Cockpit",
        title="Executive Cockpit",
        emoji="🏆",
        tagline="Your command center for strategic oversight",
        goal="Provide executives with a single-glance view of portfolio health, team performance, and strategic alignment across all projects.",
        added_value=[
            "Eliminates the need to check multiple dashboards",
            "Surfaces critical issues before they become emergencies",
            "Tracks ROI and strategic goal alignment automatically"
        ],
        helps_with=[
            "Board meeting preparation in minutes, not hours",
            "Quick daily health checks across all teams",
            "Identifying which projects need executive attention",
            "Understanding resource allocation at a glance"
        ],
        key_features=[
            "Portfolio health matrix with traffic light indicators",
            "Team velocity trends and comparisons",
            "Strategic goal tracking with progress bars",
            "Risk radar showing top concerns"
        ],
        pro_tips=[
            "Check first thing Monday morning for the week's priorities",
            "Use before stakeholder meetings for instant context",
            "Set up alerts for critical threshold breaches"
        ],
        version="2.0",
        new_features=["AI-powered risk predictions", "Customizable KPI widgets"]
    ),

    "Overview": PageMetadata(
        id="Overview",
        title="Overview",
        emoji="📊",
        tagline="The pulse of your project at a glance",
        goal="Provide a comprehensive snapshot of current project status, recent activity, and key metrics.",
        added_value=[
            "One page replaces 10 different reports",
            "Real-time data, not stale weekly updates",
            "Highlights what changed since your last visit"
        ],
        helps_with=[
            "Morning standup preparation",
            "Quick status updates to stakeholders",
            "Identifying trends before they become problems",
            "Understanding team momentum"
        ],
        key_features=[
            "Sprint progress with burndown visualization",
            "Recent activity feed with key highlights",
            "Velocity trends and predictions",
            "Quick links to problem areas"
        ],
        pro_tips=[
            "Compare 'last visit' vs 'now' for instant delta awareness",
            "Use the trend arrows to spot momentum shifts early"
        ],
        version="1.5"
    ),

    "Board": PageMetadata(
        id="Board",
        title="Board",
        emoji="📋",
        tagline="Your Kanban board, supercharged with intelligence",
        goal="Visualize work flow with AI-enhanced insights on bottlenecks, WIP limits, and cycle times.",
        added_value=[
            "See not just what's where, but what's stuck",
            "AI highlights cards that need attention",
            "Automatic WIP limit warnings"
        ],
        helps_with=[
            "Daily standups and work planning",
            "Identifying flow bottlenecks instantly",
            "Balancing workload across team members",
            "Tracking cycle time in real-time"
        ],
        key_features=[
            "Smart card highlighting based on age/blockers",
            "Swimlane views by assignee, type, or priority",
            "Cycle time indicators per card",
            "Quick actions for common operations"
        ],
        pro_tips=[
            "Enable 'heat mode' to see aging work",
            "Click column headers for WIP analytics"
        ],
        version="1.3"
    ),

    # -------------------------------------------------------------------------
    # SPRINT & DELIVERY PAGES
    # -------------------------------------------------------------------------

    "Sprint_Health": PageMetadata(
        id="Sprint_Health",
        title="Sprint Health",
        emoji="🏃",
        tagline="Is your sprint on track? Know in seconds.",
        goal="Provide real-time sprint health assessment with predictive completion analysis.",
        added_value=[
            "Predicts sprint outcome before it's too late",
            "Identifies at-risk items with time to act",
            "Shows scope creep as it happens"
        ],
        helps_with=[
            "Sprint review preparation",
            "Mid-sprint course corrections",
            "Capacity planning for next sprint",
            "Stakeholder expectation management"
        ],
        key_features=[
            "Health score with traffic light indicator",
            "Burndown with ideal vs actual comparison",
            "Scope change tracker",
            "Risk items with suggested actions"
        ],
        pro_tips=[
            "Check mid-sprint (day 5-6) for early warnings",
            "Use scope change data for retrospectives"
        ],
        version="1.4"
    ),

    "Predictions": PageMetadata(
        id="Predictions",
        title="Predictions",
        emoji="🎯",
        tagline="See the future. Change it.",
        goal="Use machine learning to predict delivery dates, identify risks, and simulate scenarios.",
        added_value=[
            "Monte Carlo simulations for realistic date ranges",
            "Risk-adjusted predictions, not wishful thinking",
            "What-if scenario planning"
        ],
        helps_with=[
            "Answering 'when will it be done?' with confidence",
            "Planning around uncertainty",
            "Negotiating scope vs timeline tradeoffs",
            "Building trust through accurate forecasting"
        ],
        key_features=[
            "Probability distribution for completion dates",
            "Confidence intervals (50%, 75%, 90%)",
            "Risk factor breakdown",
            "Historical accuracy tracking"
        ],
        pro_tips=[
            "Use the 75% confidence date for commitments",
            "Run simulations after scope changes"
        ],
        version="2.0",
        new_features=["Gemini-powered scenario narratives"]
    ),

    "Delivery_Forecast": PageMetadata(
        id="Delivery_Forecast",
        title="Delivery Forecast",
        emoji="🎲",
        tagline="Probability-based delivery intelligence",
        goal="Provide sophisticated delivery forecasting using historical velocity and Monte Carlo simulation.",
        added_value=[
            "Know the probability of hitting any date",
            "Understand what's driving uncertainty",
            "Plan for realistic scenarios, not best cases"
        ],
        helps_with=[
            "Release planning with confidence levels",
            "Resource allocation decisions",
            "Scope negotiation with data",
            "Building delivery predictability over time"
        ],
        key_features=[
            "Velocity trend analysis",
            "Monte Carlo simulation with 10,000 runs",
            "Probability curves for any target date",
            "Bottleneck identification"
        ],
        pro_tips=[
            "Focus on improving the 'worst case' scenarios",
            "Track prediction accuracy over time"
        ],
        version="1.2"
    ),

    # -------------------------------------------------------------------------
    # TEAM & PEOPLE PAGES
    # -------------------------------------------------------------------------

    "Team_Workload": PageMetadata(
        id="Team_Workload",
        title="Team Workload",
        emoji="👥",
        tagline="Balance your team. Protect your people.",
        goal="Visualize and balance workload across team members to prevent burnout and optimize throughput.",
        added_value=[
            "See overload before it causes burnout",
            "Identify capacity for new work",
            "Fair distribution with data, not gut feeling"
        ],
        helps_with=[
            "Sprint planning and capacity allocation",
            "Identifying team members who need help",
            "Making case for additional resources",
            "Performance conversations with data"
        ],
        key_features=[
            "Capacity utilization by team member",
            "Work type distribution (features vs bugs)",
            "Trend analysis over time",
            "Rebalancing suggestions"
        ],
        pro_tips=[
            "Target 70-80% utilization for sustainable pace",
            "Look for patterns, not just current state"
        ],
        version="1.3"
    ),

    "Burnout_Risk": PageMetadata(
        id="Burnout_Risk",
        title="Burnout Risk",
        emoji="🕯️",
        tagline="Protect your team's wellbeing with data",
        goal="Detect early warning signs of burnout using work patterns, overtime, and engagement metrics.",
        added_value=[
            "Catch burnout signals weeks before visible symptoms",
            "Quantify the cost of unsustainable pace",
            "Evidence-based conversations about workload"
        ],
        helps_with=[
            "Protecting team mental health proactively",
            "Justifying headcount or scope reduction",
            "1:1 conversation preparation",
            "Creating sustainable team culture"
        ],
        key_features=[
            "Individual burnout risk scores",
            "After-hours and weekend work tracking",
            "Context switching frequency",
            "Trend analysis with early warning"
        ],
        pro_tips=[
            "Check weekly, act on trends not spikes",
            "Combine with 1:1 qualitative feedback"
        ],
        version="1.5",
        new_features=["AI-generated intervention suggestions"]
    ),

    "One_on_One_Hub": PageMetadata(
        id="One_on_One_Hub",
        title="One-on-One Hub",
        emoji="👤",
        tagline="Data-powered 1:1s that matter",
        goal="Prepare for 1:1 meetings with comprehensive individual insights and AI-suggested talking points.",
        added_value=[
            "5-minute prep for meaningful 1:1s",
            "Never miss important topics",
            "Track action items and follow-ups"
        ],
        helps_with=[
            "Preparing for effective 1:1 conversations",
            "Tracking individual growth and challenges",
            "Ensuring fair recognition and support",
            "Building trust through informed conversations"
        ],
        key_features=[
            "Individual performance dashboard",
            "AI-generated talking points",
            "Recent wins and challenges summary",
            "Action item tracking across meetings"
        ],
        pro_tips=[
            "Review 10 minutes before each 1:1",
            "Let the team member see their dashboard too"
        ],
        version="1.1"
    ),

    # -------------------------------------------------------------------------
    # INTELLIGENCE & BRIEFING PAGES
    # -------------------------------------------------------------------------

    "Good_Morning": PageMetadata(
        id="Good_Morning",
        title="Good Morning",
        emoji="🌅",
        tagline="Start your day with clarity",
        goal="Provide a personalized AI-generated morning briefing with everything you need to know.",
        added_value=[
            "10-second situational awareness",
            "AI summarizes overnight changes",
            "Prioritized action items for today"
        ],
        helps_with=[
            "Starting the day focused and informed",
            "Not missing critical overnight updates",
            "Prioritizing what matters most today",
            "Reducing morning context-switching"
        ],
        key_features=[
            "AI-generated narrative briefing",
            "Delta from yesterday (what changed)",
            "Today's priority actions",
            "Team availability and blockers"
        ],
        pro_tips=[
            "Read with your morning coffee",
            "Share summary with stakeholders"
        ],
        version="2.0",
        new_features=["Gemini-powered narrative generation"]
    ),

    "Daily_Action_Intelligence": PageMetadata(
        id="Daily_Action_Intelligence",
        title="Daily Action Intelligence",
        emoji="⚡",
        tagline="Your productivity command center",
        goal="Surface the highest-impact actions you can take today with AI-prioritized recommendations.",
        added_value=[
            "Saves 35-50 minutes daily on coordination",
            "Never miss the most important action",
            "AI ranks by impact, not urgency"
        ],
        helps_with=[
            "Deciding what to focus on first",
            "Unblocking the team efficiently",
            "Maximizing your impact as a leader",
            "Reducing decision fatigue"
        ],
        key_features=[
            "AI-prioritized action cards",
            "One-click actions for common tasks",
            "Impact estimation per action",
            "Time saved tracking"
        ],
        pro_tips=[
            "Do the top 3 actions before lunch",
            "Track time saved to quantify value"
        ],
        version="1.8",
        new_features=["Gemini action explanations", "Undo capability"]
    ),

    "Scrum_Master_HQ": PageMetadata(
        id="Scrum_Master_HQ",
        title="Scrum Master HQ",
        emoji="🎖️",
        tagline="Your ceremony command center",
        goal="Provide Scrum Masters with everything needed to run effective ceremonies and protect the team.",
        added_value=[
            "One-click ceremony preparation",
            "Retrospective insights with data",
            "Team health monitoring"
        ],
        helps_with=[
            "Running better standups and retros",
            "Protecting team from interruptions",
            "Continuous improvement with metrics",
            "Stakeholder reporting"
        ],
        key_features=[
            "Ceremony countdown and prep checklists",
            "Sprint health for standups",
            "Retro topic suggestions from data",
            "Impediment tracker"
        ],
        pro_tips=[
            "Review 5 minutes before each ceremony",
            "Use data to spark retro discussions"
        ],
        version="1.2"
    ),

    # -------------------------------------------------------------------------
    # STRATEGIC PAGES
    # -------------------------------------------------------------------------

    "Strategic_Gap": PageMetadata(
        id="Strategic_Gap",
        title="Strategic Gap",
        emoji="🎯",
        tagline="Are you working on what matters?",
        goal="Analyze alignment between daily work and strategic objectives to identify gaps and drift.",
        added_value=[
            "See if busywork is crowding out strategy",
            "Quantify alignment with company goals",
            "Early warning on strategic drift"
        ],
        helps_with=[
            "Quarterly planning and prioritization",
            "Justifying strategic initiatives",
            "Explaining velocity vs progress mismatch",
            "OKR tracking and alignment"
        ],
        key_features=[
            "Work type classification (strategic vs operational)",
            "Goal alignment scoring",
            "Drift detection over time",
            "Recommendations for rebalancing"
        ],
        pro_tips=[
            "Review monthly with leadership",
            "Use for portfolio prioritization decisions"
        ],
        version="1.3"
    ),

    "Reports": PageMetadata(
        id="Reports",
        title="Reports",
        emoji="📋",
        tagline="Professional reports, one click away",
        goal="Generate polished, stakeholder-ready reports with AI-enhanced narratives.",
        added_value=[
            "Hours of report writing automated",
            "Consistent, professional format",
            "AI explains the 'so what'"
        ],
        helps_with=[
            "Stakeholder and board reporting",
            "Weekly/monthly status updates",
            "Documentation for audits",
            "Historical trend analysis"
        ],
        key_features=[
            "Multiple report templates",
            "AI narrative generation",
            "Export to PDF/PowerPoint",
            "Scheduled auto-generation"
        ],
        pro_tips=[
            "Schedule weekly reports for stakeholders",
            "Customize templates to your brand"
        ],
        version="1.4"
    ),

    # -------------------------------------------------------------------------
    # BLOCKER & DEPENDENCY PAGES
    # -------------------------------------------------------------------------

    "Blocker_Assassin": PageMetadata(
        id="Blocker_Assassin",
        title="Blocker Assassin",
        emoji="🔫",
        tagline="Kill blockers before they kill your sprint",
        goal="Identify, prioritize, and resolve blockers with maximum efficiency using cascade impact analysis.",
        added_value=[
            "See cascade impact of each blocker",
            "AI matches best resolver to each blocker",
            "SLA tracking with escalation countdowns"
        ],
        helps_with=[
            "Prioritizing which blockers to attack first",
            "Understanding true impact of delays",
            "Reducing average blocker resolution time",
            "Preventing blocker accumulation"
        ],
        key_features=[
            "Blocker severity scoring",
            "Cascade impact visualization",
            "Best resolver matching",
            "One-click resolution actions"
        ],
        pro_tips=[
            "Check twice daily during sprints",
            "Use cascade impact for prioritization"
        ],
        version="2.0",
        new_features=["AI-powered resolver matching", "Cascade simulation"]
    ),

    "Scope_Negotiator": PageMetadata(
        id="Scope_Negotiator",
        title="Scope Negotiator",
        emoji="⏱️",
        tagline="Trade scope intelligently, not emotionally",
        goal="Facilitate data-driven scope negotiations with impact analysis and trade-off visualization.",
        added_value=[
            "See true impact of scope changes",
            "Propose equivalent trade-offs automatically",
            "Document decisions for future reference"
        ],
        helps_with=[
            "Negotiating with stakeholders using data",
            "Managing scope creep proactively",
            "Finding win-win scope adjustments",
            "Protecting team from unrealistic commitments"
        ],
        key_features=[
            "Scope change impact calculator",
            "Trade-off suggestions",
            "Decision documentation",
            "Historical scope change analysis"
        ],
        pro_tips=[
            "Use before accepting scope changes",
            "Show stakeholders the trade-offs visually"
        ],
        version="1.1"
    ),

    # -------------------------------------------------------------------------
    # ADVANCED ANALYTICS PAGES
    # -------------------------------------------------------------------------

    "What_Breaks_If": PageMetadata(
        id="What_Breaks_If",
        title="What Breaks If",
        emoji="💥",
        tagline="Simulate chaos before it happens",
        goal="Run impact simulations for various scenarios (person leaves, deadline moves, scope adds) to understand vulnerabilities.",
        added_value=[
            "Pre-mortem analysis without the pain",
            "Identify single points of failure",
            "Plan for risks before they materialize"
        ],
        helps_with=[
            "Succession and bus factor planning",
            "Risk assessment for changes",
            "Building resilient teams",
            "Negotiating realistic timelines"
        ],
        key_features=[
            "Person absence simulation",
            "Deadline impact analysis",
            "Scope addition cascades",
            "Dependency failure scenarios"
        ],
        pro_tips=[
            "Run simulations quarterly for planning",
            "Use for resource request justification"
        ],
        version="1.3"
    ),

    "Resource_Shock_Absorber": PageMetadata(
        id="Resource_Shock_Absorber",
        title="Resource Shock Absorber",
        emoji="🛡️",
        tagline="Absorb disruptions without derailing delivery",
        goal="Automatically rebalance work and suggest mitigations when unexpected resource changes occur.",
        added_value=[
            "Instant rebalancing suggestions",
            "Minimize impact of unexpected absences",
            "Data-driven contingency planning"
        ],
        helps_with=[
            "Handling unexpected absences gracefully",
            "Quick recovery from disruptions",
            "Building resilient work distribution",
            "Reducing key-person dependencies"
        ],
        key_features=[
            "Automatic work redistribution suggestions",
            "Skill-based assignment matching",
            "Impact minimization algorithms",
            "Recovery timeline estimation"
        ],
        pro_tips=[
            "Keep skill matrix updated for best results",
            "Run 'what if' scenarios proactively"
        ],
        version="1.0"
    ),

    "The_Oracle": PageMetadata(
        id="The_Oracle",
        title="The Oracle",
        emoji="🔮",
        tagline="Ask anything. Get answers.",
        goal="Natural language interface to query your project data using AI.",
        added_value=[
            "Ask questions in plain English",
            "No need to build custom reports",
            "Discover insights you didn't know to ask for"
        ],
        helps_with=[
            "Ad-hoc analysis without waiting",
            "Answering stakeholder questions quickly",
            "Exploring data without SQL knowledge",
            "Finding patterns in complex data"
        ],
        key_features=[
            "Natural language query processing",
            "AI-generated visualizations",
            "Suggested follow-up questions",
            "Query history and favorites"
        ],
        pro_tips=[
            "Be specific for better answers",
            "Save frequent queries as favorites"
        ],
        version="1.5",
        new_features=["Gemini-powered natural language understanding"]
    ),

    "Relationship_Pulse": PageMetadata(
        id="Relationship_Pulse",
        title="Relationship Pulse",
        emoji="💫",
        tagline="The health of your team relationships",
        goal="Analyze collaboration patterns and relationship health across the team.",
        added_value=[
            "See collaboration patterns emerge",
            "Identify isolated team members",
            "Strengthen weak connections proactively"
        ],
        helps_with=[
            "Building cohesive teams",
            "Onboarding new team members effectively",
            "Identifying communication gaps",
            "Improving cross-team collaboration"
        ],
        key_features=[
            "Collaboration network visualization",
            "Relationship strength scoring",
            "Communication pattern analysis",
            "Team cohesion metrics"
        ],
        pro_tips=[
            "Review after team changes",
            "Use to plan pairing and mentoring"
        ],
        version="1.0"
    ),

    "Sixth_Sense": PageMetadata(
        id="Sixth_Sense",
        title="Sixth Sense",
        emoji="🧿",
        tagline="Feel what the data can't show",
        goal="AI-powered intuition engine that surfaces subtle patterns and emerging risks.",
        added_value=[
            "Catches what dashboards miss",
            "Early warning for emerging issues",
            "Pattern recognition across signals"
        ],
        helps_with=[
            "Staying ahead of problems",
            "Understanding team dynamics",
            "Making proactive decisions",
            "Building management intuition"
        ],
        key_features=[
            "Anomaly detection across metrics",
            "Sentiment analysis from communications",
            "Pattern correlation engine",
            "AI-generated insights"
        ],
        pro_tips=[
            "Trust but verify the signals",
            "Combine with your own intuition"
        ],
        version="1.1"
    ),

    # -------------------------------------------------------------------------
    # AUTONOMOUS & WEATHER PAGES
    # -------------------------------------------------------------------------

    "Autonomous_Agents": PageMetadata(
        id="Autonomous_Agents",
        title="Autonomous Agents",
        emoji="🤖",
        tagline="Work that manages itself",
        goal="Revolutionary autonomous work management where agents execute work, negotiate resources, and escalate only for true judgment calls.",
        added_value=[
            "50% reduction in coordination overhead",
            "Work continues even when you're away",
            "Focus on strategy, not status"
        ],
        helps_with=[
            "Eliminating repetitive management tasks",
            "Ensuring nothing falls through the cracks",
            "Scaling your impact across more projects",
            "Focusing on high-value decisions only"
        ],
        key_features=[
            "Intent Console - Set goals, not tasks",
            "Intervention Inbox - Only judgment calls",
            "Agent Observatory - Watch agents work",
            "Policy Layer - Set rules, not tasks"
        ],
        pro_tips=[
            "Start with Observer mode, build trust gradually",
            "Empty intervention inbox = system working"
        ],
        version="1.0",
        new_features=["Full autonomous agent system", "Policy-based governance"]
    ),

    "Project_Weather": PageMetadata(
        id="Project_Weather",
        title="Project Weather",
        emoji="🌀",
        tagline="Feel your project, don't just read it",
        goal="Revolutionary weather-based project visualization showing climate, storms, pressure, and forecasts.",
        added_value=[
            "Intuitive at-a-glance project health",
            "Predictive storm warnings",
            "Collision detection before impact"
        ],
        helps_with=[
            "Feeling the project state instantly",
            "Predicting problems before they hit",
            "Preventing resource collisions",
            "Building project intuition"
        ],
        key_features=[
            "Zone Weather - Per-team conditions",
            "Storm Tracking - Crisis management",
            "Pressure Map - Overload detection",
            "48-Hour Forecast - Predictive intelligence"
        ],
        pro_tips=[
            "Check first thing every morning",
            "Act on fronts before they collide"
        ],
        version="1.0",
        new_features=["Full weather system", "Predictive forecasting"]
    ),

    "Project_Autopilot": PageMetadata(
        id="Project_Autopilot",
        title="Project Autopilot",
        emoji="🚀",
        tagline="Hands-off project management",
        goal="Automated project management with AI-driven decisions and minimal human intervention.",
        added_value=[
            "Automated routine decisions",
            "Consistent management quality",
            "Scale without adding managers"
        ],
        helps_with=[
            "Managing multiple projects efficiently",
            "Ensuring consistent processes",
            "Freeing up time for strategic work",
            "Reducing management overhead"
        ],
        key_features=[
            "Automated task assignment",
            "Smart priority adjustments",
            "Automatic stakeholder updates",
            "Exception-based management"
        ],
        pro_tips=[
            "Review autopilot decisions daily at first",
            "Gradually increase autonomy as trust builds"
        ],
        version="1.0"
    ),

    "Next_Domino": PageMetadata(
        id="Next_Domino",
        title="Next Domino",
        emoji="🎯",
        tagline="Find the one thing that unlocks everything",
        goal="Identify the single highest-leverage action that will have the biggest cascade effect.",
        added_value=[
            "Focus on what truly moves the needle",
            "See cascade effects before acting",
            "Maximize impact per hour invested"
        ],
        helps_with=[
            "Prioritizing when everything seems urgent",
            "Finding leverage points in complex systems",
            "Explaining priorities to stakeholders",
            "Making tough trade-off decisions"
        ],
        key_features=[
            "Cascade impact analysis",
            "Leverage scoring algorithm",
            "What-if simulation",
            "Action recommendation engine"
        ],
        pro_tips=[
            "Start each day by finding the domino",
            "Trust the cascade analysis"
        ],
        version="1.0"
    ),

    "CEO_Command_Center": PageMetadata(
        id="CEO_Command_Center",
        title="CEO Command Center",
        emoji="🚀",
        tagline="The entire company at a glance",
        goal="Provide CEO-level visibility across all departments, projects, and strategic initiatives.",
        added_value=[
            "Company-wide visibility in seconds",
            "Strategic alignment tracking",
            "Cross-department coordination"
        ],
        helps_with=[
            "Board and investor reporting",
            "Strategic decision making",
            "Resource allocation across company",
            "Identifying company-wide risks"
        ],
        key_features=[
            "Multi-project portfolio view",
            "Strategic goal tracking",
            "Resource allocation overview",
            "Risk radar across all initiatives"
        ],
        pro_tips=[
            "Use for weekly leadership meetings",
            "Share view with board members"
        ],
        version="1.0"
    ),

}


# =============================================================================
# PAGE DETECTION
# =============================================================================

def get_current_page_id() -> Optional[str]:
    """
    Detect the current page from the URL or script path.
    Returns the page ID matching PAGE_REGISTRY keys.
    """
    try:
        # Get current script path
        import inspect
        frame = inspect.currentframe()
        if frame:
            caller_frame = frame.f_back.f_back  # Go up to the calling page
            if caller_frame:
                filepath = caller_frame.f_globals.get('__file__', '')
                if filepath:
                    filename = Path(filepath).stem
                    # Extract page name (remove number and emoji prefix)
                    # Pattern: "0_🏆_Executive_Cockpit" -> "Executive_Cockpit"
                    parts = filename.split('_', 2)
                    if len(parts) >= 3:
                        page_id = parts[2]
                        return page_id
                    elif len(parts) == 2:
                        return parts[1]
    except Exception:
        pass

    return None


def get_page_metadata(page_id: Optional[str] = None) -> Optional[PageMetadata]:
    """
    Get metadata for a specific page or auto-detect current page.
    """
    if page_id is None:
        page_id = get_current_page_id()

    if page_id and page_id in PAGE_REGISTRY:
        return PAGE_REGISTRY[page_id]

    # Try fuzzy matching
    if page_id:
        for key in PAGE_REGISTRY:
            if key.lower() == page_id.lower() or key.replace('_', '') == page_id.replace('_', ''):
                return PAGE_REGISTRY[key]

    return None


# =============================================================================
# PAGE GUIDE COMPONENT
# =============================================================================

def render_page_guide(page_id: Optional[str] = None):
    """
    Render a floating page guide button and expandable panel.
    Works independently of sidebar state.

    Args:
        page_id: Optional page ID. If None, auto-detects from current page.
    """
    metadata = get_page_metadata(page_id)

    if not metadata:
        return

    # Initialize session state for this page's guide
    guide_key = f"page_guide_{metadata.id}"
    if guide_key not in st.session_state:
        st.session_state[guide_key] = False

    # Build content HTML
    value_items = "".join([f"<li>{item}</li>" for item in metadata.added_value])
    helps_items = "".join([f"<li>{item}</li>" for item in metadata.helps_with])
    features_items = "".join([f"<li>{item}</li>" for item in metadata.key_features])
    tips_items = "".join([f"<li>{item}</li>" for item in metadata.pro_tips])

    new_features_html = ""
    if metadata.new_features:
        new_items = ", ".join(metadata.new_features)
        new_features_html = f"""
        <div class="page-guide-features-new">
            <div class="page-guide-features-new-title">✨ NEW IN v{metadata.version}</div>
            <div class="page-guide-features-new-list">{new_items}</div>
        </div>
        """

    # Inject CSS for floating button and panel
    st.markdown("""
    <style>
        /* Floating Guide Button */
        .floating-guide-container {
            position: fixed;
            right: 24px;
            top: 80px;
            z-index: 9999;
        }

        .floating-guide-btn {
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border: 2px solid rgba(255,255,255,0.2);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.5);
            transition: all 0.3s ease;
        }

        .floating-guide-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6);
        }

        /* Guide Panel */
        .guide-panel {
            position: fixed;
            right: 24px;
            top: 140px;
            width: 360px;
            max-height: calc(100vh - 180px);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
            z-index: 9998;
            overflow: hidden;
        }

        .guide-panel-header {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .guide-panel-emoji {
            font-size: 1.75rem;
        }

        .guide-panel-title {
            flex: 1;
        }

        .guide-panel-name {
            font-size: 1.1rem;
            font-weight: 700;
            color: white;
            margin: 0;
        }

        .guide-panel-tagline {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.8);
            margin: 0;
        }

        .guide-panel-version {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            color: white;
            font-weight: 600;
        }

        .guide-panel-content {
            padding: 20px;
            max-height: calc(100vh - 280px);
            overflow-y: auto;
        }

        .guide-section {
            margin-bottom: 16px;
        }

        .guide-section-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #3b82f6;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .guide-section-content {
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.6;
        }

        .guide-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .guide-list li {
            padding: 6px 0;
            padding-left: 20px;
            position: relative;
            color: #475569;
            font-size: 0.85rem;
            line-height: 1.5;
        }

        .guide-list li::before {
            content: '→';
            position: absolute;
            left: 0;
            color: #22c55e;
            font-weight: bold;
        }

        .guide-tips li::before {
            content: '💡';
        }

        .guide-features-new {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.1));
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 10px;
            padding: 12px;
            margin-top: 12px;
        }

        .guide-features-new-title {
            font-size: 0.7rem;
            font-weight: 700;
            color: #d97706;
            margin-bottom: 4px;
        }

        .guide-features-new-list {
            font-size: 0.8rem;
            color: #92400e;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create floating button using columns at top right
    # Use popover for better UX (Streamlit 1.33+)
    try:
        # Try using popover (newer Streamlit)
        with st.popover("📖", help="Page Guide", use_container_width=False):
            st.markdown(f"""
            <div style="min-width: 320px;">
                <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 16px; border-radius: 12px 12px 0 0; margin: -1rem -1rem 1rem -1rem;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.75rem;">{metadata.emoji}</span>
                        <div style="flex: 1;">
                            <div style="font-size: 1.1rem; font-weight: 700; color: white;">{metadata.title}</div>
                            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">{metadata.tagline}</div>
                        </div>
                        <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; color: white;">v{metadata.version}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="guide-section">
                <div class="guide-section-title">🎯 Goal</div>
                <div class="guide-section-content">{metadata.goal}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="guide-section">
                <div class="guide-section-title">✨ Added Value</div>
                <ul class="guide-list">{value_items}</ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="guide-section">
                <div class="guide-section-title">👤 Helps Team Leaders</div>
                <ul class="guide-list">{helps_items}</ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="guide-section">
                <div class="guide-section-title">⚡ Key Features</div>
                <ul class="guide-list">{features_items}</ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="guide-section">
                <div class="guide-section-title">💡 Pro Tips</div>
                <ul class="guide-list guide-tips">{tips_items}</ul>
            </div>
            """, unsafe_allow_html=True)

            if new_features_html:
                st.markdown(new_features_html, unsafe_allow_html=True)

    except Exception:
        # Fallback for older Streamlit - use expander at top
        with st.expander("📖 **Page Guide** - Click to learn about this page", expanded=False):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<div style='font-size: 2.5rem; text-align: center;'>{metadata.emoji}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"### {metadata.title}")
                st.caption(f"*{metadata.tagline}* | v{metadata.version}")

            st.markdown(f"**🎯 Goal:** {metadata.goal}")

            st.markdown("**✨ Added Value:**")
            for item in metadata.added_value:
                st.markdown(f"- {item}")

            st.markdown("**👤 Helps Team Leaders With:**")
            for item in metadata.helps_with:
                st.markdown(f"- {item}")

            st.markdown("**⚡ Key Features:**")
            for item in metadata.key_features:
                st.markdown(f"- {item}")

            st.markdown("**💡 Pro Tips:**")
            for tip in metadata.pro_tips:
                st.markdown(f"- 💡 {tip}")

            if metadata.new_features:
                st.success(f"✨ **New in v{metadata.version}:** {', '.join(metadata.new_features)}")


def render_floating_guide(page_id: Optional[str] = None):
    """
    Render a floating guide button that opens a modal/panel.
    Alternative implementation for pages that don't use sidebar.
    """
    metadata = get_page_metadata(page_id)

    if not metadata:
        return

    # Create a floating button in the top-right corner
    st.markdown("""
    <style>
        .floating-guide-btn {
            position: fixed;
            right: 24px;
            top: 80px;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.5);
            z-index: 1000;
            transition: all 0.3s ease;
        }

        .floating-guide-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6);
        }
    </style>
    """, unsafe_allow_html=True)

    # Use columns to add a button in the corner
    col1, col2, col3 = st.columns([10, 1, 1])
    with col3:
        if st.button("📖", key="floating_guide_btn", help="Page Guide"):
            st.session_state.show_floating_guide = not st.session_state.get('show_floating_guide', False)

    if st.session_state.get('show_floating_guide', False):
        with st.container():
            st.markdown(f"""
            ### {metadata.emoji} {metadata.title}
            *{metadata.tagline}*

            **Goal:** {metadata.goal}

            **Added Value:**
            """)
            for item in metadata.added_value:
                st.markdown(f"- {item}")

            st.markdown("**Helps With:**")
            for item in metadata.helps_with:
                st.markdown(f"- {item}")

            st.markdown("**Pro Tips:**")
            for tip in metadata.pro_tips:
                st.markdown(f"💡 {tip}")
