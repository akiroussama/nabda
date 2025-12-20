"""
Project Weather System - Climate Intelligence for Work

This module implements a revolutionary paradigm shift in project management:
Instead of showing the STATE of work, we show the CLIMATE of work.

You don't read data. You FEEL the weather.

Weather Elements:
- Clear Skies: Work flowing smoothly
- Clouds Forming: Early warning signs
- Rain: Active problems
- Storms: Crisis mode with cascading delays
- Tornadoes: Project-threatening events

Pressure Systems:
- High Pressure (Red): Overcommitment, deadline compression
- Low Pressure (Blue): Waiting/idle work, decision vacuums
- Fronts: When high meets low - collision imminent

Forecasting:
- 48-hour predictions based on velocity, blockers, dependencies
- Probability of delay with key decision points
- Recommended preventive actions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set
import hashlib
import random
import math
from collections import defaultdict


# =============================================================================
# ENUMS - Weather States and Severity Levels
# =============================================================================

class WeatherCondition(Enum):
    """Current weather condition for a team/project area."""
    CLEAR = "clear"              # ☀️ Everything flowing
    PARTLY_CLOUDY = "partly_cloudy"  # 🌤️ Early warnings
    CLOUDY = "cloudy"            # ⛅ Some friction
    RAIN = "rain"                # 🌧️ Active problems
    STORM = "storm"              # ⛈️ Crisis mode
    TORNADO = "tornado"          # 🌪️ Project-threatening


class StormSeverity(Enum):
    """Severity level of an active storm."""
    MINOR = 1       # Contained, 1-2 people affected
    MODERATE = 2    # Growing, 3-5 people affected
    SEVERE = 3      # Spreading, 6+ people affected
    CRITICAL = 4    # Project-threatening, executive action needed


class PressureLevel(Enum):
    """Pressure level in an area."""
    VERY_LOW = 1    # Idle/waiting
    LOW = 2         # Under capacity
    NORMAL = 3      # Healthy
    HIGH = 4        # Stretched
    CRITICAL = 5    # Overloaded


class FrontType(Enum):
    """Type of approaching front (collision)."""
    RESOURCE = "resource"        # Same person needed by multiple teams
    DEADLINE = "deadline"        # Multiple deadlines colliding
    DEPENDENCY = "dependency"    # Dependency chains colliding
    SCOPE = "scope"              # Scope changes colliding with timeline


class ForecastConfidence(Enum):
    """Confidence level of forecast."""
    HIGH = "high"        # 80%+ historical accuracy
    MEDIUM = "medium"    # 60-80% accuracy
    LOW = "low"          # <60% accuracy


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WeatherZone:
    """A zone in the project weather system (team, component, project area)."""
    id: str
    name: str
    zone_type: str               # team, component, project, epic

    # Current weather
    condition: WeatherCondition = WeatherCondition.CLEAR
    pressure: PressureLevel = PressureLevel.NORMAL
    temperature: float = 0.0     # -1.0 (cold/slow) to 1.0 (hot/fast)

    # Momentum metrics
    velocity_ratio: float = 1.0  # Current vs expected velocity
    blocker_count: int = 0
    overdue_count: int = 0

    # Health indicators
    people_count: int = 0
    active_tasks: int = 0
    completed_today: int = 0

    # Weather factors
    factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def get_emoji(self) -> str:
        """Get weather emoji for current condition."""
        return {
            WeatherCondition.CLEAR: "☀️",
            WeatherCondition.PARTLY_CLOUDY: "🌤️",
            WeatherCondition.CLOUDY: "⛅",
            WeatherCondition.RAIN: "🌧️",
            WeatherCondition.STORM: "⛈️",
            WeatherCondition.TORNADO: "🌪️",
        }.get(self.condition, "❓")

    def get_pressure_color(self) -> str:
        """Get color for pressure level."""
        return {
            PressureLevel.VERY_LOW: "#3b82f6",   # Blue
            PressureLevel.LOW: "#06b6d4",        # Cyan
            PressureLevel.NORMAL: "#22c55e",     # Green
            PressureLevel.HIGH: "#f59e0b",       # Amber
            PressureLevel.CRITICAL: "#ef4444",   # Red
        }.get(self.pressure, "#64748b")

    def get_description(self) -> str:
        """Get human-readable weather description."""
        return {
            WeatherCondition.CLEAR: "Clear skies",
            WeatherCondition.PARTLY_CLOUDY: "Clouds forming",
            WeatherCondition.CLOUDY: "Overcast",
            WeatherCondition.RAIN: "Active problems",
            WeatherCondition.STORM: "Storm active",
            WeatherCondition.TORNADO: "EMERGENCY",
        }.get(self.condition, "Unknown")


@dataclass
class Storm:
    """An active storm (crisis situation)."""
    id: str
    name: str
    zone_id: str
    severity: StormSeverity

    # Impact
    root_cause: str
    affected_people: List[str] = field(default_factory=list)
    affected_tasks: List[str] = field(default_factory=list)
    blocked_dependencies: List[str] = field(default_factory=list)

    # Duration
    started_at: datetime = field(default_factory=datetime.now)
    duration_hours: float = 0.0

    # Trajectory
    spreading: bool = False
    spread_forecast: List[str] = field(default_factory=list)  # Zone IDs
    time_to_spread_hours: float = 0.0

    # Resolution
    actions_taken: List[str] = field(default_factory=list)
    recommended_actions: List[Dict[str, str]] = field(default_factory=list)
    resolution_probability: float = 0.0

    def get_severity_emoji(self) -> str:
        """Get emoji for storm severity."""
        return {
            StormSeverity.MINOR: "🌧️",
            StormSeverity.MODERATE: "⛈️",
            StormSeverity.SEVERE: "🌪️",
            StormSeverity.CRITICAL: "🔴",
        }.get(self.severity, "⛈️")

    def get_wind_speed(self) -> str:
        """Get 'wind speed' description based on blocked people."""
        count = len(self.affected_people)
        if count == 0:
            return "Calm"
        elif count <= 2:
            return f"{count} people affected"
        elif count <= 5:
            return f"{count} people blocked"
        else:
            return f"🔴 {count} people severely impacted"


@dataclass
class Front:
    """An approaching front (collision warning)."""
    id: str
    front_type: FrontType
    description: str

    # Collision details
    colliding_elements: List[str]  # What's going to collide
    collision_date: date
    days_until_collision: int

    # Impact assessment
    impact_description: str
    affected_zones: List[str]
    severity: StormSeverity

    # Prevention
    preventable: bool = True
    prevention_actions: List[Dict[str, str]] = field(default_factory=list)

    def get_urgency_color(self) -> str:
        """Get urgency color based on days until collision."""
        if self.days_until_collision <= 1:
            return "#ef4444"  # Red - imminent
        elif self.days_until_collision <= 3:
            return "#f59e0b"  # Amber - soon
        else:
            return "#3b82f6"  # Blue - approaching


@dataclass
class Forecast:
    """48-hour weather forecast."""
    zone_id: str
    generated_at: datetime = field(default_factory=datetime.now)

    # Hourly predictions (48 hours)
    hourly_conditions: List[WeatherCondition] = field(default_factory=list)

    # Daily summaries
    today_summary: str = ""
    today_condition: WeatherCondition = WeatherCondition.CLEAR
    tomorrow_summary: str = ""
    tomorrow_condition: WeatherCondition = WeatherCondition.CLEAR

    # Risk assessment
    delay_probability: float = 0.0
    delay_trigger: str = ""
    key_decision_point: str = ""
    key_decision_time: Optional[datetime] = None

    # Confidence
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM
    accuracy_history: float = 0.75  # Historical accuracy

    # Assumptions
    assumptions: List[str] = field(default_factory=list)


@dataclass
class PressureCell:
    """A cell in the pressure map."""
    x: int
    y: int
    pressure: float              # 0.0 to 1.0
    zone_id: Optional[str] = None
    person_id: Optional[str] = None
    label: str = ""

    def get_color(self) -> str:
        """Get color based on pressure."""
        if self.pressure < 0.3:
            return "#22c55e"  # Green
        elif self.pressure < 0.5:
            return "#84cc16"  # Lime
        elif self.pressure < 0.7:
            return "#f59e0b"  # Amber
        elif self.pressure < 0.85:
            return "#f97316"  # Orange
        else:
            return "#ef4444"  # Red


@dataclass
class WeatherAlert:
    """A weather alert notification."""
    id: str
    alert_type: str              # storm_forming, front_approaching, pressure_critical
    severity: str                # info, warning, critical
    title: str
    message: str
    zone_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    actions: List[Dict[str, str]] = field(default_factory=list)


# =============================================================================
# WEATHER ENGINE - The Core Intelligence
# =============================================================================

class WeatherEngine:
    """
    The Project Weather System engine.

    Synthesizes multiple data signals to produce weather conditions,
    pressure maps, storm tracking, and 48-hour forecasts.
    """

    def __init__(self, db_connection=None, llm_client=None):
        self.db = db_connection
        self.llm_client = llm_client

        # Weather state
        self.zones: Dict[str, WeatherZone] = {}
        self.storms: Dict[str, Storm] = {}
        self.fronts: Dict[str, Front] = {}
        self.forecasts: Dict[str, Forecast] = {}
        self.alerts: List[WeatherAlert] = []

        # Pressure map
        self.pressure_grid: List[List[PressureCell]] = []
        self.grid_width = 10
        self.grid_height = 6

        # Historical data for learning
        self.storm_history: List[Storm] = []
        self.forecast_accuracy: List[Tuple[str, bool]] = []

        # Thresholds (can be calibrated per org)
        self.thresholds = {
            "blocker_storm": 2,          # Blockers to trigger storm
            "overdue_warning": 1,         # Overdue to trigger warning
            "velocity_drop_warning": 0.7, # Velocity ratio for warning
            "pressure_high": 0.8,         # Utilization for high pressure
            "pressure_critical": 1.0,     # Utilization for critical
            "front_days_warning": 5,      # Days before front to warn
        }

        # Initialize with demo data if no DB
        if not db_connection:
            self._initialize_demo_weather()

    def _initialize_demo_weather(self):
        """Initialize with realistic demo weather data."""

        # Create weather zones
        zones_data = [
            ("frontend", "Frontend", "team", WeatherCondition.CLEAR, PressureLevel.NORMAL, 1.1),
            ("backend", "Backend", "team", WeatherCondition.STORM, PressureLevel.CRITICAL, 0.6),
            ("design", "Design", "team", WeatherCondition.PARTLY_CLOUDY, PressureLevel.HIGH, 0.85),
            ("qa", "QA", "team", WeatherCondition.CLOUDY, PressureLevel.HIGH, 0.75),
            ("devops", "DevOps", "team", WeatherCondition.CLEAR, PressureLevel.NORMAL, 1.0),
            ("mobile", "Mobile", "team", WeatherCondition.RAIN, PressureLevel.HIGH, 0.65),
        ]

        for zone_id, name, zone_type, condition, pressure, velocity in zones_data:
            zone = WeatherZone(
                id=zone_id,
                name=name,
                zone_type=zone_type,
                condition=condition,
                pressure=pressure,
                velocity_ratio=velocity,
                people_count=random.randint(3, 8),
                active_tasks=random.randint(5, 20),
                completed_today=random.randint(0, 5),
                blocker_count=random.randint(0, 4) if condition in [WeatherCondition.STORM, WeatherCondition.RAIN] else 0,
                overdue_count=random.randint(0, 3) if condition != WeatherCondition.CLEAR else 0,
            )

            # Add contextual factors
            if condition == WeatherCondition.STORM:
                zone.factors = ["Critical blocker active", "3 people waiting", "Dependency chain at risk"]
                zone.warnings = ["Spreading to Frontend in ~2 hours"]
            elif condition == WeatherCondition.PARTLY_CLOUDY:
                zone.factors = ["Dependency on Backend", "Design review pending"]
                zone.warnings = ["Watch: Backend storm may spread here"]
            elif condition == WeatherCondition.RAIN:
                zone.factors = ["2 blockers active", "Velocity below target"]

            self.zones[zone_id] = zone

        # Create active storm
        storm = Storm(
            id="storm_api_integration",
            name="API Integration Blocked",
            zone_id="backend",
            severity=StormSeverity.MODERATE,
            root_cause="Waiting on external vendor response for authentication API",
            affected_people=["Alex Chen", "Jordan Smith", "Sam Wilson"],
            affected_tasks=["PROJ-234", "PROJ-245", "PROJ-251", "PROJ-256", "PROJ-259", "PROJ-261", "PROJ-270"],
            blocked_dependencies=["Frontend user auth", "Mobile login flow"],
            started_at=datetime.now() - timedelta(hours=6),
            duration_hours=6.5,
            spreading=True,
            spread_forecast=["frontend", "mobile"],
            time_to_spread_hours=2.0,
            recommended_actions=[
                {"id": "escalate", "emoji": "🔥", "label": "Escalate", "description": "Notify engineering lead"},
                {"id": "reroute", "emoji": "🔄", "label": "Reroute", "description": "Reassign blocked people temporarily"},
                {"id": "shelter", "emoji": "⏸️", "label": "Shelter", "description": "Pause affected work, protect focus"},
                {"id": "intervene", "emoji": "📞", "label": "Intervene", "description": "Contact vendor directly"},
            ],
            resolution_probability=0.65,
        )
        self.storms[storm.id] = storm

        # Create approaching fronts
        fronts_data = [
            Front(
                id="front_marketing_freeze",
                front_type=FrontType.DEADLINE,
                description="Marketing launch + Engineering freeze collision",
                colliding_elements=["Marketing Product Launch", "Q4 Code Freeze"],
                collision_date=date.today() + timedelta(days=3),
                days_until_collision=3,
                impact_description="Marketing needs final features but engineering will be in freeze mode",
                affected_zones=["frontend", "backend", "qa"],
                severity=StormSeverity.SEVERE,
                prevention_actions=[
                    {"id": "accelerate", "label": "Accelerate marketing features", "description": "Prioritize and finish before freeze"},
                    {"id": "exception", "label": "Request freeze exception", "description": "Get approval for limited changes"},
                    {"id": "delay", "label": "Delay launch", "description": "Push marketing launch by 1 week"},
                ]
            ),
            Front(
                id="front_sarah_conflict",
                front_type=FrontType.RESOURCE,
                description="Two teams need Sarah simultaneously",
                colliding_elements=["Frontend sprint review", "Design system overhaul"],
                collision_date=date.today() + timedelta(days=1),
                days_until_collision=1,
                impact_description="Sarah is the only designer who can approve both deliverables",
                affected_zones=["frontend", "design"],
                severity=StormSeverity.MODERATE,
                prevention_actions=[
                    {"id": "reschedule", "label": "Reschedule one meeting", "description": "Move frontend review to afternoon"},
                    {"id": "delegate", "label": "Delegate approval", "description": "Allow senior designer to approve design system"},
                    {"id": "parallel", "label": "Run parallel", "description": "Sarah joins both via video for critical 15 min each"},
                ]
            ),
        ]

        for front in fronts_data:
            self.fronts[front.id] = front

        # Generate forecasts
        for zone_id, zone in self.zones.items():
            self._generate_forecast(zone_id)

        # Create alerts
        self.alerts = [
            WeatherAlert(
                id="alert_1",
                alert_type="storm_active",
                severity="critical",
                title="Active Storm: Backend",
                message="API Integration blocked for 6+ hours. 3 people waiting.",
                zone_id="backend",
                actions=[
                    {"id": "view", "label": "View Storm Details"},
                    {"id": "escalate", "label": "Escalate Now"},
                ]
            ),
            WeatherAlert(
                id="alert_2",
                alert_type="front_approaching",
                severity="warning",
                title="Collision in 3 days",
                message="Marketing launch and Engineering freeze will collide.",
                zone_id=None,
                actions=[
                    {"id": "view", "label": "View Collision"},
                    {"id": "prevent", "label": "Take Action"},
                ]
            ),
        ]

        # Initialize pressure grid
        self._calculate_pressure_grid()

    def _generate_forecast(self, zone_id: str):
        """Generate 48-hour forecast for a zone."""
        zone = self.zones.get(zone_id)
        if not zone:
            return

        # Simple forecast based on current conditions
        current = zone.condition

        # Forecast logic
        if current == WeatherCondition.STORM:
            forecast = Forecast(
                zone_id=zone_id,
                today_condition=WeatherCondition.STORM,
                today_summary="Storm continues. Active resolution in progress.",
                tomorrow_condition=WeatherCondition.RAIN,
                tomorrow_summary="Storm expected to weaken if blockers resolved.",
                delay_probability=0.73,
                delay_trigger="storm not resolved by 4pm today",
                key_decision_point="Vendor response needed",
                key_decision_time=datetime.now().replace(hour=16, minute=0),
                confidence=ForecastConfidence.MEDIUM,
                assumptions=[
                    "Vendor responds within business hours",
                    "No new blockers emerge",
                    "Team maintains current velocity on unblocked work"
                ]
            )
        elif current in [WeatherCondition.RAIN, WeatherCondition.CLOUDY]:
            forecast = Forecast(
                zone_id=zone_id,
                today_condition=current,
                today_summary="Clearing expected by end of day.",
                tomorrow_condition=WeatherCondition.PARTLY_CLOUDY,
                tomorrow_summary="Improvement expected. Watch for dependency risks.",
                delay_probability=0.35,
                delay_trigger="blockers not resolved",
                confidence=ForecastConfidence.MEDIUM,
                assumptions=[
                    "Blockers addressed with current priority",
                    "No scope changes"
                ]
            )
        else:
            forecast = Forecast(
                zone_id=zone_id,
                today_condition=current,
                today_summary="Clear conditions. Steady progress expected.",
                tomorrow_condition=WeatherCondition.CLEAR,
                tomorrow_summary="Continued clear skies. Good momentum.",
                delay_probability=0.12,
                confidence=ForecastConfidence.HIGH,
                assumptions=["Current trajectory maintained"]
            )

        self.forecasts[zone_id] = forecast

    def _calculate_pressure_grid(self):
        """Calculate the pressure map grid."""
        self.pressure_grid = []

        # Create grid with realistic pressure distribution
        zone_list = list(self.zones.values())

        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                # Map grid position to zone
                zone_idx = (y * self.grid_width + x) % len(zone_list) if zone_list else 0
                zone = zone_list[zone_idx] if zone_list else None

                # Calculate pressure based on zone
                if zone:
                    base_pressure = {
                        PressureLevel.VERY_LOW: 0.1,
                        PressureLevel.LOW: 0.25,
                        PressureLevel.NORMAL: 0.4,
                        PressureLevel.HIGH: 0.7,
                        PressureLevel.CRITICAL: 0.95,
                    }.get(zone.pressure, 0.5)

                    # Add some variance
                    pressure = base_pressure + random.uniform(-0.1, 0.1)
                    pressure = max(0.0, min(1.0, pressure))
                else:
                    pressure = random.uniform(0.3, 0.6)

                cell = PressureCell(
                    x=x,
                    y=y,
                    pressure=pressure,
                    zone_id=zone.id if zone else None,
                    label=zone.name if zone else ""
                )
                row.append(cell)
            self.pressure_grid.append(row)

    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall weather system summary."""
        active_storms = [s for s in self.storms.values()]
        approaching_fronts = [f for f in self.fronts.values()]

        # Calculate overall conditions
        conditions = [z.condition for z in self.zones.values()]
        if any(c == WeatherCondition.TORNADO for c in conditions):
            overall = WeatherCondition.TORNADO
        elif any(c == WeatherCondition.STORM for c in conditions):
            overall = WeatherCondition.STORM
        elif any(c == WeatherCondition.RAIN for c in conditions):
            overall = WeatherCondition.RAIN
        elif any(c == WeatherCondition.CLOUDY for c in conditions):
            overall = WeatherCondition.CLOUDY
        elif any(c == WeatherCondition.PARTLY_CLOUDY for c in conditions):
            overall = WeatherCondition.PARTLY_CLOUDY
        else:
            overall = WeatherCondition.CLEAR

        # Calculate average pressure
        pressures = [z.pressure.value for z in self.zones.values()]
        avg_pressure = sum(pressures) / len(pressures) if pressures else 3

        # Count alerts by severity
        critical_alerts = sum(1 for a in self.alerts if a.severity == "critical" and not a.acknowledged)
        warning_alerts = sum(1 for a in self.alerts if a.severity == "warning" and not a.acknowledged)

        return {
            "overall_condition": overall,
            "overall_emoji": {
                WeatherCondition.CLEAR: "☀️",
                WeatherCondition.PARTLY_CLOUDY: "🌤️",
                WeatherCondition.CLOUDY: "⛅",
                WeatherCondition.RAIN: "🌧️",
                WeatherCondition.STORM: "⛈️",
                WeatherCondition.TORNADO: "🌪️",
            }.get(overall, "❓"),
            "active_storms": len(active_storms),
            "approaching_fronts": len(approaching_fronts),
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "zones_clear": sum(1 for z in self.zones.values() if z.condition == WeatherCondition.CLEAR),
            "zones_troubled": sum(1 for z in self.zones.values() if z.condition in [WeatherCondition.STORM, WeatherCondition.RAIN]),
            "avg_pressure": avg_pressure,
            "total_blocked_people": sum(len(s.affected_people) for s in active_storms),
        }

    def get_pressure_map_data(self) -> List[List[Dict]]:
        """Get pressure map data for visualization."""
        return [
            [
                {
                    "x": cell.x,
                    "y": cell.y,
                    "pressure": cell.pressure,
                    "color": cell.get_color(),
                    "zone_id": cell.zone_id,
                    "label": cell.label
                }
                for cell in row
            ]
            for row in self.pressure_grid
        ]

    def get_48hour_forecast(self, zone_id: str = None) -> Dict[str, Any]:
        """Get 48-hour forecast data."""
        if zone_id and zone_id in self.forecasts:
            forecast = self.forecasts[zone_id]
            zone = self.zones.get(zone_id)
            return {
                "zone": zone.name if zone else "Unknown",
                "today": {
                    "condition": forecast.today_condition.value,
                    "emoji": WeatherZone(id="", name="", zone_type="", condition=forecast.today_condition).get_emoji(),
                    "summary": forecast.today_summary
                },
                "tomorrow": {
                    "condition": forecast.tomorrow_condition.value,
                    "emoji": WeatherZone(id="", name="", zone_type="", condition=forecast.tomorrow_condition).get_emoji(),
                    "summary": forecast.tomorrow_summary
                },
                "delay_probability": forecast.delay_probability,
                "delay_trigger": forecast.delay_trigger,
                "key_decision": forecast.key_decision_point,
                "key_decision_time": forecast.key_decision_time.strftime("%I:%M %p") if forecast.key_decision_time else None,
                "confidence": forecast.confidence.value,
                "assumptions": forecast.assumptions
            }

        # Aggregate forecast
        total_delay_prob = 0
        worst_today = WeatherCondition.CLEAR
        worst_tomorrow = WeatherCondition.CLEAR

        for f in self.forecasts.values():
            total_delay_prob += f.delay_probability
            if f.today_condition.value > worst_today.value:
                worst_today = f.today_condition
            if f.tomorrow_condition.value > worst_tomorrow.value:
                worst_tomorrow = f.tomorrow_condition

        avg_delay = total_delay_prob / len(self.forecasts) if self.forecasts else 0

        return {
            "zone": "All Projects",
            "today": {
                "condition": worst_today.value,
                "emoji": WeatherZone(id="", name="", zone_type="", condition=worst_today).get_emoji(),
                "summary": "Mixed conditions across teams"
            },
            "tomorrow": {
                "condition": worst_tomorrow.value,
                "emoji": WeatherZone(id="", name="", zone_type="", condition=worst_tomorrow).get_emoji(),
                "summary": "Gradual improvement expected"
            },
            "delay_probability": avg_delay,
            "delay_trigger": "Unresolved storms",
            "confidence": "medium"
        }

    def resolve_storm(self, storm_id: str, action: str, resolved_by: str = "PM"):
        """Mark a storm as resolved or take action."""
        if storm_id in self.storms:
            storm = self.storms[storm_id]
            storm.actions_taken.append(f"{action} by {resolved_by}")

            # Update zone weather
            zone = self.zones.get(storm.zone_id)
            if zone:
                zone.condition = WeatherCondition.RAIN  # Downgrade from storm
                zone.blocker_count = max(0, zone.blocker_count - 1)

            # If fully resolved, remove storm
            if action in ["resolved", "cleared"]:
                self.storm_history.append(storm)
                del self.storms[storm_id]

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break

    def simulate_tick(self):
        """Simulate one tick of weather change (for demo/real-time updates)."""
        # Random weather evolution
        for zone in self.zones.values():
            # Small chance of condition change
            if random.random() < 0.1:
                current_idx = [
                    WeatherCondition.CLEAR,
                    WeatherCondition.PARTLY_CLOUDY,
                    WeatherCondition.CLOUDY,
                    WeatherCondition.RAIN,
                    WeatherCondition.STORM,
                ].index(zone.condition) if zone.condition != WeatherCondition.TORNADO else 4

                # Tend toward mean (regression)
                if random.random() < 0.6:
                    # Move toward clear
                    new_idx = max(0, current_idx - 1)
                else:
                    # Move toward worse
                    new_idx = min(4, current_idx + 1)

                zone.condition = [
                    WeatherCondition.CLEAR,
                    WeatherCondition.PARTLY_CLOUDY,
                    WeatherCondition.CLOUDY,
                    WeatherCondition.RAIN,
                    WeatherCondition.STORM,
                ][new_idx]

            # Update velocity slightly
            zone.velocity_ratio += random.uniform(-0.05, 0.05)
            zone.velocity_ratio = max(0.3, min(1.5, zone.velocity_ratio))

            # Update completed today
            if random.random() < 0.2:
                zone.completed_today += 1

        # Update storm duration
        for storm in self.storms.values():
            storm.duration_hours += 0.1

        # Recalculate pressure grid
        self._calculate_pressure_grid()


# =============================================================================
# DEMO DATA GENERATOR
# =============================================================================

def create_demo_weather_engine() -> WeatherEngine:
    """Create a weather engine with demo data."""
    return WeatherEngine()
