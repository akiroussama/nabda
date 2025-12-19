"""
Probabilistic Delivery Intelligence.
Monte Carlo Simulation Engine for Project Forecasting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from loguru import logger

@dataclass
class SimulationResult:
    target_date_prob: float # Probability of hitting target date
    p50_date: datetime
    p85_date: datetime
    p95_date: datetime
    simulation_dates: List[datetime]
    risk_factors: Dict[str, str] # e.g. "Estimation Bias: 2.3x"
    scenario_name: str = "Baseline"

class DeliveryForecaster:
    """
    Advanced forecasting engine using Monte Carlo simulations.
    Models uncertainties in Velocity, Estimates, and Scope.
    """

    def __init__(self):
        self._rng = np.random.default_rng(seed=42)

    def fit_distribution(self, data: List[float], dist_type='norm') -> Any:
        # Simple wrapper, can be expanded to auto-select best fit
        if len(data) < 5:
            return None
        
        if dist_type == 'norm':
            mu, std = stats.norm.fit(data)
            return stats.norm(mu, std)
        elif dist_type == 'lognorm':
            # shape, loc, scale
            shape, loc, scale = stats.lognorm.fit(data)
            return stats.lognorm(shape, loc=loc, scale=scale)
        elif dist_type == 'beta':
            a, b, loc, scale = stats.beta.fit(data)
            return stats.beta(a, b, loc=loc, scale=scale)
        return None

    def analyze_historical_performance(self, 
                                     completed_tickets: pd.DataFrame, 
                                     sprints: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze team history to extract uncertainty parameters.
        Returns dictionary with distribution parameters.
        """
        if completed_tickets.empty:
            return self._default_parameters()

        # 1. Estimation Bias (Actual vs Estimate)
        if 'time_spent_seconds' in completed_tickets.columns and 'original_estimate_seconds' in completed_tickets.columns:
            valid = completed_tickets.dropna(subset=['time_spent_seconds', 'original_estimate_seconds'])
            valid = valid[valid['original_estimate_seconds'] > 0]
            
            if not valid.empty:
                ratios = valid['time_spent_seconds'] / valid['original_estimate_seconds']
                # Filter outliers (e.g., > 10x)
                ratios = ratios[ratios < 10]
                bias_mean = ratios.mean()
                bias_std = ratios.std()
            else:
                bias_mean, bias_std = 1.2, 0.4
        else:
            bias_mean, bias_std = 1.2, 0.4 # Default pessimist

        # 2. Velocity Distribution (Throughput per week)
        # Group by week
        completed_tickets['week'] = pd.to_datetime(completed_tickets['resolved']).dt.to_period('W')
        weekly_throughput = completed_tickets.groupby('week').size()
        
        if len(weekly_throughput) > 4:
            velocity_dist = self.fit_distribution(weekly_throughput.values, 'norm') # Norm fits velocity reasonably often
            vel_mean, vel_std = weekly_throughput.mean(), weekly_throughput.std()
        else:
            velocity_dist = None
            vel_mean, vel_std = 5.0, 2.0

        # 3. Scope Creep (Tickets added after sprint start)
        # Needs sprint changelog analysis. Mocking 'creep rate' distribution for now if data missing.
        creep_mean, creep_std = 0.15, 0.10 # 15% scope creep avg

        return {
            "estimation_bias_mean": float(bias_mean),
            "estimation_bias_std": float(bias_std),
            "velocity_mean": float(vel_mean),
            "velocity_std": float(vel_std),
            "scope_creep_mean": float(creep_mean),
            "scope_creep_std": float(creep_std)
        }

    def _default_parameters(self):
        return {
            "estimation_bias_mean": 1.5,
            "estimation_bias_std": 0.5,
            "velocity_mean": 10.0,
            "velocity_std": 3.0,
            "scope_creep_mean": 0.1,
            "scope_creep_std": 0.05
        }

    def run_simulation(
        self,
        remaining_backlog_items: int,
        target_date: datetime,
        historical_params: Dict[str, Any],
        n_simulations: int = 5000,
        start_date: Optional[datetime] = None,
        # What-If Overrides
        team_size_multiplier: float = 1.0,
        scope_cut_percentage: float = 0.0,
        estimation_fix_factor: float = 1.0
    ) -> SimulationResult:
        
        start_date = start_date or datetime.now()
        
        # Apply What-Ifs
        nominal_velocity_mean = historical_params['velocity_mean'] * team_size_multiplier
        nominal_velocity_std = historical_params['velocity_std'] * np.sqrt(team_size_multiplier) # variance scales slightly differently usually
        
        initial_work_items = remaining_backlog_items * (1.0 - scope_cut_percentage)
        
        # Estimation correction (e.g., if we say we "fix bias", we reduce the mean bias towards 1.0)
        # bias_mean = 1.5. fix_factor = 0.5 -> new_bias = 1.5 - (1.5-1.0)*0.5 = 1.25
        bias_mean = historical_params['estimation_bias_mean']
        bias_eff = 1.0 + (bias_mean - 1.0) * estimation_fix_factor # Factor 1.0 means KEEP bias, 0 means REMOVE bias
        # Oops prompt logic implies "Fix bias" improves it. 
        # Let's interpret estimation_fix_factor as "Improvement": 0 = no change, 1 = perfect estimation (bias=1)
        final_bias_mean = bias_mean - (bias_mean - 1.0) * estimation_fix_factor
        
        simulated_durations_weeks = []
        
        # Vectorized Simulation? Hard because of while loop per sim. 
        # But we can approximate total work distribution / velocity distribution.
        
        # Total Work Distribution = Initial * (1 + Creep) * Bias
        # Creep ~ Normal(mean, std) clipped > 0
        # Bias ~ Normal(mean, std) clipped > 0.5
        
        creeps = self._rng.normal(
            historical_params['scope_creep_mean'], 
            historical_params['scope_creep_std'], 
            n_simulations
        )
        creeps = np.maximum(creeps, 0) # No negative creep
        
        biases = self._rng.normal(
            final_bias_mean,
            historical_params['estimation_bias_std'], # Assume std stays similar for now
            n_simulations
        )
        biases = np.maximum(biases, 0.5)

        total_work_load = initial_work_items * (1 + creeps)
        # Note: If backlog items are "points" or "hours", apply bias. If just "counts", bias applies if count represents "stories" that tend to split.
        # Let's assume input is "Estimated Points" or "Estimated Counts". Bias applies.
        total_work_load = total_work_load * biases
        
        # Velocity per week ~ Normal
        # Duration = Total Work / Velocity
        # Simulation of weekly draws is slower. 
        # Approximation: Duration = Total Work / Average_Velocity_Sample
        # Better: Duration = Total Work / Velocity_Distribution
        
        velocities = self._rng.normal(nominal_velocity_mean, nominal_velocity_std, n_simulations)
        velocities = np.maximum(velocities, 0.1) # Avoid div/0 and negative velocity
        
        durations = total_work_load / velocities # in weeks
        
        simulated_durations_weeks = durations

        # Calculate Dates
        simulation_dates = []
        target_ts = target_date.timestamp()
        hits = 0
        
        # Convert weeks to dates
        # Simulating just the end date
        seconds_per_week = 7 * 24 * 3600
        start_ts = start_date.timestamp()
        
        end_timestamps = start_ts + (simulated_durations_weeks * seconds_per_week)
        simulation_dates = pd.to_datetime(end_timestamps, unit='s')
        
        hits = np.sum(end_timestamps <= target_ts)
        prob = hits / n_simulations
        
        return SimulationResult(
            target_date_prob=prob,
            p50_date=pd.to_datetime(np.percentile(end_timestamps, 50), unit='s'),
            p85_date=pd.to_datetime(np.percentile(end_timestamps, 85), unit='s'),
            p95_date=pd.to_datetime(np.percentile(end_timestamps, 95), unit='s'),
            simulation_dates=simulation_dates,
            risk_factors={
                "Historical Bias": f"{bias_mean:.1f}x",
                "Historical Velocity Variance": f"±{int((historical_params['velocity_std']/historical_params['velocity_mean'])*100)}%",
                "Scope Creep Rate": f"{historical_params['scope_creep_mean']*100:.0f}%"
            }
        )
