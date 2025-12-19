"""
Burnout Barometer - Behavioral Pattern Analysis & Risk Scoring.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

@dataclass
class BurnoutRiskProfile:
    user_id: str
    user_name: str
    risk_score: float # 0-100
    risk_level: str # 'Low', 'Medium', 'High', 'Critical'
    deviations: Dict[str, float] # Feature -> Z-Score or % change
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    top_risk_factors: List[str]

class BurnoutAnalyzer:
    """
    Analyzes user activity patterns to detect burnout risk.
    """
    
    def __init__(self, high_risk_threshold=75, medium_risk_threshold=50):
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold

    def _calculate_metrics(self, df_activity: pd.DataFrame, period_days: int) -> Dict[str, float]:
        """
        Calculate behavioral metrics for a given activity window.
        Input df_activity should have: 'created', 'author_id', 'type' (commit/ticket/comment/etc), 'timestamp'
        For MVP, let's assume we derive this from 'issues' and 'worklogs' or 'changelog'.
        """
        if df_activity.empty:
            return {
                "ticket_volume_per_day": 0.0,
                "total_hours_per_day": 0.0,
                "weekend_ratio": 0.0,
                "after_hours_ratio": 0.0
            }
        
        # 1. Volume
        ticket_count = len(df_activity)
        volume_per_day = ticket_count / max(1, period_days)
        
        # 2. Hours (if worklogs available) - mock if not
        total_hours = df_activity['time_spent_hours'].sum() if 'time_spent_hours' in df_activity.columns else 0
        hours_per_day = total_hours / max(1, period_days)
        
        # 3. Weekend Activity
        # Check if timestamp is Saturday(5) or Sunday(6)
        weekend_mask = df_activity['timestamp'].dt.dayofweek >= 5
        weekend_tickets = df_activity[weekend_mask]
        weekend_ratio = len(weekend_tickets) / ticket_count if ticket_count > 0 else 0.0
        
        # 4. After Hours (e.g. before 9am or after 6pm)
        # Simplified: assume simple 9-18
        hours = df_activity['timestamp'].dt.hour
        after_hours_mask = (hours < 9) | (hours >= 18)
        after_hours_ratio = len(df_activity[after_hours_mask]) / ticket_count if ticket_count > 0 else 0.0
        
        return {
            "ticket_volume_per_day": volume_per_day,
            "total_hours_per_day": hours_per_day,
            "weekend_ratio": weekend_ratio,
            "after_hours_ratio": after_hours_ratio,
        }

    def analyze_team_risks(
        self, 
        df_issues: pd.DataFrame, 
        df_worklogs: pd.DataFrame, 
        users_df: pd.DataFrame
    ) -> List[BurnoutRiskProfile]:
        """
        Analyze burnout risk for all users in the users_df.
        """
        results = []
        
        # Prepare activity stream
        # Merge issues (created/updated) and worklogs (logged time)
        # This is a simplification. Ideally check changelogs for true activity.
        
        # Issue activities
        df_i = df_issues[['assignee_id', 'updated']].rename(columns={'assignee_id': 'user_id', 'updated': 'timestamp'})
        df_i['type'] = 'issue_update'
        df_i['time_spent_hours'] = 0 # Updates don't inherently have time, unless we parse
        
        # Worklog activities
        if not df_worklogs.empty:
            df_w = df_worklogs[['author_id', 'started', 'time_spent_seconds']].rename(
                columns={'author_id': 'user_id', 'started': 'timestamp'}
            )
            df_w['type'] = 'worklog'
            df_w['time_spent_hours'] = df_w['time_spent_seconds'] / 3600.0
            
            activity_stream = pd.concat([df_i, df_w[['user_id', 'timestamp', 'type', 'time_spent_hours']]])
        else:
            activity_stream = df_i
            
        activity_stream = activity_stream.dropna(subset=['user_id', 'timestamp'])
        activity_stream['timestamp'] = pd.to_datetime(activity_stream['timestamp'])
        
        # Current window: Last 30 days
        now = datetime.now()
        current_start = now - timedelta(days=30)
        
        # Baseline window: 30-180 days ago
        baseline_start = now - timedelta(days=180)
        baseline_end = current_start
        
        for _, user in users_df.iterrows():
            uid = user.get('pseudonym') or user.get('account_id') # Use what matches activity stream
            if not uid: 
                continue
                
            u_name = user.get('display_name') or uid
            
            user_activity = activity_stream[activity_stream['user_id'] == uid]
            
            if len(user_activity) < 10:
                continue # Not enough data
            
            # Split into Current vs Baseline
            current_activity = user_activity[user_activity['timestamp'] >= current_start]
            baseline_activity = user_activity[
                (user_activity['timestamp'] >= baseline_start) & 
                (user_activity['timestamp'] < baseline_end)
            ]
            
            if baseline_activity.empty:
                continue
                
            curr_metrics = self._calculate_metrics(current_activity, 30)
            base_metrics = self._calculate_metrics(baseline_activity, 150)
            
            # Calc Deviations (based on normalized daily rates)
            
            deviations = {}
            score = 0
            factors = []
            
            # 1. Workload (Volume per day)
            curr_vol = curr_metrics['ticket_volume_per_day']
            base_vol = base_metrics['ticket_volume_per_day']
            
            vol_change = (curr_vol - base_vol) / (base_vol + 0.01) # Avoid div/0
            deviations['volume_change'] = vol_change
            if vol_change > 0.5: # +50% workload intensity
                score += 20
                factors.append(f"Workload Surge (+{int(vol_change*100)}%)")
                
            # 2. Weekend Work
            weekend_change = curr_metrics['weekend_ratio'] - base_metrics['weekend_ratio']
            deviations['weekend_change'] = weekend_change
            if curr_metrics['weekend_ratio'] > 0.1 and weekend_change > 0.05:
                score += 30
                factors.append(f"Weekend Work Increase (+{int(weekend_change*100)}% pts)")
                
            # 3. After Hours
            after_hours_change = curr_metrics['after_hours_ratio'] - base_metrics['after_hours_ratio']
            deviations['after_hours_change'] = after_hours_change
            if curr_metrics['after_hours_ratio'] > 0.2 and after_hours_change > 0.1:
                score += 25
                factors.append(f"Late Night Activity (+{int(after_hours_change*100)}% pts)")

            # 4. Silence (Disengagement)
            if vol_change < -0.6:
                score += 15
                factors.append("Significant Disengagement (Activity Drop)")

            # Normalize Score
            final_score = min(100, max(0, score + 10)) # Base risk 10
            
            risk_level = "Healthy"
            if final_score >= self.high_risk_threshold:
                risk_level = "High Risk"
            elif final_score >= self.medium_risk_threshold:
                risk_level = "Elevated"
            
            results.append(BurnoutRiskProfile(
                user_id=uid,
                user_name=u_name,
                risk_score=final_score,
                risk_level=risk_level,
                deviations=deviations,
                baseline_metrics=curr_metrics, # Return current for display
                current_metrics=curr_metrics,
                top_risk_factors=factors
            ))
            
        return sorted(results, key=lambda x: x.risk_score, reverse=True)
