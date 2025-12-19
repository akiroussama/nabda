
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.burnout_models import BurnoutAnalyzer

def test_feature_2():
    print("Testing Burnout Analyzer...")
    
    # 1. Create Dummy Users
    df_users = pd.DataFrame([
        {'account_id': 'user_1', 'display_name': 'Sarah Chen', 'pseudonym': 'user_1'},
        {'account_id': 'user_2', 'display_name': 'Mike Ross', 'pseudonym': 'user_2'},
    ])
    
    # 2. Create Dummy Activity (Issues)
    # Sarah: Normal baseline, then massive surge in last 2 weeks + weekend work
    
    records = []
    
    # Baseline: 6 months ago to 1 month ago
    base_start = datetime.now() - timedelta(days=180)
    for i in range(150):
        # 2 tickets per day approx
        if i % 7 in [5, 6]: continue # Weekend rest
        
        date = base_start + timedelta(days=i)
        records.append({
            'assignee_id': 'user_1',
            'updated': date + timedelta(hours=10), # 10 AM
            'key': f'T-{i}'
        })
        records.append({
            'assignee_id': 'user_2',
            'updated': date + timedelta(hours=14), # 2 PM
            'key': f'T2-{i}'
        })

    # Recent: Last 30 days
    # Sarah goes crazy
    curr_start = datetime.now() - timedelta(days=30)
    for i in range(30):
        date = curr_start + timedelta(days=i)
        
        # High volume (5 tickets/day)
        for j in range(5):
             # Late hours: 9 PM
            records.append({
                'assignee_id': 'user_1',
                'updated': date + timedelta(hours=21), 
                'key': f'CURR-{i}-{j}'
            })
            
        # Weekend work for Sarah
        if i % 7 in [5, 6]:
            records.append({
                'assignee_id': 'user_1',
                'updated': date + timedelta(hours=15),
                'key': f'WE-{i}'
            })

        # Mike stays normal
        if i % 7 not in [5, 6]:
             records.append({
                'assignee_id': 'user_2',
                'updated': date + timedelta(hours=11),
                'key': f'CURR2-{i}'
            })

    df_issues = pd.DataFrame(records)
    df_worklogs = pd.DataFrame() # optional
    
    print(f"Generated {len(df_issues)} activity records.")
    
    analyzer = BurnoutAnalyzer()
    profiles = analyzer.analyze_team_risks(df_issues, df_worklogs, df_users)
    
    print(f"\nAnalyzed {len(profiles)} profiles.")
    
    for p in profiles:
        print(f"\nUser: {p.user_name}")
        print(f"Risk: {p.risk_level} ({p.risk_score})")
        print(f"Factors: {p.top_risk_factors}")
        print(f"Current Vol/Day: {p.current_metrics['ticket_volume_per_day']:.2f} vs Base: {p.baseline_metrics['ticket_volume_per_day']:.2f}")
        
    # Assertions
    sarah = next(p for p in profiles if p.user_id == 'user_1')
    assert sarah.risk_score > 70, "Sarah should be high risk"
    assert "Weekend Work Increase" in str(sarah.top_risk_factors)
    
    print("\nStatus: SUCCESS")

if __name__ == "__main__":
    test_feature_2()
