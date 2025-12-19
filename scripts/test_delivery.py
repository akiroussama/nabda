
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.delivery_forecast import DeliveryForecaster

def test_feature_3():
    print("Testing Delivery Forecast...")
    
    # 1. Mock Data
    dates = pd.date_range(end=datetime.now(), periods=50, freq='W')
    
    # Completed tickets
    # Estim bias = 2.0 (takes twice as long)
    df_complete = pd.DataFrame([
        {
            'resolved': d, 
            'original_estimate_seconds': 3600*4, # 4h est
            'time_spent_seconds': 3600*8 # 8h actual
        }
        for d in dates
    ])
    
    # Mock Velocity: 5 tickets/week
    # We need multiple tickets per week to get velocity count
    # Let's just mock weekly_throughput logic implicit in the analyzer
    # The analyzer groups by week.
    df_complete = pd.concat([df_complete] * 5, ignore_index=True)
    df_complete['resolved'] = pd.concat([pd.Series(dates)] * 5, ignore_index=True).values
    
    forecaster = DeliveryForecaster()
    params = forecaster.analyze_historical_performance(df_complete, pd.DataFrame())
    
    print("\n--- Model Parameters ---")
    print(params)
    
    assert params['estimation_bias_mean'] > 1.8, "Should detect ~2.0 bias"
    assert params['velocity_mean'] > 4.0, "Should detect ~5 velocity"
    
    # 2. Run Simulation
    print("\nRunning Simulation...")
    target = datetime.now() + timedelta(weeks=40)
    result = forecaster.run_simulation(
        remaining_backlog_items=100, # 100 tickets
        target_date=target,
        historical_params=params
    )
    
    print(f"Prob of hitting {target.date()}: {result.target_date_prob*100:.1f}%")
    print(f"P85 Date: {result.p85_date.date()}")
    
    # 3. What-If
    print("\nRunning What-If (Add Team)...")
    res_whatif = forecaster.run_simulation(
        remaining_backlog_items=100,
        target_date=target,
        historical_params=params,
        team_size_multiplier=1.5
    )
    
    print(f"Prob: {result.target_date_prob*100:.1f}% P50: {result.p50_date.date()}")
    print(f"New Prob: {res_whatif.target_date_prob*100:.1f}% P50: {res_whatif.p50_date.date()}")
    
    assert res_whatif.p50_date < result.p50_date, "Adding team should deliver earlier (lower P50 date)"
    
    print("\nStatus: SUCCESS")

if __name__ == "__main__":
    test_feature_3()
