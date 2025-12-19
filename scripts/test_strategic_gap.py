
import sys
import pandas as pd
from datetime import datetime
from src.intelligence.classifier import WorkClassifier
from src.features.strategic_alignment import StrategicAlignmentAnalyzer

def test_feature_1():
    print("Initializing Classifier...")
    classifier = WorkClassifier()
    
    # Create dummy tickets
    print("Creating dummy tickets...")
    df = pd.DataFrame([
        {
            "key": "TEST-1", 
            "summary": "Implement new User Profile page", 
            "description": "User needs to see their avatar.", 
            "issue_type": "Story",
            "time_spent_seconds": 3600*8,
            "created": datetime.now()
        },
        {
            "key": "TEST-2", 
            "summary": "Fix crash in payment gateway", 
            "description": "Critical null pointer exception.", 
            "issue_type": "Bug",
            "time_spent_seconds": 3600*4,
            "created": datetime.now()
        },
        {
            "key": "TEST-3", 
            "summary": "Update React dependency", 
            "description": "Bump to v18.", 
            "issue_type": "Task",
            "time_spent_seconds": 3600*2,
            "created": datetime.now()
        },
        {
            "key": "TEST-4", 
            "summary": "Refactor database schema for scalability", 
            "description": "Remove unused columns.", 
            "issue_type": "Story", # Shadow work candidate (Tech debt disguised as story)
            "time_spent_seconds": 3600*8,
            "created": datetime.now()
        }
    ])
    
    print("Classifying tickets...")
    df_classified = classifier.classify_tickets(df)
    print("Classificaton results:")
    print(df_classified[['summary', 'predicted_category', 'classification_confidence', 'is_shadow_work']])
    
    print("\nAnalyzing Strategic Alignment...")
    analyzer = StrategicAlignmentAnalyzer(classifier)
    stated_strategy = {
        "New Value": 0.7,
        "Maintenance": 0.2,
        "Tech Debt": 0.1
    }
    
    result = analyzer.calculate_alignment(df_classified, stated_strategy)
    
    print("\n--- Analysis Result ---")
    print(f"Total Drift Cost: ${result.total_drift_cost}")
    print(f"Shadow Work %: {result.shadow_work_percentage*100}%")
    print("Actual Allocation:", result.allocation_actual)
    
    print("\nStatus: SUCCESS")

if __name__ == "__main__":
    test_feature_1()
