
import sys
import os

# Add local src directory to path
sys.path.append(os.getcwd())

try:
    from src.jira_client.auth import create_jira_client_from_settings
    from config.settings import get_settings
    
    auth = create_jira_client_from_settings()
    user = auth.get_current_user()
    print(f"Connected as: {user.display_name} ({user.email})")
    
    settings = get_settings()
    print(f"Target Project Key: {settings.jira.project_key}")
    print(f"Target Board ID: {settings.jira.board_id}")
    
    if auth.test_project_access(settings.jira.project_key):
        print("SUCCESS: Access confirmed to project.")
    else:
        print("FAILURE: Cannot access project.")
        
except Exception as e:
    print(f"ERROR: {e}")
