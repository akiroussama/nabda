
import os
import json
import base64
import urllib.request
import urllib.error

def read_env(filepath):
    env_vars = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip("'").strip('"')
    except Exception as e:
        print(f"Error reading .env: {e}")
    return env_vars

def main():
    env = read_env('.env')
    
    url = env.get('JIRA_URL')
    email = env.get('JIRA_EMAIL')
    token = env.get('JIRA_API_TOKEN')
    project_key = env.get('JIRA_PROJECT_KEY')
    board_id = env.get('JIRA_BOARD_ID')

    print(f"URL: {url}")
    print(f"Email: {email}")
    print(f"Project Key: {project_key}")
    print(f"Board ID: {board_id}")

    if not (url and email and token):
        print("Missing credentials in .env")
        return

    # Prepare Auth
    auth_str = f"{email}:{token}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    
    # Check Myself
    print("\nChecking /myself...")
    req = urllib.request.Request(f"{url.rstrip('/')}/rest/api/3/myself")
    req.add_header("Authorization", f"Basic {b64_auth}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"SUCCESS: Connected as {data.get('displayName')} ({data.get('emailAddress')})")
            else:
                print(f"FAILED: Status {response.status}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        print(e.read().decode())
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    # Check Project
    if project_key:
        print(f"\nChecking Project {project_key}...")
        req = urllib.request.Request(f"{url.rstrip('/')}/rest/api/3/project/{project_key}")
        req.add_header("Authorization", f"Basic {b64_auth}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                   print(f"SUCCESS: Project {project_key} exists and is accessible.") 
        except urllib.error.HTTPError as e:
            print(f"HTTP Error accessing project: {e.code}")

if __name__ == "__main__":
    main()
