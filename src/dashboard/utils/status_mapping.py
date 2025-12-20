"""
Status Mapping Utility for French Jira Status Names

This module provides consistent status mappings between English display names
and actual French status values in the database.
"""

# French status values as they appear in the database
STATUS_TODO = 'À faire'
STATUS_IN_PROGRESS = 'En cours'
STATUS_DONE = 'Terminé(e)'

# Status lists for SQL queries
DONE_STATUSES = [STATUS_DONE]  # Only 'Terminé(e)' exists
OPEN_STATUSES = [STATUS_TODO, STATUS_IN_PROGRESS]
TODO_STATUSES = [STATUS_TODO]
IN_PROGRESS_STATUSES = [STATUS_IN_PROGRESS]

# For display purposes - English labels
STATUS_DISPLAY = {
    STATUS_TODO: 'To Do',
    STATUS_IN_PROGRESS: 'In Progress',
    STATUS_DONE: 'Done'
}

# Reverse mapping - English to French
STATUS_ENGLISH_TO_FRENCH = {
    'To Do': STATUS_TODO,
    'Backlog': STATUS_TODO,
    'Open': STATUS_TODO,
    'In Progress': STATUS_IN_PROGRESS,
    'Done': STATUS_DONE,
    'Resolved': STATUS_DONE,
    'Closed': STATUS_DONE,
    'Blocked': STATUS_TODO  # No blocked status, map to To Do
}

# SQL helpers
def get_done_status_sql():
    """Get SQL IN clause for done statuses."""
    return f"('{STATUS_DONE}')"

def get_open_status_sql():
    """Get SQL NOT IN clause value for open (non-done) statuses."""
    return f"('{STATUS_DONE}')"

def get_in_progress_sql():
    """Get SQL clause for in progress status."""
    return f"'{STATUS_IN_PROGRESS}'"

def get_todo_sql():
    """Get SQL clause for to do status."""
    return f"'{STATUS_TODO}'"

def is_done(status):
    """Check if a status represents 'done'."""
    return status == STATUS_DONE

def is_in_progress(status):
    """Check if a status represents 'in progress'."""
    return status == STATUS_IN_PROGRESS

def is_todo(status):
    """Check if a status represents 'to do'."""
    return status == STATUS_TODO

def get_display_status(status):
    """Get English display name for a French status."""
    return STATUS_DISPLAY.get(status, status)

def get_status_class(status):
    """Get CSS class for a status."""
    if status == STATUS_DONE:
        return 'done'
    elif status == STATUS_IN_PROGRESS:
        return 'progress'
    elif status == STATUS_TODO:
        return 'todo'
    else:
        return 'todo'
