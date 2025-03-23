# agent_framework/utils.py
"""
Utility functions for the agent framework
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

def get_today_date_str() -> str:
    """Get today's date as a formatted string."""
    return datetime.now().strftime("%B %d, %Y")

def save_json(data: Any, filename: str) -> str:
    """Save data to a JSON file and return the file path."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    return filename

def load_json(filename: str) -> Any:
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def format_plan_as_markdown(plan: Dict[str, Any]) -> str:
    """Format a podcast plan as a Markdown string for display."""
    md = f"# Podcast Production Plan: {plan.get('topic', 'MLB Update')}\n\n"
    
    if "key_storylines" in plan and plan["key_storylines"]:
        md += "## Key Storylines\n\n"
        for storyline in plan["key_storylines"]:
            md += f"- {storyline}\n"
        md += "\n"
    
    if "required_data_sources" in plan and plan["required_data_sources"]:
        md += "## Data Sources\n\n"
        for source in plan["required_data_sources"]:
            md += f"- {source}\n"
        md += "\n"
    
    if "specialized_agents_needed" in plan and plan["specialized_agents_needed"]:
        md += "## Agents Required\n\n"
        for agent in plan["specialized_agents_needed"]:
            md += f"- {agent}\n"
        md += "\n"
    
    if "production_notes" in plan and plan["production_notes"]:
        md += "## Production Notes\n\n"
        md += plan["production_notes"] + "\n\n"
    
    return md