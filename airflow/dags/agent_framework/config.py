# agent_framework/config.py
"""
Configuration settings for the MLB podcast agent framework
"""
from typing import Dict, List, Any, Optional
import os

# Agent system settings
AGENT_CONFIG = {
    # OpenAI API model settings
    "models": {
        "orchestrator": "gpt-4o-mini",  # Use more capable model for coordination
        "news_agent": "gpt-3.5-turbo",
        "game_analysis_agent": "gpt-3.5-turbo",
        "script_editing_agent": "gpt-4o-mini",  # Better quality for content creation
        "fan_reaction_agent": "gpt-3.5-turbo"
    },
    
    # Temperature settings for different agents
    "temperatures": {
        "orchestrator": 0.2,  # Lower temperature for more deterministic planning
        "news_agent": 0.3,
        "game_analysis_agent": 0.3,
        "script_editing_agent": 0.7,  # Higher temperature for creative writing
        "fan_reaction_agent": 0.4
    },
    
    # Capabilities and responsibilities for each agent
    "capabilities": {
        "orchestrator": [
            "Plan overall podcast content strategy",
            "Coordinate specialized agents",
            "Prioritize information gathering tasks",
            "Ensure coherent narrative structure",
            "Monitor agent outputs for quality"
        ],
        
        "news_agent": [
            "Gather MLB news from various sources",
            "Prioritize news items by relevance and recency",
            "Summarize news concisely",
            "Extract key insights from news articles",
            "Track trends across multiple news sources"
        ],
        
        "game_analysis_agent": [
            "Analyze game statistics and box scores",
            "Identify key player performances",
            "Track team trends and standings",
            "Compare current statistics to historical data",
            "Provide statistical insights for storytelling"
        ],
        
        "script_editing_agent": [
            "Generate engaging podcast scripts",
            "Edit scripts for clarity and flow",
            "Optimize content for audio delivery",
            "Apply consistent brand voice",
            "Format scripts for voice synthesis"
        ],
        
        "fan_reaction_agent": [
            "Monitor social media for fan reactions",
            "Identify trending topics among fans",
            "Gauge sentiment on teams and players",
            "Track viral content related to MLB",
            "Represent fan perspectives in content"
        ]
    },
    
    # Data sources for each agent
    "data_sources": {
        "news_agent": [
            "MLB.com",
            "ESPN MLB",
            "The Athletic",
            "Baseball Reference"
        ],
        
        "game_analysis_agent": [
            "MLB Stats API",
            "Baseball Reference",
            "FanGraphs",
            "Statcast Data"
        ],
        
        "fan_reaction_agent": [
            "Twitter/X",
            "Reddit (r/baseball)",
            "Facebook MLB groups",
            "Instagram"
        ]
    },
    
    # Memory configuration for agents (not fully implemented in current version)
    "memory": {
        "enabled": True,
        "history_length": 10,  # Number of interactions to remember
        "storage": "chroma"    # Vector store for longer-term memory
    },
    
    # Prompt templates and examples for each agent
    "prompt_templates": {
        "news_prompt": """You are an MLB news research specialist.
            
        Your task is to gather and summarize the most important MLB news based on the following requirements:
        
        Topics to focus on: {topics}
        
        Additional requirements: {requirements}
        
        Please provide a concise but comprehensive summary of the MLB news that would be relevant for a podcast.
        Format your response as a list of news items with source attribution.
        """,
        
        "game_analysis_prompt": """You are an MLB game analysis expert with deep understanding of baseball statistics.
        
        Your task is to analyze recent MLB games focusing on: {focus}
        
        Provide insights on:
        - Key player performances
        - Team trends
        - Statistical highlights
        - Strategic analysis
        
        Format your response as a well-structured analysis with specific sections.
        """,
        
        "script_editing_prompt": """You are an expert MLB podcast script editor.
        
        Review and improve the following podcast script:
        
        {script}
        
        Edit notes and requirements: {edit_notes}
        
        Make improvements to:
        - Flow and pacing
        - Clarity and engagement
        - Natural conversational tone
        - Technical accuracy
        - Voice optimization for audio delivery
        
        Provide the complete edited script, ready for recording.
        """,
        
        "fan_reaction_prompt": """You are an MLB fan sentiment analyst who specializes in social media trends.
        
        Your task is to gather and analyze fan reactions to the following MLB topics:
        {topics}
        
        Provide insights on:
        - What fans are saying about these topics
        - The general sentiment (positive, negative, mixed)
        - Notable or trending fan opinions
        - Any viral content or memes related to these topics
        
        Format your response as a summary of fan reactions that would be valuable for a podcast.
        """
    }
}

# MLB content focus settings
MLB_CONTENT_CONFIG = {
    # Topic weights for content prioritization
    "topic_weights": {
        "game_results": 0.8,
        "player_performances": 0.9,
        "team_standings": 0.7,
        "trades_and_transactions": 0.8,
        "injuries": 0.6,
        "prospects": 0.5,
        "historical_context": 0.4
    },
    
    # Key teams to always prioritize (adjust based on current season interest)
    "priority_teams": [
        "New York Yankees",
        "Los Angeles Dodgers",
        "Boston Red Sox",
        "Chicago Cubs",
        "Houston Astros"
    ],
    
    # Current storylines to track
    "active_storylines": [
        "Pennant races",
        "MVP race",
        "Cy Young contenders",
        "Rookie standouts",
        "Record chases",
        "Playoff implications"
    ],
    
    # Content templates for different podcast segments
    "segment_templates": {
        "intro": "Welcome to {podcast_name}, your daily dose of MLB insights and analysis. I'm your host, and today we're covering {main_topics}.",
        
        "news_roundup": "Let's dive into the latest news from around the league. {news_summary}",
        
        "game_recap": "In yesterday's action, {game_highlights}. The standout performance came from {standout_player} who {player_achievement}.",
        
        "deep_dive": "Today's deep dive focuses on {deep_dive_topic}. {detailed_analysis}",
        
        "fan_corner": "In our fan corner segment, here's what baseball fans are talking about today: {fan_reactions}",
        
        "outro": "That's all for today's episode of {podcast_name}. Join us tomorrow for more MLB coverage. Until then, keep enjoying the beautiful game of baseball!"
    }
}

# Voice synthesis configuration
VOICE_CONFIG = {
    # ElevenLabs voice settings
    "voice_id": os.getenv("ELEVENLABS_VOICE_ID", ""),
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.6,
        "similarity_boost": 0.8,
        "style": 0.25,
        "use_speaker_boost": True
    },
    
    # Text formatting for better TTS results
    "format_replacements": {
        "MLB": "M L B",
        "HR": "home run",
        "RBI": "R B I",
        "ERA": "E R A",
        "AL": "A L",
        "NL": "N L",
        "vs.": "versus",
        "vs": "versus"
    },
    
    # Pause settings
    "pauses": {
        "sentence": ". ... ",
        "question": "? ... ",
        "exclamation": "! ... ",
        "paragraph": "\n...\n"
    }
}

# Podcast production settings
PODCAST_CONFIG = {
    "name": "MLB Daily Digest",
    "tagline": "Your daily dose of baseball insights and analysis",
    "episode_length_minutes": 15,
    "format": "monologue",  # alternatively: "interview", "panel", etc.
    "publishing_schedule": "daily",
    "target_audience": "Passionate MLB fans who want in-depth analysis",
    
    # Default structure (percentages of total time)
    "structure": {
        "intro": 0.05,            # 5% of time
        "news_roundup": 0.25,     # 25% of time
        "game_analysis": 0.35,    # 35% of time
        "feature_segment": 0.20,  # 20% of time
        "fan_reactions": 0.10,    # 10% of time
        "outro": 0.05             # 5% of time
    }
}

def get_config(config_name: str) -> Dict[str, Any]:
    """Get a specific configuration dictionary."""
    config_map = {
        "agent": AGENT_CONFIG,
        "mlb_content": MLB_CONTENT_CONFIG,
        "voice": VOICE_CONFIG,
        "podcast": PODCAST_CONFIG
    }
    return config_map.get(config_name, {})