# agent_framework/tools.py
"""
Tools for the agent framework to use with LangChain
"""
from typing import Dict, List, Optional, Any
import logging

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from agent_framework.specialized_agents import (
    NewsGatheringAgent,
    GameAnalysisAgent,
    ScriptEditingAgent,
    FanReactionAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsInput(BaseModel):
    """Input for the news gathering tool."""
    topics: str = Field(
        ...,
        description="Comma-separated list of MLB topics to gather news on"
    )
    requirements: Optional[str] = Field(
        None,
        description="Any specific requirements for the news gathering"
    )

class NewsTool(BaseTool):
    """Tool for gathering MLB news."""
    name: str = "news_gathering_tool"
    description: str = """
    Use this tool to gather recent MLB news on specific topics.
    Input should include topics (comma-separated) and any specific requirements.
    """
    args_schema: Optional[type[BaseModel]] = NewsInput
    news_agent: NewsGatheringAgent

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, news_agent: NewsGatheringAgent):
        """Initialize with a NewsGatheringAgent."""
        super().__init__()
        self.news_agent = news_agent
    
    def _run(self, topics: str, requirements: Optional[str] = None) -> str:
        """Run the news gathering tool."""
        topics_list = [t.strip() for t in topics.split(",")]
        news_items = self.news_agent.gather_news(
            topics=topics_list,
            requirements=requirements or ""
        )
        
        # Format the results for return
        results = []
        for item in news_items:
            headline = item.get("headline", "Untitled")
            content = item.get("content", "")
            source = item.get("source", "MLB News")
            results.append(f"HEADLINE: {headline}\nCONTENT: {content}\nSOURCE: {source}")
        
        return "\n\n".join(results)
    
    def __init__(self, news_agent: NewsGatheringAgent, **kwargs):
            """Override init and forward all args to BaseTool via kwargs."""
            super().__init__(news_agent=news_agent, **kwargs)

class GameStatsInput(BaseModel):
    """Input for the game analysis tool."""
    focus: str = Field(
        ...,
        description="What aspect of games to focus on (e.g., 'pitching performance', 'team standings')"
    )
    teams: Optional[str] = Field(
        None,
        description="Comma-separated list of teams to focus on (leave empty for all teams)"
    )

class GameStatsTool(BaseTool):
    """Tool for analyzing MLB game statistics."""
    name: str = "game_analysis_tool"
    description: str = """
    Use this tool to analyze MLB game statistics and performances.
    Input should include a focus area and optionally specific teams.
    """
    args_schema: Optional[type[BaseModel]] = GameStatsInput
    game_agent: GameAnalysisAgent

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, game_agent: GameAnalysisAgent):
        """Initialize with a GameAnalysisAgent."""
        super().__init__()
        self.game_agent = game_agent
    
    def _run(self, focus: str, teams: Optional[str] = None) -> str:
        """Run the game analysis tool."""
        team_list = None
        if teams:
            team_list = [t.strip() for t in teams.split(",")]
        
        analysis = self.game_agent.analyze_games(
            focus=focus,
            team_filter=team_list
        )
        
        # Format the results for return
        results = []
        for section, content in analysis.items():
            section_title = section.replace("_", " ").upper()
            results.append(f"{section_title}:\n{content}")
        
        return "\n\n".join(results)
    
    def __init__(self, game_agent: GameAnalysisAgent, **kwargs):
            """Override init and forward all args to BaseTool via kwargs."""
            super().__init__(game_agent=game_agent, **kwargs)
    


class ScriptEditInput(BaseModel):
    """Input for the script editing tool."""
    script: str = Field(
        ...,
        description="The podcast script to edit and improve"
    )
    requirements: Optional[str] = Field(
        None,
        description="Specific requirements or notes for the script editing"
    )

class ScriptEditingTool(BaseTool):
    """Tool for editing and improving podcast scripts."""
    name: str = "script_editing_tool"
    description: str = """
    Use this tool to edit and improve podcast scripts.
    Input should include the script to edit and any specific requirements.
    """
    args_schema: Optional[type[BaseModel]] = ScriptEditInput
    script_agent: ScriptEditingAgent

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, script_agent: ScriptEditingAgent):
        """Initialize with a ScriptEditingAgent."""
        super().__init__()
        self.script_agent = script_agent
    
    def _run(self, script: str, requirements: Optional[str] = None) -> str:
        """Run the script editing tool."""
        edited_script = self.script_agent.refine_script(
            script=script,
            edit_notes=requirements or ""
        )
        
        return edited_script

    def __init__(self, script_agent: ScriptEditingAgent, **kwargs):
            """Override init and forward all args to BaseTool via kwargs."""
            super().__init__(script_agent=script_agent, **kwargs)


class FanReactionInput(BaseModel):
    """Input for the fan reaction tool."""
    topics: str = Field(
        ...,
        description="Comma-separated list of MLB topics to gather fan reactions on"
    )
    platforms: Optional[str] = Field(
        None,
        description="Comma-separated list of social media platforms to focus on"
    )

class FanReactionTool(BaseTool):
    """Tool for gathering fan reactions from social media."""
    name: str = "fan_reaction_tool"
    description: str = """
    Use this tool to gather fan reactions and sentiment from social media.
    Input should include topics and optionally specific platforms.
    """
    args_schema: Optional[type[BaseModel]] = FanReactionInput
    fan_agent: FanReactionAgent

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, fan_agent: FanReactionAgent):
        """Initialize with a FanReactionAgent."""
        super().__init__()
        self.fan_agent = fan_agent
    
    def _run(self, topics: str, platforms: Optional[str] = None) -> str:
        """Run the fan reaction tool."""
        topics_list = [t.strip() for t in topics.split(",")]
        
        platform_list = None
        if platforms:
            platform_list = [p.strip() for p in platforms.split(",")]
        
        reactions = self.fan_agent.gather_reactions(
            topics=topics_list,
            platforms=platform_list
        )
        
        # Format the results for return
        formatted = "FAN REACTIONS:\n" + "\n".join([f"- {r}" for r in reactions])
        
        return formatted
    def __init__(self, fan_agent: FanReactionAgent, **kwargs):
            """Override init and forward all args to BaseTool via kwargs."""
            super().__init__(fan_agent=fan_agent, **kwargs)


class RAGQueryInput(BaseModel):
    """Input for the RAG query tool."""
    query: str = Field(
        ...,
        description="The question or query to answer using the knowledge base"
    )

class RAGQueryTool(BaseTool):
    """Tool for querying the knowledge base using RAG."""
    name: str = "rag_query_tool"
    description: str = """
    Use this tool to query the MLB knowledge base for specific information.
    Input should be a clear, specific question.
    """
    args_schema: Optional[type[BaseModel]] = RAGQueryInput
    
    def _run(self, query: str) -> str:
        """Run the RAG query tool."""
        # Here you would integrate with your existing RAG pipeline
        # For now, import it directly for simplicity
        from mlb_pipeline.pipeline import rag_pipeline
        
        try:
            answer = rag_pipeline(query)
            return f"RAG QUERY RESULT:\n{answer}"
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return "Error executing RAG query. The knowledge base may be unavailable."