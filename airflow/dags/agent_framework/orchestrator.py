# agent_framework/orchestrator.py
"""
Orchestrator Agent for MLB Podcast System
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.tools import (
    NewsTool,
    GameStatsTool,
    ScriptEditingTool,
    FanReactionTool,
    RAGQueryTool
)
from agent_framework.specialized_agents import (
    NewsGatheringAgent,
    GameAnalysisAgent,
    ScriptEditingAgent,
    FanReactionAgent
)
from agent_framework.utils import get_today_date_str

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PodcastTaskPlan(BaseModel):
    """Structure for the podcast production plan created by the orchestrator."""
    topic: str = Field(description="The main topic for the podcast episode")
    required_data_sources: List[str] = Field(
        description="List of data sources that need to be queried"
    )
    specialized_agents_needed: List[str] = Field(
        description="List of specialized agents that need to be invoked"
    )
    context_requirements: Dict[str, str] = Field(
        description="Specific information needed from each agent"
    )
    key_storylines: List[str] = Field(
        description="Important storylines to include in the podcast"
    )
    production_notes: Optional[str] = Field(
        description="Additional notes for podcast production",
        default=None
    )

class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates the specialized agents
    and determines what information to gather for podcast creation.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        
        # Initialize specialized agents
        self.news_agent = NewsGatheringAgent(openai_api_key)
        self.game_analysis_agent = GameAnalysisAgent(openai_api_key)
        self.script_agent = ScriptEditingAgent(openai_api_key)
        self.fan_reaction_agent = FanReactionAgent(openai_api_key)
        
        # Initialize tools for the orchestrator
        self.tools = [
            NewsTool(news_agent=self.news_agent),
            GameStatsTool(game_agent=self.game_analysis_agent),
            ScriptEditingTool(script_agent=self.script_agent),
            FanReactionTool(fan_agent=self.fan_reaction_agent),
            RAGQueryTool(),
        ]
        
        # Configure the orchestrator agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LangChain agent with the appropriate tools and prompt."""
        # Create LLM
        llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4o-mini",
            api_key=self.openai_api_key
        )
        
        # Define the system prompt for the orchestrator
        system_prompt = """You are an MLB podcast planning expert responsible for coordinating a podcast production system.
        
        Your job is to:
        1. Plan what information and data needs to be collected for today's MLB podcast episode
        2. Decide which specialized agents should be used to gather and process that information
        3. Create a structured plan for the podcast production
        4. Oversee the information gathering process by invoking the right tools in the right order
        5. Synthesize the collected information into a coherent podcast production plan
        
        Today's date is {today_date}. Make sure your podcast planning is relevant to recent MLB events and news.
        
        You have access to several specialized agents through tools:
        - News Gathering Agent: Collects and summarizes MLB news from various sources
        - Game Analysis Agent: Analyzes recent game statistics and player performances
        - Script Editing Agent: Helps refine podcast scripts for better quality
        - Fan Reaction Agent: Gathers fan sentiments and reactions from social media
        - RAG Query Tool: Allows querying the knowledge base for specific information
        
        Create a plan that will result in an engaging, informative podcast for baseball fans.
        """
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Setup the LangChain agent
        tool_functions = [format_tool_to_openai_function(t) for t in self.tools]
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x.get("intermediate_steps", [])
                ),
                "today_date": lambda x: get_today_date_str(),
            }
            | prompt
            | llm.bind_functions(tool_functions)
            | OpenAIFunctionsAgentOutputParser()
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
    
    def create_podcast_plan(
        self, 
        topic: Optional[str] = None, 
        special_focus: Optional[str] = None,
        chat_history: List = None
    ) -> PodcastTaskPlan:
        """
        Create a podcast production plan by orchestrating the specialized agents.
        
        Args:
            topic: Optional specific topic for the podcast
            special_focus: Optional area to emphasize (e.g., "player profiles", "team standings")
            chat_history: Any prior conversation history to consider
            
        Returns:
            A structured podcast production plan
        """
        if chat_history is None:
            chat_history = []
        
        # Create the input for the orchestrator
        if topic:
            if special_focus:
                input_query = f"Create a podcast production plan about {topic} with special focus on {special_focus}."
            else:
                input_query = f"Create a podcast production plan about {topic}."
        else:
            input_query = "Create a podcast production plan about the most relevant MLB topics for today."
        
        # Execute the agent to get a plan
        logger.info(f"Creating podcast plan with input: {input_query}")
        result = self.agent_executor.invoke({
            "input": input_query,
            "chat_history": chat_history
        })
        
        # Parse the result into our structured format
        # This is a simplified version - in production, you'd want more robust parsing
        try:
            # Extract plan details from the agent's response
            # In a real implementation, you'd use a more structured approach to extract this information
            plan_dict = self._extract_plan_from_response(result["output"])
            return PodcastTaskPlan(**plan_dict)
        except Exception as e:
            logger.error(f"Error parsing podcast plan: {e}")
            # Return a basic plan if parsing fails
            return PodcastTaskPlan(
                topic="MLB Daily Update",
                required_data_sources=["MLB.com", "ESPN"],
                specialized_agents_needed=["NewsGatheringAgent", "GameAnalysisAgent"],
                context_requirements={"NewsGatheringAgent": "Get latest MLB news"},
                key_storylines=["Recent game results", "Player highlights"]
            )
    
    def _extract_plan_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract structured plan data from the agent's text response.
        In a production system, this would be more robust.
        """
        # This is a placeholder for more sophisticated parsing
        # In a real system, you might use regex, additional LLM calls, or structured outputs
        
        # Default values
        plan = {
            "topic": "MLB Daily Update",
            "required_data_sources": [],
            "specialized_agents_needed": [],
            "context_requirements": {},
            "key_storylines": []
        }
        
        # Very simple parsing - in production you'd want something more robust
        if "Topic:" in response:
            parts = response.split("Topic:")
            if len(parts) > 1:
                topic_line = parts[1].split("\n")[0].strip()
                plan["topic"] = topic_line
        
        # Extract data sources
        if "Data Sources:" in response:
            sources_section = response.split("Data Sources:")[1].split("\n\n")[0]
            sources = [s.strip().strip("- ") for s in sources_section.split("\n") if s.strip()]
            plan["required_data_sources"] = sources
        
        # Extract agents
        if "Agents:" in response:
            agents_section = response.split("Agents:")[1].split("\n\n")[0]
            agents = [a.strip().strip("- ") for a in agents_section.split("\n") if a.strip()]
            plan["specialized_agents_needed"] = agents
        
        # Extract storylines
        if "Storylines:" in response:
            storylines_section = response.split("Storylines:")[1].split("\n\n")[0]
            storylines = [s.strip().strip("- ") for s in storylines_section.split("\n") if s.strip()]
            plan["key_storylines"] = storylines
        
        # Extract notes if present
        if "Notes:" in response:
            notes_section = response.split("Notes:")[1].split("\n\n")[0].strip()
            plan["production_notes"] = notes_section
            
        return plan
        
    def execute_podcast_plan(self, plan: PodcastTaskPlan) -> Dict[str, Any]:
        """
        Execute the podcast production plan by coordinating the specialized agents.
        
        Args:
            plan: The podcast production plan to execute
            
        Returns:
            Dictionary containing all gathered information and the final podcast script
        """
        results = {"plan": plan.dict()}
        
        # Gather information from each required agent
        for agent_name in plan.specialized_agents_needed:
            logger.info(f"Invoking {agent_name} for podcast production")
            
            if agent_name == "NewsGatheringAgent":
                results["news"] = self.news_agent.gather_news(
                    topics=plan.key_storylines,
                    requirements=plan.context_requirements.get("NewsGatheringAgent", "")
                )
            
            elif agent_name == "GameAnalysisAgent":
                results["game_analysis"] = self.game_analysis_agent.analyze_games(
                    focus=plan.context_requirements.get("GameAnalysisAgent", "")
                )
            
            elif agent_name == "FanReactionAgent":
                results["fan_reactions"] = self.fan_reaction_agent.gather_reactions(
                    topics=plan.key_storylines
                )
        
        # Generate initial script based on gathered information
        logger.info("Generating initial podcast script")
        script_content = self._generate_initial_script(results, plan)
        results["initial_script"] = script_content
        
        # Refine the script using the ScriptEditingAgent
        logger.info("Refining podcast script")
        results["final_script"] = self.script_agent.refine_script(
            script_content,
            edit_notes=plan.production_notes
        )
        
        return results
    
    def _generate_initial_script(self, results: Dict[str, Any], plan: PodcastTaskPlan) -> str:
        """
        Generate an initial podcast script based on the gathered information.
        
        Args:
            results: The information gathered from specialized agents
            plan: The podcast production plan
            
        Returns:
            An initial podcast script
        """
        # Combine all the gathered information into a context for script generation
        context = {
            "topic": plan.topic,
            "key_storylines": plan.key_storylines,
            "news": results.get("news", []),
            "game_analysis": results.get("game_analysis", {}),
            "fan_reactions": results.get("fan_reactions", [])
        }
        
        # Use the script agent to generate the initial script
        script = self.script_agent.generate_script(
            topic=plan.topic,
            context=context,
            notes=plan.production_notes
        )
        
        return script











