# agent_framework/specialized_agents.py
"""
Implementation of specialized agents for different MLB podcast tasks
"""
from typing import List, Dict, Any, Optional
import logging

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsGatheringAgent:
    """Agent specialized in gathering and summarizing MLB news."""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Define the prompt template for news gathering
        self.prompt = ChatPromptTemplate.from_template(
            """You are an MLB news research specialist.
            
            Your task is to gather and summarize the most important MLB news based on the following requirements:
            
            Topics to focus on: {topics}
            
            Additional requirements: {requirements}
            
            Please provide a concise but comprehensive summary of the MLB news that would be relevant for a podcast.
            Format your response as a list of news items with source attribution.
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def gather_news(
        self,
        topics: List[str],
        requirements: str = "",
        sources: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        Gather and summarize MLB news.
        
        Args:
            topics: List of topics to focus on
            requirements: Additional specific requirements
            sources: Optional list of specific sources to use
            
        Returns:
            List of news items with summaries and sources
        """
        # Here you would typically call external APIs, web scrapers, etc.
        # For this example, we'll simulate it with an LLM call
        
        topics_str = ", ".join(topics)
        if not topics_str:
            topics_str = "latest MLB news, trending stories, and significant updates"
        
        logger.info(f"Gathering news on topics: {topics_str}")
        
        # In a real implementation, integrate with your existing scraping functions
        result = self.chain.invoke({
            "topics": topics_str,
            "requirements": requirements
        })
        
        # Process the result into structured news items
        # In a real implementation, you'd parse this more robustly
        news_items = self._parse_news_items(result["text"])
        
        return news_items
    
    def _parse_news_items(self, text: str) -> List[Dict[str, str]]:
        """Parse the LLM output into structured news items."""
        # Simple parsing logic - would be more robust in production
        lines = text.split("\n")
        news_items = []
        
        current_item = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("- ") or line.startswith("* "):
                # Start a new news item
                if current_item:
                    news_items.append(current_item)
                current_item = {
                    "headline": line.strip("- *").strip(),
                    "content": "",
                    "source": "MLB News"  # Default source
                }
            elif current_item is not None:
                # Add to the current news item
                if "Source:" in line:
                    current_item["source"] = line.split("Source:")[1].strip()
                else:
                    if current_item["content"]:
                        current_item["content"] += " " + line
                    else:
                        current_item["content"] = line
        
        # Add the last item if it exists
        if current_item:
            news_items.append(current_item)
            
        return news_items


class GameAnalysisAgent:
    """Agent specialized in analyzing MLB game statistics and performances."""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Define the prompt template for game analysis
        self.prompt = ChatPromptTemplate.from_template(
            """You are an MLB game analysis expert with deep understanding of baseball statistics.
            
            Your task is to analyze recent MLB games focusing on: {focus}
            
            Provide insights on:
            - Key player performances
            - Team trends
            - Statistical highlights
            - Strategic analysis
            
            Format your response as a well-structured analysis with specific sections.
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze_games(
        self,
        focus: str = "recent games",
        team_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze MLB games and statistics.
        
        Args:
            focus: What aspect of the games to focus on
            team_filter: Optional list of teams to filter for
            
        Returns:
            Dictionary containing analysis of games
        """
        # In a real implementation, you would fetch actual game data from APIs
        logger.info(f"Analyzing games with focus: {focus}")
        
        # Filter by teams if provided
        team_focus = ""
        if team_filter:
            teams_str = ", ".join(team_filter)
            team_focus = f" with emphasis on {teams_str}"
        
        result = self.chain.invoke({
            "focus": focus + team_focus
        })
        
        # Process the result into a structured analysis
        analysis = self._structure_analysis(result["text"])
        
        return analysis
    
    def _structure_analysis(self, text: str) -> Dict[str, Any]:
        """Structure the analysis text into sections."""
        # Simple section parsing - would be more robust in production
        sections = {}
        current_section = "overview"
        sections[current_section] = ""
        
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header (simplistic check)
            if line.isupper() or (line.endswith(":") and len(line) < 50):
                # Start a new section
                current_section = line.lower().strip(": ").replace(" ", "_")
                sections[current_section] = ""
            else:
                # Add to current section
                if sections[current_section]:
                    sections[current_section] += "\n" + line
                else:
                    sections[current_section] = line
        
        return sections


class ScriptEditingAgent:
    """Agent specialized in generating and refining podcast scripts."""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.4,
            model="gpt-4o-mini",  # Using more advanced model for creative writing
            api_key=openai_api_key
        )
        
        # Define the prompt template for script generation
        self.generation_prompt = ChatPromptTemplate.from_template(
            """You are an expert MLB podcast script writer.
            
            Your task is to create an engaging podcast script about: {topic}
            
            Use the following information to craft your script:
            {context}
            
            Additional notes: {notes}
            
            Create a podcast script that:
            - Has a natural, conversational tone without speaker labels
            - Flows smoothly between topics
            - Engages the audience with interesting insights and analysis
            - Includes appropriate references to the provided information
            - Is appropriate for a 10-15 minute segment
            
            Format the script as a clean monologue without section headings.
            """
        )
        
        # Define the prompt template for script editing
        self.editing_prompt = ChatPromptTemplate.from_template(
            """You are an expert MLB podcast script editor.
            
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
            """
        )
        
        self.generation_chain = LLMChain(llm=self.llm, prompt=self.generation_prompt)
        self.editing_chain = LLMChain(llm=self.llm, prompt=self.editing_prompt)
    
    def generate_script(
        self,
        topic: str,
        context: Dict[str, Any],
        notes: str = ""
    ) -> str:
        """
        Generate a podcast script based on the provided context.
        
        Args:
            topic: The main topic of the script
            context: Dictionary containing information to include in the script
            notes: Additional notes/requirements for the script
            
        Returns:
            Generated podcast script
        """
        # Format the context for the prompt
        context_str = self._format_context_for_prompt(context)
        
        logger.info(f"Generating script on topic: {topic}")
        
        result = self.generation_chain.invoke({
            "topic": topic,
            "context": context_str,
            "notes": notes or "Make it engaging and conversational."
        })
        
        return result["text"]
    
    def refine_script(
        self,
        script: str,
        edit_notes: str = ""
    ) -> str:
        """
        Refine and improve an existing podcast script.
        
        Args:
            script: The script to refine
            edit_notes: Specific editing requirements
            
        Returns:
            Improved version of the script
        """
        logger.info("Refining podcast script")
        
        result = self.editing_chain.invoke({
            "script": script,
            "edit_notes": edit_notes or "Optimize for natural speech, engagement, and clarity."
        })
        
        return result["text"]
    
    def _format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """Format the context dictionary into a string for the prompt."""
        sections = []
        
        if "topic" in context:
            sections.append(f"TOPIC: {context['topic']}")
        
        if "key_storylines" in context and context["key_storylines"]:
            storylines = "\n".join([f"- {s}" for s in context["key_storylines"]])
            sections.append(f"KEY STORYLINES:\n{storylines}")
        
        if "news" in context and context["news"]:
            news_items = "\n".join([
                f"- {item.get('headline', 'Untitled')}: {item.get('content', '')}"
                for item in context["news"]
            ])
            sections.append(f"NEWS ITEMS:\n{news_items}")
        
        if "game_analysis" in context and context["game_analysis"]:
            analysis = []
            for section, content in context["game_analysis"].items():
                section_title = section.replace("_", " ").upper()
                analysis.append(f"{section_title}:\n{content}")
            sections.append("GAME ANALYSIS:\n" + "\n\n".join(analysis))
        
        if "fan_reactions" in context and context["fan_reactions"]:
            reactions = "\n".join([f"- {r}" for r in context["fan_reactions"]])
            sections.append(f"FAN REACTIONS:\n{reactions}")
        
        return "\n\n".join(sections)


class FanReactionAgent:
    """Agent specialized in gathering and analyzing fan reactions from social media."""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.4,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Define the prompt template for fan reaction analysis
        self.prompt = ChatPromptTemplate.from_template(
            """You are an MLB fan sentiment analyst who specializes in social media trends.
            
            Your task is to gather and analyze fan reactions to the following MLB topics:
            {topics}
            
            Provide insights on:
            - What fans are saying about these topics
            - The general sentiment (positive, negative, mixed)
            - Notable or trending fan opinions
            - Any viral content or memes related to these topics
            
            Format your response as a summary of fan reactions that would be valuable for a podcast.
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def gather_reactions(
        self,
        topics: List[str],
        platforms: List[str] = None
    ) -> List[str]:
        """
        Gather fan reactions from social media on specific topics.
        
        Args:
            topics: List of topics to gather reactions on
            platforms: Optional list of specific platforms to focus on
            
        Returns:
            List of fan reaction insights
        """
        # In a real implementation, you would integrate with social media APIs
        topics_str = ", ".join(topics) if topics else "recent MLB games and news"
        
        logger.info(f"Gathering fan reactions on: {topics_str}")
        
        # Add platform specification if provided
        platform_focus = ""
        if platforms:
            platforms_str = ", ".join(platforms)
            platform_focus = f" focusing on {platforms_str}"
        
        result = self.chain.invoke({
            "topics": topics_str + platform_focus
        })
        
        # Process the result into fan reaction insights
        reactions = self._extract_reactions(result["text"])
        
        return reactions
    
    def _extract_reactions(self, text: str) -> List[str]:
        """Extract individual fan reaction insights from the text."""
        # Simple bullet point extraction - would be more robust in production
        reactions = []
        
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                reactions.append(line.strip("- *").strip())
        
        return reactions