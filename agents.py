# agents.py - UPDATED FOR ADAPTIVE PIPELINE
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, LLM

# Updated imports for adaptive tools
from tools.adaptive_weaviate_tools import adaptive_business_context_analyzer, adaptive_schema_discovery_engine
from tools.athena_tools import sql_generation_tool, athena_execution_tool, data_analysis_tool

# Load environment variables
load_dotenv()

class AdaptiveForecastingAgents:
    """Enhanced agents with adaptive AI-driven pipeline"""
    
    def __init__(self):
        # Load agent configurations from YAML
        config_path = Path(__file__).parent / "config" / "agents.yaml"
        try:
            with open(config_path, 'r') as file:
                self.agents_config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"agents.yaml not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing agents.yaml: {e}")
        
        # Configure AWS Bedrock LLM with Claude
        self.llm = LLM(
            model=f"bedrock/{os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-3-5-haiku-20241022-v1:0')}",
            temperature=0.1,
            aws_region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        
        # Map adaptive tools
        self.adaptive_tools_map = {
            # STAGE 1: Adaptive Context Analysis
            "adaptive_business_context_analyzer": adaptive_business_context_analyzer,
            "adaptive_schema_discovery_engine": adaptive_schema_discovery_engine,
            
            # STAGE 3+: SQL Generation and Execution (unchanged)
            "sql_generation_tool": sql_generation_tool, 
            "athena_execution_tool": athena_execution_tool,
            "data_analysis_tool": data_analysis_tool
        }
    
    def _create_agent(self, agent_name: str, tools: list = None) -> Agent:
        """Create an agent from YAML configuration."""
        config = self.agents_config.get(agent_name)
        if not config:
            raise ValueError(f"Agent '{agent_name}' not found in agents.yaml")
        
        # Ensure required fields exist
        required_fields = ['role', 'goal', 'backstory']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' for agent '{agent_name}'")
        
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=tools or [],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def adaptive_context_analyst(self) -> Agent:
        """
        STAGE 1 & 2: Adaptive Context Analyst
        
        This agent now handles BOTH stages of the adaptive pipeline:
        - Stage 1: AI-driven query triage and context enrichment
        - Stage 2: Precision schema discovery with mandatory filtering
        
        Tools:
        1. adaptive_business_context_analyzer (Stage 1)
        2. adaptive_schema_discovery_engine (Stage 2)
        """
        return self._create_agent(
            'adaptive_context_analyst',
            tools=[
                self.adaptive_tools_map['adaptive_business_context_analyzer'],
                self.adaptive_tools_map['adaptive_schema_discovery_engine']
            ]
        )
    
    def sql_developer(self) -> Agent:
        """STAGE 3: SQL Developer (unchanged)"""
        return self._create_agent(
            'sql_developer',
            tools=[self.adaptive_tools_map['sql_generation_tool']]
        )
    
    def athena_executor(self) -> Agent:
        """STAGE 4A: Athena Executor (unchanged)"""
        return self._create_agent(
            'athena_executor',
            tools=[self.adaptive_tools_map['athena_execution_tool']]
        )
    
    def data_insights_analyst(self) -> Agent:
        """STAGE 4B: Data Insights Analyst (unchanged)"""
        return self._create_agent(
            'data_insights_analyst',
            tools=[self.adaptive_tools_map['data_analysis_tool']]
        )