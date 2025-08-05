# agents.py - Enhanced Agentic Data Science System
import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import (
    DataAnalysisTool, DataCleaningTool, FeatureEngineeringTool,
    ModelTrainingTool, ModelValidationTool, DataSplittingTool,
    VisualizationTool, PatternRecognitionTool, RelationshipAnalysisTool,
    MultiDatasetAnalysisTool, ConversationTool, InsightGeneratorTool
)


class EnhancedDataScienceAgents:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    def conversation_orchestrator_agent(self):
        """Main agent that orchestrates user conversation and delegates tasks"""
        return Agent(
            role="Conversation Orchestrator",
            goal="Guide users through data science workflow with intelligent questioning and task delegation",
            backstory="""You are an expert data science consultant who excels at understanding 
            user needs, asking the right questions, and coordinating with specialist agents. 
            You guide users through complex data science workflows by breaking down problems 
            into manageable steps and ensuring clear communication throughout the process.""",
            tools=[ConversationTool(), InsightGeneratorTool()],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

    def senior_data_analyst_agent(self):
        """Enhanced data analyst with deep analytical capabilities"""
        return Agent(
            role="Senior Data Analyst",
            goal="Provide comprehensive data analysis with actionable insights and strategic recommendations",
            backstory="""You are a senior data analyst with 10+ years of experience in 
            statistical analysis, data exploration, and business intelligence. You excel at 
            identifying hidden patterns, data quality issues, and providing strategic insights 
            that drive business decisions. You always ask clarifying questions to understand 
            the business context.""",
            tools=[DataAnalysisTool(), VisualizationTool(), PatternRecognitionTool()],
            llm=self.llm,
            verbose=True
        )

    def data_cleaner_agent(self):
        """Enhanced data cleaning agent with intelligent strategies"""
        return Agent(
            role="Data Cleaning Specialist",
            goal="Intelligently clean and preprocess data with context-aware strategies",
            backstory="""You are a meticulous data cleaning expert who understands that 
            different datasets require different cleaning strategies. You analyze data 
            characteristics before applying cleaning techniques and always explain your 
            reasoning. You consider business context when making cleaning decisions.""",
            tools=[DataCleaningTool()],
            llm=self.llm,
            verbose=True
        )

    def feature_engineer_agent(self):
        """Advanced feature engineering agent with domain awareness"""
        return Agent(
            role="Feature Engineering Expert",
            goal="Create intelligent features that capture domain knowledge and improve model performance",
            backstory="""You are a skilled feature engineering expert with deep understanding 
            of domain-specific feature creation. You don't just apply standard transformations 
            - you think creatively about feature interactions, temporal patterns, and 
            business-relevant derived features. You always explain the reasoning behind 
            feature choices.""",
            tools=[FeatureEngineeringTool()],
            llm=self.llm,
            verbose=True
        )

    def model_strategist_agent(self):
        """Strategic model selection and training agent"""
        return Agent(
            role="Model Strategist",
            goal="Select optimal modeling strategies based on data characteristics and business objectives",
            backstory="""You are an experienced ML strategist who understands that model 
            selection is not just about accuracy - it's about business value, interpretability, 
            deployment constraints, and maintenance costs. You recommend modeling approaches 
            that align with business objectives and technical constraints.""",
            tools=[ModelTrainingTool(), DataSplittingTool()],
            llm=self.llm,
            verbose=True
        )

    def model_validator_agent(self):
        """Comprehensive model validation specialist"""
        return Agent(
            role="Model Validation Specialist",
            goal="Rigorously evaluate models with comprehensive validation strategies",
            backstory="""You are a model validation expert who goes beyond simple accuracy 
            metrics. You understand bias, fairness, robustness, and real-world performance 
            considerations. You design validation strategies that reflect actual deployment 
            conditions and identify potential failure modes.""",
            tools=[ModelValidationTool(), VisualizationTool()],
            llm=self.llm,
            verbose=True
        )

    def relationship_discovery_agent(self):
        """Agent specialized in finding relationships between datasets"""
        return Agent(
            role="Relationship Discovery Expert",
            goal="Identify and analyze relationships, patterns, and connections across multiple datasets",
            backstory="""You are a relationship discovery expert who excels at finding 
            hidden connections between seemingly unrelated datasets. You understand join 
            keys, foreign key relationships, temporal alignments, and semantic similarities. 
            You can identify opportunities for data enrichment and cross-dataset insights.""",
            tools=[RelationshipAnalysisTool(), MultiDatasetAnalysisTool(), PatternRecognitionTool()],
            llm=self.llm,
            verbose=True
        )

    def visualization_specialist_agent(self):
        """Specialized agent for creating insightful visualizations"""
        return Agent(
            role="Visualization Specialist",
            goal="Create compelling and informative visualizations that reveal data insights",
            backstory="""You are a visualization expert who understands that great 
            visualizations tell stories and drive decisions. You choose the right chart 
            types for different data characteristics and audiences. You create both 
            exploratory visualizations for analysis and presentation-ready charts for 
            stakeholders.""",
            tools=[VisualizationTool(), PatternRecognitionTool()],
            llm=self.llm,
            verbose=True
        )

    def multi_dataset_coordinator_agent(self):
        """Agent that coordinates analysis across multiple related datasets"""
        return Agent(
            role="Multi-Dataset Coordinator",
            goal="Orchestrate analysis across multiple related datasets to find comprehensive insights",
            backstory="""You are a multi-dataset analysis expert who understands how to 
            coordinate analysis across related datasets. You identify opportunities for 
            data joining, cross-validation, and ensemble approaches. You ensure consistency 
            in preprocessing and feature engineering across datasets while respecting their 
            unique characteristics.""",
            tools=[MultiDatasetAnalysisTool(), RelationshipAnalysisTool(), DataAnalysisTool()],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )