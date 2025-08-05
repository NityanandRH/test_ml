# crew.py - Enhanced Crew with Conversational Flow and Multi-Dataset Support
from crewai import Crew, Process
from agents import EnhancedDataScienceAgents
from tasks import EnhancedDataScienceTasks
from typing import Dict, List, Any, Optional
import json
import os


class EnhancedDataScienceCrew:
    def __init__(self, openai_api_key: str):
        self.agents = EnhancedDataScienceAgents(openai_api_key)
        self.tasks = EnhancedDataScienceTasks()
        self.conversation_history = []
        self.analysis_context = {}

    def handle_user_conversation(self, user_input: str, context: Dict = None) -> str:
        """Handle conversational interaction with the user"""
        try:
            # Initialize conversation orchestrator
            orchestrator = self.agents.conversation_orchestrator_agent()

            # Create conversation task
            conversation_task = self.tasks.conversation_orchestration_task(
                orchestrator, user_input, context or self.analysis_context
            )

            # Create crew for conversation handling
            conversation_crew = Crew(
                agents=[orchestrator],
                tasks=[conversation_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute conversation
            result = conversation_crew.kickoff()

            # Update conversation history
            self.conversation_history.append({
                'user_input': user_input,
                'agent_response': result,
                'context': context or {}
            })

            return result

        except Exception as e:
            return f"Error in conversation handling: {str(e)}"

    def run_comprehensive_analysis_pipeline(self, data_path: str, target_column: str = None,
                                            business_context: str = "", user_preferences: Dict = None) -> Dict:
        """Run comprehensive analysis pipeline with user interaction"""
        try:
            # Initialize agents
            orchestrator = self.agents.conversation_orchestrator_agent()
            data_analyst = self.agents.senior_data_analyst_agent()
            cleaner = self.agents.data_cleaner_agent()
            engineer = self.agents.feature_engineer_agent()
            visualizer = self.agents.visualization_specialist_agent()
            pattern_expert = self.agents.relationship_discovery_agent()

            # Create tasks
            analysis_task = self.tasks.comprehensive_data_analysis_task(
                data_analyst, data_path, target_column, "comprehensive", business_context
            )

            visualization_task = self.tasks.visualization_and_insights_task(
                visualizer, data_path, "comprehensive", "business", target_column
            )

            pattern_task = self.tasks.pattern_recognition_and_anomaly_detection_task(
                pattern_expert, data_path, "all", "medium"
            )

            # Create crew
            analysis_crew = Crew(
                agents=[orchestrator, data_analyst, visualizer, pattern_expert],
                tasks=[analysis_task, visualization_task, pattern_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute analysis
            result = analysis_crew.kickoff()

            # Update context
            self.analysis_context.update({
                'data_path': data_path,
                'target_column': target_column,
                'business_context': business_context,
                'analysis_completed': True
            })

            return {
                'status': 'completed',
                'result': result,
                'context': self.analysis_context
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'context': self.analysis_context
            }

    def run_interactive_cleaning_pipeline(self, data_path: str, user_preferences: Dict = None,
                                          business_context: str = "") -> Dict:
        """Run interactive data cleaning with user guidance"""
        try:
            # Initialize agents
            orchestrator = self.agents.conversation_orchestrator_agent()
            cleaner = self.agents.data_cleaner_agent()
            analyst = self.agents.senior_data_analyst_agent()

            # Create cleaning task
            cleaning_task = self.tasks.intelligent_data_cleaning_task(
                cleaner, data_path, "intelligent", business_context, user_preferences
            )

            # Create validation task
            validation_task = self.tasks.comprehensive_data_analysis_task(
                analyst, data_path.replace('.csv', '_cleaned.csv'),
                self.analysis_context.get('target_column'), "quality_focused", business_context
            )

            # Create crew
            cleaning_crew = Crew(
                agents=[orchestrator, cleaner, analyst],
                tasks=[cleaning_task, validation_task],
                process=Process.sequential,
                verbose=True
            )

            result = cleaning_crew.kickoff()

            # Update context
            self.analysis_context.update({
                'cleaned_data_path': data_path.replace('.csv', '_cleaned.csv'),
                'cleaning_completed': True
            })

            return {
                'status': 'completed',
                'result': result,
                'context': self.analysis_context
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def run_feature_engineering_pipeline(self, data_path: str, target_column: str,
                                         engineering_strategy: str = "comprehensive",
                                         domain_context: str = "") -> Dict:
        """Run intelligent feature engineering pipeline"""
        try:
            # Initialize agents
            engineer = self.agents.feature_engineer_agent()
            analyst = self.agents.senior_data_analyst_agent()

            # Create feature engineering task
            engineering_task = self.tasks.advanced_feature_engineering_task(
                engineer, data_path, target_column, engineering_strategy, domain_context
            )

            # Create feature analysis task
            feature_analysis_task = self.tasks.comprehensive_data_analysis_task(
                analyst, data_path.replace('.csv', '_engineered.csv'),
                target_column, "feature_focused", domain_context
            )

            # Create crew
            engineering_crew = Crew(
                agents=[engineer, analyst],
                tasks=[engineering_task, feature_analysis_task],
                process=Process.sequential,
                verbose=True
            )

            result = engineering_crew.kickoff()

            # Update context
            self.analysis_context.update({
                'engineered_data_path': data_path.replace('.csv', '_engineered.csv'),
                'feature_engineering_completed': True
            })

            return {
                'status': 'completed',
                'result': result,
                'context': self.analysis_context
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def run_modeling_pipeline(self, train_data_path: str, target_column: str,
                              business_objective: str = "", constraints: Dict = None) -> Dict:
        """Run comprehensive modeling pipeline"""
        try:
            # Initialize agents
            strategist = self.agents.model_strategist_agent()
            validator = self.agents.model_validator_agent()

            # Create modeling strategy task
            strategy_task = self.tasks.intelligent_model_strategy_task(
                strategist, train_data_path, target_column, business_objective, constraints
            )

            # Create validation task
            validation_task = self.tasks.comprehensive_model_validation_task(
                validator,
                train_data_path.replace('_train.csv', '_model.pkl'),
                train_data_path.replace('_train.csv', '_test.csv'),
                target_column, "comprehensive"
            )

            # Create crew
            modeling_crew = Crew(
                agents=[strategist, validator],
                tasks=[strategy_task, validation_task],
                process=Process.sequential,
                verbose=True
            )

            result = modeling_crew.kickoff()

            # Update context
            self.analysis_context.update({
                'model_path': train_data_path.replace('_train.csv', '_model.pkl'),
                'modeling_completed': True
            })

            return {
                'status': 'completed',
                'result': result,
                'context': self.analysis_context
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def run_multi_dataset_analysis(self, dataset_paths: List[str],
                                   target_columns: List[str] = None,
                                   analysis_focus: str = "comprehensive") -> Dict:
        """Run comprehensive multi-dataset analysis"""
        try:
            # Initialize agents
            coordinator = self.agents.multi_dataset_coordinator_agent()
            relationship_expert = self.agents.relationship_discovery_agent()
            analyst = self.agents.senior_data_analyst_agent()

            # Create multi-dataset analysis task
            multi_analysis_task = self.tasks.multi_dataset_relationship_analysis_task(
                relationship_expert, dataset_paths, analysis_focus
            )

            # Create coordination task (if target columns provided)
            coordination_task = None
            if target_columns:
                coordination_task = self.tasks.unified_modeling_coordination_task(
                    coordinator, dataset_paths, target_columns, "ensemble"
                )

            # Create crew
            agents = [coordinator, relationship_expert, analyst]
            tasks = [multi_analysis_task]

            if coordination_task:
                tasks.append(coordination_task)

            multi_dataset_crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )

            result = multi_dataset_crew.kickoff()

            # Update context
            self.analysis_context.update({
                'multi_dataset_paths': dataset_paths,
                'multi_dataset_analysis_completed': True,
                'target_columns': target_columns
            })

            return {
                'status': 'completed',
                'result': result,
                'context': self.analysis_context
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def run_full_pipeline(self, data_path: str, target_column: str,
                          problem_description: str = "", user_preferences: Dict = None) -> Dict:
        """Run complete end-to-end pipeline with user interaction"""
        try:
            pipeline_results = {
                'stages_completed': [],
                'results': {},
                'context': {}
            }

            # Stage 1: Comprehensive Analysis
            print("🔍 Starting comprehensive data analysis...")
            analysis_result = self.run_comprehensive_analysis_pipeline(
                data_path, target_column, problem_description, user_preferences
            )
            pipeline_results['stages_completed'].append('analysis')
            pipeline_results['results']['analysis'] = analysis_result

            if analysis_result['status'] == 'error':
                return pipeline_results

            # Stage 2: Interactive Data Cleaning
            print("🧹 Starting interactive data cleaning...")
            cleaning_result = self.run_interactive_cleaning_pipeline(
                data_path, user_preferences, problem_description
            )
            pipeline_results['stages_completed'].append('cleaning')
            pipeline_results['results']['cleaning'] = cleaning_result

            if cleaning_result['status'] == 'error':
                return pipeline_results

            # Stage 3: Advanced Feature Engineering
            print("🔧 Starting advanced feature engineering...")
            cleaned_path = self.analysis_context.get('cleaned_data_path', data_path)
            engineering_result = self.run_feature_engineering_pipeline(
                cleaned_path, target_column, "comprehensive", problem_description
            )
            pipeline_results['stages_completed'].append('feature_engineering')
            pipeline_results['results']['feature_engineering'] = engineering_result

            if engineering_result['status'] == 'error':
                return pipeline_results

            # Stage 4: Comprehensive Modeling
            print("🤖 Starting comprehensive modeling...")
            engineered_path = self.analysis_context.get('engineered_data_path', cleaned_path)

            # Split data first (using existing tool)
            from tools import DataSplittingTool
            splitter = DataSplittingTool()
            split_result = splitter._run(engineered_path, target_column, 0.2)

            train_path = engineered_path.replace('.csv', '_train.csv')
            modeling_result = self.run_modeling_pipeline(
                train_path, target_column, problem_description, user_preferences
            )
            pipeline_results['stages_completed'].append('modeling')
            pipeline_results['results']['modeling'] = modeling_result

            # Update final context
            pipeline_results['context'] = self.analysis_context
            pipeline_results['status'] = 'completed'

            return pipeline_results

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'stages_completed': pipeline_results.get('stages_completed', []),
                'context': self.analysis_context
            }

    def generate_insights_and_recommendations(self, analysis_results: Dict,
                                              user_context: str = "",
                                              business_domain: str = "") -> str:
        """Generate conversational insights from analysis results"""
        try:
            # Initialize insight generator agent
            orchestrator = self.agents.conversation_orchestrator_agent()

            # Create insight generation task
            insight_task = self.tasks.conversational_insight_generation_task(
                orchestrator, analysis_results, user_context, business_domain
            )

            # Create crew
            insight_crew = Crew(
                agents=[orchestrator],
                tasks=[insight_task],
                process=Process.sequential,
                verbose=True
            )

            return insight_crew.kickoff()

        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history

    def get_analysis_context(self) -> Dict:
        """Get the current analysis context"""
        return self.analysis_context

    def reset_context(self):
        """Reset the analysis context and conversation history"""
        self.conversation_history = []
        self.analysis_context = {}

    def save_session(self, file_path: str):
        """Save current session to file"""
        session_data = {
            'conversation_history': self.conversation_history,
            'analysis_context': self.analysis_context
        }

        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

    def load_session(self, file_path: str):
        """Load session from file"""
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)

            self.conversation_history = session_data.get('conversation_history', [])
            self.analysis_context = session_data.get('analysis_context', {})

            return True
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return False