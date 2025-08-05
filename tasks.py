# tasks.py - Enhanced Tasks with Conversational Flow
from crewai import Task
from typing import Dict, List, Any


class EnhancedDataScienceTasks:
    def conversation_orchestration_task(self, agent, user_input: str, context: Dict = None):
        """Task for the conversation orchestrator to manage user interaction"""
        return Task(
            description=f"""
            Process the user input: "{user_input}"

            Context: {context or {} }

            Your responsibilities:
            1. Understand user intent and requirements
            2. Ask clarifying questions if needed
            3. Delegate appropriate tasks to specialist agents
            4. Coordinate the overall workflow
            5. Provide clear, actionable responses
            6. Guide the user through the data science process

            Always consider the business context and user's level of expertise.
            """,
            expected_output="Contextual response with next steps or delegation to appropriate agents",
            agent=agent
        )

    def comprehensive_data_analysis_task(self, agent, data_path: str, target_column: str = None,
                                         analysis_depth: str = "comprehensive", business_context: str = ""):
        """Enhanced data analysis task with business context"""
        return Task(
            description=f"""
            Perform comprehensive analysis of the dataset at {data_path}
            Target column: {target_column or "Not specified"}
            Analysis depth: {analysis_depth}
            Business context: {business_context}

            Your analysis should include:
            1. **Data Structure Analysis**:
               - Dataset dimensions and memory usage
               - Column types and characteristics
               - Data quality assessment

            2. **Statistical Analysis**:
               - Descriptive statistics for numerical columns
               - Distribution analysis and skewness detection
               - Correlation analysis between features

            3. **Data Quality Assessment**:
               - Missing value patterns and impact
               - Duplicate records identification
               - Outlier detection and severity

            4. **Target Variable Analysis** (if specified):
               - Target distribution and balance
               - Problem type identification (classification/regression)
               - Target-feature relationships

            5. **Business Insights**:
               - Actionable insights based on business context
               - Data collection recommendations
               - Feature engineering opportunities

            6. **Next Steps Recommendations**:
               - Data cleaning priorities
               - Feature engineering suggestions
               - Modeling approach recommendations

            Present findings in a clear, business-friendly format with specific recommendations.
            """,
            expected_output="Comprehensive data analysis report with business insights and actionable recommendations",
            agent=agent
        )

    def intelligent_data_cleaning_task(self, agent, data_path: str, cleaning_strategy: str = "intelligent",
                                       business_context: str = "", user_preferences: Dict = None):
        """Task for intelligent data cleaning with user interaction"""
        return Task(
            description=f"""
            Clean the dataset at {data_path} using {cleaning_strategy} strategy
            Business context: {business_context}
            User preferences: {user_preferences or {} }

            Your cleaning approach should:
            1. **Analyze Data Characteristics**:
               - Assess missing value patterns and causes
               - Identify outlier types and business relevance
               - Evaluate duplicate records impact

            2. **Apply Context-Aware Cleaning**:
               - Choose appropriate imputation strategies per column
               - Handle outliers based on business domain
               - Preserve important data relationships

            3. **User Interaction**:
               - Ask user about tolerance for data loss
               - Confirm cleaning decisions for critical columns
               - Explain trade-offs of different approaches

            4. **Quality Validation**:
               - Validate cleaning results
               - Document all cleaning steps taken
               - Assess impact on data distribution

            5. **Generate Recommendations**:
               - Suggest additional cleaning steps if needed
               - Recommend data collection improvements
               - Identify potential data quality monitoring needs

            Provide detailed logging of all cleaning operations and their rationale.
            """,
            expected_output="Cleaned dataset with comprehensive cleaning log and quality assessment",
            agent=agent
        )

    def advanced_feature_engineering_task(self, agent, data_path: str, target_column: str,
                                          engineering_strategy: str = "comprehensive",
                                          domain_context: str = "", user_requirements: Dict = None):
        """Advanced feature engineering task with domain awareness"""
        return Task(
            description=f"""
            Engineer features for the dataset at {data_path} with target {target_column}
            Strategy: {engineering_strategy}
            Domain context: {domain_context}
            User requirements: {user_requirements or {} }

            Your feature engineering should include:
            1. **Intelligent Encoding Strategy**:
               - Analyze categorical variable cardinality
               - Apply appropriate encoding (one-hot, label, target, frequency)
               - Handle high-cardinality categories intelligently

            2. **Numerical Feature Enhancement**:
               - Apply appropriate scaling based on distribution
               - Create polynomial features for key variables
               - Generate ratio and interaction features

            3. **Domain-Specific Features**:
               - Create business-relevant derived features
               - Apply domain knowledge for feature creation
               - Generate temporal features from date columns

            4. **Feature Selection and Optimization**:
               - Remove redundant and highly correlated features
               - Apply feature importance ranking
               - Optimize feature set size vs. performance

            5. **User Interaction**:
               - Ask about specific domain features to create
               - Confirm feature creation strategies
               - Explain feature engineering decisions

            6. **Validation and Documentation**:
               - Validate feature distributions
               - Document all feature transformations
               - Provide feature interpretation guide

            Focus on creating features that will improve model performance while maintaining interpretability.
            """,
            expected_output="Feature-engineered dataset with detailed feature documentation and transformation log",
            agent=agent
        )

    def multi_dataset_relationship_analysis_task(self, agent, dataset_paths: List[str],
                                                 analysis_focus: str = "comprehensive"):
        """Task to analyze relationships across multiple datasets"""
        return Task(
            description=f"""
            Analyze relationships and patterns across multiple datasets: {dataset_paths}
            Analysis focus: {analysis_focus}

            Your analysis should cover:
            1. **Schema Analysis**:
               - Compare column structures across datasets
               - Identify common and similar columns
               - Analyze data type compatibility

            2. **Join Opportunity Analysis**:
               - Identify potential join keys
               - Assess data overlap and join feasibility
               - Recommend optimal join strategies

            3. **Pattern Recognition**:
               - Find common patterns across datasets
               - Identify complementary information
               - Detect data consistency issues

            4. **Integration Opportunities**:
               - Suggest data integration strategies
               - Recommend feature sharing approaches
               - Identify ensemble modeling opportunities

            5. **Cross-Dataset Insights**:
               - Find relationships between different data sources
               - Identify data enrichment opportunities
               - Suggest unified modeling approaches

            6. **Business Value Assessment**:
               - Evaluate business value of dataset combinations
               - Recommend prioritization of integration efforts
               - Suggest data governance improvements

            Provide actionable recommendations for leveraging multiple datasets effectively.
            """,
            expected_output="Comprehensive multi-dataset relationship analysis with integration recommendations",
            agent=agent
        )

    def intelligent_model_strategy_task(self, agent, train_data_path: str, target_column: str,
                                        business_objective: str = "", constraints: Dict = None):
        """Task for intelligent model selection and strategy"""
        return Task(
            description=f"""
            Develop optimal modeling strategy for {train_data_path} with target {target_column}
            Business objective: {business_objective}
            Constraints: {constraints or {} }

            Your strategy should include:
            1. **Problem Characterization**:
               - Determine problem type (classification/regression)
               - Assess data size and complexity
               - Identify modeling challenges

            2. **Model Selection Strategy**:
               - Choose appropriate algorithms based on data characteristics
               - Consider business constraints (interpretability, speed, accuracy)
               - Plan ensemble strategies if beneficial

            3. **Training Approach**:
               - Design validation strategy
               - Plan hyperparameter optimization
               - Consider computational constraints

            4. **Business Alignment**:
               - Align model selection with business objectives
               - Consider deployment requirements
               - Plan for model maintenance and updates

            5. **Risk Assessment**:
               - Identify potential overfitting risks
               - Consider data drift and model decay
               - Plan monitoring and validation strategies

            6. **User Interaction**:
               - Ask about accuracy vs interpretability trade-offs
               - Confirm computational budget and timeline
               - Discuss deployment environment constraints

            Provide a comprehensive modeling strategy with clear rationale for all decisions.
            """,
            expected_output="Comprehensive modeling strategy with trained models and performance analysis",
            agent=agent
        )

    def comprehensive_model_validation_task(self, agent, model_path: str, test_data_path: str,
                                            target_column: str, validation_focus: str = "comprehensive"):
        """Enhanced model validation task with bias detection"""
        return Task(
            description=f"""
            Perform comprehensive validation of model at {model_path} using test data {test_data_path}
            Target column: {target_column}
            Validation focus: {validation_focus}

            Your validation should include:
            1. **Performance Metrics**:
               - Calculate appropriate metrics for problem type
               - Provide confidence intervals for metrics
               - Compare against baseline models

            2. **Model Behavior Analysis**:
               - Analyze prediction distributions
               - Identify model biases and blind spots
               - Assess model calibration

            3. **Feature Importance Analysis**:
               - Analyze feature contributions to predictions
               - Identify most influential features
               - Validate feature importance stability

            4. **Robustness Testing**:
               - Test model performance on different data subsets
               - Analyze performance across different feature ranges
               - Identify potential failure modes

            5. **Business Impact Assessment**:
               - Translate technical metrics to business value
               - Assess model's alignment with business objectives
               - Identify potential business risks

            6. **Deployment Readiness**:
               - Assess model's production readiness
               - Identify monitoring requirements
               - Recommend model governance practices

            Provide actionable recommendations for model improvement and deployment.
            """,
            expected_output="Comprehensive model validation report with deployment recommendations",
            agent=agent
        )

    def visualization_and_insights_task(self, agent, data_path: str, analysis_focus: str = "comprehensive",
                                        target_audience: str = "technical", target_column: str = None):
        """Task for creating insightful visualizations"""
        return Task(
            description=f"""
            Create comprehensive visualizations for dataset at {data_path}
            Analysis focus: {analysis_focus}
            Target audience: {target_audience}
            Target column: {target_column or "Not specified"}

            Your visualization suite should include:
            1. **Data Quality Visualizations**:
               - Missing value patterns and heatmaps
               - Data distribution plots
               - Outlier detection charts

            2. **Exploratory Data Analysis Charts**:
               - Feature distribution plots
               - Correlation matrices and heatmaps
               - Target variable analysis (if specified)

            3. **Relationship Analysis**:
               - Feature-target relationships
               - Multi-dimensional scatter plots
               - Categorical variable relationships

            4. **Pattern Recognition Charts**:
               - Time series patterns (if applicable)
               - Clustering visualization
               - Anomaly detection plots

            5. **Business-Focused Insights**:
               - Key performance indicator trends
               - Business metric distributions
               - Actionable insight highlights

            6. **Interactive Elements**:
               - Filterable dashboards
               - Drill-down capabilities
               - Dynamic parameter adjustment

            Tailor visualizations to the target audience and ensure they tell a clear story.
            """,
            expected_output="Comprehensive visualization suite with business insights and interactive elements",
            agent=agent
        )

    def pattern_recognition_and_anomaly_detection_task(self, agent, data_path: str,
                                                       pattern_types: str = "all",
                                                       sensitivity: str = "medium"):
        """Task for advanced pattern recognition and anomaly detection"""
        return Task(
            description=f"""
            Perform advanced pattern recognition and anomaly detection on {data_path}
            Pattern types to analyze: {pattern_types}
            Detection sensitivity: {sensitivity}

            Your analysis should include:
            1. **Temporal Pattern Detection**:
               - Identify time-based patterns and seasonality
               - Detect trend changes and breakpoints
               - Analyze cyclical behaviors

            2. **Statistical Anomaly Detection**:
               - Apply multiple anomaly detection algorithms
               - Identify univariate and multivariate outliers
               - Assess anomaly severity and business impact

            3. **Clustering and Segmentation**:
               - Identify natural data clusters
               - Detect unusual cluster memberships
               - Analyze cluster characteristics

            4. **Relationship Pattern Analysis**:
               - Identify unusual feature relationships
               - Detect correlation changes over time
               - Find hidden interaction patterns

            5. **Business Rule Validation**:
               - Check for business rule violations
               - Identify data consistency issues
               - Detect process anomalies

            6. **Actionable Insights**:
               - Prioritize anomalies by business impact
               - Recommend investigation priorities
               - Suggest process improvements

            Focus on patterns that have clear business implications and actionable outcomes.
            """,
            expected_output="Comprehensive pattern analysis report with prioritized anomalies and business recommendations",
            agent=agent
        )

    def unified_modeling_coordination_task(self, agent, dataset_paths: List[str],
                                           target_columns: List[str], modeling_strategy: str = "ensemble"):
        """Task for coordinating modeling across multiple datasets"""
        return Task(
            description=f"""
            Coordinate unified modeling approach across datasets: {dataset_paths}
            Target columns: {target_columns}
            Modeling strategy: {modeling_strategy}

            Your coordination should include:
            1. **Multi-Dataset Strategy**:
               - Develop unified modeling approach
               - Plan data integration and feature sharing
               - Design ensemble or stacking strategies

            2. **Model Architecture Design**:
               - Choose appropriate model architectures
               - Plan feature sharing and transfer learning
               - Design model combination strategies

            3. **Training Coordination**:
               - Coordinate training across datasets
               - Ensure consistent preprocessing
               - Manage computational resources

            4. **Cross-Validation Strategy**:
               - Design cross-dataset validation
               - Plan holdout strategies
               - Ensure fair performance comparison

            5. **Performance Optimization**:
               - Optimize ensemble weights
               - Fine-tune individual models
               - Balance model complexity vs performance

            6. **Business Value Maximization**:
               - Align models with business objectives
               - Optimize for business metrics
               - Consider deployment and maintenance costs

            Provide a comprehensive strategy for leveraging multiple datasets effectively.
            """,
            expected_output="Unified modeling strategy with coordinated training results and performance analysis",
            agent=agent
        )

    def conversational_insight_generation_task(self, agent, analysis_results: Dict,
                                               user_context: str, business_domain: str = ""):
        """Task for generating conversational insights and recommendations"""
        return Task(
            description=f"""
            Generate conversational insights from analysis results: {analysis_results}
            User context: {user_context}
            Business domain: {business_domain}

            Your insight generation should include:
            1. **Key Findings Summary**:
               - Distill complex analysis into key insights
               - Highlight most important discoveries
               - Prioritize findings by business impact

            2. **Actionable Recommendations**:
               - Provide specific next steps
               - Suggest concrete actions
               - Include implementation guidance

            3. **Business Context Translation**:
               - Translate technical findings to business language
               - Connect insights to business objectives
               - Assess potential business impact

            4. **Risk and Opportunity Assessment**:
               - Identify potential risks in the data/models
               - Highlight opportunities for improvement
               - Suggest mitigation strategies

            5. **Interactive Q&A Preparation**:
               - Anticipate follow-up questions
               - Prepare detailed explanations
               - Ready alternative approaches

            6. **Next Steps Guidance**:
               - Suggest logical next analysis steps
               - Recommend additional data collection
               - Guide modeling decisions

            Present insights in a conversational, accessible format that drives decision-making.
            """,
            expected_output="Conversational insights report with actionable recommendations and next steps guidance",
            agent=agent
        )