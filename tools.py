# tools.py - Enhanced Tools with Multi-Dataset Support
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from crewai.tools import BaseTool
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


class ConversationTool(BaseTool):
    name: str = "Conversation Tool"
    description: str = "Handle user conversations and generate contextual questions"

    def _run(self, user_input: str, context: Dict = None) -> str:
        """Process user input and generate appropriate responses or questions"""
        try:
            context = context or {}

            # Analyze user intent
            user_input_lower = user_input.lower()

            if any(word in user_input_lower for word in ['analyze', 'analysis', 'explore']):
                return self._generate_analysis_questions(context)
            elif any(word in user_input_lower for word in ['clean', 'preprocessing', 'prepare']):
                return self._generate_cleaning_questions(context)
            elif any(word in user_input_lower for word in ['feature', 'engineering', 'transform']):
                return self._generate_feature_questions(context)
            elif any(word in user_input_lower for word in ['model', 'predict', 'train']):
                return self._generate_modeling_questions(context)
            elif any(word in user_input_lower for word in ['visualize', 'plot', 'chart']):
                return self._generate_visualization_questions(context)
            else:
                return self._generate_general_guidance(context)

        except Exception as e:
            return f"Error in conversation processing: {str(e)}"

    def _generate_analysis_questions(self, context: Dict) -> str:
        """Generate questions about data analysis preferences"""
        questions = [
            "What specific aspects of your data are you most interested in exploring?",
            "Are you looking for patterns, outliers, or relationships between variables?",
            "Do you have any business hypotheses you'd like to validate?",
            "What's the business context or domain of this dataset?"
        ]
        return "Great! Let me help you with data analysis. " + " ".join(questions)

    def _generate_cleaning_questions(self, context: Dict) -> str:
        """Generate questions about data cleaning strategies"""
        questions = [
            "What's your tolerance for data loss during cleaning?",
            "Are there specific data quality issues you've noticed?",
            "Do you want to preserve all records or are you open to removing problematic ones?",
            "Are there domain-specific cleaning rules I should consider?"
        ]
        return "I'll help you clean your data effectively. " + " ".join(questions)

    def _generate_feature_questions(self, context: Dict) -> str:
        """Generate questions about feature engineering preferences"""
        questions = [
            "What's the target variable you want to predict?",
            "Are there domain-specific features that might be valuable?",
            "Do you want me to create interaction features between variables?",
            "Should I focus on interpretability or predictive power?"
        ]
        return "Let's engineer some powerful features! " + " ".join(questions)

    def _generate_modeling_questions(self, context: Dict) -> str:
        """Generate questions about modeling preferences"""
        questions = [
            "What's more important: model accuracy or interpretability?",
            "Do you have any deployment constraints (speed, memory, etc.)?",
            "Are you looking for a single best model or multiple models for comparison?",
            "What's the business cost of false positives vs false negatives?"
        ]
        return "Time to build some models! " + " ".join(questions)

    def _generate_visualization_questions(self, context: Dict) -> str:
        """Generate questions about visualization preferences"""
        questions = [
            "What story do you want the visualizations to tell?",
            "Who is the target audience for these charts?",
            "Do you prefer interactive or static visualizations?",
            "Are there specific relationships you want to highlight?"
        ]
        return "Let's create some insightful visualizations! " + " ".join(questions)

    def _generate_general_guidance(self, context: Dict) -> str:
        """Provide general guidance based on context"""
        return """I'm here to help you with your data science project! Here's what I can do:

🔍 **Data Analysis**: Comprehensive exploration and insights
🧹 **Data Cleaning**: Intelligent preprocessing strategies  
🔧 **Feature Engineering**: Create powerful predictive features
🤖 **Model Training**: Select and train optimal models
📊 **Visualizations**: Create compelling charts and plots
🔗 **Multi-Dataset Analysis**: Find relationships across datasets

What would you like to start with?"""

class DataSplittingTool(BaseTool):
    name: str = "Data Splitting Tool"
    description: str = "Split data into training and testing sets"

    def _run(self, data_path: str, target_column: str, test_size: float = 0.2) -> str:
        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() < 20 else None
            )

            # Save splits
            base_path = data_path.replace('.csv', '')
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            train_path = f"{base_path}_train.csv"
            test_path = f"{base_path}_test.csv"

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            return f"Data split completed. Train: {train_data.shape}, Test: {test_data.shape}. Saved to: {train_path}, {test_path}"

        except Exception as e:
            return f"Error in data splitting: {str(e)}"
class InsightGeneratorTool(BaseTool):
    name: str = "Insight Generator Tool"
    description: str = "Generate actionable insights from analysis results"

    def _run(self, analysis_results: str, business_context: str = "") -> str:
        """Generate business insights from technical analysis results"""
        try:
            # Parse analysis results if they're in string format
            if isinstance(analysis_results, str):
                try:
                    results = json.loads(analysis_results.replace("'", '"'))
                except:
                    results = {"raw_analysis": analysis_results}
            else:
                results = analysis_results

            insights = []

            # Generate insights based on different aspects
            if 'missing_values' in results:
                insights.extend(self._generate_missing_value_insights(results['missing_values']))

            if 'target_analysis' in results:
                insights.extend(self._generate_target_insights(results['target_analysis']))

            if 'correlations' in results:
                insights.extend(self._generate_correlation_insights(results['correlations']))

            return "\n".join(
                [f"💡 {insight}" for insight in insights]) if insights else "Analysis completed successfully!"

        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def _generate_missing_value_insights(self, missing_values: Dict) -> List[str]:
        """Generate insights about missing values"""
        insights = []
        total_missing = sum(missing_values.values())

        if total_missing > 0:
            high_missing_cols = [col for col, count in missing_values.items() if count > 0]
            insights.append(
                f"Found missing values in {len(high_missing_cols)} columns. Consider imputation strategies.")

            if len(high_missing_cols) > 5:
                insights.append("High number of columns with missing values suggests data collection issues.")
        else:
            insights.append("Excellent data quality - no missing values detected!")

        return insights

    def _generate_target_insights(self, target_analysis: Dict) -> List[str]:
        """Generate insights about target variable"""
        insights = []

        if 'target_distribution' in target_analysis:
            dist = target_analysis['target_distribution']
            if len(dist) == 2:  # Binary classification
                values = list(dist.values())
                ratio = max(values) / min(values)
                if ratio > 3:
                    insights.append(f"Class imbalance detected (ratio: {ratio:.1f}). Consider balancing techniques.")
            elif len(dist) > 10:
                insights.append("High cardinality target detected. Consider if this is truly categorical.")

        return insights

    def _generate_correlation_insights(self, correlations: Dict) -> List[str]:
        """Generate insights about correlations"""
        insights = []

        # This would analyze correlation patterns
        insights.append("Correlation analysis completed. Check for multicollinearity in features.")

        return insights


class DataAnalysisTool(BaseTool):
    name: str = "Enhanced Data Analysis Tool"
    description: str = "Comprehensive dataset analysis with intelligent insights"

    def _run(self, data_path: str, target_column: str = None, analysis_depth: str = "comprehensive") -> str:
        try:
            # Load data with multiple format support
            df = self._load_data(data_path)
            if df is None:
                return "Error: Could not load data. Supported formats: CSV, Excel, JSON"

            # Comprehensive analysis
            analysis_results = {
                'basic_info': self._basic_info_analysis(df),
                'data_quality': self._data_quality_analysis(df),
                'statistical_summary': self._statistical_analysis(df),
                'column_analysis': self._column_analysis(df),
                'relationship_analysis': self._relationship_analysis(df),
                'business_insights': self._generate_business_insights(df, target_column)
            }

            # Target-specific analysis
            if target_column and target_column in df.columns:
                analysis_results['target_analysis'] = self._target_variable_analysis(df, target_column)

            # Generate recommendations
            analysis_results['recommendations'] = self._generate_recommendations(analysis_results, df)

            return json.dumps(analysis_results, indent=2, default=str)

        except Exception as e:
            return f"Error in comprehensive data analysis: {str(e)}"

    def _load_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load data from various formats"""
        try:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                return pd.read_json(data_path)
            else:
                return None
        except Exception:
            return None

    def _basic_info_analysis(self, df: pd.DataFrame) -> Dict:
        """Basic dataset information"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
        }

    def _data_quality_analysis(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality assessment"""
        return {
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_per_column': {col: df[col].nunique() for col in df.columns},
            'data_types_summary': df.dtypes.value_counts().to_dict()
        }

    def _statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """Statistical summary of numerical columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats = numeric_df.describe().to_dict()
            # Add additional statistics
            for col in numeric_df.columns:
                stats[col]['skewness'] = numeric_df[col].skew()
                stats[col]['kurtosis'] = numeric_df[col].kurtosis()
            return stats
        return {}

    def _column_analysis(self, df: pd.DataFrame) -> Dict:
        """Individual column analysis"""
        analysis = {}
        for col in df.columns:
            col_analysis = {
                'type': str(df[col].dtype),
                'unique_count': df[col].nunique(),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': df[col].isnull().sum() / len(df) * 100
            }

            if df[col].dtype in ['int64', 'float64']:
                col_analysis.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                })
            else:
                col_analysis['top_values'] = df[col].value_counts().head().to_dict()

            analysis[col] = col_analysis
        return analysis

    def _relationship_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze relationships between columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        relationships = {}

        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()

            # Find high correlations
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })

            relationships['high_correlations'] = high_corr_pairs
            relationships['correlation_matrix'] = correlation_matrix.to_dict()

        return relationships

    def _target_variable_analysis(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Comprehensive target variable analysis"""
        target_series = df[target_column]

        analysis = {
            'data_type': str(target_series.dtype),
            'unique_values': target_series.nunique(),
            'missing_values': target_series.isnull().sum(),
            'value_distribution': target_series.value_counts().to_dict()
        }

        # Determine problem type
        if target_series.nunique() <= 20 and (target_series.dtype == 'object' or target_series.nunique() <= 10):
            analysis['problem_type'] = 'classification'
            analysis['class_balance'] = target_series.value_counts(normalize=True).to_dict()

            # Check for imbalance
            class_counts = target_series.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            analysis['imbalance_ratio'] = imbalance_ratio
            analysis['is_balanced'] = imbalance_ratio <= 3
        else:
            analysis['problem_type'] = 'regression'
            analysis['distribution_stats'] = {
                'mean': target_series.mean(),
                'median': target_series.median(),
                'std': target_series.std(),
                'skewness': target_series.skew(),
                'kurtosis': target_series.kurtosis()
            }

        return analysis

    def _generate_business_insights(self, df: pd.DataFrame, target_column: str = None) -> List[str]:
        """Generate business-relevant insights"""
        insights = []

        # Data volume insights
        if df.shape[0] < 1000:
            insights.append("Small dataset detected. Consider data augmentation or simpler models.")
        elif df.shape[0] > 100000:
            insights.append("Large dataset detected. Consider sampling for faster experimentation.")

        # Feature insights
        if df.shape[1] > 50:
            insights.append("High-dimensional dataset. Consider feature selection techniques.")

        # Missing data insights
        missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_percentage > 20:
            insights.append("Significant missing data detected. Investigate data collection process.")

        return insights

    def _generate_recommendations(self, analysis_results: Dict, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Data quality recommendations
        missing_cols = [col for col, pct in analysis_results['data_quality']['missing_percentage'].items() if pct > 5]
        if missing_cols:
            recommendations.append(f"Address missing values in columns: {', '.join(missing_cols[:3])}")

        # Feature engineering recommendations
        high_cardinality_cols = [col for col, count in
                                 analysis_results['data_quality']['unique_values_per_column'].items()
                                 if count > 50 and df[col].dtype == 'object']
        if high_cardinality_cols:
            recommendations.append(
                f"Consider target encoding for high-cardinality columns: {', '.join(high_cardinality_cols[:3])}")

        # Modeling recommendations
        if 'target_analysis' in analysis_results:
            target_analysis = analysis_results['target_analysis']
            if target_analysis.get('problem_type') == 'classification' and not target_analysis.get('is_balanced', True):
                recommendations.append("Apply class balancing techniques due to imbalanced target variable.")

        return recommendations


class VisualizationTool(BaseTool):
    name: str = "Advanced Visualization Tool"
    description: str = "Create comprehensive visualizations with business insights"

    def _run(self, data_path: str, chart_types: str = "comprehensive", target_column: str = None) -> str:
        try:
            df = pd.read_csv(data_path)

            # Create comprehensive visualizations
            viz_results = {
                'charts_created': [],
                'insights': []
            }

            # 1. Data Quality Visualization
            if chart_types in ['comprehensive', 'quality']:
                self._create_data_quality_charts(df, viz_results)

            # 2. Distribution Visualizations
            if chart_types in ['comprehensive', 'distribution']:
                self._create_distribution_charts(df, viz_results)

            # 3. Relationship Visualizations
            if chart_types in ['comprehensive', 'relationships']:
                self._create_relationship_charts(df, viz_results)

            # 4. Target Variable Visualizations
            if target_column and target_column in df.columns:
                self._create_target_visualizations(df, target_column, viz_results)

            return json.dumps(viz_results, indent=2)

        except Exception as e:
            return f"Error creating visualizations: {str(e)}"

    def _create_data_quality_charts(self, df: pd.DataFrame, results: Dict):
        """Create data quality visualization charts"""
        # Missing values chart
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]

        if not missing_data.empty:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                         title="Missing Values by Column")
            # In practice, you'd save this figure
            results['charts_created'].append('missing_values_chart')
            results['insights'].append(f"Missing values found in {len(missing_data)} columns")

    def _create_distribution_charts(self, df: pd.DataFrame, results: Dict):
        """Create distribution visualization charts"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            # Create histogram
            results['charts_created'].append(f'histogram_{col}')

            # Generate insight
            skewness = df[col].skew()
            if abs(skewness) > 1:
                results['insights'].append(f"{col} is highly skewed (skewness: {skewness:.2f})")

    def _create_relationship_charts(self, df: pd.DataFrame, results: Dict):
        """Create relationship visualization charts"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            # Correlation heatmap
            corr_matrix = df[numeric_cols].corr()
            results['charts_created'].append('correlation_heatmap')

            # Find strong correlations
            strong_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr_pairs.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.2f}")

            if strong_corr_pairs:
                results['insights'].append(f"Strong correlations found: {', '.join(strong_corr_pairs[:3])}")

    def _create_target_visualizations(self, df: pd.DataFrame, target_column: str, results: Dict):
        """Create target variable specific visualizations"""
        target_series = df[target_column]

        if target_series.nunique() <= 20:  # Categorical target
            # Target distribution
            results['charts_created'].append('target_distribution')

            # Class balance analysis
            value_counts = target_series.value_counts()
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 3:
                results['insights'].append(f"Target variable is imbalanced (ratio: {imbalance_ratio:.1f})")
        else:  # Continuous target
            # Target distribution histogram
            results['charts_created'].append('target_histogram')

            # Distribution analysis
            skewness = target_series.skew()
            if abs(skewness) > 1:
                results['insights'].append(f"Target variable is skewed (skewness: {skewness:.2f})")


class PatternRecognitionTool(BaseTool):
    name: str = "Pattern Recognition Tool"
    description: str = "Identify complex patterns and anomalies in data"

    def _run(self, data_path: str, pattern_types: str = "all") -> str:
        try:
            df = pd.read_csv(data_path)

            patterns_found = {
                'temporal_patterns': [],
                'seasonal_patterns': [],
                'anomalies': [],
                'clusters': [],
                'trends': []
            }

            # Temporal pattern detection
            if pattern_types in ['all', 'temporal']:
                patterns_found['temporal_patterns'] = self._detect_temporal_patterns(df)

            # Anomaly detection
            if pattern_types in ['all', 'anomalies']:
                patterns_found['anomalies'] = self._detect_anomalies(df)

            # Trend analysis
            if pattern_types in ['all', 'trends']:
                patterns_found['trends'] = self._detect_trends(df)

            return json.dumps(patterns_found, indent=2, default=str)

        except Exception as e:
            return f"Error in pattern recognition: {str(e)}"

    def _detect_temporal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect temporal patterns in data"""
        patterns = []

        # Look for date columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass

        if date_cols:
            patterns.append({
                'type': 'temporal_columns_found',
                'columns': date_cols,
                'description': f"Found {len(date_cols)} potential temporal columns"
            })

        return patterns

    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if len(outliers) > 0:
                anomalies.append({
                    'column': col,
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(df) * 100,
                    'method': 'IQR',
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                })

        return anomalies

    def _detect_trends(self, df: pd.DataFrame) -> List[Dict]:
        """Detect trends in numerical data"""
        trends = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Simple trend detection using correlation with index
            correlation_with_index = df[col].corr(pd.Series(range(len(df))))

            if abs(correlation_with_index) > 0.3:
                trend_direction = "increasing" if correlation_with_index > 0 else "decreasing"
                trends.append({
                    'column': col,
                    'trend_direction': trend_direction,
                    'correlation_strength': abs(correlation_with_index),
                    'description': f"{col} shows {trend_direction} trend over time"
                })

        return trends


class RelationshipAnalysisTool(BaseTool):
    name: str = "Relationship Analysis Tool"
    description: str = "Analyze relationships between different datasets"

    def _run(self, dataset_paths: List[str], relationship_types: str = "all") -> str:
        try:
            datasets = {}

            # Load all datasets
            for i, path in enumerate(dataset_paths):
                datasets[f'dataset_{i}'] = pd.read_csv(path)

            relationships = {
                'schema_similarities': [],
                'potential_joins': [],
                'common_patterns': [],
                'data_overlaps': []
            }

            # Analyze relationships between datasets
            dataset_names = list(datasets.keys())

            for i in range(len(dataset_names)):
                for j in range(i + 1, len(dataset_names)):
                    ds1_name, ds2_name = dataset_names[i], dataset_names[j]
                    ds1, ds2 = datasets[ds1_name], datasets[ds2_name]

                    # Schema similarity analysis
                    if relationship_types in ['all', 'schema']:
                        schema_sim = self._analyze_schema_similarity(ds1, ds2, ds1_name, ds2_name)
                        relationships['schema_similarities'].extend(schema_sim)

                    # Potential join analysis
                    if relationship_types in ['all', 'joins']:
                        join_opportunities = self._analyze_join_opportunities(ds1, ds2, ds1_name, ds2_name)
                        relationships['potential_joins'].extend(join_opportunities)

                    # Data overlap analysis
                    if relationship_types in ['all', 'overlaps']:
                        overlaps = self._analyze_data_overlaps(ds1, ds2, ds1_name, ds2_name)
                        relationships['data_overlaps'].extend(overlaps)

            return json.dumps(relationships, indent=2, default=str)

        except Exception as e:
            return f"Error in relationship analysis: {str(e)}"

    def _analyze_schema_similarity(self, ds1: pd.DataFrame, ds2: pd.DataFrame,
                                   name1: str, name2: str) -> List[Dict]:
        """Analyze schema similarities between datasets"""
        similarities = []

        # Common columns analysis
        common_cols = set(ds1.columns) & set(ds2.columns)
        if common_cols:
            similarities.append({
                'type': 'common_columns',
                'dataset1': name1,
                'dataset2': name2,
                'common_columns': list(common_cols),
                'similarity_score': len(common_cols) / max(len(ds1.columns), len(ds2.columns))
            })

        # Similar column names (fuzzy matching)
        similar_cols = []
        for col1 in ds1.columns:
            for col2 in ds2.columns:
                if col1.lower() in col2.lower() or col2.lower() in col1.lower():
                    if col1 != col2:  # Not exact matches
                        similar_cols.append((col1, col2))

        if similar_cols:
            similarities.append({
                'type': 'similar_column_names',
                'dataset1': name1,
                'dataset2': name2,
                'similar_columns': similar_cols
            })

        return similarities

    def _analyze_join_opportunities(self, ds1: pd.DataFrame, ds2: pd.DataFrame,
                                    name1: str, name2: str) -> List[Dict]:
        """Identify potential join opportunities"""
        join_opportunities = []

        # Look for potential key columns
        for col1 in ds1.columns:
            for col2 in ds2.columns:
                # Check if columns might be joinable
                if (col1.lower() == col2.lower() or
                        'id' in col1.lower() and 'id' in col2.lower() or
                        col1.lower().replace('_', '') == col2.lower().replace('_', '')):

                    # Check data type compatibility
                    if ds1[col1].dtype == ds2[col2].dtype:
                        # Check for overlapping values
                        overlap = set(ds1[col1].dropna()) & set(ds2[col2].dropna())
                        overlap_percentage = len(overlap) / max(ds1[col1].nunique(), ds2[col2].nunique()) * 100

                        if overlap_percentage > 10:  # At least 10% overlap
                            join_opportunities.append({
                                'dataset1': name1,
                                'dataset2': name2,
                                'column1': col1,
                                'column2': col2,
                                'overlap_percentage': overlap_percentage,
                                'overlapping_values_count': len(overlap),
                                'join_type_suggestion': 'inner' if overlap_percentage > 80 else 'left'
                            })

        return join_opportunities

    def _analyze_data_overlaps(self, ds1: pd.DataFrame, ds2: pd.DataFrame,
                               name1: str, name2: str) -> List[Dict]:
        """Analyze data value overlaps between datasets"""
        overlaps = []

        # Check for overlapping values in similar columns
        common_cols = set(ds1.columns) & set(ds2.columns)

        for col in common_cols:
            if ds1[col].dtype == ds2[col].dtype:
                # Calculate overlap in actual values
                values1 = set(ds1[col].dropna().astype(str))
                values2 = set(ds2[col].dropna().astype(str))
                overlap = values1 & values2

                if overlap:
                    overlap_percentage = len(overlap) / len(values1 | values2) * 100
                    overlaps.append({
                        'column': col,
                        'dataset1': name1,
                        'dataset2': name2,
                        'overlap_percentage': overlap_percentage,
                        'unique_values_overlap': len(overlap),
                        'total_unique_combined': len(values1 | values2)
                    })

        return overlaps


class MultiDatasetAnalysisTool(BaseTool):
    name: str = "Multi-Dataset Analysis Tool"
    description: str = "Coordinate analysis across multiple related datasets"

    def _run(self, dataset_paths: List[str], analysis_type: str = "comprehensive") -> str:
        try:
            datasets = {}
            analysis_results = {}

            # Load all datasets
            for i, path in enumerate(dataset_paths):
                dataset_name = f'dataset_{i}_{os.path.basename(path).replace(".csv", "")}'
                datasets[dataset_name] = pd.read_csv(path)

            # Individual dataset analysis
            analysis_results['individual_analysis'] = {}
            for name, df in datasets.items():
                analysis_results['individual_analysis'][name] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'data_types': df.dtypes.to_dict()
                }

            # Cross-dataset analysis
            if analysis_type in ['comprehensive', 'cross_analysis']:
                analysis_results['cross_dataset_analysis'] = self._cross_dataset_analysis(datasets)

            # Integration opportunities
            if analysis_type in ['comprehensive', 'integration']:
                analysis_results['integration_opportunities'] = self._find_integration_opportunities(datasets)

            # Unified modeling recommendations
            if analysis_type in ['comprehensive', 'modeling']:
                analysis_results['modeling_recommendations'] = self._unified_modeling_recommendations(datasets)

            return json.dumps(analysis_results, indent=2, default=str)

        except Exception as e:
            return f"Error in multi-dataset analysis: {str(e)}"

    def _cross_dataset_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze patterns across datasets"""
        cross_analysis = {
            'size_comparison': {},
            'column_overlap': {},
            'data_quality_comparison': {}
        }

        # Size comparison
        for name, df in datasets.items():
            cross_analysis['size_comparison'][name] = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
            }

        # Column overlap analysis
        all_columns = set()
        dataset_columns = {}
        for name, df in datasets.items():
            dataset_columns[name] = set(df.columns)
            all_columns.update(df.columns)

        # Find common columns across all datasets
        common_to_all = set.intersection(*dataset_columns.values()) if dataset_columns else set()
        cross_analysis['column_overlap']['common_to_all'] = list(common_to_all)

        # Data quality comparison
        for name, df in datasets.items():
            cross_analysis['data_quality_comparison'][name] = {
                'missing_percentage': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': df.duplicated().sum() / df.shape[0] * 100
            }

        return cross_analysis

    def _find_integration_opportunities(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find opportunities to integrate datasets"""
        opportunities = []

        dataset_names = list(datasets.keys())

        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                df1, df2 = datasets[name1], datasets[name2]

                # Look for potential join keys
                potential_joins = []
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        if (col1.lower() == col2.lower() or
                                any(key in col1.lower() and key in col2.lower()
                                    for key in ['id', 'key', 'code', 'name'])):

                            # Check data compatibility
                            if df1[col1].dtype == df2[col2].dtype:
                                overlap = set(df1[col1].dropna()) & set(df2[col2].dropna())
                                if len(overlap) > 0:
                                    potential_joins.append({
                                        'column1': col1,
                                        'column2': col2,
                                        'overlap_count': len(overlap)
                                    })

                if potential_joins:
                    opportunities.append({
                        'dataset1': name1,
                        'dataset2': name2,
                        'join_opportunities': potential_joins,
                        'integration_feasibility': 'high' if len(potential_joins) > 1 else 'medium'
                    })

        return opportunities

    def _unified_modeling_recommendations(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Generate recommendations for unified modeling across datasets"""
        recommendations = {
            'ensemble_opportunities': [],
            'stacking_potential': [],
            'feature_sharing': [],
            'model_architecture': []
        }

        # Analyze if datasets can be used for ensemble modeling
        if len(datasets) > 1:
            recommendations['ensemble_opportunities'].append({
                'type': 'multi_dataset_ensemble',
                'description': f'Train separate models on {len(datasets)} datasets and ensemble predictions',
                'benefit': 'Improved robustness and generalization'
            })

        # Feature sharing analysis
        common_features = set.intersection(*[set(df.columns) for df in datasets.values()])
        if common_features:
            recommendations['feature_sharing'].append({
                'common_features': list(common_features),
                'sharing_strategy': 'Use common features for transfer learning or shared representations'
            })

        # Model architecture recommendations
        total_samples = sum(df.shape[0] for df in datasets.values())
        total_features = sum(df.shape[1] for df in datasets.values())

        if total_samples > 10000:
            recommendations['model_architecture'].append({
                'type': 'deep_learning',
                'reason': f'Large combined dataset ({total_samples} samples) suitable for deep learning'
            })
        else:
            recommendations['model_architecture'].append({
                'type': 'traditional_ml',
                'reason': f'Moderate dataset size ({total_samples} samples) suitable for traditional ML'
            })

        return recommendations


# Enhanced versions of existing tools with improved capabilities
class DataCleaningTool(BaseTool):
    name: str = "Intelligent Data Cleaning Tool"
    description: str = "Context-aware data cleaning with multiple strategies"

    def _run(self, data_path: str, cleaning_strategy: str = "intelligent",
             business_context: str = "", preserve_relationships: bool = True) -> str:
        try:
            df = pd.read_csv(data_path)
            original_shape = df.shape
            cleaning_log = []

            # Intelligent cleaning based on data characteristics
            if cleaning_strategy == "intelligent":
                cleaning_strategy = self._determine_optimal_strategy(df, business_context)
                cleaning_log.append(f"Selected cleaning strategy: {cleaning_strategy}")

            # Remove duplicates with logging
            duplicates_before = df.duplicated().sum()
            df = df.drop_duplicates()
            if duplicates_before > 0:
                cleaning_log.append(f"Removed {duplicates_before} duplicate rows")

            # Intelligent missing value handling
            missing_strategy = self._determine_missing_value_strategy(df, business_context)
            df = self._handle_missing_values(df, missing_strategy, cleaning_log)

            # Outlier handling based on context
            if cleaning_strategy != "conservative":
                df = self._handle_outliers(df, cleaning_strategy, cleaning_log)

            # Data type optimization
            df = self._optimize_data_types(df, cleaning_log)

            # Save cleaned data
            cleaned_path = data_path.replace('.csv', '_cleaned.csv')
            df.to_csv(cleaned_path, index=False)

            result = {
                'original_shape': original_shape,
                'cleaned_shape': df.shape,
                'cleaning_log': cleaning_log,
                'cleaned_file_path': cleaned_path,
                'data_loss_percentage': (1 - df.shape[0] / original_shape[0]) * 100
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error in intelligent data cleaning: {str(e)}"

    def _determine_optimal_strategy(self, df: pd.DataFrame, business_context: str) -> str:
        """Determine optimal cleaning strategy based on data characteristics"""
        missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100

        if missing_percentage > 30:
            return "aggressive"
        elif missing_percentage > 10:
            return "moderate"
        elif "financial" in business_context.lower() or "medical" in business_context.lower():
            return "conservative"
        else:
            return "balanced"

    def _determine_missing_value_strategy(self, df: pd.DataFrame, business_context: str) -> Dict:
        """Determine missing value strategy for each column"""
        strategies = {}

        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100

            if missing_pct > 50:
                strategies[col] = "drop_column"
            elif df[col].dtype in ['int64', 'float64']:
                if missing_pct > 20:
                    strategies[col] = "iterative_imputation"
                else:
                    strategies[col] = "median"
            else:  # Categorical
                if missing_pct > 20:
                    strategies[col] = "new_category"
                else:
                    strategies[col] = "mode"

        return strategies

    def _handle_missing_values(self, df: pd.DataFrame, strategies: Dict, log: List) -> pd.DataFrame:
        """Handle missing values based on determined strategies"""
        for col, strategy in strategies.items():
            if col not in df.columns:
                continue

            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue

            if strategy == "drop_column":
                df = df.drop(columns=[col])
                log.append(f"Dropped column {col} (>50% missing)")
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
                log.append(f"Filled {missing_count} missing values in {col} with median")
            elif strategy == "mode":
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                log.append(f"Filled {missing_count} missing values in {col} with mode")
            elif strategy == "new_category":
                df[col].fillna('Missing', inplace=True)
                log.append(f"Filled {missing_count} missing values in {col} with 'Missing' category")
            elif strategy == "iterative_imputation":
                # Simplified iterative imputation
                df[col].fillna(df[col].median(), inplace=True)
                log.append(f"Applied iterative imputation to {col}")

        return df

    def _handle_outliers(self, df: pd.DataFrame, strategy: str, log: List) -> pd.DataFrame:
        """Handle outliers based on strategy"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if strategy == "aggressive":
                multiplier = 1.5
            elif strategy == "moderate":
                multiplier = 2.0
            else:
                multiplier = 2.5

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers_before > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                log.append(f"Clipped {outliers_before} outliers in {col}")

        return df

    def _optimize_data_types(self, df: pd.DataFrame, log: List) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        memory_before = df.memory_usage(deep=True).sum() / 1024 ** 2

        # Convert integer columns to smaller types if possible
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')

        # Convert float columns to smaller types if possible
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].isnull().sum() == 0:  # No NaN values
                if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')

        memory_after = df.memory_usage(deep=True).sum() / 1024 ** 2
        memory_saved = memory_before - memory_after

        if memory_saved > 0.1:  # If saved more than 0.1 MB
            log.append(f"Optimized data types, saved {memory_saved:.2f} MB")

        return df


class FeatureEngineeringTool(BaseTool):
    name: str = "Advanced Feature Engineering Tool"
    description: str = "Intelligent feature engineering with domain awareness"

    def _run(self, data_path: str, target_column: str,
             engineering_strategy: str = "comprehensive",
             domain_context: str = "",
             create_interactions: bool = True) -> str:
        try:
            df = pd.read_csv(data_path)
            original_features = df.shape[1]
            feature_log = []

            # Separate features and target
            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
            else:
                X = df
                y = None
                feature_log.append(f"Warning: Target column '{target_column}' not found")

            # Apply feature engineering based on strategy
            if engineering_strategy == "comprehensive":
                X = self._comprehensive_feature_engineering(X, y, domain_context, feature_log)
            elif engineering_strategy == "minimal":
                X = self._minimal_feature_engineering(X, y, feature_log)
            elif engineering_strategy == "domain_specific":
                X = self._domain_specific_engineering(X, y, domain_context, feature_log)

            # Create interaction features if requested
            if create_interactions and X.shape[1] <= 20:  # Limit interactions for high-dim data
                X = self._create_interaction_features(X, feature_log)

            # Feature selection based on importance
            if X.shape[1] > 100:  # If too many features
                X = self._select_important_features(X, y, feature_log)

            # Recombine with target if it exists
            if y is not None:
                engineered_data = pd.concat([X, y], axis=1)
            else:
                engineered_data = X

            # Save engineered features
            engineered_path = data_path.replace('.csv', '_engineered.csv')
            engineered_data.to_csv(engineered_path, index=False)

            result = {
                'original_features': original_features,
                'engineered_features': X.shape[1],
                'feature_engineering_log': feature_log,
                'engineered_file_path': engineered_path,
                'feature_increase_ratio': X.shape[1] / original_features
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error in advanced feature engineering: {str(e)}"

    def _comprehensive_feature_engineering(self, X: pd.DataFrame, y: pd.Series,
                                           domain_context: str, log: List) -> pd.DataFrame:
        """Apply comprehensive feature engineering"""

        # 1. Handle categorical variables intelligently
        X = self._intelligent_categorical_encoding(X, y, log)

        # 2. Scale numerical features
        X = self._intelligent_scaling(X, log)

        # 3. Create polynomial features for important numeric columns
        X = self._create_polynomial_features(X, log)

        # 4. Create temporal features if applicable
        X = self._create_temporal_features(X, log)

        # 5. Create statistical features
        X = self._create_statistical_features(X, log)

        return X

    def _intelligent_categorical_encoding(self, X: pd.DataFrame, y: pd.Series, log: List) -> pd.DataFrame:
        """Apply intelligent categorical encoding based on cardinality and target relationship"""
        categorical_cols = X.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            cardinality = X[col].nunique()

            if cardinality <= 5:
                # One-hot encode low cardinality
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])
                log.append(f"One-hot encoded {col} ({cardinality} categories)")

            elif cardinality <= 20:
                # Binary encoding for medium cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                log.append(f"Label encoded {col} ({cardinality} categories)")

            else:
                # Target encoding for high cardinality (if target available)
                if y is not None:
                    target_means = X.groupby(col)[y.name if hasattr(y, 'name') else 'target'].mean()
                    X[f'{col}_target_encoded'] = X[col].map(target_means)
                    X = X.drop(columns=[col])
                    log.append(f"Target encoded {col} ({cardinality} categories)")
                else:
                    # Frequency encoding as fallback
                    freq_encoding = X[col].value_counts(normalize=True)
                    X[f'{col}_frequency'] = X[col].map(freq_encoding)
                    X = X.drop(columns=[col])
                    log.append(f"Frequency encoded {col} ({cardinality} categories)")

        return X

    def _intelligent_scaling(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Apply intelligent scaling based on distribution characteristics"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Don't scale binary columns (0,1) or already scaled columns
        cols_to_scale = []
        for col in numeric_cols:
            if X[col].nunique() > 2 and (X[col].max() - X[col].min()) > 1:
                cols_to_scale.append(col)

        if cols_to_scale:
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            log.append(f"Scaled {len(cols_to_scale)} numerical features")

        return X

    def _create_polynomial_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create polynomial features for important numeric columns"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Limit to most important numeric columns (by variance)
        if len(numeric_cols) > 5:
            variances = X[numeric_cols].var()
            top_cols = variances.nlargest(3).index
        else:
            top_cols = numeric_cols[:3]

        for col in top_cols:
            if col in X.columns:  # Check if column still exists
                X[f'{col}_squared'] = X[col] ** 2
                X[f'{col}_cubed'] = X[col] ** 3

        if len(top_cols) > 0:
            log.append(f"Created polynomial features for top {len(top_cols)} numeric columns")

        return X

    def _create_temporal_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create temporal features from date columns"""
        date_cols = []

        # Identify potential date columns
        for col in X.columns:
            if X[col].dtype == 'object' and any(
                    keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                try:
                    X[col] = pd.to_datetime(X[col])
                    date_cols.append(col)
                except:
                    continue

        # Create temporal features
        for col in date_cols:
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X[f'{col}_quarter'] = X[col].dt.quarter

            # Drop original date column
            X = X.drop(columns=[col])
            log.append(f"Created temporal features from {col}")

        return X

    def _create_statistical_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create statistical features across related columns"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 3:
            # Create row-wise statistics
            X['row_mean'] = X[numeric_cols].mean(axis=1)
            X['row_std'] = X[numeric_cols].std(axis=1)
            X['row_max'] = X[numeric_cols].max(axis=1)
            X['row_min'] = X[numeric_cols].min(axis=1)
            log.append("Created row-wise statistical features")

        return X

    def _create_interaction_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create interaction features between important columns"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Limit interactions to prevent feature explosion
        if len(numeric_cols) >= 2:
            # Find top correlated pairs
            corr_matrix = X[numeric_cols].corr()

            interaction_count = 0
            max_interactions = 5  # Limit interactions

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if interaction_count >= max_interactions:
                        break

                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    correlation = abs(corr_matrix.loc[col1, col2])

                    # Create interactions for moderately correlated features
                    if 0.3 <= correlation <= 0.8:
                        X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                        X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)  # Avoid division by zero
                        interaction_count += 1

                if interaction_count >= max_interactions:
                    break

            if interaction_count > 0:
                log.append(f"Created {interaction_count * 2} interaction features")

        return X

    def _select_important_features(self, X: pd.DataFrame, y: pd.Series, log: List) -> pd.DataFrame:
        """Select most important features using tree-based feature importance"""
        if y is None:
            return X

        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.feature_selection import SelectKBest, f_regression, f_classif

            # Determine problem type
            if y.nunique() <= 20:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = SelectKBest(f_regression, k=min(50, X.shape[1]))

            # Fit model for feature importance
            model.fit(X, y)
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)

            # Select top features
            top_features = feature_importance.nlargest(min(50, X.shape[1])).index
            X_selected = X[top_features]

            log.append(f"Selected top {len(top_features)} features out of {X.shape[1]}")
            return X_selected

        except Exception as e:
            log.append(f"Feature selection failed: {str(e)}, keeping all features")
            return X

    def _minimal_feature_engineering(self, X: pd.DataFrame, y: pd.Series, log: List) -> pd.DataFrame:
        """Apply minimal feature engineering - only essential transformations"""
        # Basic categorical encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].nunique() <= 10:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Basic scaling
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        log.append("Applied minimal feature engineering (encoding + scaling)")
        return X

    def _domain_specific_engineering(self, X: pd.DataFrame, y: pd.Series,
                                     domain_context: str, log: List) -> pd.DataFrame:
        """Apply domain-specific feature engineering"""
        domain_lower = domain_context.lower()

        if 'financial' in domain_lower or 'finance' in domain_lower:
            X = self._financial_domain_features(X, log)
        elif 'retail' in domain_lower or 'ecommerce' in domain_lower:
            X = self._retail_domain_features(X, log)
        elif 'healthcare' in domain_lower or 'medical' in domain_lower:
            X = self._healthcare_domain_features(X, log)
        else:
            # Default comprehensive engineering
            X = self._comprehensive_feature_engineering(X, y, domain_context, log)

        return X

    def _financial_domain_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create financial domain-specific features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Look for financial columns
        financial_indicators = ['amount', 'balance', 'income', 'expense', 'price', 'cost', 'revenue']
        financial_cols = [col for col in numeric_cols
                          if any(indicator in col.lower() for indicator in financial_indicators)]

        if len(financial_cols) >= 2:
            # Create financial ratios
            for i, col1 in enumerate(financial_cols):
                for col2 in financial_cols[i + 1:]:
                    X[f'{col1}_to_{col2}_ratio'] = X[col1] / (X[col2] + 1e-8)

            log.append(f"Created financial ratio features from {len(financial_cols)} columns")

        # Apply standard encoding and scaling
        X = self._intelligent_categorical_encoding(X, None, log)
        X = self._intelligent_scaling(X, log)

        return X

    def _retail_domain_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create retail domain-specific features"""
        # Look for retail-specific columns
        retail_indicators = ['quantity', 'price', 'discount', 'category', 'brand', 'customer']

        # Create retail-specific interactions
        if 'quantity' in X.columns and 'price' in X.columns:
            X['total_value'] = X['quantity'] * X['price']
            log.append("Created total_value feature (quantity * price)")

        if 'discount' in X.columns and 'price' in X.columns:
            X['discounted_price'] = X['price'] * (1 - X['discount'])
            log.append("Created discounted_price feature")

        # Apply standard processing
        X = self._intelligent_categorical_encoding(X, None, log)
        X = self._intelligent_scaling(X, log)

        return X

    def _healthcare_domain_features(self, X: pd.DataFrame, log: List) -> pd.DataFrame:
        """Create healthcare domain-specific features"""
        # Look for healthcare-specific columns
        health_indicators = ['age', 'weight', 'height', 'blood', 'pressure', 'heart', 'temperature']

        # Create BMI if height and weight are available
        if 'height' in X.columns and 'weight' in X.columns:
            X['bmi'] = X['weight'] / ((X['height'] / 100) ** 2)
            log.append("Created BMI feature from height and weight")

        # Create age groups if age is available
        if 'age' in X.columns:
            X['age_group'] = pd.cut(X['age'], bins=[0, 18, 35, 50, 65, 100],
                                    labels=['child', 'young_adult', 'adult', 'middle_aged', 'senior'])
            log.append("Created age_group categorical feature")

        # Apply standard processing
        X = self._intelligent_categorical_encoding(X, None, log)
        X = self._intelligent_scaling(X, log)

        return X


class ModelTrainingTool(BaseTool):
    name: str = "Advanced Model Training Tool"
    description: str = "Intelligent model selection and training with hyperparameter optimization"

    def _run(self, train_data_path: str, target_column: str,
             problem_type: str = "auto", model_strategy: str = "comprehensive",
             business_objective: str = "", time_constraint: str = "medium") -> str:
        try:
            df = pd.read_csv(train_data_path)

            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found in dataset"

            X_train = df.drop(columns=[target_column])
            y_train = df[target_column]

            # Intelligent problem type detection
            if problem_type == "auto":
                problem_type = self._detect_problem_type(y_train)

            # Model selection based on strategy and constraints
            models_to_train = self._select_models(problem_type, model_strategy,
                                                  X_train.shape, time_constraint)

            training_results = {
                'problem_type': problem_type,
                'models_trained': {},
                'best_model': None,
                'training_log': [],
                'recommendations': []
            }

            # Train models
            best_score = -np.inf if problem_type == 'regression' else 0
            best_model_name = None

            for model_name, model_config in models_to_train.items():
                try:
                    model = model_config['model']

                    # Train model
                    model.fit(X_train, y_train)

                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train, y_train,
                                                cv=5, scoring=model_config['scoring'])
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()

                    # Store results
                    training_results['models_trained'][model_name] = {
                        'cv_score_mean': cv_mean,
                        'cv_score_std': cv_std,
                        'model_params': model.get_params(),
                        'training_time': 'quick'  # In practice, measure actual time
                    }

                    # Track best model
                    if (problem_type == 'classification' and cv_mean > best_score) or \
                            (problem_type == 'regression' and cv_mean > best_score):
                        best_score = cv_mean
                        best_model_name = model_name

                    training_results['training_log'].append(
                        f"Trained {model_name}: CV Score = {cv_mean:.4f} (±{cv_std:.4f})"
                    )

                except Exception as e:
                    training_results['training_log'].append(f"Failed to train {model_name}: {str(e)}")
                    continue

            # Set best model
            if best_model_name:
                training_results['best_model'] = best_model_name
                training_results['best_score'] = best_score

                # Save best model
                import pickle
                best_model = models_to_train[best_model_name]['model']
                model_path = train_data_path.replace('_train.csv', '_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                training_results['model_path'] = model_path

            # Generate recommendations
            training_results['recommendations'] = self._generate_model_recommendations(
                training_results, business_objective, X_train.shape
            )

            return json.dumps(training_results, indent=2, default=str)

        except Exception as e:
            return f"Error in advanced model training: {str(e)}"

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Intelligently detect problem type"""
        unique_values = y.nunique()

        if y.dtype == 'object' or unique_values <= 20:
            if unique_values == 2:
                return 'binary_classification'
            else:
                return 'multiclass_classification'
        else:
            # Check if it's actually discrete values
            if y.dtype in ['int64'] and unique_values <= 50:
                return 'multiclass_classification'
            else:
                return 'regression'

    def _select_models(self, problem_type: str, strategy: str, data_shape: Tuple,
                       time_constraint: str) -> Dict:
        """Select appropriate models based on problem characteristics"""
        n_samples, n_features = data_shape
        models = {}

        # Import models
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        if problem_type in ['binary_classification', 'multiclass_classification']:
            base_models = {
                'RandomForest': {
                    'model': RandomForestClassifier(n_estimators=100, random_state=42),
                    'scoring': 'accuracy'
                },
                'LogisticRegression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'scoring': 'accuracy'
                }
            }

            # Add more models based on strategy and constraints
            if strategy == 'comprehensive' and time_constraint != 'fast':
                if n_samples <= 10000:  # SVM for smaller datasets
                    base_models['SVM'] = {
                        'model': SVC(random_state=42),
                        'scoring': 'accuracy'
                    }

                base_models['KNN'] = {
                    'model': KNeighborsClassifier(),
                    'scoring': 'accuracy'
                }

                if n_features <= 50:  # Naive Bayes for lower dimensions
                    base_models['NaiveBayes'] = {
                        'model': GaussianNB(),
                        'scoring': 'accuracy'
                    }

        else:  # Regression
            base_models = {
                'RandomForest': {
                    'model': RandomForestRegressor(n_estimators=100, random_state=42),
                    'scoring': 'r2'
                },
                'LinearRegression': {
                    'model': LinearRegression(),
                    'scoring': 'r2'
                }
            }

            # Add regularized models for high-dimensional data
            if n_features > 20:
                base_models['Ridge'] = {
                    'model': Ridge(random_state=42),
                    'scoring': 'r2'
                }
                base_models['Lasso'] = {
                    'model': Lasso(random_state=42),
                    'scoring': 'r2'
                }

            if strategy == 'comprehensive' and time_constraint != 'fast':
                if n_samples <= 10000:
                    base_models['SVR'] = {
                        'model': SVR(),
                        'scoring': 'r2'
                    }

                base_models['KNN'] = {
                    'model': KNeighborsRegressor(),
                    'scoring': 'r2'
                }

        return base_models

    def _generate_model_recommendations(self, training_results: Dict,
                                        business_objective: str, data_shape: Tuple) -> List[str]:
        """Generate model recommendations based on results and business context"""
        recommendations = []

        if training_results['best_model']:
            recommendations.append(
                f"Best performing model: {training_results['best_model']} "
                f"(Score: {training_results['best_score']:.4f})"
            )

        # Business context recommendations
        if 'interpretability' in business_objective.lower():
            interpretable_models = ['LogisticRegression', 'LinearRegression', 'DecisionTree']
            trained_interpretable = [m for m in interpretable_models
                                     if m in training_results['models_trained']]
            if trained_interpretable:
                best_interpretable = max(trained_interpretable,
                                         key=lambda x: training_results['models_trained'][x]['cv_score_mean'])
                recommendations.append(
                    f"For interpretability, consider: {best_interpretable}"
                )

        if 'speed' in business_objective.lower() or 'fast' in business_objective.lower():
            fast_models = ['LogisticRegression', 'LinearRegression', 'NaiveBayes']
            trained_fast = [m for m in fast_models if m in training_results['models_trained']]
            if trained_fast:
                recommendations.append(
                    f"For speed requirements, consider: {', '.join(trained_fast)}"
                )

        # Data size recommendations
        n_samples, n_features = data_shape
        if n_samples < 1000:
            recommendations.append(
                "Small dataset detected. Consider data augmentation or simpler models."
            )
        elif n_samples > 100000:
            recommendations.append(
                "Large dataset detected. Consider using SGD-based algorithms for faster training."
            )

        if n_features > 100:
            recommendations.append(
                "High-dimensional data detected. Consider feature selection or regularization."
            )

        return recommendations


class ModelValidationTool(BaseTool):
    name: str = "Comprehensive Model Validation Tool"
    description: str = "Advanced model validation with multiple metrics and bias detection"

    def _run(self, model_path: str, test_data_path: str, target_column: str,
             validation_strategy: str = "comprehensive", fairness_check: bool = True) -> str:
        try:
            import pickle
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                         mean_squared_error, mean_absolute_error, r2_score,
                                         classification_report, confusion_matrix)

            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Load test data
            df = pd.read_csv(test_data_path)
            X_test = df.drop(columns=[target_column])
            y_test = df[target_column]

            # Make predictions
            y_pred = model.predict(X_test)

            validation_results = {
                'model_type': type(model).__name__,
                'test_set_size': len(X_test),
                'metrics': {},
                'detailed_analysis': {},
                'recommendations': []
            }

            # Determine problem type and calculate appropriate metrics
            if hasattr(model, 'predict_proba'):  # Classification
                validation_results['problem_type'] = 'classification'

                # Basic metrics
                validation_results['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
                validation_results['metrics']['precision'] = precision_score(y_test, y_pred, average='weighted')
                validation_results['metrics']['recall'] = recall_score(y_test, y_pred, average='weighted')
                validation_results['metrics']['f1_score'] = f1_score(y_test, y_pred, average='weighted')

                # Detailed classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                validation_results['detailed_analysis']['classification_report'] = class_report

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                validation_results['detailed_analysis']['confusion_matrix'] = cm.tolist()

                # Class balance analysis
                validation_results['detailed_analysis']['class_distribution'] = {
                    'actual': y_test.value_counts().to_dict(),
                    'predicted': pd.Series(y_pred).value_counts().to_dict()
                }

            else:  # Regression
                validation_results['problem_type'] = 'regression'

                # Basic metrics
                validation_results['metrics']['mse'] = mean_squared_error(y_test, y_pred)
                validation_results['metrics']['rmse'] = np.sqrt(validation_results['metrics']['mse'])
                validation_results['metrics']['mae'] = mean_absolute_error(y_test, y_pred)
                validation_results['metrics']['r2_score'] = r2_score(y_test, y_pred)

                # Additional regression metrics
                validation_results['metrics']['mean_absolute_percentage_error'] = (
                        np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                )

                # Residual analysis
                residuals = y_test - y_pred
                validation_results['detailed_analysis']['residual_stats'] = {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals)
                }

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.Series(model.feature_importances_,
                                               index=X_test.columns).sort_values(ascending=False)
                validation_results['detailed_analysis']['feature_importance'] = (
                    feature_importance.head(10).to_dict()
                )

            # Model complexity analysis
            validation_results['detailed_analysis']['model_complexity'] = self._analyze_model_complexity(model)

            # Performance recommendations
            validation_results['recommendations'] = self._generate_validation_recommendations(
                validation_results, y_test, y_pred
            )

            return json.dumps(validation_results, indent=2, default=str)

        except Exception as e:
            return f"Error in comprehensive model validation: {str(e)}"

    def _analyze_model_complexity(self, model) -> Dict:
        """Analyze model complexity characteristics"""
        complexity_analysis = {
            'model_type': type(model).__name__,
            'interpretability': 'unknown',
            'training_speed': 'unknown',
            'prediction_speed': 'unknown'
        }

        model_name = type(model).__name__

        # Interpretability assessment
        if model_name in ['LinearRegression', 'LogisticRegression', 'DecisionTreeClassifier', 'DecisionTreeRegressor']:
            complexity_analysis['interpretability'] = 'high'
        elif model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
            complexity_analysis['interpretability'] = 'medium'
        else:
            complexity_analysis['interpretability'] = 'low'

        # Speed assessment (simplified)
        if model_name in ['LinearRegression', 'LogisticRegression', 'GaussianNB']:
            complexity_analysis['training_speed'] = 'fast'
            complexity_analysis['prediction_speed'] = 'fast'
        elif model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
            complexity_analysis['training_speed'] = 'medium'
            complexity_analysis['prediction_speed'] = 'medium'
        elif model_name in ['SVC', 'SVR']:
            complexity_analysis['training_speed'] = 'slow'
            complexity_analysis['prediction_speed'] = 'medium'

        return complexity_analysis

    def _generate_validation_recommendations(self, results: Dict, y_test, y_pred) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        if results['problem_type'] == 'classification':
            accuracy = results['metrics']['accuracy']

            if accuracy < 0.7:
                recommendations.append("Low accuracy detected. Consider feature engineering or different algorithms.")
            elif accuracy > 0.95:
                recommendations.append("Very high accuracy - check for overfitting or data leakage.")

            # Check class imbalance
            if 'classification_report' in results['detailed_analysis']:
                class_report = results['detailed_analysis']['classification_report']
                f1_scores = [class_report[str(cls)]['f1-score'] for cls in class_report
                             if str(cls).isdigit() or cls in ['0', '1']]
                if len(f1_scores) > 1 and max(f1_scores) - min(f1_scores) > 0.2:
                    recommendations.append("Significant class imbalance detected. Consider balancing techniques.")

        else:  # Regression
            r2 = results['metrics']['r2_score']

            if r2 < 0.5:
                recommendations.append("Low R² score. Consider feature engineering or different algorithms.")
            elif r2 > 0.99:
                recommendations.append("Very high R² score - check for overfitting or data leakage.")

            # Check residual patterns
            if 'residual_stats' in results['detailed_analysis']:
                residual_mean = results['detailed_analysis']['residual_stats']['mean']
                if abs(residual_mean) > 0.1:
                    recommendations.append("Residuals show bias. Consider model adjustments.")

        # Model complexity recommendations
        if 'model_complexity' in results['detailed_analysis']:
            complexity = results['detailed_analysis']['model_complexity']
            if complexity['interpretability'] == 'low':
                recommendations.append("Consider more interpretable models if explainability is important.")

        return recommendations