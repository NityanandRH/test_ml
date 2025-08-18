import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import tempfile
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="🔍",
    layout="wide"
)

# Global variables to store data
if 'current_dataframe' not in st.session_state:
    st.session_state.current_dataframe = None
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Tool 1: File Analysis Tool (Modified for Streamlit)
class FileAnalysisTool(BaseTool):
    name: str = "File Analysis"
    description: str = """
    Analyze CSV/Excel files and answer questions about the data.
    Input format: 'question' (file is already loaded in session)
    """

    def _run(self, question: str) -> str:
        try:
            df = st.session_state.current_dataframe
            if df is None:
                return "Error: No data loaded. Please upload a file first."

            question = question.strip().lower()

            # Analyze based on question
            if "number of columns" in question or "how many columns" in question:
                return f"The dataset has {len(df.columns)} columns"

            elif "number of rows" in question or "how many rows" in question:
                return f"The dataset has {len(df)} rows"

            elif "column names" in question or "what are the columns" in question:
                return f"Columns in the dataset: {list(df.columns)}"

            elif "null values" in question:
                col_name = self._extract_column_name(question, df.columns)
                if col_name:
                    null_count = df[col_name].isnull().sum()
                    return f"Column '{col_name}' has {null_count} null values"
                else:
                    null_info = df.isnull().sum()
                    return f"Null values per column:\n{null_info.to_string()}"

            elif "max" in question and ("of" in question or "in" in question):
                col_name = self._extract_column_name(question, df.columns)
                if col_name:
                    if df[col_name].dtype in ['object', 'string']:
                        return f"Column '{col_name}' is not numeric. Cannot calculate max."
                    max_val = df[col_name].max()
                    return f"Maximum value in column '{col_name}': {max_val}"
                else:
                    return "Please specify which column you want the maximum of"

            elif "min" in question and ("of" in question or "in" in question):
                col_name = self._extract_column_name(question, df.columns)
                if col_name:
                    if df[col_name].dtype in ['object', 'string']:
                        return f"Column '{col_name}' is not numeric. Cannot calculate min."
                    min_val = df[col_name].min()
                    return f"Minimum value in column '{col_name}': {min_val}"
                else:
                    return "Please specify which column you want the minimum of"

            elif "mean" in question or "average" in question:
                col_name = self._extract_column_name(question, df.columns)
                if col_name:
                    if df[col_name].dtype in ['object', 'string']:
                        return f"Column '{col_name}' is not numeric. Cannot calculate mean."
                    mean_val = df[col_name].mean()
                    return f"Mean value in column '{col_name}': {mean_val}"
                else:
                    return "Please specify which column you want the mean of"

            elif "describe" in question or "statistics" in question or "summary" in question:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    return f"Basic statistics:\n{df[numeric_cols].describe().to_string()}"
                else:
                    return "No numeric columns found for statistical summary"

            elif "info" in question or "data types" in question:
                info_str = f"Dataset Info:\n"
                info_str += f"Shape: {df.shape}\n"
                info_str += f"Data types:\n{df.dtypes.to_string()}\n"
                info_str += f"Memory usage: {df.memory_usage().sum()} bytes"
                return info_str

            elif "correlation" in question or "corr" in question:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    return f"Correlation matrix:\n{corr_matrix.to_string()}"
                else:
                    return "Need at least 2 numeric columns for correlation analysis"

            else:
                return f"I can help with: column count, row count, column names, null values, max/min/mean of columns, statistics, info, and correlations."

        except Exception as e:
            return f"Error: {str(e)}"

    def _extract_column_name(self, question: str, columns) -> Optional[str]:
        """Extract column name from user question"""
        question_lower = question.lower()

        for col in columns:
            if col.lower() in question_lower:
                return col

        words = question_lower.split()
        for word in words:
            for col in columns:
                if word in col.lower() or col.lower() in word:
                    return col

        return None


# Tool 2: User Plot Tool (Modified for Streamlit)
class UserPlotTool(BaseTool):
    name: str = "User Plot"
    description: str = """
    Create visualizations based on user requests and display in Streamlit.
    Input format: user's natural language plot request
    """

    def _run(self, user_request: str) -> str:
        try:
            df = st.session_state.current_dataframe
            if df is None:
                return "Error: No data loaded. Please upload a file first."

            request_lower = user_request.lower()

            # Extract plot type and columns from natural language
            plot_type, columns = self._parse_plot_request(request_lower, df.columns)

            if not plot_type:
                return f"Could not understand plot request. Available columns: {list(df.columns)}"

            # Validate columns
            valid_cols = []
            for col in columns:
                matched_col = self._find_column(col, df.columns)
                if matched_col:
                    valid_cols.append(matched_col)

            if not valid_cols:
                suggestions = list(df.columns)[:5]
                return f"Columns not found. Available columns: {suggestions}"

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            if plot_type == "distribution":
                col = valid_cols[0]
                if df[col].dtype in ['object', 'string', 'category']:
                    df[col].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                else:
                    ax.hist(df[col].dropna(), bins=30, alpha=0.7)
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')

            elif plot_type == "scatter":
                if len(valid_cols) >= 2:
                    col1, col2 = valid_cols[0], valid_cols[1]
                    if df[col1].dtype not in ['object', 'string'] and df[col2].dtype not in ['object', 'string']:
                        ax.scatter(df[col1], df[col2], alpha=0.6)
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        ax.set_title(f'Scatter Plot: {col1} vs {col2}')
                    else:
                        return f"Scatter plot requires numeric columns. {col1} type: {df[col1].dtype}, {col2} type: {df[col2].dtype}"
                else:
                    return "Scatter plot requires two columns. Please specify both columns."

            elif plot_type == "count":
                col = valid_cols[0]
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Count Plot of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)

            plt.tight_layout()

            # Display plot in Streamlit
            st.pyplot(fig)
            plt.close(fig)  # Close to free memory

            return f"✅ Created {plot_type} plot for columns: {valid_cols}"

        except Exception as e:
            return f"Error creating plot: {str(e)}"

    def _parse_plot_request(self, request: str, columns):
        """Parse natural language plot request"""
        plot_type = None
        found_columns = []

        # Detect plot type
        if any(word in request for word in ["distribution", "histogram", "dist"]):
            plot_type = "distribution"
        elif any(word in request for word in ["scatter", "relationship", "vs", "against"]):
            plot_type = "scatter"
        elif any(word in request for word in ["count", "frequency", "bar"]):
            plot_type = "count"
        else:
            plot_type = "distribution"  # default

        # Extract column names
        for col in columns:
            if col.lower() in request:
                found_columns.append(col)

        return plot_type, found_columns

    def _find_column(self, user_input: str, columns) -> Optional[str]:
        """Find the best matching column name"""
        user_input_lower = user_input.lower()

        for col in columns:
            if col.lower() == user_input_lower:
                return col

        for col in columns:
            if user_input_lower in col.lower() or col.lower() in user_input_lower:
                return col

        return None


# Create Agent
@st.cache_resource
def get_agent():
    return Agent(
        role='Data Analyst',
        goal='Analyze datasets and create visualizations based on user requests',
        backstory="""You are an expert data analyst who can examine datasets, 
        answer questions about data characteristics, and create appropriate visualizations.""",
        tools=[FileAnalysisTool(), UserPlotTool()],
        verbose=False,  # Set to False to reduce noise in Streamlit
        allow_delegation=False,
        max_iter=5
    )


def process_question(question: str) -> str:
    """Process user question using CrewAI agent"""
    try:
        agent = get_agent()

        task = Task(
            description=f"User asks: '{question}'. Use appropriate tools to analyze data or create visualizations.",
            expected_output="Clear answer with analysis or visualization",
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return str(result)

    except Exception as e:
        return f"Error processing question: {str(e)}"


# Streamlit UI
def main():
    st.title("🔍 Data Analysis Assistant")
    st.markdown("Upload your data and ask questions to get insights and visualizations!")

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 File Upload")

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to start analysis"
        )

        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Store in session state
                st.session_state.current_dataframe = df
                st.session_state.current_filename = uploaded_file.name

                st.success(f"✅ Loaded {uploaded_file.name}")
                st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

                # Show column info
                st.subheader("📋 Dataset Info")
                st.write("**Columns:**")
                for i, col in enumerate(df.columns, 1):
                    st.write(f"{i}. {col} ({df[col].dtype})")

                # Show preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    # Main chat interface
    st.header("💬 Chat with your Data")

    if st.session_state.current_dataframe is not None:
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                if "plot" in chat["answer"].lower() and "✅" in chat["answer"]:
                    # If it was a successful plot, the plot would have been displayed already
                    pass

        # Chat input
        user_question = st.chat_input("Ask about your data (e.g., 'How many columns?', 'Show distribution of sales')")

        if user_question:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_question)

            # Process and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = process_question(user_question)
                st.write(response)

            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response
            })

        # Example questions
        st.subheader("💡 Example Questions")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Data Analysis:**
            - How many columns are there?
            - What are the column names?
            - Show me null values
            - What's the max of sales column?
            - Give me basic statistics
            """)

        with col2:
            st.markdown("""
            **Visualizations:**
            - Show distribution of price
            - Create scatter plot between cost and profit
            - Give me count plot of category
            - Show relationship between x and y
            """)

    else:
        st.info("👆 Please upload a CSV or Excel file in the sidebar to start analysis")


if __name__ == "__main__":
    main()
