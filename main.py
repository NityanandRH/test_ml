# main.py - Main Application with Interactive Chat Interface
import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from crew import EnhancedDataScienceCrew
import tempfile
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Agentic Data Science Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    .system-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .pipeline-status {
        padding: 0.5rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .status-not-started { background: #ffebee; color: #c62828; }
    .status-running { background: #fff3e0; color: #ef6c00; }
    .status-completed { background: #e8f5e8; color: #2e7d32; }
    .status-error { background: #ffebee; color: #c62828; }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'crew' not in st.session_state:
        # Try to initialize with API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.session_state.crew = EnhancedDataScienceCrew(api_key)
            st.session_state.ai_initialized = True
        else:
            st.session_state.crew = None
            st.session_state.ai_initialized = False

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'uploaded_datasets' not in st.session_state:
        st.session_state.uploaded_datasets = {}

    if 'pipeline_status' not in st.session_state:
        st.session_state.pipeline_status = "Not Started"

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

    if 'current_context' not in st.session_state:
        st.session_state.current_context = {}


# Helper functions
def add_message_to_chat(role: str, content: str, message_type: str = "normal"):
    """Add a message to the chat history"""
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": message_type
    })


def display_chat_message(message: dict):
    """Display a chat message with appropriate styling"""
    timestamp = message.get("timestamp", "")
    content = message["content"]

    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div>
                <strong>👤 You ({timestamp}):</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif message["role"] == "system":
        st.markdown(f"""
        <div class="chat-message system-message">
            <div>
                <strong>🔧 System ({timestamp}):</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:  # assistant
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div>
                <strong>🤖 AI Assistant ({timestamp}):</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return the path"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)

        # Save file
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def process_user_input(user_input: str):
    """Process user input and get response from the crew"""
    if not st.session_state.ai_initialized or st.session_state.crew is None:
        return "❌ AI system not initialized. Please check your OpenAI API key in the environment variables."

    try:
        # Get current context
        context = st.session_state.crew.get_analysis_context()
        context.update(st.session_state.current_context)

        # Process with crew
        response = st.session_state.crew.handle_user_conversation(user_input, context)

        # Update current context
        st.session_state.current_context = st.session_state.crew.get_analysis_context()

        return response
    except Exception as e:
        return f"❌ Error processing request: {str(e)}"


# Initialize session state
initialize_session_state()

# Main App
st.markdown('<h1 class="main-header">🤖 Agentic Data Science Assistant</h1>', unsafe_allow_html=True)

# System status display
if st.session_state.ai_initialized:
    st.markdown("""
    <div class="success-box">
        <strong>✅ Agentic AI System Active</strong><br>
        Multi-agent data science crew is ready to assist you with comprehensive analysis!
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ AI System Not Initialized</strong><br>
        Please ensure OPENAI_API_KEY is set in your .env file. Basic functionality available with fallback responses.
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")

    # Dataset Management
    st.subheader("📊 Dataset Management")

    uploaded_files = st.file_uploader(
        "Upload your datasets",
        type=['csv', 'xlsx', 'json'],
        accept_multiple_files=True,
        help="Upload one or more datasets for analysis"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_datasets:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    # Load and preview data
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file_path)
                        elif uploaded_file.name.endswith('.json'):
                            df = pd.read_json(file_path)

                        st.session_state.uploaded_datasets[uploaded_file.name] = {
                            'path': file_path,
                            'dataframe': df,
                            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # Add system message
                        add_message_to_chat(
                            "system",
                            f"📊 Dataset '{uploaded_file.name}' uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                            "system"
                        )

                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {str(e)}")

    # Display uploaded datasets
    if st.session_state.uploaded_datasets:
        st.subheader("📁 Uploaded Datasets")
        for name, info in st.session_state.uploaded_datasets.items():
            with st.expander(f"📄 {name}"):
                st.write(f"**Shape:** {info['dataframe'].shape}")
                st.write(f"**Uploaded:** {info['upload_time']}")
                st.write(
                    f"**Columns:** {', '.join(info['dataframe'].columns[:5])}{'...' if len(info['dataframe'].columns) > 5 else ''}")

                # Quick actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Analyze", key=f"analyze_{name}"):
                        add_message_to_chat("user", f"Please analyze the dataset '{name}'")
                        response = process_user_input(
                            f"Please perform comprehensive analysis of the dataset at {info['path']}")
                        add_message_to_chat("assistant", response)
                        st.rerun()

                with col2:
                    if st.button(f"🎨 Visualize", key=f"viz_{name}"):
                        add_message_to_chat("user", f"Create visualizations for dataset '{name}'")
                        response = process_user_input(
                            f"Create comprehensive visualizations for the dataset at {info['path']}")
                        add_message_to_chat("assistant", response)
                        st.rerun()

    st.markdown("---")

    # Pipeline Control
    st.subheader("🔬 Pipeline Control")

    # Pipeline status
    status_class = f"status-{st.session_state.pipeline_status.lower().replace(' ', '-')}"
    st.markdown(f"""
    <div class="pipeline-status {status_class}">
        Pipeline Status: {st.session_state.pipeline_status}
    </div>
    """, unsafe_allow_html=True)

    # Quick Pipeline Actions
    if st.session_state.uploaded_datasets:
        dataset_names = list(st.session_state.uploaded_datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Pipeline", dataset_names)

        if selected_dataset:
            dataset_info = st.session_state.uploaded_datasets[selected_dataset]
            df = dataset_info['dataframe']

            # Target column selection
            target_column = st.selectbox(
                "Select Target Column (Optional)",
                ["None"] + df.columns.tolist()
            )
            target_column = None if target_column == "None" else target_column

            # Business context
            business_context = st.text_area(
                "Business Context (Optional)",
                placeholder="Describe your business problem or domain..."
            )

            # Pipeline actions
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🚀 Run Full Pipeline"):
                    st.session_state.pipeline_status = "Running"
                    add_message_to_chat("system", "🚀 Starting full pipeline execution...", "system")

                    try:
                        result = st.session_state.crew.run_full_pipeline(
                            dataset_info['path'],
                            target_column,
                            business_context
                        )

                        if result['status'] == 'completed':
                            st.session_state.pipeline_status = "Completed"
                            st.session_state.analysis_results = result
                            add_message_to_chat("assistant",
                                                "✅ Full pipeline completed successfully! Check the results in the Analysis tab.")
                        else:
                            st.session_state.pipeline_status = "Error"
                            add_message_to_chat("assistant",
                                                f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.session_state.pipeline_status = "Error"
                        add_message_to_chat("assistant", f"❌ Pipeline error: {str(e)}")

                    st.rerun()

            with col2:
                if st.button("🔗 Multi-Dataset Analysis") and len(st.session_state.uploaded_datasets) > 1:
                    add_message_to_chat("system", "🔗 Starting multi-dataset analysis...", "system")

                    try:
                        dataset_paths = [info['path'] for info in st.session_state.uploaded_datasets.values()]
                        result = st.session_state.crew.run_multi_dataset_analysis(dataset_paths)

                        if result['status'] == 'completed':
                            add_message_to_chat("assistant",
                                                "✅ Multi-dataset analysis completed! Found relationships and patterns across your datasets.")
                        else:
                            add_message_to_chat("assistant",
                                                f"❌ Multi-dataset analysis failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        add_message_to_chat("assistant", f"❌ Multi-dataset analysis error: {str(e)}")

                    st.rerun()

    st.markdown("---")

    # Session Management
    st.subheader("💾 Session Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Session"):
            try:
                session_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                if st.session_state.crew:
                    st.session_state.crew.save_session(session_file)
                    st.success(f"Session saved as {session_file}")
                else:
                    st.warning("No active crew to save")
            except Exception as e:
                st.error(f"Error saving session: {str(e)}")

    with col2:
        if st.button("🗑️ Reset Session"):
            st.session_state.conversation_history = []
            st.session_state.uploaded_datasets = {}
            st.session_state.pipeline_status = "Not Started"
            st.session_state.analysis_results = {}
            st.session_state.current_context = {}
            if st.session_state.crew:
                st.session_state.crew.reset_context()
            st.success("Session reset!")
            st.rerun()

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat Assistant", "📊 Data Explorer", "🔬 Analysis Results", "📈 Multi-Dataset Insights"])

# Tab 1: Chat Assistant
with tab1:
    st.header("💬 Conversational AI Assistant")
    st.markdown("""
    **Your intelligent data science partner is ready to help!** 

    Ask me anything about data analysis, feature engineering, modeling, or insights. I can:
    - 🔍 Analyze your datasets comprehensively
    - 🧹 Clean and preprocess your data intelligently  
    - 🔧 Engineer powerful features automatically
    - 🤖 Build and validate ML models
    - 🎨 Create insightful visualizations
    - 🔗 Find relationships across multiple datasets
    """)

    # Chat interface
    chat_container = st.container()

    # Display conversation history
    with chat_container:
        for message in st.session_state.conversation_history:
            display_chat_message(message)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Ask me anything about your data:",
                placeholder="e.g., 'Analyze my dataset and find the most important features for prediction'",
                key="chat_input"
            )

        with col2:
            send_button = st.form_submit_button("Send 🚀")

    if send_button and user_input:
        # Add user message
        add_message_to_chat("user", user_input)

        # Process input and get response
        with st.spinner("🤖 AI agents are working on your request..."):
            response = process_user_input(user_input)
            add_message_to_chat("assistant", response)

        st.rerun()

    # Quick Action Buttons
    st.subheader("⚡ Quick Actions")

    if st.session_state.uploaded_datasets:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔍 **Analyze All Data**", key="quick_analyze"):
                add_message_to_chat("user", "Please analyze all my uploaded datasets")
                response = process_user_input("Perform comprehensive analysis of all uploaded datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

        with col2:
            if st.button("🎨 **Create Visualizations**", key="quick_viz"):
                add_message_to_chat("user", "Create visualizations for my data")
                response = process_user_input("Create comprehensive visualizations for all datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

        with col3:
            if st.button("🔧 **Feature Engineering**", key="quick_features"):
                add_message_to_chat("user", "Help me with feature engineering")
                response = process_user_input("Suggest and implement feature engineering strategies for my datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

        with col4:
            if st.button("🤖 **Model Recommendations**", key="quick_models"):
                add_message_to_chat("user", "What models should I use?")
                response = process_user_input("Recommend optimal machine learning models for my datasets")
                add_message_to_chat("assistant", response)
                st.rerun()
    else:
        st.info("👆 Upload some datasets first to unlock quick actions!")

    # Example questions
    with st.expander("💡 Example Questions to Get Started"):
        st.markdown("""
        **Data Analysis:**
        - "What patterns do you see in my data?"
        - "Are there any quality issues I should address?"
        - "Which features are most correlated with my target variable?"

        **Data Cleaning:**
        - "How should I handle missing values in my dataset?"
        - "What's the best way to deal with outliers?"
        - "Can you clean my data intelligently?"

        **Feature Engineering:**
        - "What new features should I create?"
        - "How can I improve my model's performance with better features?"
        - "Should I use polynomial features or interactions?"

        **Modeling:**
        - "What's the best algorithm for my problem?"
        - "How can I improve my model's accuracy?"
        - "Should I use ensemble methods?"

        **Multi-Dataset Analysis:**
        - "How are my datasets related?"
        - "Can I combine these datasets for better insights?"
        - "What patterns exist across all my data?"
        """)

# Tab 2: Data Explorer
with tab2:
    st.header("📊 Interactive Data Explorer")

    if st.session_state.uploaded_datasets:
        # Dataset selector
        dataset_names = list(st.session_state.uploaded_datasets.keys())
        selected_dataset = st.selectbox("Select Dataset to Explore", dataset_names, key="explorer_dataset")

        if selected_dataset:
            dataset_info = st.session_state.uploaded_datasets[selected_dataset]
            df = dataset_info['dataframe']

            # Dataset overview
            st.subheader("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.shape[0]}</h3>
                    <p>Total Rows</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.shape[1]}</h3>
                    <p>Total Columns</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                missing_values = df.isnull().sum().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{missing_values}</h3>
                    <p>Missing Values</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{memory_mb:.1f} MB</h3>
                    <p>Memory Usage</p>
                </div>
                """, unsafe_allow_html=True)

            # Data preview
            st.subheader("👀 Data Preview")
            show_rows = st.slider("Number of rows to display:", 5, min(100, len(df)), 10)
            st.dataframe(df.head(show_rows), use_container_width=True)

            # Column information
            st.subheader("📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

            # Statistical summary
            st.subheader("📈 Statistical Summary")
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary.")

            # Interactive actions
            st.subheader("🔧 Interactive Actions")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(f"🔍 Deep Analysis", key=f"deep_analysis_{selected_dataset}"):
                    add_message_to_chat("user", f"Perform deep analysis of {selected_dataset}")
                    response = process_user_input(
                        f"Perform comprehensive deep analysis of the dataset {dataset_info['path']}")
                    add_message_to_chat("assistant", response)
                    st.rerun()

            with col2:
                if st.button(f"🧹 Clean Data", key=f"clean_data_{selected_dataset}"):
                    add_message_to_chat("user", f"Clean the dataset {selected_dataset}")
                    response = process_user_input(f"Intelligently clean the dataset at {dataset_info['path']}")
                    add_message_to_chat("assistant", response)
                    st.rerun()

            with col3:
                if st.button(f"🎨 Visualize", key=f"visualize_{selected_dataset}"):
                    add_message_to_chat("user", f"Create visualizations for {selected_dataset}")
                    response = process_user_input(f"Create comprehensive visualizations for {dataset_info['path']}")
                    add_message_to_chat("assistant", response)
                    st.rerun()
    else:
        st.info("📁 Upload datasets in the sidebar to start exploring!")

# Tab 3: Analysis Results
with tab3:
    st.header("🔬 Analysis Results & Insights")

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        # Pipeline summary
        st.subheader("🏗️ Pipeline Summary")
        if 'stages_completed' in results:
            stages = results['stages_completed']
            total_stages = 4  # analysis, cleaning, feature_engineering, modeling
            progress = len(stages) / total_stages

            st.progress(progress)
            st.write(f"Completed stages: {', '.join(stages)} ({len(stages)}/{total_stages})")

            # Detailed results for each stage
            if 'results' in results:
                for stage, stage_result in results['results'].items():
                    with st.expander(f"📊 {stage.replace('_', ' ').title()} Results"):
                        if stage_result['status'] == 'completed':
                            st.success("✅ Stage completed successfully")
                            if 'result' in stage_result:
                                st.text_area("Results:", str(stage_result['result']), height=200)
                        else:
                            st.error(f"❌ Stage failed: {stage_result.get('error', 'Unknown error')}")

        # Context information
        if 'context' in results:
            st.subheader("📋 Analysis Context")
            context = results['context']

            col1, col2 = st.columns(2)
            with col1:
                if 'data_path' in context:
                    st.write(f"**Original Data:** {os.path.basename(context['data_path'])}")
                if 'target_column' in context:
                    st.write(f"**Target Column:** {context['target_column']}")

            with col2:
                if 'cleaned_data_path' in context:
                    st.write(f"**Cleaned Data:** {os.path.basename(context['cleaned_data_path'])}")
                if 'engineered_data_path' in context:
                    st.write(f"**Engineered Data:** {os.path.basename(context['engineered_data_path'])}")

        # Generate insights
        st.subheader("💡 AI-Generated Insights")
        if st.button("🧠 Generate Business Insights"):
            with st.spinner("Generating insights..."):
                try:
                    insights = st.session_state.crew.generate_insights_and_recommendations(
                        results,
                        "Business stakeholder seeking actionable insights",
                        "General business analysis"
                    )
                    st.markdown("### 🎯 Key Insights & Recommendations")
                    st.write(insights)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")

    else:
        st.info("🚀 Run a pipeline first to see analysis results!")

        # Quick start buttons
        if st.session_state.uploaded_datasets:
            st.subheader("🎯 Quick Start")
            dataset_names = list(st.session_state.uploaded_datasets.keys())

            for dataset_name in dataset_names:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{dataset_name}**")
                with col2:
                    if st.button(f"🚀 Analyze", key=f"quick_start_{dataset_name}"):
                        dataset_info = st.session_state.uploaded_datasets[dataset_name]
                        add_message_to_chat("system", f"Starting analysis of {dataset_name}...", "system")

                        try:
                            result = st.session_state.crew.run_comprehensive_analysis_pipeline(
                                dataset_info['path'],
                                business_context="Quick analysis"
                            )
                            st.session_state.analysis_results = result
                            add_message_to_chat("assistant", "✅ Analysis completed! Check the Analysis Results tab.")
                        except Exception as e:
                            add_message_to_chat("assistant", f"❌ Analysis failed: {str(e)}")

                        st.rerun()

# Tab 4: Multi-Dataset Insights
with tab4:
    st.header("📈 Multi-Dataset Insights & Relationships")

    if len(st.session_state.uploaded_datasets) > 1:
        st.subheader("🔗 Dataset Relationships")

        # Basic relationship overview
        dataset_names = list(st.session_state.uploaded_datasets.keys())
        st.write(f"**Uploaded Datasets:** {len(dataset_names)}")

        # Dataset comparison table
        comparison_data = []
        for name, info in st.session_state.uploaded_datasets.items():
            df = info['dataframe']
            comparison_data.append({
                'Dataset': name,
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Numeric Cols': len(df.select_dtypes(include=['number']).columns),
                'Categorical Cols': len(df.select_dtypes(include=['object']).columns),
                'Missing %': f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Common columns analysis
        st.subheader("🔍 Common Columns Analysis")
        all_columns = set()
        dataset_columns = {}

        for name, info in st.session_state.uploaded_datasets.items():
            cols = set(info['dataframe'].columns)
            dataset_columns[name] = cols
            all_columns.update(cols)

        # Find common columns
        common_columns = set.intersection(*dataset_columns.values()) if dataset_columns else set()

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Unique Columns:** {len(all_columns)}")
            st.write(f"**Common to All Datasets:** {len(common_columns)}")

            if common_columns:
                st.write("**Common Columns:**")
                for col in sorted(common_columns):
                    st.write(f"• {col}")

        with col2:
            # Similar column names
            similar_cols = []
            dataset_names = list(dataset_columns.keys())

            for i in range(len(dataset_names)):
                for j in range(i + 1, len(dataset_names)):
                    name1, name2 = dataset_names[i], dataset_names[j]
                    cols1, cols2 = dataset_columns[name1], dataset_columns[name2]

                    for col1 in cols1:
                        for col2 in cols2:
                            if col1 != col2 and (col1.lower() in col2.lower() or col2.lower() in col1.lower()):
                                similar_cols.append(f"{col1} ↔ {col2}")

            if similar_cols:
                st.write("**Similar Column Names:**")
                for sim in similar_cols[:10]:  # Show first 10
                    st.write(f"• {sim}")

        # Multi-dataset actions
        st.subheader("🔧 Multi-Dataset Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔗 **Find Relationships**", key="find_relationships"):
                add_message_to_chat("user", "Find relationships between my datasets")
                response = process_user_input("Analyze relationships and connections between all uploaded datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

        with col2:
            if st.button("🎯 **Integration Strategy**", key="integration_strategy"):
                add_message_to_chat("user", "How can I integrate these datasets?")
                response = process_user_input("Suggest strategies for integrating and combining my multiple datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

        with col3:
            if st.button("🤖 **Unified Modeling**", key="unified_modeling"):
                add_message_to_chat("user", "How can I build models across multiple datasets?")
                response = process_user_input("Recommend unified modeling approaches for my multiple datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

        # Advanced multi-dataset analysis
        with st.expander("🔬 Advanced Multi-Dataset Analysis"):
            st.markdown("""
            **Available Analysis Types:**

            🔍 **Relationship Discovery**
            - Find join opportunities between datasets
            - Identify overlapping data and values
            - Discover hidden connections

            📊 **Cross-Dataset Patterns**
            - Identify common trends across datasets
            - Find complementary information
            - Detect data consistency issues

            🎯 **Integration Opportunities**
            - Recommend optimal join strategies
            - Suggest feature sharing approaches
            - Plan ensemble modeling strategies

            🤖 **Unified Modeling**
            - Multi-dataset ensemble approaches
            - Cross-dataset feature engineering
            - Transfer learning opportunities
            """)

            # Custom analysis options
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                ["Comprehensive", "Relationship Discovery", "Pattern Analysis", "Integration Planning"]
            )

            if st.button("🚀 Run Custom Analysis"):
                add_message_to_chat("user", f"Run {analysis_type} on my datasets")
                response = process_user_input(f"Perform {analysis_type} across all uploaded datasets")
                add_message_to_chat("assistant", response)
                st.rerun()

    elif len(st.session_state.uploaded_datasets) == 1:
        st.info("📁 Upload multiple datasets to see relationship analysis and cross-dataset insights!")

        # Single dataset advanced analysis
        st.subheader("🔍 Advanced Single Dataset Analysis")
        dataset_name = list(st.session_state.uploaded_datasets.keys())[0]
        dataset_info = st.session_state.uploaded_datasets[dataset_name]

        st.write(f"**Current Dataset:** {dataset_name}")
        st.write("Upload additional datasets to unlock:")
        st.markdown("""
        - 🔗 Cross-dataset relationship discovery
        - 📊 Pattern analysis across multiple sources  
        - 🎯 Data integration strategies
        - 🤖 Multi-dataset ensemble modeling
        - 📈 Comprehensive business insights
        """)

        # Suggest what to upload next
        df = dataset_info['dataframe']
        suggestions = []

        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            suggestions.append("📅 **Time-series data** - to analyze temporal patterns")

        if any('id' in col.lower() for col in df.columns):
            suggestions.append("🔗 **Related transactional data** - to join with ID columns")

        if df.select_dtypes(include=['object']).shape[1] > 0:
            suggestions.append("📊 **Reference/lookup tables** - to enrich categorical data")

        if suggestions:
            st.subheader("💡 Suggested Additional Datasets")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")

    else:
        st.info("📂 Upload datasets in the sidebar to start multi-dataset analysis!")

        # Benefits of multi-dataset analysis
        st.subheader("🌟 Benefits of Multi-Dataset Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🔍 Discover Hidden Insights**
            - Find relationships across data sources
            - Uncover patterns invisible in single datasets
            - Identify data quality issues through cross-validation

            **🎯 Better Decision Making**
            - Get comprehensive view of your business
            - Reduce bias from single data source
            - Make data-driven decisions with confidence
            """)

        with col2:
            st.markdown("""
            **🤖 Enhanced Modeling**
            - Build more robust predictive models
            - Use ensemble methods across datasets
            - Apply transfer learning techniques

            **⚡ Operational Efficiency**
            - Automate data integration processes
            - Standardize analysis across teams
            - Scale insights across organization
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🤖 <strong>Agentic Data Science Assistant</strong> | 
    Powered by Multi-Agent AI System | 
    Built with CrewAI, OpenAI, and Streamlit
    <br>
    <small>Your intelligent partner for comprehensive data science workflows</small>
</div>
""", unsafe_allow_html=True)

# Auto-scroll chat to bottom (JavaScript)
st.markdown("""
<script>
function scrollToBottom() {
    var chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}
setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)  # main.py - Main Application with Interactive Chat Interface
import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from crew import EnhancedDataScienceCrew
import tempfile
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Agentic Data Science Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    .system-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        padding: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    } 
</style>
""", unsafe_allow_html=True)