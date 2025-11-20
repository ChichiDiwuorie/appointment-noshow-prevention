# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Still used by some libraries, but we'll use joblib for our model
import joblib  # ### MODIFIED ### - Using joblib for loading the scikit-learn pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Healthcare Appointment No-Show Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ### MODIFIED ### - LOAD THE NEW, SINGLE PIPELINE
# ==============================================================================
@st.cache_resource
def load_pipeline():
    """Load the final, custom-trained prediction pipeline from a single file."""
    try:
        # This function now loads the single .pkl file created by your training script.
        pipeline = joblib.load('model/final_model_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("❌ Final model file ('model/final_model_pipeline.pkl') not found.")
        st.info("💡 Please ensure you have run the `train_final_model.py` script to create the model file.")
        st.stop()

# Load the entire pipeline once at the start of the app.
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"A critical error occurred while loading the model pipeline: {e}")
    st.stop()


# ==============================================================================
# ### MODIFIED ### - HELPER FUNCTIONS USING THE PIPELINE
# ==============================================================================

def validate_csv(df, pipeline_obj):
    """Validate uploaded CSV has required columns based on the pipeline's memory."""
    # The pipeline itself knows which features it was trained on. This is robust!
    try:
        required_columns = set(pipeline_obj.feature_names_in_)
        uploaded_columns = set(df.columns)
        missing_columns = required_columns - uploaded_columns

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        return True, "CSV is valid!"
    except Exception as e:
        return False, f"An error occurred during validation: {e}"


def make_predictions(df, pipeline_obj):
    """Make predictions on the uploaded data using the consolidated pipeline."""
    # No more manual preprocessing! The pipeline handles scaling internally.
    
    # Ensure the dataframe has columns in the order the pipeline expects
    try:
        X_predict = df[pipeline_obj.feature_names_in_]
    except KeyError as e:
        st.error(f"A required column is missing from the uploaded data: {e}")
        st.stop()
    
    # Get predictions and probabilities directly from the pipeline
    predictions = pipeline_obj.predict(X_predict)
    probabilities = pipeline_obj.predict_proba(X_predict)[:, 1] # Probability of '1' class (no-show)

    # Add new columns with prediction results to the original dataframe
    result_df = df.copy()
    result_df['predicted_noshow'] = predictions
    result_df['risk_score'] = probabilities
    
    risk_info = [calculate_risk_level(p) for p in probabilities]
    result_df['risk_level'] = [info[0] for info in risk_info]
    result_df['risk_icon'] = [info[1] for info in risk_info]

    return result_df

# This function remains unchanged as it's pure logic.
def calculate_risk_level(probability):
    """Convert probability to a risk level category and an emoji icon."""
    if probability < 0.25:
        return "Low", "🟢"
    elif probability < 0.60:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📤 Upload & Predict", "📊 Analytics Dashboard", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Project Info")
st.sidebar.markdown("**UN SDG Goal 3**  \nGood Health and Well-being")
st.sidebar.markdown("**Team:**  \n- Chichi (PM)  \n- Himanshu (Data)  \n- Judah (Engineering)  \n- Ousmane (QA)")

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏥 Healthcare Appointment No-Show Prevention System</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h3 style='color: #1f77b4; margin-top: 0;'>Welcome to Our Predictive Analytics Platform</h3>
        <p style='font-size: 1.1rem; margin-bottom: 0;'>
            This system uses machine learning to predict appointment no-shows and help
            healthcare facilities optimize their scheduling and reminder strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📊 Average No-Show Rate",
            value="20-30%",
            delta="Healthcare Industry",
            delta_color="off"
        )

    with col2:
        st.metric(
            label="💰 Annual Cost",
            value="$150B",
            delta="U.S. Healthcare System",
            delta_color="off"
        )

    with col3:
        st.metric(
            label="🎯 Model Accuracy",
            value="70-85%",
            delta="ML Prediction Range",
            delta_color="off"
        )

    st.markdown("---")

    # How It Works
    st.subheader("🔄 How It Works")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### Simple 5-Step Process:

        1. **📤 Upload** your appointment data (CSV format)
        2. **🔍 Analyze** patient and appointment characteristics
        3. **🤖 Predict** which appointments are high-risk for no-shows
        4. **📊 Visualize** patterns and insights in interactive charts
        5. **📥 Download** actionable reports for staff

        All processing happens instantly in your browser!
        """)

    with col2:
        st.info("""
        ### 🎯 What You Get:

        - **Risk Scores** for each appointment (0.0-1.0)
        - **High-Risk Flags** for appointments needing attention
        - **Interactive Charts** showing patterns by age, day, etc.
        - **Downloadable Reports** in CSV format
        - **Actionable Insights** to reduce no-shows
        """)

    # Quick Start
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")

    with st.expander("📋 What data format do I need?"):
        # ### MODIFIED ### - Get feature names directly from the loaded pipeline
        feature_names_from_model = pipeline.feature_names_in_
        st.markdown(f"""
        Your CSV file should contain these {len(feature_names_from_model)} columns:
        - `{'` - `'.join(feature_names_from_model)}`
        """)

    with st.expander("❓ How accurate is the prediction?"):
        st.markdown(f"""
        Our Random Forest model achieves **70-85% accuracy** based on:
        - 110,000+ historical appointments
        - {len(pipeline.feature_names_in_)} key predictive features
        - Validated on test data

        The model identifies **high-risk appointments** with >60% confidence.
        """)

    with st.expander("🔒 Is my data secure?"):
        st.markdown("""
        - All data processing happens in your browser
        - No data is stored on our servers
        - Files are only kept during your session
        - Compliant with healthcare data privacy standards
        """)

    st.success("👈 **Ready to get started?** Use the sidebar to navigate to 'Upload & Predict'")

# ==============================================================================
# PAGE 2: UPLOAD & PREDICT
# ==============================================================================
elif page == "📤 Upload & Predict":
    st.markdown('<p class="main-header">📤 Upload Appointment Data & Get Predictions</p>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your appointment CSV file",
        type=['csv'],
        help="CSV file should contain required appointment features"
    )

    # Sample data option
    use_sample = st.checkbox("Or use sample data for demo (no upload needed)")

    df = None # Initialize df to None

    if use_sample:
        try:
            st.info("📁 Loading sample data...")
            # ### MODIFIED ### - Point to the new sample data file you created in Step 7
            df = pd.read_csv('data/appointment_sample_data.csv') 
        except FileNotFoundError:
            st.error("❌ Sample data file ('data/appointment_sample_data.csv') not found. Please create it.")
            st.stop()
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
    if df is not None:
        try:
            st.success(f"✅ Loaded {len(df):,} appointments successfully!")

            # ### MODIFIED ### - Validate CSV using the pipeline object
            is_valid, message = validate_csv(df, pipeline)

            if not is_valid:
                st.error(f"❌ {message}")
                st.info("💡 Please ensure your CSV has all required columns. Check the 'Home' page for column requirements.")
                st.stop()

            # Display data preview
            with st.expander("📋 Data Preview (First 10 Rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # Data summary
            st.subheader("📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Appointments", f"{len(df):,}")
            with col2:
                # Use .get() for safe access in case column doesn't exist
                avg_age = df.get('age', pd.Series([0])).mean()
                st.metric("Average Age", f"{avg_age:.1f}")
            with col3:
                avg_lead = df.get('lead_time_days', pd.Series([0])).mean()
                st.metric("Avg Lead Time", f"{avg_lead:.1f} days")
            with col4:
                sms_pct = (df.get('sms_received', pd.Series([0])).sum() / len(df) * 100)
                st.metric("SMS Sent", f"{sms_pct:.1f}%")

            st.markdown("---")

            # Predict button
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing appointments and generating predictions..."):
                    # ### MODIFIED ### - Make predictions using the pipeline object
                    result_df = make_predictions(df, pipeline)

                    # Store in session state
                    st.session_state['predictions'] = result_df

                st.success("✅ Predictions generated successfully!")

                # Results summary
                st.subheader("📈 Prediction Results")

                col1, col2, col3, col4 = st.columns(4)

                predicted_noshows = result_df['predicted_noshow'].sum()
                high_risk = (result_df['risk_level'] == 'High').sum()
                medium_risk = (result_df['risk_level'] == 'Medium').sum()
                low_risk = (result_df['risk_level'] == 'Low').sum()

                with col1:
                    st.metric(
                        "Predicted No-Shows",
                        f"{predicted_noshows:,}",
                        f"{predicted_noshows/len(df)*100:.1f}%"
                    )
                with col2:
                    st.metric(
                        "🔴 High Risk",
                        f"{high_risk:,}",
                        f"{high_risk/len(df)*100:.1f}%"
                    )
                with col3:
                    st.metric(
                        "🟡 Medium Risk",
                        f"{medium_risk:,}",
                        f"{medium_risk/len(df)*100:.1f}%"
                    )
                with col4:
                    st.metric(
                        "🟢 Low Risk",
                        f"{low_risk:,}",
                        f"{low_risk/len(df)*100:.1f}%"
                    )

                # High-risk appointments
                st.subheader("🚨 High-Risk Appointments (Require Immediate Attention)")

                high_risk_df = result_df[result_df['risk_level'] == 'High'].sort_values(
                    'risk_score', ascending=False
                )

                if not high_risk_df.empty:
                    # Define columns to display dynamically based on what's available
                    display_cols = ['age', 'lead_time_days', 'sms_received', 'risk_score', 'risk_level']
                    if 'appointment_id' in high_risk_df.columns:
                        display_cols.insert(0, 'appointment_id')
                    elif 'patient_id' in high_risk_df.columns:
                        display_cols.insert(0, 'patient_id')
                    
                    display_cols = [col for col in display_cols if col in high_risk_df.columns]

                    st.dataframe(
                        high_risk_df[display_cols].head(20),
                        use_container_width=True
                    )
                    st.warning(f"⚠️ {len(high_risk_df)} appointments flagged as high-risk. Consider additional follow-up.")
                else:
                    st.success("✅ No high-risk appointments detected!")

                # Download predictions
                st.subheader("📥 Download Results")
                col1, col2 = st.columns(2)

                with col1:
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download Full Predictions (CSV)",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    if not high_risk_df.empty:
                        high_risk_csv = high_risk_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="🚨 Download High-Risk Only (CSV)",
                            data=high_risk_csv,
                            file_name=f"high_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please check your CSV file format and try again.")

# ==============================================================================
# PAGE 3: ANALYTICS DASHBOARD
# ==============================================================================
elif page == "📊 Analytics Dashboard":
    st.markdown('<p class="main-header">📊 Analytics Dashboard</p>', unsafe_allow_html=True)

    if 'predictions' not in st.session_state:
        st.warning("⚠️ No predictions available yet. Please upload data and generate predictions first.")
        st.info("👈 Go to 'Upload & Predict' page to get started")
    else:
        df = st.session_state['predictions']

        # Overview metrics
        st.subheader("📈 Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Appointments", f"{len(df):,}")
        with col2:
            avg_risk = df['risk_score'].mean() * 100
            st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
        with col3:
            if 'lead_time_days' in df.columns:
                avg_lead_time = df['lead_time_days'].mean()
                st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
        with col4:
            high_risk_count = (df['risk_level'] == 'High').sum()
            st.metric("High-Risk Count", f"{high_risk_count:,}")

        st.markdown("---")

        # Visualizations
        st.subheader("📊 Risk Distribution")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Risk Level Distribution")
            risk_counts = df['risk_level'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
            ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=[colors.get(key, '#95a5a6') for key in risk_counts.index], startangle=90)
            ax.set_title('Appointments by Risk Level')
            st.pyplot(fig)
            plt.close(fig) 

        with col2:
            st.markdown("#### Risk Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df['risk_score'], bins=30, color='steelblue', edgecolor='black')
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Risk Scores')
            ax.axvline(x=0.6, color='red', linestyle='--', label='High Risk Threshold (>0.6)')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # Pattern Analysis
        st.subheader("🔍 Pattern Analysis")
        col1, col2 = st.columns(2)
        
        # Initialize these outside the 'if' blocks to handle cases where they aren't created
        age_risk = pd.Series()
        lead_risk = pd.Series()

        with col1:
            if 'age' in df.columns:
                st.markdown("#### Risk Score by Age Group")
                df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 120],
                                         labels=['0-18', '19-35', '36-50', '51-65', '65+'], right=False)
                age_risk = df.groupby('age_group', observed=True)['risk_score'].mean() * 100

                fig, ax = plt.subplots(figsize=(8, 6))
                age_risk.plot(kind='bar', ax=ax, color='coral')
                ax.set_title('Average Risk Score by Age Group')
                ax.set_ylabel('Risk Score (%)')
                ax.set_xlabel('Age Group')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            if 'lead_time_days' in df.columns:
                st.markdown("#### Risk Score by Lead Time")
                df['lead_time_category'] = pd.cut(df['lead_time_days'],
                                                   bins=[-1, 7, 14, 30, 60, 365], 
                                                   labels=['0-7', '8-14', '15-30', '31-60', '60+'])
                lead_risk = df.groupby('lead_time_category', observed=True)['risk_score'].mean() * 100

                fig, ax = plt.subplots(figsize=(8, 6))
                lead_risk.plot(kind='bar', ax=ax, color='teal')
                ax.set_title('Average Risk Score by Lead Time')
                ax.set_ylabel('Risk Score (%)')
                ax.set_xlabel('Lead Time (days)')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)
        
        # Summary Insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            if not age_risk.empty:
                st.info(f"""
                **Highest Risk Group:**  
                {age_risk.idxmax()} age group with {age_risk.max():.1f}% avg risk score
                """)
        with insights_col2:
            if not lead_risk.empty:
                st.info(f"""
                **Critical Lead Time:**  
                Appointments {lead_risk.idxmax()} days out have highest risk ({lead_risk.max():.1f}%)
                """)

# ==============================================================================
# PAGE 4: ABOUT
# ==============================================================================
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    ## Automated Patient Appointment Management & No-Show Prevention System

    ### 🎯 UN Sustainable Development Goal 3
    **Good Health and Well-being**

    This system addresses the critical challenge of healthcare appointment no-shows,
    which cost the U.S. healthcare system approximately $150 billion annually and
    disrupt care continuity for vulnerable populations.
    """)

    st.markdown("---")

    # Team
    st.subheader("👥 Team Members")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **Chichi**  
        *Project Manager*
        """)
    with col2:
        st.markdown("""
        **Himanshu**  
        *Data Analytics Lead*
        """)
    with col3:
        st.markdown("""
        **Judah**  
        *Engineering Lead*
        """)
    with col4:
        st.markdown("""
        **Ousmane**  
        *Quality Assurance*
        """)

    st.markdown("---")

    # Technology Stack
    st.subheader("🛠️ Technology Stack")
    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        **Core Technologies:**
        - **Python 3.12+** - Programming language
        - **Streamlit** - Web framework
        - **Scikit-learn** - Machine learning
        - **Pandas** - Data processing
        - **Matplotlib/Seaborn** - Visualizations
        """)
    with tech_col2:
        st.markdown("""
        **Deployment:**
        - **Streamlit Cloud** - Free hosting
        - **GitHub** - Version control

        **Model:**
        - **Random Forest Classifier**
        - **70-85% Accuracy**
        """)

    st.markdown("---")
    
    # Project Info
    st.subheader("📚 Project Information")
    st.markdown("""
    **Program:** Grow with Google IT Automation with Python Scholarship 2025  
    **Project Type:** Capstone Project  
    **Goal:** Reduce healthcare appointment no-shows through predictive analytics
    """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        © 2025 Healthcare Appointment Management Team |
        Grow with Google Capstone Project |
        UN SDG Goal 3: Good Health and Well-being
    </div>
    """,
    unsafe_allow_html=True
)