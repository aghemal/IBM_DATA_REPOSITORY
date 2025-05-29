import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from io import StringIO
import tempfile
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Data Quality Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0c0080;
    }
    footer {
        visibility: hidden;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        
        # Convert all columns to string representation for Arrow compatibility
        for col in df.select_dtypes(include=['object', 'datetime64']).columns:
            df[col] = df[col].astype(str)
        
        # Ensure numeric columns are proper float types
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
# Enhanced null value check
def check_null_values(df):
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    null_stats = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage': null_percent.round(2)
    })
    return null_stats.sort_values('Null Count', ascending=False)

# Enhanced duplicate check
def check_duplicates(df):
    dupes = df.duplicated()
    return {
        'count': dupes.sum(),
        'percentage': (dupes.sum() / len(df)) * 100,
        'indices': np.where(dupes)[0].tolist()
    }

# Advanced anomaly detection
def detect_anomalies(df, columns, contamination=0.05):
    df_numeric = df[columns].select_dtypes(include=[np.number])
    if df_numeric.empty:
        return pd.DataFrame()

    # Advanced imputation
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=150,
        max_samples='auto'
    )
    anomalies = model.fit_predict(df_imputed)
    df['anomaly_score'] = model.decision_function(df_imputed)
    return df[anomalies == -1].sort_values('anomaly_score')

# Data profiling
def data_profile(df):
    profile = pd.DataFrame({
        'dtype': df.dtypes,
        'unique': df.nunique(),
        'zeros': (df == 0).sum(),
        'negative': (df.select_dtypes(include=[np.number]) < 0).sum()
    })
    return profile

# Streamlit app
def main():
    st.title("🔍 Advanced AI Data Quality Monitoring Dashboard")
    st.markdown("""
    *Upload your dataset* to perform comprehensive data quality analysis including missing values detection, 
    duplicate identification, and AI-powered anomaly detection.
    """)

    # File upload with drag and drop
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="file_uploader"
    )

    if uploaded_file is not None:
        with st.spinner('Analyzing your data...'):
            data = load_data(uploaded_file)
            
            if data is not None:
                # Sidebar controls
                st.sidebar.header("Analysis Settings")
                st.sidebar.subheader("Data Overview")
                st.sidebar.write(f"📊 *Shape:* {data.shape[0]} rows × {data.shape[1]} columns")
                
                # Data sampling option
                sample_size = st.sidebar.slider(
                    "Sample size for preview",
                    min_value=5,
                    max_value=min(100, len(data)),
                    value=10
                )
                
                # Main tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Data Overview", 
                    "❓ Missing Values", 
                    "♻ Duplicates", 
                    "🔍 Anomaly Detection"
                ])

                with tab1:
                    st.subheader("Data Preview")
                    st.dataframe(data.head(sample_size))

                    st.subheader("Data Profile")
                    profile = data_profile(data)
                    st.dataframe(profile.style.background_gradient(cmap='Blues'))

                    # Column type distribution
                    st.subheader("Column Types Distribution")
                    dtype_counts = data.dtypes.value_counts()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    dtype_counts.plot(kind='bar', ax=ax, color=sns.color_palette("pastel"))
                    plt.xticks(rotation=45)
                    plt.title("Distribution of Data Types")
                    plt.tight_layout()
                    st.pyplot(fig)

                with tab2:
                    st.subheader("Missing Values Analysis")
                    null_stats = check_null_values(data)
                    
                    # Metrics cards
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Missing Values</div>
                                <div class="metric-value">{null_stats['Null Count'].sum()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Max Missing %</div>
                                <div class="metric-value">{null_stats['Null Percentage'].max():.2f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    st.dataframe(null_stats.style.background_gradient(
                        cmap='Reds', subset=['Null Count', 'Null Percentage']
                    ))

                    # Visualization
                    st.subheader("Missing Values Heatmap")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(data.isnull(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
                    plt.title("Missing Values Pattern", pad=20)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)

                with tab3:
                    st.subheader("Duplicate Analysis")
                    dup_stats = check_duplicates(data)
                    
                    # Metrics cards
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Duplicate Rows</div>
                                <div class="metric-value">{dup_stats['count']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Percentage</div>
                                <div class="metric-value">{dup_stats['percentage']:.2f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    if dup_stats['count'] > 0:
                        st.warning(f"⚠ Found {dup_stats['count']} duplicate rows ({dup_stats['percentage']:.2f}% of data)")
                        if st.checkbox("Show duplicate rows"):
                            st.dataframe(data.iloc[dup_stats['indices']])
                    else:
                        st.success("✅ No duplicate rows found")

                # Updated Anomaly Detection Section (replace the existing tab4 content)
                with tab4:
                    st.subheader("Anomaly Detection")
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        st.warning("No numeric columns found for anomaly detection")
                    else:
                        contamination = st.slider(
                            "Anomaly contamination fraction",
                            min_value=0.01,
                            max_value=0.5,
                            value=0.05,
                            step=0.01,
                            help="Expected proportion of anomalies in the data"
                        )
                        
                        selected_cols = st.multiselect(
                            "Select columns for anomaly detection",
                            numeric_cols,
                            default=numeric_cols  # Default to all numeric columns
                        )
                        
                        if selected_cols:
                            with st.spinner('Detecting anomalies...'):
                                anomalies = detect_anomalies(data, selected_cols, contamination)
                                
                                st.metric(
                                    "Anomalies Detected",
                                    f"{len(anomalies)} ({len(anomalies)/len(data)*100:.2f}%)"
                                )
                                
                                if not anomalies.empty:
                                    st.dataframe(anomalies.sort_values('anomaly_score'))
                                    
                                    # Download button for anomalies
                                    csv = anomalies.to_csv(index=False)
                                    st.download_button(
                                        label="Download Anomalies as excel file",
                                        data=csv,
                                        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime='text/csv'
                                    )
                                    
                                    # Visualization with scrollable boxplot
                                    st.subheader("Anomaly Distribution")
                                    
                                    # Create tabs for different visualizations
                                    viz_tab1, viz_tab2 = st.tabs(["Anomaly Scores", "Feature Distribution"])
                                    
                                    with viz_tab1:
                                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                                        sns.histplot(
                                            data=data,
                                            x='anomaly_score',
                                            bins=30,
                                            kde=True,
                                            ax=ax1
                                        )
                                        ax1.axvline(x=data['anomaly_score'].quantile(contamination), color='red', linestyle='--')
                                        ax1.set_title("Distribution of Anomaly Scores")
                                        st.pyplot(fig1)
                                    
                                    with viz_tab2:
                                        st.markdown("*Feature Distribution with Potential Outliers*")
                                        
                                        # Calculate how many rows of plots we need (3 per row)
                                        n_cols = 3
                                        n_features = len(selected_cols)
                                        n_rows = (n_features + n_cols - 1) // n_cols
                                        
                                        # Create a grid of boxplots
                                        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
                                        
                                        if n_rows == 1:
                                            axes = axes.reshape(1, -1)
                                        
                                        for idx, col in enumerate(selected_cols):
                                            row = idx // n_cols
                                            col_pos = idx % n_cols
                                            
                                            # Create boxplot for each feature
                                            sns.boxplot(
                                                data=data,
                                                y=col,
                                                ax=axes[row, col_pos],
                                                color='skyblue'
                                            )
                                            
                                            # Highlight anomalies if they exist
                                            if not anomalies.empty:
                                                y_vals = anomalies[col].dropna()
                                                if not y_vals.empty:
                                                    x_vals = [0] * len(y_vals)
                                                    axes[row, col_pos].scatter(
                                                        x_vals,
                                                        y_vals,
                                                        color='red',
                                                        alpha=0.5,
                                                        label='Anomaly'
                                                    )
                                            
                                            axes[row, col_pos].set_title(col)
                                            axes[row, col_pos].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                                        
                                        # Hide any empty subplots
                                        for idx in range(n_features, n_rows * n_cols):
                                            row = idx // n_cols
                                            col_pos = idx % n_cols
                                            fig2.delaxes(axes[row, col_pos])
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig2)
                                        
                                        # Add description
                                        st.markdown("""
                                        *Interpretation:*
                                        - The boxplots show the distribution of each selected feature
                                        - Red dots represent values identified as anomalies
                                        - The box represents the interquartile range (IQR)
                                        - Whiskers extend to 1.5*IQR from the box
                                        """)
                                else:
                                    st.success("✅ No anomalies detected with current settings")

                # Data download section
                st.sidebar.markdown("---")
                def generate_report(data, null_stats, dup_stats, anomalies=None):
                    """Generate a comprehensive data quality report"""
                    report = f"""
                    ====================================
                    DATA QUALITY REPORT
                    ====================================
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Dataset Shape: {data.shape[0]} rows × {data.shape[1]} columns
                    
                    ----------------------------
                    MISSING VALUES SUMMARY
                    ----------------------------
                    Total Missing Values: {null_stats['Null Count'].sum()}
                    Columns with Missing Values: {(null_stats['Null Count'] > 0).sum()}
                    
                    Top 5 Columns with Most Missing Values:
                    {null_stats.head(5).to_string()}
                    
                    ----------------------------
                    DUPLICATE ROWS SUMMARY
                    ----------------------------
                    Total Duplicate Rows: {dup_stats['count']}
                    Percentage of Duplicates: {dup_stats['percentage']:.2f}%
                    
                    First 5 Duplicate Row Indices: {dup_stats['indices'][:5] if dup_stats['count'] > 0 else 'None'}
                    """
                    
                    if anomalies is not None and not anomalies.empty:
                        report += f"""
                        ----------------------------
                        ANOMALY DETECTION SUMMARY
                        ----------------------------
                        Total Anomalies Detected: {len(anomalies)}
                        Percentage of Anomalies: {(len(anomalies)/len(data))*100:.2f}%
                        
                        Columns Used for Detection: {', '.join(anomalies.columns.drop('anomaly_score'))}
                        """
                    
                    return report

                # In your Streamlit app (replace the report generation section):
                if st.sidebar.button("Generate Data Quality Report"):
                    with st.spinner('Generating report...'):
                        # Ensure we're working with string representations of dtypes
                        null_stats_str = null_stats.astype(str)
                        dup_stats_str = {
                            'count': str(dup_stats['count']),
                            'percentage': f"{dup_stats['percentage']:.2f}%",
                            'indices': str(dup_stats['indices'][:10])  # Only include first 10 indices
                        }
                        
                        # Generate the report text
                        report_text = generate_report(data, null_stats, dup_stats, anomalies if 'anomalies' in locals() else None)
                        
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmpfile:
                            tmpfile.write(report_text)
                            tmp_path = tmpfile.name
                        
                        # Provide download button
                        with open(tmp_path, 'rb') as f:
                            st.sidebar.download_button(
                                label="Download Full Report",
                                data=f,
                                file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime='text/plain'
                            )
                        
                        # Clean up
                        os.unlink(tmp_path)
                        st.sidebar.success("Report generated successfully!")
if __name__ == "__main__":
    main()
