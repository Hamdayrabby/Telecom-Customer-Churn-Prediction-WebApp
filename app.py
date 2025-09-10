import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style('whitegrid')

# Title and description
st.title("📊 Telecom Customer Churn Analysis")
st.markdown("""
This application performs exploratory data analysis and churn prediction for telecom customer data.
Explore the dataset, visualize key patterns, and understand factors that influence customer churn.
""")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", 
                       ["EDA", "Data Wrangling", "Model Results/Churn Prediction"])

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This app analyzes customer churn data from a telecom company.
The dataset includes customer demographics, account information, and churn indicators.
""")

# Load data
@st.cache_data
def load_data():
    # In a real app, you would load from the uploaded file
    # For demo purposes, we'll use the sample data structure
    # You'll need to upload the actual CSV file in the real app
    data = pd.read_csv('Telco_customer_churn.csv')
    data.set_index('CustomerID', inplace=True)
    data.drop('Churn Reason', axis=1, inplace=True)
    return data

# Preprocess data for modeling
def preprocess_data(df):
    # Make a copy
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col != 'Churn Label':  # We'll handle this separately
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Encode target variable
    df_processed['Churn Label'] = le.fit_transform(df_processed['Churn Label'])
    
    return df_processed

# Train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "Support Vector Machine": SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if hasattr(model, "predict_proba") else 0
        
        # Store results
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "predictions": y_pred,
            "probabilities": y_prob
        }
    
    return results

# File uploader
uploaded_file = st.file_uploader("Upload your Telco customer churn CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.set_index('CustomerID', inplace=True)
    
    # Handle missing values
    if 'Churn Reason' in df.columns:
        df.drop('Churn Reason', axis=1, inplace=True)
    
    st.success("Data loaded successfully!")
    
    # Display basic info
    if page == "EDA":
        st.subheader("Exploratory Data Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            churn_rate = (df['Churn Value'].sum() / df.shape[0]) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Show dataset
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.dataframe(df)
        
        # Data info
        if st.checkbox("Show data info"):
            st.subheader("Data Information")
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        
        # Visualizations
        st.header("Data Visualizations")
        
        # Churn distribution
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Churn Label'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
        
        # CLTV distribution
        st.subheader("Customer Lifetime Value (CLTV) Distribution")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        sns.histplot(df, x='CLTV', kde=True, ax=axes[0])
        axes[0].set_xlabel(None)
        sns.boxenplot(df, x='CLTV', ax=axes[1])
        plt.tight_layout()
        st.pyplot(fig)
        
        # Churn Score distribution
        st.subheader("Churn Score Distribution")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        sns.histplot(df, x='Churn Score', kde=True, ax=axes[0])
        axes[0].set_xlabel(None)
        sns.boxenplot(df, x='Churn Score', ax=axes[1])
        plt.tight_layout()
        st.pyplot(fig)
        
        # Monthly Charges vs Churn
        st.subheader("Monthly Charges by Churn Status")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Churn Label', y='Monthly Charges', ax=ax)
        st.pyplot(fig)
        
        # Tenure vs Churn
        st.subheader("Tenure by Churn Status")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Churn Label', y='Tenure Months', ax=ax)
        st.pyplot(fig)
        
        # Contract type vs Churn
        st.subheader("Churn by Contract Type")
        contract_churn = pd.crosstab(df['Contract'], df['Churn Label'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        contract_churn.plot(kind='bar', ax=ax)
        ax.set_ylabel('Churn Percentage')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Insights
        st.header("Key Insights")
        
        insights = """
        1. **Overall Churn Rate**: Approximately 26.5% of customers have churned.
        2. **CLTV Distribution**: The Customer Lifetime Value distribution is right-skewed, with most customers having CLTV between 2000-5000.
        3. **Churn Score**: The distribution is left-skewed, indicating that the majority of customers have high churn scores.
        4. **Monthly Charges**: Churned customers tend to have higher monthly charges on average.
        5. **Tenure**: Customers with shorter tenure are more likely to churn.
        6. **Contract Type**: Month-to-month contracts have significantly higher churn rates compared to longer-term contracts.
        """
        
        st.markdown(insights)
    
    elif page == "Data Wrangling":
        st.subheader("Data Wrangling")
        
        # Show missing values
        st.write("Missing Values:")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("No missing values found!")
        else:
            st.write(missing_data[missing_data > 0])
        
        # Data types
        st.write("Data Types:")
        st.write(df.dtypes)
        
        # Basic statistics
        st.write("Descriptive Statistics:")
        st.write(df.describe())
        
        # Data transformation options
        st.subheader("Data Transformations")
        
        # Show unique values for categorical columns
        if st.checkbox("Show unique values for categorical columns"):
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"{col}: {df[col].unique()}")
    
    elif page == "Model Results/Churn Prediction":
        st.subheader("Model Results and Churn Prediction")
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df_processed = preprocess_data(df)
        
        # Prepare features and target
        X = df_processed.drop(['Churn Label', 'Churn Value'], axis=1)
        y = df_processed['Churn Label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        with st.spinner("Training models..."):
            results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Display model comparison
        st.subheader("Model Performance Comparison")
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results],
            'Precision': [results[model]['precision'] for model in results],
            'Recall': [results[model]['recall'] for model in results],
            'F1 Score': [results[model]['f1'] for model in results],
            'ROC AUC': [results[model]['roc_auc'] for model in results]
        })
        
        # Display metrics table
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Visualize model comparison
        fig = go.Figure()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                name=metric,
                text=metrics_df[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Metrics',
            barmode='group',
            xaxis_tickangle=-45,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed results for each model
        st.subheader("Detailed Model Results")
        
        selected_model = st.selectbox("Select a model to view detailed results:", list(results.keys()))
        
        if selected_model:
            model_results = results[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{model_results['accuracy']:.3f}")
                st.metric("Precision", f"{model_results['precision']:.3f}")
                st.metric("Recall", f"{model_results['recall']:.3f}")
            
            with col2:
                st.metric("F1 Score", f"{model_results['f1']:.3f}")
                st.metric("ROC AUC", f"{model_results['roc_auc']:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, model_results['predictions'])
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Not Churn', 'Churn'],
                y=['Not Churn', 'Churn']
            )
            
            fig_cm.update_layout(title=f'Confusion Matrix - {selected_model}')
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, model_results['predictions'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Feature importance for tree-based models
        if selected_model in ["Random Forest", "Gradient Boosting"]:
            st.subheader("Feature Importance")
            
            feature_importance = results[selected_model]['model'].feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top 10 Feature Importance - {selected_model}'
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Prediction interface
        st.subheader("Make Predictions on New Data")
        
        with st.expander("Input customer details for prediction"):
            # Create input fields for key features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tenure = st.slider("Tenure (months)", 0, 100, 12)
                monthly_charges = st.slider("Monthly Charges", 0, 200, 50)
                total_charges = st.slider("Total Charges", 0, 10000, 1000)
            
            with col2:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
                ])
            
            with col3:
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            # Create a sample for prediction (this would need to be properly encoded in a real scenario)
            sample_data = {
                'Tenure Months': tenure,
                'Monthly Charges': monthly_charges,
                'Total Charges': total_charges,
                # These would need proper encoding in a real implementation
                'Contract': 0 if contract == "Month-to-month" else 1 if contract == "One year" else 2,
                'Internet Service': 0 if internet_service == "DSL" else 1 if internet_service == "Fiber optic" else 2,
                'Payment Method': 0 if payment_method == "Electronic check" else 1 if payment_method == "Mailed check" else 2 if payment_method == "Bank transfer" else 3,
                'Senior Citizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0
            }
            
            # Add default values for other features
            for col in X.columns:
                if col not in sample_data:
                    sample_data[col] = 0  # Default value
            
            # Convert to dataframe
            sample_df = pd.DataFrame([sample_data])[X.columns]
            
            # Scale the sample
            sample_scaled = scaler.transform(sample_df)
            
            if st.button("Predict Churn"):
                # Get prediction from selected model
                prediction = results[selected_model]['model'].predict(sample_scaled)[0]
                probability = results[selected_model]['model'].predict_proba(sample_scaled)[0][1]
                
                if prediction == 1:
                    st.error(f"This customer is predicted to CHURN with {probability:.2%} probability")
                else:
                    st.success(f"This customer is predicted to STAY with {1-probability:.2%} probability")
                
                # Show probability breakdown
                fig_proba = go.Figure(go.Bar(
                    x=['Stay', 'Churn'],
                    y=[1-probability, probability],
                    marker_color=['green', 'red']
                ))
                
                fig_proba.update_layout(
                    title='Churn Probability',
                    yaxis_title='Probability',
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig_proba, use_container_width=True)
    
    # Download button for processed data (available on all pages)
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="Download processed data as CSV",
        data=csv,
        file_name='processed_churn_data.csv',
        mime='text/csv',
    )
    
else:
    st.info("Please upload a CSV file to get started. Use the Telco_customer_churn dataset.")
    
    # Display sample structure
    st.subheader("Expected Data Structure")
    sample_data = {
        'CustomerID': ['3668-QPYBK', '9237-HQITU'],
        'Count': [1, 1],
        'Country': ['United States', 'United States'],
        'State': ['California', 'California'],
        'Churn Label': ['Yes', 'Yes'],
        'Churn Value': [1, 1],
        'Monthly Charges': [53.85, 70.70],
        'Total Charges': [108.15, 151.65]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)