import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
import shap
import joblib

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

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", 
                       ["Data Overview", "EDA", "Data Preprocessing", "Model Comparison", "Advanced Models", "Prediction", "Model Interpretation"])

st.sidebar.header("About")
st.sidebar.markdown("""
This app analyzes customer churn data from a telecom company.
The dataset includes customer demographics, account information, and churn indicators.
""")

# Load data function
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Preprocess data function
@st.cache_data
def preprocess_data(df):
    # Make a copy
    df_processed = df.copy()
    
    # Set CustomerID as index
    df_processed.set_index('CustomerID', inplace=True)
    
    # Handle missing values in Churn Reason
    if 'Churn Reason' in df_processed.columns:
        df_processed.drop('Churn Reason', axis=1, inplace=True)
    
    # Drop unnecessary columns
    cols_to_drop = ['Count', 'Country', 'State', 'Lat Long', 'Zip Code', 'Churn Label']
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed.drop(col, axis=1, inplace=True)
    
    # Handle Total Charges
    if 'Total Charges' in df_processed.columns:
        # Convert to numeric, forcing errors to NaN
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
        # Fill missing values with Monthly Charges * Tenure Months
        mask = df_processed['Total Charges'].isna()
        df_processed.loc[mask, 'Total Charges'] = (
            df_processed.loc[mask, 'Monthly Charges'] * df_processed.loc[mask, 'Tenure Months']
        )
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed

# Train basic models function
@st.cache_resource
def train_basic_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
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

# Train advanced models with hyperparameter tuning
@st.cache_resource
def train_advanced_models(X_train, X_test, y_train, y_test):
    # Define models with their parameter grids for tuning
    models = {
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(random_state=42, silent=True),
            'params': {
                'iterations': [100, 200],
                'depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        },
        'LGBM': {
            'model': LGBMClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6],
                'num_leaves': [31, 50],
                'early_stopping_rounds': [10]
            }
        },
        'Bagging': {
            'model': BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.8, 1.0],
                'max_features': [0.8, 1.0]
            }
        }
    }
    
    results = {}
    
    for name, model_info in models.items():
        st.write(f"Tuning {name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model_info['model'], 
            model_info['params'], 
            cv=3, 
            scoring='f1', 
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else [0] * len(y_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if hasattr(best_model, "predict_proba") else 0
        
        # Store results
        results[name] = {
            "model": best_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "predictions": y_pred,
            "probabilities": y_prob,
            "best_params": grid_search.best_params_
        }
    
    # Train stacking classifier
    st.write("Training Stacking Classifier...")
    
    # Base models
    base_models = [
        ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, 
                              use_label_encoder=False, eval_metric='logloss')),
        ('cat', CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=42, silent=True)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, num_leaves=50, random_state=42))
    ]
    
    # Meta-model
    meta_model = LogisticRegression(random_state=42)
    
    # Stacking classifier
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        n_jobs=-1
    )
    
    # Train the stacking model
    stacking_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stacking_model.predict(X_test)
    y_prob = stacking_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results["Stacking"] = {
        "model": stacking_model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "predictions": y_pred,
        "probabilities": y_prob,
        "best_params": "Stacking of base models"
    }
    
    return results

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your Telco customer churn CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Display basic info
    if page == "Data Overview":
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            churn_rate = (df['Churn Value'].sum() / df.shape[0]) * 100 if 'Churn Value' in df.columns else 0
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Show dataset
        st.subheader("Raw Data")
        st.dataframe(df.head())
        
        # Data info
        st.subheader("Data Information")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Column descriptions
        st.subheader("Column Descriptions")
        col_descriptions = {
            'CustomerID': 'Unique identifier for each customer',
            'CLTV': 'Customer Lifetime Value',
            'Churn Label': 'Yes = customer left, No = customer remained',
            'Churn Value': '1 = customer left, 0 = customer remained (Target)',
            'Contract': 'Customer\'s current contract type',
            'Dependents': 'Indicates if customer lives with any dependents',
            'Device Protection': 'Indicates if customer subscribes to device protection',
            'Gender': 'Customer\'s gender',
            'Internet Service': 'Indicates if customer subscribes to Internet service',
            'Monthly Charges': 'The amount the customer is billed monthly',
            'Total Charges': 'Cumulative amount billed to the customer',
            'Tenure Months': 'Total months the customer has been with the company'
        }
        
        for col, desc in col_descriptions.items():
            if col in df.columns:
                st.write(f"**{col}**: {desc}")
    
    elif page == "EDA":
        st.subheader("Exploratory Data Analysis")
        
        # Churn distribution
        st.subheader("Churn Distribution")
        if 'Churn Label' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            df['Churn Label'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)
        else:
            st.warning("Churn Label column not found in the dataset")
        
        # CLTV distribution
        if 'CLTV' in df.columns:
            st.subheader("Customer Lifetime Value (CLTV) Distribution")
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            sns.histplot(df, x='CLTV', kde=True, ax=axes[0])
            axes[0].set_xlabel(None)
            sns.boxenplot(df, x='CLTV', ax=axes[1])
            plt.tight_layout()
            st.pyplot(fig)
        
        # # Churn Score distribution
        # if 'Churn Score' in df.columns:
        #     st.subheader("Churn Score Distribution")
        #     fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        #     sns.histplot(df, x='Churn Score', kde=True, ax=axes[0])
        #     axes[0].set_xlabel(None)
        #     sns.boxenplot(df, x='Churn Score', ax=axes[1])
        #     plt.tight_layout()
        #     st.pyplot(fig)
        
        # Monthly Charges vs Churn
        if 'Monthly Charges' in df.columns and 'Churn Value' in df.columns:
            st.subheader("Monthly Charges by Churn Status")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='Churn Value', y='Monthly Charges', ax=ax)
            ax.set_xticklabels(['Not Churn', 'Churn'])
            st.pyplot(fig)
        
        # Tenure vs Churn
        if 'Tenure Months' in df.columns and 'Churn Value' in df.columns:
            st.subheader("Tenure by Churn Status")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='Churn Value', y='Tenure Months', ax=ax)
            ax.set_xticklabels(['Not Churn', 'Churn'])
            st.pyplot(fig)
        
        # Contract type vs Churn
        if 'Contract' in df.columns and 'Churn Value' in df.columns:
            st.subheader("Churn by Contract Type")
            contract_churn = pd.crosstab(df['Contract'], df['Churn Value'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            contract_churn.plot(kind='bar', ax=ax)
            ax.set_ylabel('Churn Percentage')
            ax.set_xlabel('Contract Type')
            ax.legend(['Not Churn', 'Churn'])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
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
    
    elif page == "Data Preprocessing":
        st.subheader("Data Preprocessing")
        
        with st.spinner("Preprocessing data..."):
            df_processed = preprocess_data(df)
        
        st.success("Data preprocessing completed!")
        
        # Show processed data
        st.subheader("Processed Data")
        st.dataframe(df_processed.head())
        
        # Show data info
        st.subheader("Processed Data Information")
        buffer = StringIO()
        df_processed.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Show missing values
        st.subheader("Missing Values After Preprocessing")
        missing_data = df_processed.isnull().sum()
        if missing_data.sum() == 0:
            st.success("No missing values found!")
        else:
            st.warning(f"Missing values found: {missing_data[missing_data > 0]}")
        
        # Data types
        st.subheader("Data Types After Preprocessing")
        st.write(df_processed.dtypes)
        
        # Basic statistics
        st.subheader("Descriptive Statistics")
        st.write(df_processed.describe())
    
    elif page == "Model Comparison":
        st.subheader("Model Comparison")
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df_processed = preprocess_data(df)
        
        # Prepare features and target
        if 'Churn Value' in df_processed.columns:
            X = df_processed.drop('Churn Value', axis=1)
            y = df_processed['Churn Value']
        else:
            st.error("Churn Value column not found in the processed data")
            st.stop()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        with st.spinner("Training models..."):
            results = train_basic_models(X_train_scaled, X_test_scaled, y_train_smote, y_test)
        
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
            st.plotly_chart(fc_cm, use_container_width=True)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, model_results['predictions'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Feature importance for tree-based models
        if selected_model in ["Random Forest", "Gradient Boosting", "AdaBoost", "XGBoost", "LightGBM", "Decision Tree"]:
            st.subheader("Feature Importance")
            
            try:
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
            except Exception as e:
                st.warning(f"Could not display feature importance: {e}")
    
    elif page == "Advanced Models":
        st.subheader("Advanced Models with Hyperparameter Tuning")
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df_processed = preprocess_data(df)
        
        # Prepare features and target
        if 'Churn Value' in df_processed.columns:
            X = df_processed.drop('Churn Value', axis=1)
            y = df_processed['Churn Value']
        else:
            st.error("Churn Value column not found in the processed data")
            st.stop()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
        
        # Train advanced models
        with st.spinner("Training advanced models with hyperparameter tuning (this may take a while)..."):
            results = train_advanced_models(X_train_scaled, X_test_scaled, y_train_smote, y_test)
        
        # Display model comparison
        st.subheader("Advanced Model Performance Comparison")
        
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
            title='Advanced Model Performance Metrics',
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
            
            # Show best parameters for tuned models
            if 'best_params' in model_results:
                st.subheader("Best Hyperparameters")
                st.write(model_results['best_params'])
            
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
            
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, model_results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=700, height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Precision-Recall Curve
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, model_results['probabilities'])
            
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve'))
            fig_pr.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=700, height=500
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # Store results for SHAP analysis
        if 'advanced_results' not in st.session_state:
            st.session_state.advanced_results = results
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.X_test = X_test
            st.session_state.feature_names = X.columns
    
    elif page == "Prediction":
        st.subheader("Churn Prediction")
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df_processed = preprocess_data(df)
        
        # Prepare features and target
        if 'Churn Value' in df_processed.columns:
            X = df_processed.drop('Churn Value', axis=1)
            y = df_processed['Churn Value']
        else:
            st.error("Churn Value column not found in the processed data")
            st.stop()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
        
        # Train a model for prediction (using the best model from advanced models if available)
        if 'advanced_results' in st.session_state:
            results = st.session_state.advanced_results
            # Get the best model based on F1 score
            best_model_name = max(results, key=lambda x: results[x]['f1'])
            model = results[best_model_name]['model']
            st.info(f"Using the best model: {best_model_name}")
        else:
            # Train a Random Forest model as fallback
            with st.spinner("Training model..."):
                model = RandomForestClassifier(random_state=42, n_estimators=100)
                model.fit(X_train_scaled, y_train_smote)
            st.info("Using Random Forest model (advanced models not available)")
        
        # Get feature importances for the top features
        try:
            feature_importance = model.feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            top_features = importance_df.head(10)['Feature'].tolist()
        except:
            # If feature importances are not available, use a predefined list of important features
            top_features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Contract', 'Internet Service', 
                           'Online Security', 'Tech Support', 'Payment Method', 'Paperless Billing', 'Senior Citizen']
        
        # Prediction interface
        st.subheader("Make Predictions on New Data")
        st.markdown("Adjust the feature values below to predict whether a customer will stay or leave.")
        
        with st.expander("Input customer details for prediction", expanded=True):
            # Create input fields for the top features
            col1, col2, col3 = st.columns(3)
            
            input_values = {}
            
            with col1:
                if 'Tenure Months' in top_features:
                    input_values['Tenure Months'] = st.slider("Tenure (months)", 0, 100, 12)
                if 'Monthly Charges' in top_features:
                    input_values['Monthly Charges'] = st.slider("Monthly Charges", 0, 200, 50)
                if 'Total Charges' in top_features:
                    input_values['Total Charges'] = st.slider("Total Charges", 0, 10000, 1000)
                if 'CLTV' in top_features:
                    input_values['CLTV'] = st.slider("CLTV", 0, 10000, 4000)
            
            with col2:
                if 'Contract' in top_features:
                    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                    input_values['Contract'] = 0 if contract == "Month-to-month" else 1 if contract == "One year" else 2
                if 'Internet Service' in top_features:
                    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No", "Cable"])
                    input_values['Internet Service'] = 0 if internet_service == "DSL" else 1 if internet_service == "Fiber optic" else 2 if internet_service == "No" else 3
                if 'Payment Method' in top_features:
                    payment_method = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
                    ])
                    input_values['Payment Method'] = 0 if payment_method == "Electronic check" else 1 if payment_method == "Mailed check" else 2 if payment_method == "Bank transfer" else 3
                if 'Paperless Billing' in top_features:
                    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                    input_values['Paperless Billing'] = 1 if paperless_billing == "Yes" else 0
            
            with col3:
                if 'Senior Citizen' in top_features:
                    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                    input_values['Senior Citizen'] = 1 if senior_citizen == "Yes" else 0
                if 'Partner' in top_features:
                    partner = st.selectbox("Partner", ["No", "Yes"])
                    input_values['Partner'] = 1 if partner == "Yes" else 0
                if 'Dependents' in top_features:
                    dependents = st.selectbox("Dependents", ["No", "Yes"])
                    input_values['Dependents'] = 1 if dependents == "Yes" else 0
                if 'Gender' in top_features:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    input_values['Gender'] = 0 if gender == "Male" else 1
            
            # Add input fields for other top features that might not be covered above
            other_features = [f for f in top_features if f not in input_values]
            if other_features:
                st.subheader("Other Important Features")
                for feature in other_features:
                    if df_processed[feature].dtype == 'object' or df_processed[feature].nunique() < 10:
                        # Categorical feature
                        options = df_processed[feature].unique()
                        input_values[feature] = st.selectbox(feature, options)
                    else:
                        # Numerical feature
                        min_val = float(df_processed[feature].min())
                        max_val = float(df_processed[feature].max())
                        default_val = float(df_processed[feature].median())
                        input_values[feature] = st.slider(feature, min_val, max_val, default_val)
            
            # Create a sample for prediction with default values for all features
            sample_data = {}
            for feature in X.columns:
                if feature in input_values:
                    sample_data[feature] = input_values[feature]
                else:
                    # Set to median for numerical features, mode for categorical
                    if df_processed[feature].dtype == 'object' or df_processed[feature].nunique() < 10:
                        sample_data[feature] = df_processed[feature].mode()[0]
                    else:
                        sample_data[feature] = df_processed[feature].median()
            
            # Convert to dataframe
            sample_df = pd.DataFrame([sample_data])[X.columns]
            
            # Scale the sample
            sample_scaled = scaler.transform(sample_df)
            
            if st.button("Predict Churn", type="primary"):
                # Get prediction
                prediction = model.predict(sample_scaled)[0]
                probability = model.predict_proba(sample_scaled)[0][1]
                
                if prediction == 1:
                    st.error(f"**Prediction: CHURN**")
                    st.metric("Churn Probability", f"{probability:.2%}")
                    st.write("**Recommended actions:**")
                    st.write("- Offer retention discount or special promotion")
                    st.write("- Provide personalized service review")
                    st.write("- Consider contract upgrade options")
                    st.write("- Assign dedicated account manager")
                else:
                    st.success(f"**Prediction: STAY**")
                    st.metric("Churn Probability", f"{probability:.2%}")
                    st.write("**Recommended actions:**")
                    st.write("- Continue with current service plan")
                    st.write("- Consider upselling premium features")
                    st.write("- Maintain regular engagement")
                    st.write("- Monitor for any changes in usage patterns")
                
                # Show probability breakdown
                fig_proba = go.Figure(go.Bar(
                    x=['Stay', 'Churn'],
                    y=[1-probability, probability],
                    marker_color=['green', 'red']
                ))
                
                fig_proba.update_layout(
                    title='Churn Probability Breakdown',
                    yaxis_title='Probability',
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig_proba, use_container_width=True)
                
                # Show feature importance for this prediction
                try:
                    st.subheader("Feature Impact on This Prediction")
                    
                    # Use SHAP to explain the prediction
                    explainer = shap.Explainer(model, X_train_scaled)
                    shap_values = explainer(sample_scaled)
                    
                    # Plot the force plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate detailed explanation: {e}")
    
    elif page == "Model Interpretation":
        st.subheader("Model Interpretation with SHAP")
        
        if 'advanced_results' not in st.session_state:
            st.warning("Please train advanced models on the 'Advanced Models' page first.")
        else:
            results = st.session_state.advanced_results
            X_test_scaled = st.session_state.X_test_scaled
            X_test = st.session_state.X_test
            feature_names = st.session_state.feature_names
            
            st.subheader("SHAP Analysis")
            
            # Let user select a model for SHAP analysis
            model_names = list(results.keys())
            selected_model = st.selectbox("Select a model for SHAP analysis:", model_names)
            
            if selected_model:
                model = results[selected_model]['model']
                
                # Sample data for faster computation
                sample_size = min(100, X_test_scaled.shape[0])
                X_sample = X_test_scaled[:sample_size]
                
                with st.spinner("Computing SHAP values..."):
                    try:
                        # Create SHAP explainer
                        if hasattr(model, 'predict_proba'):
                            explainer = shap.Explainer(model, X_sample)
                            shap_values = explainer(X_sample)
                        else:
                            st.warning("Selected model doesn't support SHAP explanation.")
                            shap_values = None
                    except Exception as e:
                        st.error(f"Error computing SHAP values: {e}")
                        shap_values = None
                
                if shap_values is not None:
                    # SHAP summary plot
                    st.subheader("SHAP Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, max_display=10, show=False)
                    st.pyplot(fig)
                    
                    # SHAP feature importance
                    st.subheader("Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.bar(shap_values, max_display=10, show=False)
                    st.pyplot(fig)
                    
                    # Let user select a feature for dependence plot
                    st.subheader("SHAP Dependence Plot")
                    selected_feature = st.selectbox("Select a feature to see its dependence:", feature_names, index=0)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.dependence_plot(selected_feature, shap_values.values, X_sample, feature_names=feature_names, show=False)
                    st.pyplot(fig)
                    
                    # Individual prediction explanations
                    st.subheader("Individual Prediction Explanations")
                    sample_idx = st.slider("Select a sample to explain:", 0, sample_size-1, 0)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
                    st.pyplot(fig)
    
    # Download button for processed data
    if page != "Data Overview":
        csv = df_processed.to_csv().encode('utf-8')
        st.sidebar.download_button(
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
        'Total Charges': [108.15, 151.65],
        'Tenure Months': [5, 10],
        'Contract': ['Month-to-month', 'One year'],
        'Internet Service': ['DSL', 'Fiber optic']
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
    
    st.subheader("Column Reference")
    col_ref = """
    - **CustomerID**: Unique identifier for each customer
    - **CLTV**: Customer Lifetime Value
    - **Churn Label**: Yes = customer left, No = customer remained
    - **Churn Value**: 1 = customer left, 0 = customer remained (Target)
    - **Contract**: Customer's current contract type
    - **Dependents**: Indicates if customer lives with any dependents
    - **Device Protection**: Indicates if customer subscribes to device protection
    - **Gender**: Customer's gender
    - **Internet Service**: Indicates if customer subscribes to Internet service
    - **Monthly Charges**: The amount the customer is billed monthly
    - **Total Charges**: Cumulative amount billed to the customer
    - **Tenure Months**: Total months the customer has been with the company
    """
    st.markdown(col_ref)

