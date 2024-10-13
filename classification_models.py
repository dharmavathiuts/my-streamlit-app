import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, confusion_matrix
import streamlit as st

def run():
    st.title("Classification Models")

    # Load the preprocessed feature set and data
    st.write("Loading dataset...")
    file_path = '/content/drive/My Drive/ass2_amazonfoodreview/Step5_Combined_Features.parquet'
    Features = pd.read_parquet(file_path)
    Data_Sample = pd.read_parquet('/content/drive/My Drive/ass2_amazonfoodreview/Step2_Combined_Data.parquet')

    # Define X (features) and y (target)
    y = Data_Sample['Helpfulness Label']
    X = Features
    
    # Perform train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=456)
    
    # Logistic Regression Model
    st.write("Training Logistic Regression Model...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    pred_val_lr = lr_model.predict(X_test)
    lr_model_results = classification_report(y_true=y_test, y_pred=pred_val_lr, output_dict=True, zero_division=0)
    lr_model_precision = precision_score(y_test, pred_val_lr)
    lr_model_recall = recall_score(y_test, pred_val_lr)
    lr_model_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

    # Decision Tree Model
    st.write("Training Decision Tree Model...")
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    pred_val_dt = dt_model.predict(X_test)
    dt_model_results = classification_report(y_true=y_test, y_pred=pred_val_dt, output_dict=True, zero_division=0)
    dt_model_precision = precision_score(y_test, pred_val_dt)
    dt_model_recall = recall_score(y_test, pred_val_dt)
    dt_model_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])

    # SVC Model
    st.write("Training SVC Model...")
    svc_model = SVC(probability=True)
    svc_model.fit(X_train, y_train)
    pred_val_svc = svc_model.predict(X_test)
    svc_model_results = classification_report(y_true=y_test, y_pred=pred_val_svc, output_dict=True, zero_division=0)
    svc_model_precision = precision_score(y_test, pred_val_svc)
    svc_model_recall = recall_score(y_test, pred_val_svc)
    svc_model_auc = roc_auc_score(y_test, svc_model.predict_proba(X_test)[:, 1])

    # XGBoost Model
    st.write("Training XGBoost Model...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    pred_val_xgb = xgb_model.predict(X_test)
    xgb_model_results = classification_report(y_true=y_test, y_pred=pred_val_xgb, output_dict=True, zero_division=0)
    xgb_model_precision = precision_score(y_test, pred_val_xgb)
    xgb_model_recall = recall_score(y_test, pred_val_xgb)
    xgb_model_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

    # Model comparison results
    st.write("Model Comparison Results:")
    model_comparison = {
        "Model": ["Logistic Regression", "Decision Tree", "SVC", "XGBoost"],
        "Accuracy": [
            lr_model_results['accuracy'] * 100,
            dt_model_results['accuracy'] * 100,
            svc_model_results['accuracy'] * 100,
            xgb_model_results['accuracy'] * 100
        ],
        "Precision": [
            lr_model_precision,
            dt_model_precision,
            svc_model_precision,
            xgb_model_precision
        ],
        "Recall": [
            lr_model_recall,
            dt_model_recall,
            svc_model_recall,
            xgb_model_recall
        ],
        "F1-score": [
            lr_model_results['weighted avg']['f1-score'],
            dt_model_results['weighted avg']['f1-score'],
            svc_model_results['weighted avg']['f1-score'],
            xgb_model_results['weighted avg']['f1-score']
        ],
        "AUC": [
            lr_model_auc,
            dt_model_auc,
            svc_model_auc,
            xgb_model_auc
        ]
    }

    # Create a DataFrame for the comparison
    comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(comparison_df)

    # Plot confusion matrix for XGBoost (example)
    st.write("Confusion Matrix for XGBoost:")
    conf_matrix_xgb = confusion_matrix(y_test, pred_val_xgb)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax,
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    ax.set_title('Confusion Matrix - XGBoost')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)
