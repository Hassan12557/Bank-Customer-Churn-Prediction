"""
Churn Prediction Modeling Pipeline
Author: Data Science Team
Description: Trains and compares Logistic Regression, Random Forest, and XGBoost models
             using time-series validation splits and custom business metrics.
"""

import os  # <<--- FIXED: Added missing import to resolve NameError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from xgboost import XGBClassifier

# Set clean aesthetic style for visualization popups inside PyCharm
sns.set_theme(style="whitegrid")
plt.rcParams['figure.max_open_warning'] = 50


def load_and_split_data():
    """
    Loads automated output vectors directly from the data pipeline,
    completely bypassing manual processing loops.
    """
    train_path = "data/processed/train_processed.csv"
    test_path = "data/processed/test_processed.csv"

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Automated data path arrays are missing. Run data_pipeline.py first!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df['ChurnStatus']
    X_train_scaled = train_df.drop(columns=['ChurnStatus'])

    y_test = test_df['ChurnStatus']
    X_test_scaled = test_df.drop(columns=['ChurnStatus'])

    print(f"✓ Production data successfully loaded via automated data pipelines.")
    return X_train_scaled.values, X_test_scaled.values, y_train, y_test, X_train_scaled.columns

def run_modeling_pipeline():
    # 1. Pipeline Input Phase
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_split_data()

    # 2. Model Blueprint Parameters Setup
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    }

    results = {}

    # 3. Training & Iterative Diagnostic Testing Engine
    for model_name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f" Executing Pipeline Block: {model_name}")
        print(f"{'=' * 70}")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Inference metrics extraction
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Operational Performance Matrices
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f} | Precision-Recall AUC: {pr_auc:.4f}")

        # 4. Business Risk Concentration Analysis (Top 20% Risk Bracket)
        test_df = pd.DataFrame({
            'true_label': y_test.values,
            'pred_proba': y_pred_proba
        }).sort_values('pred_proba', ascending=False)

        top_20_count = int(len(test_df) * 0.2)
        top_20_df = test_df.head(top_20_count)

        actual_churners_in_top20 = top_20_df['true_label'].sum()
        total_actual_churners = test_df['true_label'].sum()
        pct_captured_in_top20 = (
                    actual_churners_in_top20 / total_actual_churners * 100) if total_actual_churners > 0 else 0

        print(f"\nBusiness Metrics (Top 20% Concentrated Risk Allocation):")
        print(
            f"  - Captured Churners: {actual_churners_in_top20}/{total_actual_churners} ({pct_captured_in_top20:.1f}%)")

        # Financial Impact Optimization Equations
        retention_value = 500  # Financial reward yield value mapping per retained customer
        intervention_cost = 50  # Operational intervention campaign cost factor allocation

        if total_actual_churners > 0:
            expected_profit = (actual_churners_in_top20 * retention_value) - (top_20_count * intervention_cost)
            print(f"  - Projected Expected Campaign Profit (Top 20% Segment): ${expected_profit:,.2f}")
            print(f"  - Margin Yield Per Contact Target Vector: ${expected_profit / top_20_count:.2f}")

        # 5. Financial Boundary Threshold Calibration
        fn_cost = 500  # Cost of a False Negative (Unidentified Churner Loss)
        fp_cost = 50  # Cost of a False Positive (Wasted Retention Outreach Budget)

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_net_value = -np.inf

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()

            # Global Utility Equation calculation
            net_value = (tp * retention_value) - (fp * fp_cost) - (fn * fn_cost)

            if net_value > best_net_value:
                best_net_value = net_value
                best_threshold = threshold

        print(f"\nThreshold Calibration Optimization:")
        print(f"  - Structurally Optimal Boundary Cutoff: {best_threshold:.2f}")
        print(f"  - Net Campaign Value Generated at Threshold: ${best_net_value:,.2f}")

        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
        print(f"  - Risk Matrix Volume [TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}]")

        # 6. Standard Subplot Visualizations Generation
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0].plot(fpr, tpr, color='#4C72B0', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axes[0].set_title(f'ROC Curve: {model_name}', fontweight='bold')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')

        # Plot Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1].plot(recall, precision, color='#C44E52', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
        axes[1].set_title(f'PR Curve: {model_name}', fontweight='bold')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left')

        plt.tight_layout()
        plt.show()

    # 7. Model Assessment & Comparative Ranking Summary Matrix
    print("\n" + "=" * 70)
    print("                GLOBAL RUNTIME COMPARISON SUMMARY")
    print("=" * 70)
    comparison_df = pd.DataFrame(index=models.keys())

    for model_name in models.keys():
        prob_vec = results[model_name]['y_pred_proba']
        pred_vec = results[model_name]['y_pred']

        comparison_df.loc[model_name, 'ROC-AUC'] = roc_auc_score(y_test, prob_vec)
        comparison_df.loc[model_name, 'PR-AUC'] = average_precision_score(y_test, prob_vec)
        comparison_df.loc[model_name, 'Accuracy'] = np.mean(pred_vec == y_test)

    print(comparison_df.round(4))
    print(f"\n🥇 Operational Standard Recommendation via ROC-AUC: {comparison_df['ROC-AUC'].idxmax()}")

    # 8. Structural Feature Importance Profiling Engine
    print("\n" + "=" * 70)
    print("              TREE-BASED FEATURE IMPORTANCE METRICS")
    print("=" * 70)
    for model_name in ['Random Forest', 'XGBoost']:
        if model_name in results:
            trained_model = results[model_name]['model']

            if hasattr(trained_model, 'feature_importances_'):
                importance_matrix = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': trained_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                print(f"\nTop Features Analysis ({model_name}):")
                print(importance_matrix.head(10).to_string(index=False))

                # Horizontal Structural Plotting Sequence
                plt.figure(figsize=(10, 5))
                top_visual_features = importance_matrix.head(12)
                sns.barplot(x='Importance', y='Feature', data=top_visual_features, palette="viridis", edgecolor='black')
                plt.title(f'Feature Importance Weights - {model_name}', fontweight='bold', pad=12)
                plt.xlabel('Relative Information Gain Metric Scale')
                plt.ylabel('Evaluated Features Matrix Column')
                plt.tight_layout()
                plt.show()


if __name__ == '__main__':
    # FIXED: Cleared double-parentheses syntax error to ensure clean script execution
    run_modeling_pipeline()