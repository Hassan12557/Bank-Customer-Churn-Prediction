import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global visualization aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 13})

# Path navigation from notebooks/ to data/Processed/
DATA_PATH = os.path.join("..", "data", "Processed", "bank_data_preprocessed.csv")

if os.path.exists(DATA_PATH):
    df_cleaned = pd.read_csv(DATA_PATH)
    print(f"📊 Dataset successfully loaded: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
else:
    raise FileNotFoundError(f"Missing preprocessed dataset at: {os.path.abspath(DATA_PATH)}")


def calculate_descriptive_stats(df, column_name):
    """Generates an organized dataframe of location and variability metrics."""
    return pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Mode', 'Range', 'Variance', 'Standard Deviation', 'IQR'],
        'Value': [
            df[column_name].mean(),
            df[column_name].median(),
            df[column_name].mode()[0],
            df[column_name].max() - df[column_name].min(),
            df[column_name].var(),
            df[column_name].std(),
            df[column_name].quantile(0.75) - df[column_name].quantile(0.25)
        ]
    }).set_index('Metric')


def plot_distribution_trio(df, column_name, box_color='#4C72B0', hist_color='#55A868', kde_color='#C44E52'):
    """Renders side-by-side boxplot, histogram, and density configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Boxplot
    sns.boxplot(y=df[column_name], ax=axes[0], color=box_color, width=0.4)
    axes[0].set_title(f'Boxplot of {column_name}', fontweight='bold')
    axes[0].set_ylabel(column_name)

    # 2. Histogram
    axes[1].hist(df[column_name], bins=20, edgecolor='black', color=hist_color, alpha=0.7)
    axes[1].set_title(f'Histogram of {column_name}', fontweight='bold')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Frequency')

    # 3. Kernel Density Estimate
    sns.kdeplot(df[column_name], ax=axes[2], fill=True, color=kde_color, linewidth=2)
    axes[2].set_title(f'Density Plot of {column_name}', fontweight='bold')
    axes[2].set_xlabel(column_name)
    axes[2].set_ylabel('Density')

    plt.tight_layout()
    plt.show()


# --- Run Analysis for Age ---
print("\n=== AGE METRICS ===")
print(calculate_descriptive_stats(df_cleaned, 'Age').round(2))
plot_distribution_trio(df_cleaned, 'Age')

# --- Run Analysis for AmountSpent ---
print("\n=== AMOUNTSPENT METRICS ===")
print(calculate_descriptive_stats(df_cleaned, 'AmountSpent').round(2))
plot_distribution_trio(df_cleaned, 'AmountSpent', box_color='#8172B3', hist_color='#CCB974', kde_color='#64B5CD')

# Verify if the pipeline preserved activity frequency metrics
if 'LoginFrequency' in df_cleaned.columns:
    print("=== LOGINFREQUENCY METRICS ===")
    print(calculate_descriptive_stats(df_cleaned, 'LoginFrequency').round(2))
    plot_distribution_trio(df_cleaned, 'LoginFrequency', box_color='#E15759', hist_color='#F28E2B', kde_color='#499894')
else:
    print("ℹ️ Note: LoginFrequency column was intentionally dropped during preprocessing or pipeline cleaning steps.")

if 'ChurnStatus' in df_cleaned.columns:
    # 1. Average Age across Classes
    churn_by_age = df_cleaned.groupby('ChurnStatus')['Age'].mean()

    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=churn_by_age.index, y=churn_by_age.values, palette=['#4C72B0', '#C44E52'], hue=churn_by_age.index,
                legend=False)
    plt.title('Average Age by Churn Status', fontweight='bold', pad=15)
    plt.xlabel('Customer Churn Status')
    plt.ylabel('Mean Age (Years)')
    plt.gca().set_xticklabels(['Retained (0)', 'Churned (1)'])
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 2. Binning Segmented Cohorts
    df_cleaned['AgeGroup'] = pd.cut(
        df_cleaned['Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    )

    age_churn_cross = pd.crosstab(df_cleaned['AgeGroup'], df_cleaned['ChurnStatus'])

    # Grouped Representation
    ax = age_churn_cross.plot(kind='bar', figsize=(11, 5.5), color=['#4C72B0', '#C44E52'], edgecolor='black', width=0.7)
    plt.title('Churn Volume Distribution Across Age Demographics', fontweight='bold', pad=15)
    plt.xlabel('Age Category Cohort')
    plt.ylabel('Total Accounts Evaluated')
    plt.legend(['Retained (0)', 'Churned (1)'], frameon=True)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)

    # Annotate absolute counts on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{int(p.get_height()):,}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5),
                        textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Memory management: Drop categorical column to maintain numeric pipeline integrity
    df_cleaned.drop(columns=['AgeGroup'], inplace=True, errors='ignore')
else:
    print("❌ Error: Target array 'ChurnStatus' not isolated in working DataFrame structure.")


def analyze_categorical_feature(dataframe, column_name, palette_colors=None):
    """
    Prints baseline category counts/cardinality metrics and plots an
    uncluttered, beautifully formatted categorical distribution pie chart.
    """
    if column_name not in dataframe.columns:
        print(f"ℹ️ Note: Column '{column_name}' is not present in the current dataframe context.")
        return

    # Calculate metrics
    mode_val = dataframe[column_name].mode()[0]
    cardinality = dataframe[column_name].nunique()
    value_counts = dataframe[column_name].value_counts()

    print(f"\n=== {column_name.upper()} METRICS ===")
    print(f"Primary Variant (Mode) : {mode_val}")
    print(f"Unique Categories      : {cardinality}")

    # Render Layout
    plt.figure(figsize=(6, 6))
    if palette_colors is None:
        palette_colors = sns.color_palette("Pastel1", n_colors=cardinality)

    plt.pie(
        value_counts,
        labels=value_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=palette_colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
    )
    plt.title(f'{column_name} Distribution Context', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# --- Run Categorical Pipeline across Raw Unencoded Target Metrics ---
analyze_categorical_feature(df_cleaned, 'Gender', palette_colors=['#FF9999', '#66B2FF'])
analyze_categorical_feature(df_cleaned, 'MaritalStatus', palette_colors=['#99FF99', '#FF9999', '#66B2FF', '#FFCC99'])
analyze_categorical_feature(df_cleaned, 'ProductCategory')
analyze_categorical_feature(df_cleaned, 'IncomeLevel', palette_colors=['#FF9999', '#99FF99', '#66B2FF'])
analyze_categorical_feature(df_cleaned, 'InteractionType', palette_colors=['#FFCC99', '#99FF99', '#B3FFF2'])
analyze_categorical_feature(df_cleaned, 'ResolutionStatus', palette_colors=['#99FF99', '#FF9999'])
analyze_categorical_feature(df_cleaned, 'ServiceUsage')

if 'Gender' in df_cleaned.columns and 'ChurnStatus' in df_cleaned.columns:
    # Build standard cross-tabulation table
    gender_churn = pd.crosstab(df_cleaned['Gender'], df_cleaned['ChurnStatus'])

    # Normalize index entries for readability
    gender_churn.index = ['Female' if idx in ['F', 'Female', 0] else 'Male' for idx in gender_churn.index]

    print("=" * 45)
    print("      CHURN VOLUMETRICS BY CUSTOMER GENDER     ")
    print("=" * 45)
    print("Absolute Cross-Tabulation Matrix:")
    print(gender_churn)
    print("\nNormalized Percentage Proportions (%):")
    gender_churn_percent = gender_churn.div(gender_churn.sum(axis=1), axis=0) * 100
    print(gender_churn_percent.round(2))

    # Build Stacked Percentage Representation
    ax = gender_churn_percent.plot(kind='bar', stacked=True, color=['#4C72B0', '#C44E52'], figsize=(8, 5.5),
                                   edgecolor='black')
    plt.title('Proportional Churn Intensity Across Gender Segments', fontweight='bold', pad=15)
    plt.xlabel('Customer Identified Gender')
    plt.ylabel('Percentage Proportion (%)')
    plt.legend(['Retained (0)', 'Churned (1)'], loc='upper right', frameon=True)
    plt.xticks(rotation=0)
    plt.ylim(0, 110)  # Room at the top for labels
    plt.grid(axis='y', alpha=0.3)

    # Annotate inner segments with explicit percentage labels
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()
        x = patch.get_x()
        y = patch.get_y()
        if height > 5:  # Only print label if the segment bar space is visually large enough
            ax.text(x + width / 2, y + height / 2, f"{height:.1f}%",
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Could not evaluate Bivariate Categorical Interaction: Check column availability in 'df'.")

# 1. Churn breakdown by Marital Status
marital_churn = pd.crosstab(df_cleaned['MaritalStatus'], df_cleaned['ChurnStatus'])
marital_churn.columns = ['Not Churned', 'Churned']
print("=== CHURN BY MARITAL STATUS ===")
print(marital_churn)

marital_churn.plot(kind='bar', figsize=(9, 5), color=['#4C72B0', '#C44E52'], edgecolor='black')
plt.title('Churn Count Variance by Marital Status', fontweight='bold')
plt.xlabel('Marital Status')
plt.ylabel('Account Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Gender vs Marital Status Interaction
gender_marital = pd.crosstab(df_cleaned['Gender'], df_cleaned['MaritalStatus'])
gender_marital.index = ['Female', 'Male'] if 'F' in df_cleaned['Gender'].unique() else gender_marital.index

print("\n=== MARITAL STATUS DISTRIBUTION BY GENDER ===")
print(gender_marital)

gender_marital.plot(kind='bar', figsize=(9, 5), stacked=True, edgecolor='black', color=sns.color_palette("Pastel1"))
plt.title('Marital Status Compositions across Gender Segments', fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Account Count')
plt.legend(title='Marital Status')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Deep dive into joint feature effects on retention
grouped = df_cleaned.groupby(['Gender', 'MaritalStatus', 'ChurnStatus']).size().unstack(fill_value=0)
grouped.columns = ['Not Churned', 'Churned']

print("=== THREE-DIMENSIONAL RETENTION MATRIX ===")
print(grouped)

grouped.plot(kind='bar', figsize=(12, 5.5), color=['#4C72B0', '#C44E52'], edgecolor='black', width=0.8)
plt.title('Churn Intensity Across Intersected Demographics', fontweight='bold', pad=15)
plt.xlabel('Demographic Clusters (Gender, Marital Status)')
plt.ylabel('Total Accounts')
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
