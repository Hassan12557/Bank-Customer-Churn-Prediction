import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for nice visualizations
sns.set_theme(style="whitegrid")

# Relative path from the 'notebooks' directory back up to the data directory
DATA_PATH = os.path.join("..", "data", "Processed", "bank_data_preprocessed.csv")

if os.path.exists(DATA_PATH):
    df_cleaned = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded preprocessed dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
else:
    print(f"❌ Could not find preprocessed data at: {os.path.abspath(DATA_PATH)}")
    print("Please ensure you have run your preprocessing.py script first to generate this file.")

    # Estimates of location and variability for Age column
age_stats = {
    'Mean': df_cleaned['Age'].mean(),
    'Median': df_cleaned['Age'].median(),
    'Mode': df_cleaned['Age'].mode()[0],
    'Range': df_cleaned['Age'].max() - df_cleaned['Age'].min(),
    'Variance': df_cleaned['Age'].var(),
    'Standard_Deviation': df_cleaned['Age'].std(),
    'IQR': df_cleaned['Age'].quantile(0.75) - df_cleaned['Age'].quantile(0.25)
}

print("=" * 30)
print("     AGE COLUMN STATISTICS     ")
print("=" * 30)
for stat, value in age_stats.items():
    print(f"{stat:18s}: {value:.2f}")

# Set up the figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Boxplot (Using seaborn for consistency and look)
sns.boxplot(y=df_cleaned['Age'], ax=axes[0], color='#4C72B0', width=0.4)
axes[0].set_title('Boxplot of Age', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Age')

# 2. Histogram
axes[1].hist(df_cleaned['Age'], bins=20, edgecolor='black', color='#55A868', alpha=0.7)
axes[1].set_title('Histogram of Age', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')

# 3. Density plot
sns.kdeplot(df_cleaned['Age'], ax=axes[2], fill=True, color='#C44E52', linewidth=2)
axes[2].set_title('Density Plot of Age', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Density')

plt.tight_layout()
plt.show()