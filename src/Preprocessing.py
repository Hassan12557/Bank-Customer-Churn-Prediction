import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df="/Bank-Customer Churn-Prediction/data/Raw/Combined.csv"
# Check if the churn dataset is balanced
churn_counts = df['ChurnStatus'].value_counts()
churn_ratio = churn_counts[1] / churn_counts[0] if 0 in churn_counts else 0
is_balanced = 0.3 <= churn_ratio <= 0.7
print(f"Class distribution:\n{churn_counts}")
print(f"Churn ratio (1/0): {churn_ratio:.3f}")
print(f"Is balanced: {is_balanced}")
#Drop Unnecessary Columns
# Drop unnecessary columns
columns_to_drop = ['CustomerID', 'TransactionID', 'LastLoginDate',
                    'TransactionDate', 'InteractionDate','InteractionID']
df_cleaned = df.drop(columns=columns_to_drop)

# Print the columns after dropping
print(df_cleaned.columns.tolist())

# Find missing values after dropping columns
missing_values = df_cleaned.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Fix missing and null values
# First check which columns have missing values
missing_cols = df_cleaned.columns[df_cleaned.isnull().any()].tolist()

# Fill numeric columns with median
for col in df_cleaned.select_dtypes(include=['int64', 'float64']).columns:
    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Fill categorical columns with mode
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

print("Missing values fixed. Remaining nulls:", df_cleaned.isnull().sum().sum())

#hot encoding categorical data
# Initialize label encoder
le = LabelEncoder()

# Label Encoding for Gender
df_cleaned['Gender'] = le.fit_transform(df_cleaned['Gender'])

# Label Encoding for ResolutionStatus
df_cleaned['ResolutionStatus'] = le.fit_transform(df_cleaned['ResolutionStatus'])

# One-Hot Encoding for specified columns
one_hot_cols = ['MaritalStatus', 'ProductCategory', 'IncomeLevel', 'InteractionType']
df_cleaned = pd.get_dummies(df_cleaned, columns=one_hot_cols, drop_first=True)

# Note: ServiceUsage was already dropped in previous steps, so I'll include it if it exists
if 'ServiceUsage' in df_cleaned.columns:
    df_cleaned = pd.get_dummies(df_cleaned, columns=['ServiceUsage'], drop_first=True)
else:
    print("Note: ServiceUsage column was already dropped earlier")
#ONE SMALL ISSUE: Boolean Values as Strings
# ❌ CURRENT: Boolean values as strings 'TRUE'/'FALSE'
MaritalStatus_Married: 'FALSE'  # Should be 0
MaritalStatus_Single: 'TRUE'    # Should be 1

# ✅ FIX: Convert to integers
df_cleaned = df_cleaned.replace({'TRUE': 1, 'FALSE': 0})
# OR
df_cleaned = df_cleaned.applymap(lambda x: 1 if x == 'TRUE' else 0 if x == 'FALSE' else x)

# Convert TRUE/FALSE strings to 1/0 integers
bool_columns = [col for col in df_cleaned.columns
                if df_cleaned[col].dtype == 'object'
                and set(df_cleaned[col].unique()).issubset({'TRUE', 'FALSE'})]

print(f"Converting {len(bool_columns)} boolean columns:")
for col in bool_columns:
    df_cleaned[col] = df_cleaned[col].map({'TRUE': 1, 'FALSE': 0}).astype(int)
    print(f"  ✓ {col}")

# Verify all numerical now
print(f"\n✅ Final data types:")
print(df_cleaned.dtypes.value_counts())


# Fix TRUE/FALSE → 1/0 conversion
bool_columns = df_cleaned.select_dtypes(include=['bool']).columns
df_cleaned[bool_columns] = df_cleaned[bool_columns].astype(int)

#Split the train and test
#scaling only training data
#Transform both train and test with same scaler
# Split features and target
X = df_cleaned.drop('ChurnStatus', axis=1)
y = df_cleaned['ChurnStatus']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data with same scaler
X_test_scaled = scaler.transform(X_test)


#Summary:
# Run this in your preprocessing file
import numpy as np
print("="*60)
print("FINAL PREPROCESSED DATASET CHECK")
print("="*60)
print(f"Shape: {df_cleaned.shape}")
print(f"Rows: {df_cleaned.shape[0]}, Columns: {df_cleaned.shape[1]}")

print("\n📊 COLUMNS LIST:")
for i, col in enumerate(df_cleaned.columns.tolist(), 1):
    dtype = df_cleaned[col].dtype
    unique = df_cleaned[col].nunique()
    print(f"{i:2d}. {col:30s} | {str(dtype):10s} | {unique:3d} unique values")

print("\n🎯 TARGET DISTRIBUTION:")
churn_rate = df_cleaned['ChurnStatus'].mean()
print(f"Churn rate: {churn_rate:.2%}")
print(f"Active (0): {sum(df_cleaned['ChurnStatus'] == 0):,}")
print(f"Churned (1): {sum(df_cleaned['ChurnStatus'] == 1):,}")

print("\n🔍 DATA TYPES SUMMARY:")
print(df_cleaned.dtypes.value_counts())

print("\n⚠️ CHECK FOR ISSUES:")
# 1. Any non-numeric columns?
non_numeric = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"❌ Non-numeric columns: {non_numeric}")
else:
    print("✅ All columns numeric")

# 2. Any columns with too many unique values (potential IDs)?
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() > 0.8 * len(df_cleaned):
        print(f"⚠️  {col}: {df_cleaned[col].nunique():,} unique values - check if it's an ID")

# 3. Missing values?
missing = df_cleaned.isnull().sum().sum()
if missing > 0:
    print(f"❌ {missing} missing values remaining")
else:
    print("✅ No missing values")

# Save your cleaned dataframe to CSV
df_cleaned.to_csv('bank_data_preprocessed_ready_for_modeling.csv', index=False)
print("✅ Dataset saved as: 'bank_data_preprocessed_ready_for_modeling.csv'")
print(f"Size: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")


df_cleaned.to_excel('bank_data_preprocessed.xlsx', index=False)
print("✅ Dataset saved as Excel: 'bank_data_preprocessed.xlsx'")
