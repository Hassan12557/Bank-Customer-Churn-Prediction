&#x20;# EDA Summary - Class Distribution Check



Class distribution:

ChurnStatus

0    5460

1    1352

Name: count, dtype: int64



Churn ratio (1/0): 0.248



Is balanced: False



\# Data Preprocessing Summary



\## Columns Dropped

\- CustomerID

\- TransactionID

\- LastLoginDate

\- TransactionDate

\- InteractionDate

\- InteractionID



\## Missing Values Treatment

\- Numeric columns filled with median

\- Categorical columns filled with mode

\- Remaining nulls: 0



\## Encoding Techniques Applied

\- \*\*Label Encoding\*\*: Gender, ResolutionStatus

\- \*\*One-Hot Encoding (drop\_first=True)\*\*: MaritalStatus, ProductCategory, IncomeLevel, InteractionType



\## Boolean Conversion Fix

\- Converted 'TRUE'/'FALSE' string values to integers (1/0)

\- Applied to all boolean columns from one-hot encoding



\## Final Data Types (After Conversion)

\- bool: 13 columns

\- int64: 5 columns

\- float64: 1 column



\# Final Preprocessed Dataset Check



\## Dataset Overview

\- Shape: (6812, 19)

\- Rows: 6812

\- Columns: 19



\## Target Distribution

\- Churn rate: 19.85%

\- Active (0): 5,460

\- Churned (1): 1,352



\## Statistical Analysis



\### AmountSpent Statistics

\- Mean: 251.62

\- Median: 251.85

\- Mode: 179.18

\- Range: 494.68

\- Variance: 20420.89

\- Standard Deviation: 142.90

\- IQR: 248.18



\### LoginFrequency Statistics

\- Mean: 25.72

\- Median: 26.00

\- Mode: 30.00

\- Range: 48.00

\- Variance: 197.74

\- Standard Deviation: 14.06

\- IQR: 24.00



\## Data Quality Validation

\- All columns numeric

\- No missing values

\- Train-test split: 80/20 with stratification

\- StandardScaler applied (fit on train, transform on test)



\## Data Types Summary

\- int64: 18 columns

\- float64: 1 column



\# Exploratory Data Analysis (EDA) Findings



\## Demographic Analysis



\### Gender Distribution

\- Female: 47.4%

\- Male: 52.6%

\- Gender mode: Female



\### Churn by Gender

\- Female churn rate: 18.32% (656 churned out of 3,580)

\- Male churn rate: 21.53% (696 churned out of 3,232)

\- \*\*Finding\*\*: Male customers churn at a higher rate than females



\### Marital Status Distribution (Before Encoding)

\- Widowed: 27.8% (mode)

\- Divorced: 26.3%

\- Married: 23.4%

\- Single: 22.4%



\### Churn by Marital Status

\- Divorced: 335 churned

\- Married: 337 churned

\- Single: 332 churned

\- Widowed: 348 churned

\- All marital statuses show similar churn counts (332-348)



\### Marital Status by Gender

\- Female: Highest in Widowed (1,067), then Divorced (897)

\- Male: Evenly distributed across categories



\### Age Group Distribution by Gender

\- Female: Highest in 50-60 age group (822)

\- Male: Highest in 40-50 age group (665)



\### Churn by Age Group

\- Highest churn: 50-60 age group (329 churned)

\- Lowest churn: 20-30 age group (221 churned)



\## Product \& Behavioral Analysis



\### Product Category Distribution (Before Encoding)

\- Books: 20.7% (mode)

\- Electronics: 20.1%

\- Other: 20.1%

\- Clothing: 19.8%

\- Furniture: 19.4%



\### Income Level Distribution

\- High: 34.4%

\- Medium: 33.3%

\- Low: 32.4%

\- Nearly equal distribution across all three levels



\### Income Level by Gender

\- Female: Relatively balanced across High (1,179), Low (1,172), Medium (1,229)

\- Male: Slightly lower in Low (1,035) and Medium (1,036) compared to High (1,161)



\### Interaction Type Analysis

\- Complaint: 1,769 interactions (1,399 not churned, 370 churned)

\- Feedback: 1,921 interactions (1,523 not churned, 398 churned)

\- Inquiry: 1,514 interactions (1,210 not churned, 304 churned)



\### Resolution Status

\- Resolved: 2,693 (2,135 not churned, 558 churned)

\- Unresolved: 2,511 (1,997 not churned, 514 churned)



\## Complex Interaction Analysis



\### Churn by Gender and Marital Status Combination

\- Highest churn risk: M\_Married (185 churned), M\_Single (182 churned)

\- Lowest churn risk: F\_Married (152 churned), F\_Single (150 churned)



\### Churn by Interaction Type and Marital Status

\- Highest risk combinations: Married\_Complaint (119 churned), Single\_Feedback (119 churned)

\- Lowest risk: Widowed\_Inquiry (55 churned)



\## Correlation Analysis



\### Numerical Correlations with ChurnStatus

\- Age: Weak positive correlation (0.045), statistically significant (p=0.0002)

\- AmountSpent: Negligible correlation (0.005), not significant (p=0.6731)

\- LoginFrequency: Weak negative correlation (-0.100), statistically significant (p=0.0000)

\- \*\*Key insight\*\*: LoginFrequency shows the strongest relationship with churn



\### Categorical Correlation Matrix (Cramér's V)

\- Gender and MaritalStatus: Weak correlation (0.047)

\- MaritalStatus and InteractionType: Moderate correlation (0.082)

\- IncomeLevel and ResolutionStatus: Weak correlation (0.057)

\- Most categorical variables show weak to moderate correlations

\- No strong multicollinearity detected among features



\## Key Business Insights from EDA



1\. \*\*Gender Gap\*\*: Male customers churn at 21.53% vs 18.32% for females

2\. \*\*Age Risk\*\*: 50-60 age group shows highest absolute churn numbers

3\. \*\*Login Frequency\*\*: Negative correlation (-0.10) indicates less engaged customers are more likely to churn

4\. \*\*Interaction Patterns\*\*: Complaint and Feedback interactions show similar churn rates

5\. \*\*Marital Status\*\*: Relatively even churn distribution across all categories

6\. \*\*Income\*\*: Balanced distribution suggests income alone isn't a strong predictor

