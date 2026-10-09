# TASK 5: EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = sns.load_dataset('titanic')

# 1. Basic Data Exploration
print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nSURVIVAL COUNT")
print(df['survived'].value_counts())

# 2. Handle Missing Values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# 3. Univariate Analysis - Survival Count
sns.countplot(data=df, x='survived')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 4. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# 5. Fare Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='fare', bins=30, kde=True)
plt.title('Fare Distribution')
plt.show()

# 6. Bivariate Analysis - Survival by Gender
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival by Gender')
plt.show()

# 7. Survival by Passenger Class
sns.countplot(data=df, x='class', hue='survived')
plt.title('Survival by Passenger Class')
plt.show()

# 8. Age vs Fare Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age', y='fare', hue='survived')
plt.title('Age vs Fare by Survival')
plt.show()

# 9. Detect Outliers Using Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='fare')
plt.title('Fare Outliers')
plt.show()

# 10. Correlation Heatmap
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 11. Pairplot
sns.pairplot(
    df,
    vars=['age', 'fare', 'sibsp', 'parch'],
    hue='survived',
    diag_kind='hist'
)
plt.show()

# 12. Grouped Analysis
print("\nSURVIVAL RATE BY GENDER")
print(df.groupby('sex')['survived'].mean())

print("\nSURVIVAL RATE BY PASSENGER CLASS")
print(df.groupby('class', observed=True)['survived'].mean())

print("\nSURVIVAL RATE BY GENDER AND CLASS")
print(df.groupby(['sex', 'class'], observed=True)['survived'].mean())

# 13. Skewness Analysis
print("\nSKEWNESS OF AGE AND FARE")
print(df[['age', 'fare']].skew())

# 14. Final Summary
print("\nKEY FINDINGS")
print("1. The dataset contains both survivors and non-survivors.")
print("2. Female passengers generally had higher survival rates.")
print("3. Passenger class was associated with survival outcomes.")
print("4. Fare distribution is right-skewed.")
print("5. Boxplots help identify potential outliers.")
print("6. Heatmaps show correlations between numerical variables.")
print("7. EDA helps identify patterns, trends, and anomalies.")

print("\nEDA COMPLETED SUCCESSFULLY")
