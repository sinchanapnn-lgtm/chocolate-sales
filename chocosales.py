# CHCOLATE SALES PROJECT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv('/content/Chocolate Sales (2).csv')
a.info()

# STEP 3 – Dataset Loading & Display
import pandas as pd

# Load the dataset into variable 'a'
# We use the filename provided in the context
file_path = 'Chocolate Sales (2).csv'

try:
    a = pd.read_csv(file_path)

    # Display the first five rows
    print("--- First 5 Rows ---")
    print(a.head())

    # Display the last five rows
    print("\n--- Last 5 Rows ---")
    print(a.tail())

    # Display the dataset shape
    print("\n--- Dataset Shape ---")
    print(a.shape)

    # Display dataset information (column names, data types, non-null counts)
    print("\n--- Dataset Information ---")
    a.info()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found in the current directory.")


    # STEP 4 – Dataset Structure Analysis
import pandas as pd

# Define the file path
file_path = 'Chocolate Sales (2).csv'

try:
    # Load the dataset
    a = pd.read_csv(file_path)

    # 1. Column Names and Data Types
    print("--- Column Names and Data Types ---")
    print(a.dtypes)

    # 2. Number of Unique Values per Column
    # This helps identify categorical vs. high-cardinality columns
    print("\n--- Number of Unique Values per Column ---")
    print(a.nunique())

    # 3. Missing Values per Column
    print("\n--- Missing Values per Column ---")
    print(a.isnull().sum())

    # 4. Basic Statistical Summary for Numerical Columns
    # By default, this covers the 'Boxes Shipped' column (int64)
    print("\n--- Statistical Summary (Numerical) ---")
    print(a.describe())

    # 5. Frequency Distributions for Categorical Columns
    # This loops through all object/string columns to show the count of each category
    print("\n--- Frequency Distributions (Categorical) ---")
    categorical_cols = a.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nTop 10 values for {col}:")
        print(a[col].value_counts().head(10))

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the file is uploaded to the environment.")

    #STEP 5 – Data Cleaning
import pandas as pd
import numpy as np

# 1. Handling Missing Values (Updated syntax to remove FutureWarning)
a = a.ffill().bfill()

# 2. Removing Duplicates
a = a.drop_duplicates().reset_index(drop=True)

# 3. Renaming Columns Consistently (Snake Case)
a.columns = [col.strip().lower().replace(" ", "_") for col in a.columns]

# 4. Standardizing Text & 5. Trimming Extra Spaces
for col in a.select_dtypes(include=['object']).columns:
    a[col] = a[col].astype(str).str.strip().str.title()

# 6. Converting Data Types & 7. Fixing Inconsistent Formats
# Amount: Use raw string r'' to remove SyntaxWarning for \$
a['amount'] = a['amount'].replace(r'[\$,]', '', regex=True).astype(float)

# Date: Use format='mixed' or dayfirst=True to recover the 1,992 missing dates
a['date'] = pd.to_datetime(a['date'], dayfirst=True, errors='coerce')

# Fill any remaining date gaps that couldn't be parsed
a['date'] = a['date'].ffill().bfill()

# 8. Correcting Incorrect Entries (e.g., negative values)
a['amount'] = a['amount'].abs()
a['boxes_shipped'] = a['boxes_shipped'].abs()

# 9. Fixing Invalid Categorical Values
country_map = {'Usa': 'United States', 'Uk': 'United Kingdom'}
a['country'] = a['country'].replace(country_map)

# 10. Normalizing Labels
a['product'] = a['product'].str.replace(r'\s+', ' ', regex=True)

# 11. Resolving Mixed Data Types
a['boxes_shipped'] = pd.to_numeric(a['boxes_shipped'], errors='coerce').fillna(0).astype(int)

# 12. Detecting and Handling Outliers (IQR Method)
Q1 = a['amount'].quantile(0.25)
Q3 = a['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
a['amount'] = np.clip(a['amount'], lower_bound, upper_bound)
# 13. Validating Cleaned Data
print("--- Final Cleaned Information ---")
print(a.info())
print("\n--- Missing Values Check ---")
print(a.isnull().sum())

# STEP 6 – Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# 1. Feature Engineering (Creating new insights from existing data)
# Creating 'price_per_box' to understand unit value
a['price_per_box'] = a['amount'] / a['boxes_shipped'].replace(0, 1)

# 2. Handling Date-Time Features
# Extracting Year, Month, and Day of Week from the cleaned date
a['year'] = a['date'].dt.year
a['month'] = a['date'].dt.month
a['day_of_week'] = a['date'].dt.dayofweek # 0=Monday, 6=Sunday

# 3. Binning Continuous Variables
# Grouping 'boxes_shipped' into 'Small', 'Medium', and 'Large' orders
bins = [0, 50, 200, a['boxes_shipped'].max()]
labels = ['Small', 'Medium', 'Large']
a['order_size_category'] = pd.cut(a['boxes_shipped'], bins=bins, labels=labels)

# 4. Encoding Categorical Variables
# Label Encoding for ordinal categories (Order Size)
le = LabelEncoder()
a['order_size_encoded'] = le.fit_transform(a['order_size_category'])

# One-Hot Encoding for nominal categories (Country) to avoid bias
a = pd.get_dummies(a, columns=['country'], prefix='country')

# 5. Applying Transformations
# Log Transformation on 'amount' to reduce skewness and stabilize variance
# We add 1 to avoid log(0)
a['amount_log'] = np.log1p(a['amount'])

# Square Root Transformation on 'boxes_shipped'
a['boxes_sqrt'] = np.sqrt(a['boxes_shipped'])

# 6. Scaling Numerical Features
# Standardization (Mean=0, Std=1) for the Log-transformed amount
scaler_std = StandardScaler()
a['amount_scaled'] = scaler_std.fit_transform(a[['amount_log']])

# Min-Max Scaling (0 to 1 range) for boxes_shipped
scaler_minmax = MinMaxScaler()
a['boxes_minmax'] = scaler_minmax.fit_transform(a[['boxes_shipped']])

# 7. Final Validation of Preprocessed Data
print("--- Preprocessing Complete ---")
print(f"New Column Count: {len(a.columns)}")
print("\n--- Sample of Transformed Data ---")
print(a[['amount', 'amount_log', 'amount_scaled', 'boxes_shipped', 'order_size_category']].head())

# STEP 7 – Exploratory Data Analysis (EDA)
# Check current column names to be sure
print(a.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Univariate Analysis: Distribution of Sales Amount
plt.figure(figsize=(10, 5))
sns.histplot(a['amount'], kde=True, color='brown')
plt.title('Distribution of Sales Amount')
plt.show()

# 2. Bivariate Analysis: Amount vs Boxes Shipped
# Purpose: To see the relationship between volume and revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='boxes_shipped', y='amount', data=a)
plt.title('Boxes Shipped vs. Sales Amount')
plt.show()

# 3. Time-Series Analysis: Monthly Trends
# Using the 'month' column created in the preprocessing step
plt.figure(figsize=(10, 5))
a.groupby('month')['amount'].sum().plot(kind='line', marker='o')
plt.title('Total Revenue by Month')
plt.ylabel('Total Sales ($)')
plt.show()

# 4. Correlation Heatmap (Focusing on Numeric Columns)
plt.figure(figsize=(10, 8))
numeric_a = a.select_dtypes(include=['float64', 'int64', 'int32'])
sns.heatmap(numeric_a.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# STEP 8 – Feature Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Coefficient of Variation (CV) Analysis
# Purpose: Measures relative variability. High CV means the feature has a wide range of values.
numeric_cols = a.select_dtypes(include=[np.number]).columns
cv_values = (a[numeric_cols].std() / a[numeric_cols].mean()) * 100
print("--- Coefficient of Variation (%) ---")
print(cv_values)

# 2. Target Correlation Analysis (Influence on Outcome)
# Purpose: Directly measures how much each feature influences the 'amount'.
plt.figure(figsize=(8, 6))
correlations = a[numeric_cols].corr()['amount'].sort_values(ascending=False)
correlations.drop('amount').plot(kind='bar', color='teal')
plt.title('Feature Influence on Sales Amount')
plt.ylabel('Correlation Coefficient')
plt.show()

# 3. Categorical Significance (Boxen Plots)
# Purpose: Boxen plots show the distribution of 'amount' across different products
# to see which product is the most consistent revenue driver.
plt.figure(figsize=(12, 6))
sns.boxenplot(x='product', y='amount', data=a)
plt.xticks(rotation=45)
plt.title('Revenue Stability by Product')
plt.show()

# 4. Feature Redundancy Check (Multicollinearity)
# Purpose: If two features are too similar, one is redundant.
# We look for correlations > 0.8.
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(a[numeric_cols].corr(), dtype=bool))
sns.heatmap(a[numeric_cols].corr(), mask=mask, annot=True, cmap='vlag')
plt.title('Redundancy Heatmap (Multicollinearity)')
plt.show()

# 5. Temporal Relevance (Year-over-Year Growth)
# Purpose: Checks if the 'year' feature actually matters or if sales are stagnant.
yoy_growth = a.groupby('year')['amount'].sum()
print("\n--- Year-over-Year Revenue ---")
print(yoy_growth)

# data visulisation
# --- STEP 1: FIX DATA STRUCTURE FOR VISUALIZATION ---
# If 'country' was one-hot encoded, we reconstruct it for plotting purposes
if 'country' not in a.columns:
    country_cols = [col for col in a.columns if col.startswith('country_')]
    if country_cols:
        # Reverses one-hot encoding back to a single 'country' column for the charts
        a['country'] = a[country_cols].idxmax(axis=1).str.replace('country_', '')

# Ensure other categorical columns exist; if not, use the original ones
# We'll also suppress warnings to keep the output clean
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- STEP 2: GENERATE VISUALIZATIONS ---
sns.set_theme(style="whitegrid")

# 1-4. Distribution Grid
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.histplot(a['amount'], kde=True, ax=axes[0, 0], color='brown').set_title('1. Amount Distribution')
sns.kdeplot(a['boxes_shipped'], fill=True, ax=axes[0, 1], color='orange').set_title('2. Boxes Shipped Density')
sns.countplot(x='month', data=a, ax=axes[1, 0], hue='month', palette='viridis', legend=False).set_title('3. Monthly Transactions')
sns.countplot(x='year', data=a, ax=axes[1, 1], hue='year', palette='magma', legend=False).set_title('4. Yearly Transactions')
plt.tight_layout()

# 5. Bar Plot: Revenue by Product
plt.figure(figsize=(14, 6))
sns.barplot(x='product', y='amount', data=a, estimator=sum, hue='product', palette='rocket', legend=False)
plt.title('5. Total Revenue by Product')
plt.xticks(rotation=45)
plt.show()

# 6. Violin Plot: Shipping Volume per Country
#
plt.figure(figsize=(14, 6))
sns.violinplot(x='country', y='boxes_shipped', data=a, hue='country', palette='Set2', legend=False)
plt.title('6. Shipping Volume Density per Country')
plt.show()

# 7. Box Plot: Sales Person Variability
plt.figure(figsize=(16, 6))
sns.boxplot(x='sales_person', y='amount', data=a, hue='sales_person', palette='husl', legend=False)
plt.title('7. Sales Person Performance Variability')
plt.xticks(rotation=45)
plt.show()

# 8. Scatter Plot: Revenue vs. Volume
#
plt.figure(figsize=(12, 7))
sns.scatterplot(x='boxes_shipped', y='amount', hue='product', size='amount', sizes=(20, 200), data=a, alpha=0.6)
plt.title('8. Revenue vs. Volume Relationship')
plt.show()

# 9. Correlation Heatmap (Numeric Only)
#
plt.figure(figsize=(10, 8))
numeric_a = a.select_dtypes(include=[np.number])
sns.heatmap(numeric_a.corr(), annot=True, cmap='coolwarm', fmt=".2f").set_title('9. Feature Correlation Heatmap')
plt.show()


# STEP 10 – Insight Generation

import pandas as pd
import numpy as np

# 1. Product Performance Insight
# Calculation: Total revenue and average unit price per product
product_stats = a.groupby('product').agg({
    'amount': 'sum',
    'boxes_shipped': 'sum',
    'price_per_box': 'mean'
}).sort_values(by='amount', ascending=False)

print("--- Top Performing Products ---")
print(product_stats)

# 2. Regional Market Share
# Calculation: Percentage of total revenue contributed by each country
total_revenue = a['amount'].sum()
country_revenue = a.groupby('country')['amount'].sum().sort_values(ascending=False)
country_share = (country_revenue / total_revenue) * 100

print("\n--- Market Share by Country (%) ---")
print(country_share)

# 3. Sales Person Efficiency
# Calculation: Revenue generated per box shipped (Efficiency Ratio)
sales_efficiency = a.groupby('sales_person').agg({
    'amount': 'sum',
    'boxes_shipped': 'sum'
})
sales_efficiency['revenue_per_box'] = sales_efficiency['amount'] / sales_efficiency['boxes_shipped']
sales_efficiency = sales_efficiency.sort_values(by='revenue_per_box', ascending=False)

print("\n--- Sales Person Efficiency (Revenue per Box) ---")
print(sales_efficiency)

# 4. Seasonal Demand Analysis
# Calculation: Average sales per month to find peak seasons
monthly_avg = a.groupby('month')['amount'].mean()
peak_month = monthly_avg.idxmax()
low_month = monthly_avg.idxmin()

print(f"\n--- Seasonality Insights ---")
print(f"Peak Demand Month: {peak_month} (Avg Sales: ${monthly_avg.max():.2f})")
print(f"Lowest Demand Month: {low_month} (Avg Sales: ${monthly_avg.min():.2f})")


#STEP 11 – Statistical Analysis

import pandas as pd
import numpy as np
from scipy import stats

# 1. Measures of Central Tendency & Dispersion (Every Column)
# Purpose: To understand the "typical" sale and the "spread" of our data.
stats_summary = a.describe(include='all').T
stats_summary['skewness'] = a.select_dtypes(include=[np.number]).skew()
stats_summary['kurtosis'] = a.select_dtypes(include=[np.number]).kurtosis()

print("--- Central Tendency & Dispersion Summary ---")
print(stats_summary[['mean', '50%', 'std', 'skewness']])

# 2. Normality Testing (Shapiro-Wilk or D'Agostino's K^2)
# Purpose: To see if 'amount' follows a normal distribution.
# Many statistical models assume normality.
k2, p_norm = stats.normaltest(a['amount'])
print(f"\n--- Normality Test (Amount) ---")
print(f"P-value: {p_norm:.4f}")
if p_norm < 0.05:
    print("Result: Data is NOT normally distributed (skewed).")
else:
    print("Result: Data follows a Normal Distribution.")

# 3. Hypothesis Testing: Correlation (Pearson's R)
# Purpose: Is the relationship between 'boxes_shipped' and 'amount' statistically significant?
corr_coeff, p_corr = stats.pearsonr(a['boxes_shipped'], a['amount'])
print(f"\n--- Correlation Significance ---")
print(f"Pearson Correlation: {corr_coeff:.4f}, P-value: {p_corr:.4f}")

# 4. Hypothesis Testing: T-Test (Group Comparison)
# Purpose: Compare if the average sales in two different years are significantly different.
# We'll split the data by the 'year' feature we engineered earlier.
if a['year'].nunique() >= 2:
    years = a['year'].unique()
    group1 = a[a['year'] == years[0]]['amount']
    group2 = a[a['year'] == years[1]]['amount']
    t_stat, p_ttest = stats.ttest_ind(group1, group2)
    print(f"\n--- T-Test (Comparing {years[0]} vs {years[1]}) ---")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_ttest:.4f}")

# 5. Chi-Square Test of Independence (Categorical Relationships)
# Purpose: Is there a relationship between 'Country' and the type of 'Product' preferred?
contingency_table = pd.crosstab(a['country'], a['product'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\n--- Chi-Square Test (Country vs Product) ---")
print(f"P-value: {p_chi2:.4f}")

# STEP 12 – Model Building / Prediction.

# Modeling Strategy & AssumptionsObjective: Predict the amount (Sales Revenue) based on all other available features.Assumptions:The relationships between features (like boxes_shipped and product) are consistent over time.Outliers have been handled (using the clipping we performed in Step 5) to prevent model distortion.Preprocessing Logic:Features ($X$): Includes boxes_shipped, month, year, and encoded versions of country and product.Target ($y$): The amount column.Split: 80% Training data (to teach the model) and 20% Testing data (to verify accuracy).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Prepare Features (X) and Target (y)
# We drop 'amount' and 'date' (already represented by year/month)
X = a.drop(['amount', 'date', 'amount_log', 'amount_scaled'], axis=1, errors='ignore')

# Select only numeric columns (including one-hot encoded ones)
X = X.select_dtypes(include=[np.number])
y = a['amount']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Initialization and Training
# We use 100 decision trees to ensure stability in predictions
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Model Performance Metrics ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# 6. Feature Importance Analysis
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Top Features Influencing Prediction ---")
print(importances.head(5))


# STEP 13 – Model Evaluation Prompt

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# 1. Calculate Comprehensive Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
med_ae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Median Absolute Error: ${med_ae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 2. Residual Analysis Plot
# Purpose: To check if errors are random. Patterns here suggest model bias.
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot: Errors vs. Predicted Values')
plt.xlabel('Predicted Amount ($)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()

# 3. Prediction Error Distribution
# Purpose: Visualizes how many of our predictions were 'close' vs 'far off'.
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Error Magnitude ($)')
plt.show()

# 4. Actual vs. Predicted Comparison Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Sales Amount')
plt.xlabel('Actual Sales ($)')
plt.ylabel('Predicted Sales ($)')
plt.show()