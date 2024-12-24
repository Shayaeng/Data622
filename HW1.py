# Instructions: Exploratory analysis and essay
# Due October 20th (extended 1 week)

# Pre-work

#Visit the following website and explore the range of sizes of this dataset (from 100 to 5 million records):
#https://excelbianalytics.com/wp/downloads-18-sample-csv-files-data-sets-for-testing-sales/ or
#(new) https://www.kaggle.com/datasets
#Select 2 files to download
#Based on your computer's capabilities (memory, CPU), select 2 files you can handle (recommended one small, one large)
#Download the files
#Review the structure and content of the tables, and think about the data sets (structure, size, dependencies, labels, etc)
#Consider the similarities and differences in the two data sets you have downloaded
#Think about how to analyze and predict an outcome based on the datasets available
#Based on the data you have, think which two machine learning algorithms presented so far could be used to analyze the data
#Deliverable

#Essay (minimum 500 word document)
#Write a short essay explaining your selection of algorithms and how they relate to the data and what you are trying to do
#Exploratory Analysis using R or Python (submit code + errors + analysis as notebook or copy/paste to document)
#Explore how to analyze and predict an outcome based on the data available. This will be an exploratory exercise, so feel free to show errors and warnings that raise during the analysis. Test the code with both datasets selected and compare the results.
#Answer questions such as:

#Are the columns of your data correlated?
#Are there labels in your data? Did that impact your choice of algorithm?
#What are the pros and cons of each algorithm you selected?
#How your choice of algorithm relates to the datasets (was your choice of algorithm impacted by the datasets you chose)?
#Which result will you trust if you need to make a business decision?
#Do you think an analysis could be prone to errors when using too much data, or when using the least amount possible?
#How does the analysis between data sets compare?

#The ask is for you to do an Exploratory Data Analysis on a dataset of your choosing, train models, and then provide insight and conclusions about what you did. This is a typical ask in the life of a data scientist (except you don’t get to choose your dataset).

#As part of that the ask is for you to look at another data set and think about how the size of the dataset would have changed your EDA, training, and conclusions. You don’t have to analyze the second data set to the same extent as the first – just discuss the impact of data set size e.g. on the choice of your algorithm. For example, how would a much smaller data set impact your choice of using a Decision Tree.

# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import IterativeImputer


# Data Sets

data1 = pd.read_csv("C:\\Users\\shaya\\OneDrive\\Documents\\repos\\Data622\\credit-risk\\credit_risk_dataset.csv") # https://www.kaggle.com/datasets/itshappy/ps4e9-original-data-loan-approval-prediction/data
data2 = pd.read_csv("C:\\Users\\shaya\\OneDrive\\Documents\\repos\\Data622\\data2\\Training Data.csv")

# Data Exploration

# Data 1
print(data1.info())

# The data set has 32,581 rows and 12 columns. The columns are divided into three data types: int64, float64, and object. Columns 'person_age', 'person_income', 'loan_amnt', 'loan_status', and 'cb_person_cred_hist_length' are int64 data types. Columns 'person_home_ownership', 'loan_intent', 'loan_grade', and 'cb_person_default_on_file' are object data types. Columns 'person_emp_length', 'loan_int_rate', and 'loan_percent_income' are float64 data types. The majority of columns do not have any missing values. Column 'person_emp_length' has 895 missing values, or arpund 2.7% of the data. Column 'loan_int_rate' has 3,116 missing values, or around 9.6% of the data. 

print(data1.describe())

# The provided summary statistics reveal several notable insights about the dataset. The loan amounts and income levels exhibit significant variability, with average loan amounts around \$15.13 million and incomes averaging approximately \$5.06 million. The standard deviations are high, suggesting a wide range of financial situations among the applicants. Furthermore, the average number of dependents per applicant is about 2.5, indicating moderate family responsibilities.

# The CIBIL scores show a broad distribution from 300 to 900, highlighting diverse credit profiles, with an average score around 600. Loan terms also vary greatly, averaging roughly 11 years, reflecting different loan agreements' durations. Asset values, including residential, commercial, and luxury assets, display substantial ranges, with some unexpected outliers, such as negative residential asset values, necessitating further investigation and cleaning for accurate analysis.

# Concerning potential skewness, both loan amounts and income appear to be positively skewed, as suggested by high means and standard deviations. This skewness indicates a concentration of values at the lower end, with a few significantly higher values pulling the averages upward. Similarly, the asset values exhibit signs of skewness, emphasizing the need for thorough data cleaning and visualization to ensure accurate insights.

# Moreover, the presence of negative values in the residential_assets_value is particularly concerning. Negative asset values are typically indicative of data entry errors or anomalies, requiring careful examination and correction to maintain data integrity. Similarly, the maximum age is 144 years, which is likely an outlier or data entry error, necessitating further investigation and cleaning to ensure accurate analysis. The maximum value of 123 years for the person_emp_length column is equally suspicious.

## Univariate Analysis Functions

## Univariate Analysis Functions

# Function to generate histograms for the numerical variables
def plot_histograms(data, columns, rows=None, cols=None):
    if not rows and not cols:
        # Calculate rows and cols based on the number of columns
        cols = min(len(columns), 4)  # Adjust maximum columns as needed
        rows = (len(columns) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))

    for ax, col in zip(axes.ravel(), columns):
        sns.histplot(data[col], bins=30, kde=True, ax=ax, stat="density")  
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide empty subplots
    for i in range(len(columns), rows * cols):
        axes.flatten()[i].set_visible(False)

    plt.tight_layout()
    plt.show()




# Function to generate boxplots for the numerical variables
def plot_boxplots(data, columns, rows=None, cols=None):
    if not rows and not cols:
        # Calculate rows and cols based on the number of columns
        cols = min(len(columns), 4)  # Adjust maximum columns as needed
        rows = (len(columns) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))

    for ax, col in zip(axes.ravel(), columns):
        sns.boxplot(y=data[col], ax=ax, color='steelblue')
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide empty subplots
    for i in range(len(columns), rows * cols):
        axes.flatten()[i].set_visible(False)

    plt.tight_layout()
    plt.show()



# Function to generate barplots for the categorical variables
def plot_barplots(data, columns, rows=None, cols=None):

    if not rows and not cols:
        # Calculate rows and cols based on the number of columns
        cols = min(len(columns), 4)  # Adjust maximum columns as needed
        rows = (len(columns) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))

    for ax, col in zip(axes.ravel(), columns):
        sns.countplot(x=col, data=data, ax=ax)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Hide empty subplots
    for i in range(len(columns), rows * cols):
        axes.flatten()[i].set_visible(False)

    plt.tight_layout()
    plt.show()

# Create lists of the different column types
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
target_col = ['loan_status']

plot_histograms(data1, numeric_cols + target_col)

plot_boxplots(data1, numeric_cols)

# Some takeaways from the above plots:
# - Many of the variables are right skewed, with very similar shapes. This makes sense as they can all be very dependent on the age of the applicant. Age, employment length and credit history length are logically going to be extremely correlated. These relationships will be further explored in the next section.
# - The income variable is extremely right skewed with almost everyone earning less than a million but some earning much more. While income is likely a very important variable in determining loan approval, it is likely that the model will have to be able to handle the right skewness of this variable. Winsoring will be considered.
# - The target variable, loan status, is imbalanced. This will need to be addressed in the modeling section.
# - The boxplots further reinforce one single extreme outlier in the age and employment length variables. They also highlight the the single extreme outlier in the income variable. These outliers will need to be addressed in the data cleaning section.

plot_barplots(data1, categorical_cols)

# Some takeaways from the above barplots:
# - The vast majority of applicants are either renters or mortgage holders. While this is not surprising, it is important to not that the model might not perform very well on applicants in the other two categories.
# - The loan intent variable is fairly evenly distributed. This is good as it means the model will have a good amount of data to learn from for each category.
# - The loan grade variable is also fairly evenly distributed. Since it is an ordinal scale, this will possibly be converted to a a numeric variable in the data cleaning section.
# - The default on file variable is extremely imbalanced. As expected, a very small percentage of applicants have a default on file. This will be addressed in the modeling section. Additionally, this variable will likely be converted to a binary variable in the data cleaning section.

# Bivariate Functions

# Function to generate pairplots for the numerical variables
def plot_pairplots(data, columns, target_col):
    sns.pairplot(data, vars=columns, hue=target_col, plot_kws={'alpha': 0.5})
    plt.show()

# Function to generate boxplots for the numerical variables grouped by the target variable
def plot_boxplots_grouped(data, columns, target_col, rows=None, cols=None):
    if not rows and not cols:
        # Calculate rows and cols based on the number of columns
        cols = min(len(columns), 4)  # Adjust maximum columns as needed
        rows = (len(columns) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))

    for ax, col in zip(axes.ravel(), columns):
        sns.boxplot(x=target_col, y=col, data=data, ax=ax, palette='viridis')
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide empty subplots
    for i in range(len(columns), rows * cols):
        axes.flatten()[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# Function to generate barplots for the categorical variables grouped by the target variable
def plot_barplots_grouped(data, columns, target_col, rows=None, cols=None):
    if not rows and not cols:
        # Calculate rows and cols based on the number of columns
        cols = min(len(columns), 4)  # Adjust maximum columns as needed
        rows = (len(columns) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))

    for ax, col in zip(axes.ravel(), columns):
        sns.countplot(x=col, hue=target_col, data=data, ax=ax)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Hide empty subplots
    for i in range(len(columns), rows * cols):
        axes.flatten()[i].set_visible(False)

    plt.tight_layout()
    plt.show()

# Function to generate correlation heatmap
def plot_correlation_heatmap(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()


plot_pairplots(data1, numeric_cols, target_col[0])

# The pairplots show that the numerical variables are not very correlated with the target variable. The loan interest rate and the loan percent income did have the clearest delineation between the two classes. These can be further verified using the grouped boxplots below. We see most of the plots look the same in the two groups besides the two mentioned. As observed earlier, the age of a person and the age of his credit history are extremely correlated, for obvious reasons.

plot_boxplots_grouped(data1, numeric_cols, target_col[0])

# Data Cleaning
# - Check for NAs and impute if necessary
# - Check cause of outliers and handle if necessary
# - Convert cb_person_default_on_file to binary
# - Convert loan_grade to numeric
# - One hot encode the remaining two categorical variables

# Check for NAs
print(data1.isna().sum())

# Impute missing values in numeric columns
imputer = IterativeImputer(max_iter=10, random_state=0)
numeric_data = data1[numeric_cols + target_col]
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Convert the imputed data back to a DataFrame
numeric_data_imputed = pd.DataFrame(numeric_data_imputed, columns=numeric_cols+target_col)

# Combine the imputed numeric data with the original categorical data
data_imputed = pd.concat([numeric_data_imputed, data1[categorical_cols]], axis=1)

# Verify the imputed data
print(data_imputed.info())

# Check cause of age and employment length outliers
# Filter the rows where either person_age or person_emp_length is above 100
outliers = data_imputed[(data_imputed['person_age'] > 100) | (data_imputed['person_emp_length'] > 100)]

# Print the outliers
print(outliers)

# The outliers in the person_age and person_emp_length columns are likely data entry errors. The ages of 123 and 144 are unrealistic, and the employment length of 123 years is also implausible. Since there are only a few such rows, they can simply be deleted from the dataset.

# Remove the outliers
data_cleaned = data_imputed.drop(outliers.index)

# Verify the cleaned data
print(data_cleaned.info())

# Convert cb_person_default_on_file to binary
data_cleaned['cb_person_default_on_file'] = data_cleaned['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

# Convert loan_grade to numeric and set as ordinal
data_cleaned['loan_grade'] = data_cleaned['loan_grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

# One-hot encode the remaining categorical variables
data_cleaned = pd.get_dummies(data_cleaned, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

# Verify the cleaned data
print(data_cleaned.info())


plot_correlation_heatmap(data_cleaned)

# The above correlation heatmap shows that the variables in this dataset do not seem to be very correlated with each other, albeit with some exceptions. However, the target variable, loan_status, does have some moderate correlation with some of the variables. The loan intent columns all seem to have very slight correaltion, if any, yet are adding significant dimensions to the dataset. 

# Modeling
# We will try two models: logistic regression and random forest. The logistic regression model is a simple model that is easy to interpret and can provide insights into the relationships between the variables and the target variable. The random forest model is a more complex model that can capture non-linear relationships and interactions between the variables, potentially improving the predictive performance.
# - Split the data into training and testing sets
# - Scale the data
# - Train the models
# - Evaluate the models

# Load Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Split the data into training and testing sets
X = data_cleaned.drop(target_col, axis=1)
y = data_cleaned[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train_scaled, y_train.values.ravel())

# Predict the target variable
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Evaluate the logistic regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test, y_pred_log_reg)
confusion_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)

print('Logistic Regression Model')
print('Accuracy:', accuracy_log_reg)
print('Precision:', precision_log_reg)
print('Recall:', recall_log_reg)
print('F1 Score:', f1_log_reg)
print('ROC AUC Score:', roc_auc_log_reg)
print('Confusion Matrix:')
print(confusion_matrix_log_reg)

# Train the random forest model
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train_scaled, y_train.values.ravel())

# Predict the target variable
y_pred_random_forest = random_forest.predict(X_test_scaled)

# Evaluate the random forest model
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
precision_random_forest = precision_score(y_test, y_pred_random_forest)
recall_random_forest = recall_score(y_test, y_pred_random_forest)
f1_random_forest = f1_score(y_test, y_pred_random_forest)
roc_auc_random_forest = roc_auc_score(y_test, y_pred_random_forest)
confusion_matrix_random_forest = confusion_matrix(y_test, y_pred_random_forest)

print('Random Forest Model')
print('Accuracy:', accuracy_random_forest)
print('Precision:', precision_random_forest)
print('Recall:', recall_random_forest)
print('F1 Score:', f1_random_forest)
print('ROC AUC Score:', roc_auc_random_forest)
print('Confusion Matrix:')
print(confusion_matrix_random_forest)

# Data 2
print(data2.info())

# The data set has 252000 rows and 13 columns.