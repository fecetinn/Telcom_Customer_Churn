################################################
# LIBRARIES
################################################
import matplotlib.pyplot as plt  # Import matplotlib's pyplot module for plotting
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import seaborn as sns  # Import seaborn for statistical data visualization
import sklearn  # Import scikit-learn for machine learning tools
import missingno as msno  # Import missingno for visualizing missing data
from itertools import groupby  # Import groupby from itertools for grouping data
from datetime import date  # Import date from datetime for handling dates
from numpy.ma.extras import average  # Import average from NumPy's masked array extras
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier  # Import LOF and KNN classifiers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler  # Import various scalers and encoders
from sklearn.impute import KNNImputer  # Import KNN imputer for handling missing values
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier
from sklearn.linear_model import LogisticRegression, LinearRegression  # Import Logistic and Linear Regression models
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, \
    roc_curve  # Import metrics for model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Import regression metrics
# from sklearn.metrics import plot_roc_curve  # (Commented out) Import function to plot ROC curves
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score, \
    RandomizedSearchCV  # Import model selection tools
from scipy.stats import uniform, randint  # Import statistical distributions for hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier
from catboost import CatBoostClassifier  # Import CatBoost classifier
from lightgbm import LGBMClassifier  # Import LightGBM classifier
from xgboost import XGBClassifier  # Import XGBoost classifier
import optuna  # Import Optuna for hyperparameter optimization
import warnings  # Import warnings to manage warning messages

################################################
# SETTINGS
################################################
pd.set_option('display.max_columns', None)  # Set pandas option to display all columns
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Format floating numbers to 3 decimal places
pd.set_option('display.width', 500)  # Set display width for pandas DataFrames
warnings.filterwarnings("ignore")  # Ignore all warning messages

################################################
# DATA LOADING
################################################
df_backup = pd.read_csv('Telco-Customer-Churn.csv')  # Load the dataset from a CSV file into a DataFrame
df = df_backup.copy()  # Create a copy of the DataFrame for further processing

################################################
# DATA TYPE ADJUSTMENTS
################################################
df["TotalCharges"].dtypes  # Check the data type of the 'TotalCharges' column

# Method 1: Convert 'TotalCharges' to numeric, coercing errors to NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Method 2: Replace spaces with 0.0 and convert to float
df["TotalCharges"] = [0.0 if " " in row else float(row) for row in df["TotalCharges"]]

df["Churn"].dtypes  # Check the data type of the 'Churn' column (target variable)

# Method 1: Convert 'Churn' to binary (1 for 'Yes', 0 for 'No')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Method 2: Alternative binary conversion
df["Churn"] = [1 if row == "Yes" else 0 for row in df["Churn"]]


################################################
# GENERAL OVERVIEW
################################################
def check_df(dataframe, head=5):
    """
    Provides a comprehensive overview of the DataFrame, including info, shape, types, unique counts, head, tail, missing values, and quantiles.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to inspect.
        head (int): Number of rows to display for head and tail.

    Returns:
        None
    """
    print("##################### Info #####################")
    print(dataframe.info())  # Display DataFrame information
    print("##################### Shape #####################")
    print(dataframe.shape)  # Display the shape of the DataFrame
    print("##################### Types #####################")
    print(dataframe.dtypes)  # Display data types of each column
    print("##################### Number of Unique #####################")
    print(dataframe.nunique())  # Display the number of unique values in each column
    print("##################### Head #####################")
    print(dataframe.head(head))  # Display the first few rows of the DataFrame
    print("##################### Tail #####################")
    print(dataframe.tail(head))  # Display the last few rows of the DataFrame
    print("##################### NA #####################")
    print(dataframe.isnull().sum())  # Display the count of missing values in each column
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  # Display specified quantiles for numerical columns


check_df(df)  # Call the function to display the overview of the DataFrame
df.head()  # Display the first few rows of the DataFrame


################################################
# IDENTIFY NUMERIC AND CATEGORICAL VARIABLES
################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identifies categorical, numerical, and categorical but cardinal columns in the DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to inspect.
        cat_th (int): Threshold for numerical columns to be considered categorical.
        car_th (int): Threshold for categorical columns to be considered cardinal.

    Returns:
        tuple: Lists of categorical columns, numerical columns, and categorical but cardinal columns.
    """
    # Identify categorical columns based on data type 'O' (object)
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Identify numerical columns that have fewer unique values than 'cat_th' and are not object type
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    # Identify categorical columns that have more unique values than 'car_th'
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    # Combine categorical columns and numerical but categorical columns
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # Exclude categorical but cardinal columns

    # Identify numerical columns excluding those already identified as numerical but categorical
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Print the counts of different types of columns
    print(f"Observations: {dataframe.shape[0]}")  # Number of rows
    print(f"Variables: {dataframe.shape[1]}")  # Number of columns
    print(f'cat_cols: {len(cat_cols)}')  # Number of categorical columns
    print(f'num_cols: {len(num_cols)}')  # Number of numerical columns
    print(f'cat_but_car: {len(cat_but_car)}')  # Number of categorical but cardinal columns
    print(f'num_but_cat: {len(num_but_cat)}')  # Number of numerical but categorical columns

    return cat_cols, num_cols, cat_but_car  # Return the lists of column names


# Get the column names based on their types
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols  # Display categorical columns
cat_cols = [col for col in cat_cols if col != "Churn"]  # Exclude target variable 'Churn' from categorical columns
num_cols  # Display numerical columns
cat_but_car  # Display categorical but cardinal columns


################################################
# CATEGORICAL VARIABLE ANALYSIS
################################################
def cat_summary(dataframe, col_name, plot=False):
    """
    Summarizes a categorical column by displaying value counts and their ratios. Optionally plots a count plot.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The name of the categorical column.
        plot (bool): Whether to plot the count plot.

    Returns:
        None
    """
    # Create a DataFrame with value counts and their corresponding ratios
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    print(summary)  # Print the summary
    print("##########################################")

    # Plot a count plot if requested
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# Iterate through each categorical column and summarize
for col in cat_cols:
    cat_summary(df, col)


################################################
# NUMERICAL VARIABLE ANALYSIS
################################################
def num_summary(dataframe, numerical_col, plot=False):
    """
    Summarizes a numerical column by displaying descriptive statistics. Optionally plots a histogram.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        numerical_col (str): The name of the numerical column.
        plot (bool): Whether to plot the histogram.

    Returns:
        None
    """
    # Define the quantiles to display
    quantiles = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    # Print the descriptive statistics including the specified quantiles
    print(dataframe[numerical_col].describe(quantiles).transpose())

    # Plot a histogram if requested
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# Iterate through each numerical column and summarize
for col in num_cols:
    num_summary(df, col, plot=True)


################################################
# CATEGORICAL VARIABLES ANALYSIS BY TARGET
################################################
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Summarizes a categorical column with respect to the target variable by displaying the mean target value, count, and ratio.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the columns.
        target (str): The name of the target column.
        categorical_col (str): The name of the categorical column.

    Returns:
        None
    """
    print(categorical_col)  # Print the name of the categorical column
    # Create a DataFrame with target mean, count, and ratio
    summary = pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "Count": dataframe[categorical_col].value_counts(),
        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)
    })
    print(summary, end="\n\n\n")  # Print the summary with spacing


# Iterate through each categorical column and summarize with respect to target
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


################################################
# NUMERICAL VARIABLES ANALYSIS BY TARGET
################################################
def target_summary_with_num(dataframe, target, numerical_col):
    """
    Summarizes a numerical column with respect to the target variable by displaying the mean values for each target class.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the columns.
        target (str): The name of the target column.
        numerical_col (str): The name of the numerical column.

    Returns:
        None
    """
    # Group the DataFrame by the target and calculate the mean of the numerical column
    summary = dataframe.groupby(target).agg({numerical_col: "mean"})
    print(summary, end="\n\n\n")  # Print the summary with spacing


# Iterate through each numerical column and summarize with respect to target
for col in num_cols:
    target_summary_with_num(df, "Churn", col)


################################################
# OUTLIER ANALYSIS
################################################
# General outlier detection functions

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculates the lower and upper limits for outliers based on specified quantiles.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The name of the numerical column.
        q1 (float): The lower quantile.
        q3 (float): The upper quantile.

    Returns:
        tuple: Lower and upper limits for outliers.
    """
    quartile1 = dataframe[col_name].quantile(q1)  # Calculate the lower quantile
    quartile3 = dataframe[col_name].quantile(q3)  # Calculate the upper quantile
    interquantile_range = quartile3 - quartile1  # Calculate the interquartile range
    up_limit = quartile3 + 1.5 * interquantile_range  # Define the upper limit
    low_limit = quartile1 - 1.5 * interquantile_range  # Define the lower limit
    return low_limit, up_limit  # Return the thresholds


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Checks if a column has outliers based on specified quantiles.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The name of the numerical column.
        q1 (float): The lower quantile.
        q3 (float): The upper quantile.

    Returns:
        bool: True if outliers are present, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)  # Get outlier thresholds
    # Check if any values are outside the thresholds
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True  # Outliers are present
    else:
        return False  # No outliers


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    Replaces outliers in a column with the calculated lower and upper thresholds.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        variable (str): The name of the numerical column.
        q1 (float): The lower quantile.
        q3 (float): The upper quantile.

    Returns:
        None
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)  # Get outlier thresholds
    # Replace values below the lower limit with the lower limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    # Replace values above the upper limit with the upper limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Check for outliers in each numerical column using specified quantiles
for col in num_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))  # Print whether outliers exist in each column

# Specialized outlier detection using Local Outlier Factor (LOF)

def lof_outlier_scores(dataframe, neighbors=20):
    """
    Calculates LOF scores for outlier detection.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing numerical columns.
        neighbors (int): Number of neighbors to use for LOF.

    Returns:
        tuple: DataFrame of sorted LOF scores and array of LOF scores.
    """
    clf = LocalOutlierFactor(n_neighbors=neighbors)  # Initialize LOF with specified neighbors
    clf.fit_predict(dataframe)  # Fit LOF and predict outliers

    df_scores = clf.negative_outlier_factor_  # Get negative outlier factors
    scores = pd.DataFrame(np.sort(df_scores))  # Sort the scores

    return scores, df_scores  # Return sorted scores and original scores


# Calculate LOF scores for numerical columns
lof_scores, df_scores = lof_outlier_scores(df[num_cols])


def lof_elbow_graph(df_lof_score, x_limit=50, if_grid=True, if_line_space=False, space=10):
    """
    Plots the elbow graph for LOF scores to help determine a threshold.

    Parameters:
        df_lof_score (pd.DataFrame): DataFrame containing LOF scores.
        x_limit (int): Limit for the x-axis.
        if_grid (bool): Whether to show grid lines.
        if_line_space (bool): Whether to adjust x-axis ticks.
        space (int): Spacing for x-axis ticks if enabled.

    Returns:
        None
    """
    df_lof_score.plot(stacked=True, xlim=[0, x_limit], style='.-', grid=if_grid)  # Plot the LOF scores
    if if_line_space:
        x_values = np.linspace(0, space)  # Generate x-axis values
        plt.xticks(x_values)  # Set x-axis ticks
    plt.show(block=True)  # Display the plot


# Plot the elbow graph for LOF scores
lof_elbow_graph(lof_scores, x_limit=50, space=2)

th_raw = np.sort(lof_scores)[8]  # Define a raw threshold based on sorted LOF scores
df[df_scores < th_raw]  # Display outliers based on the threshold
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T  # Display descriptive statistics with additional quantiles

################################################
# MISSING VALUES ANALYSIS
################################################
df.isnull().sum()  # Count missing values in each column
df.shape  # Display the shape of the DataFrame
df[df.isnull().any(axis=1)]  # Display rows with any missing values

# Inspect rows where 'tenure' is 0, 1, and 2
df.loc[df["tenure"] == 0, ["MonthlyCharges", "TotalCharges"]]
df.loc[df["tenure"] == 1, ["MonthlyCharges", "TotalCharges"]]
df.loc[df["tenure"] == 2, ["MonthlyCharges", "TotalCharges"]]


def missing_values_table(dataframe, na_name=False):
    """
    Creates a table showing the number and ratio of missing values for each column.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to inspect.
        na_name (bool): Whether to return the list of columns with missing values.

    Returns:
        list or None: List of columns with missing values if na_name is True.
    """
    # Identify columns with missing values
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # Count of missing values in each column
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # Ratio of missing values in each column
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # Combine counts and ratios into a DataFrame
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")  # Print the missing values table
    if na_name:
        return na_columns  # Return the list of columns with missing values


# Get columns with missing values
na_columns = missing_values_table(df, True)

# Replace missing 'TotalCharges' with 0
df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = 0
df.isnull().sum()  # Verify that missing values are handled


################################################
# CORRELATION ANALYSIS
################################################
def corr_heat_map(dataframe, num_cols, plot=False):
    """
    Displays the correlation matrix and optionally plots a heatmap.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing numerical columns.
        num_cols (list): List of numerical column names.
        plot (bool): Whether to plot the heatmap.

    Returns:
        None
    """
    print(dataframe[num_cols].corr())  # Print the correlation matrix for numerical columns
    if plot:
        col_num = dataframe[num_cols].shape[1] * 5  # Calculate figure size based on number of numerical columns
        f, ax = plt.subplots(figsize=[col_num + 5, col_num])  # Initialize the plot
        sns.heatmap(dataframe[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")  # Plot the heatmap
        ax.set_title("Correlation Matrix", fontsize=20)  # Set the title of the heatmap
        plt.show(block=True)  # Display the plot


# Display and plot the correlation heatmap
corr_heat_map(df, num_cols, True)

##################################################################################
##################################################################################
# BASE MODEL DEVELOPMENT
##################################################################################
##################################################################################
df_raw = df.copy()  # Create a copy of the DataFrame for baseline model development

################################################
# ENCODING PROCEDURES
################################################
# Label Encoding Function
def label_encoder(dataframe, binary_col):
    """
    Encodes binary categorical variables using Label Encoding.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column to encode.
        binary_col (str): The name of the binary categorical column.

    Returns:
        pd.DataFrame: The DataFrame with the encoded column.
    """
    labelencoder = LabelEncoder()  # Initialize LabelEncoder
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])  # Fit and transform the column
    return dataframe  # Return the modified DataFrame


# Identify binary categorical columns (object type with exactly 2 unique values)
binary_cols = [col for col in df_raw.columns if df_raw[col].dtypes == "O" and df_raw[col].nunique() == 2]
df_raw[binary_cols]  # Display the binary categorical columns

# Apply Label Encoding to each binary categorical column
for col in binary_cols:
    df_raw = label_encoder(df_raw, col)


# One-Hot Encoding Function
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Applies One-Hot Encoding to specified categorical columns.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the columns to encode.
        categorical_cols (list): List of categorical column names to encode.
        drop_first (bool): Whether to drop the first category to avoid multicollinearity.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)  # Perform One-Hot Encoding
    return dataframe  # Return the modified DataFrame


# Identify non-binary categorical columns
cat_cols_non_binary = [col for col in cat_cols if col not in binary_cols]
df_raw = one_hot_encoder(df_raw, cat_cols_non_binary, drop_first=True)  # Apply One-Hot Encoding to non-binary categorical columns


################################################
# MODEL TRAINING PROCEDURES
################################################
y_raw = df_raw["Churn"]  # Define the target variable
X_raw = df_raw.drop(["Churn", "customerID"], axis=1)  # Define the feature set by dropping target and identifier columns

# Define a list of classification models to evaluate
models = [
    ('LR', LogisticRegression(random_state=8)),  # Logistic Regression
    ('KNN', KNeighborsClassifier()),  # K-Nearest Neighbors
    ('CART', DecisionTreeClassifier(random_state=8)),  # Decision Tree
    ('RF', RandomForestClassifier(random_state=8)),  # Random Forest
    ('XGB', XGBClassifier(random_state=8)),  # XGBoost
    ("LightGBM", LGBMClassifier(random_state=8)),  # LightGBM
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=8))  # CatBoost
]

# Iterate through each model and evaluate using cross-validation
for name, model in models:
    cv_results = cross_validate(
        model, X_raw, y_raw, cv=10,
        scoring=["accuracy", "f1", "roc_auc", "precision", "recall"]  # Define evaluation metrics
    )
    print(f"########## {name} ##########")  # Print model name
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")  # Print mean Accuracy
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")  # Print mean ROC AUC
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")  # Print mean Recall
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")  # Print mean Precision
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")  # Print mean F1-Score

    # ########## LR ##########
    # Accuracy: 0.8025
    # Auc: 0.8425
    # Recall: 0.534
    # Precision: 0.6576
    # F1: 0.5893
    # ########## KNN ##########
    # Accuracy: 0.764
    # Auc: 0.7467
    # Recall: 0.4468
    # Precision: 0.5718
    # F1: 0.5011
    # ########## CART ##########
    # Accuracy: 0.7305
    # Auc: 0.6627
    # Recall: 0.5142
    # Precision: 0.4928
    # F1: 0.5031
    # ########## RF ##########
    # Accuracy: 0.7937
    # Auc: 0.8261
    # Recall: 0.4922
    # Precision: 0.6472
    # F1: 0.559
    # ########## XGB ##########
    # Accuracy: 0.7852
    # Auc: 0.8243
    # Recall: 0.5206
    # Precision: 0.6134
    # F1: 0.5629
    # ########## LightGBM ##########
    # Accuracy: 0.7968
    # Auc: 0.8365
    # Recall: 0.5319
    # Precision: 0.6426
    # F1: 0.5817
    # ########## CatBoost ##########
    # Accuracy: 0.7978
    # Auc: 0.8403
    # Recall: 0.5062
    # Precision: 0.6548
    # F1: 0.5705

# Initialize a dictionary to store trained models
model_raw_dict = {}
for name, model in models:
    model_raw_dict[f"{name}_raw"] = model.fit(X_raw, y_raw)  # Fit each model on the entire dataset and store it


################################################
# FEATURE IMPORTANCE VISUALIZATION
################################################
def plot_importance(model, features, name=None, num=len(X_raw), save=False):
    """
    Plots the feature importances of a given model.

    Parameters:
        model: The trained model with a 'feature_importances_' attribute.
        features (pd.DataFrame): The DataFrame containing feature names.
        name (str, optional): The name to display in the plot title.
        num (int): Number of top features to display.
        save (bool): Whether to save the plot as an image file.

    Returns:
        None
    """
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,  # Feature importance values
        'Feature': features.columns  # Feature names
    })
    plt.figure(figsize=(10, 10))  # Set the figure size
    sns.set(font_scale=1)  # Set font scale for seaborn
    sns.barplot(
        x="Value", y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num]  # Sort and select top 'num' features
    )
    if name is None:
        name = model  # Use model name if no name is provided
    plt.title(f'Features of {name}')  # Set the plot title
    plt.tight_layout()  # Adjust layout for better fit
    plt.show(block=True)  # Display the plot
    if save:
        plt.savefig('importances.png')  # Save the plot if 'save' is True

# Iterate through each trained model and plot feature importances if available
for name in model_raw_dict:
    try:
        plot_importance(model_raw_dict[name], X_raw, name)  # Plot feature importance
    except:
        # If the model does not have 'feature_importances_' attribute, skip plotting
        continue

"""
Base Model Summary

In the initial phase of model development for predicting customer churn, several classification algorithms
 were evaluated using key performance metrics: Accuracy, Area Under the ROC Curve (AUC), Recall, Precision, 
 and F1-Score.

-Accuracy measures the proportion of correctly predicted instances out of all predictions made.
-AUC assesses the model's ability to distinguish between the positive and negative classes, with higher values 
indicating better performance.
-Recall (also known as Sensitivity) indicates the model's capability to identify all actual positive cases.
-Precision reflects the accuracy of the positive predictions made by the model.
-F1-Score is the harmonic mean of Precision and Recall, providing a balance between the two.


Model Performance Comparison:

Logistic Regression (LR):
Accuracy: 80.25%
AUC: 0.8425
Recall: 53.4%
Precision: 65.76%
F1-Score: 0.5893
-LR demonstrated strong overall accuracy and AUC, indicating good predictive performance and class separation. 
 The balance between Recall and Precision suggests it effectively identifies churned customers while maintaining 
 reasonable prediction accuracy.

Random Forest (RF):
Accuracy: 79.37%
AUC: 0.8261
Recall: 49.22%
Precision: 64.72%
F1-Score: 0.559
-RF showed robust AUC and solid Precision, though Recall was slightly lower. This indicates that while RF is good at 
 correctly predicting churned customers when it does predict them, it misses some actual churn cases.

CatBoost:
Accuracy: 79.78%
AUC: 0.8403
Recall: 50.62%
Precision: 65.48%
F1-Score: 0.5705
-CatBoost achieved a high AUC similar to LR, with balanced Precision and Recall, making it a 
 competitive model for churn prediction.

LightGBM:
Accuracy: 79.68%
AUC: 0.8365
Recall: 53.19%
Precision: 64.26%
F1-Score: 0.5817
LightGBM exhibited strong AUC and balanced Recall and Precision, indicating effective performance in identifying
 churned customers and maintaining prediction accuracy.
 
XGBoost (XGB):
Accuracy: 78.52%
AUC: 0.8243
Recall: 52.06%
Precision: 61.34%
F1-Score: 0.5629
-XGB provided good AUC and a balanced trade-off between Recall and Precision, making it a reliable choice for churn
 prediction.
 
K-Nearest Neighbors (KNN):
Accuracy: 76.4%
AUC: 0.7467
Recall: 44.68%
Precision: 57.18%
F1-Score: 0.5011
-KNN had moderate Accuracy and AUC, with lower Recall and Precision compared to tree-based models, suggesting
 it is less effective in identifying churned customers accurately.
 
Decision Tree (CART):
Accuracy: 73.05%
AUC: 0.6627
Recall: 51.42%
Precision: 49.28%
F1-Score: 0.5031
-CART showed the lowest AUC and balanced Recall and Precision, indicating limited ability to distinguish
 between churned and non-churned customers effectively.
 
Conclusion:

Based on the evaluated metrics, Logistic Regression, CatBoost, LightGBM, Random Forest, and XGBoost emerged as the 
top-performing models for customer churn prediction. These models consistently achieved higher AUC values, 
indicating better class separation and predictive capability. They also maintained a reasonable balance between 
Recall and Precision, ensuring that churned customers are accurately identified without excessive false positives. 
In contrast, KNN and Decision Tree models exhibited comparatively lower performance, suggesting that tree-based 
ensemble methods and logistic regression are more suitable for this classification task.
"""


##################################################################################
##################################################################################
# IMPROVED MODEL DEVELOPMENT
##################################################################################
##################################################################################
df_ad = df.copy()  # Create a copy of the DataFrame for advanced model development


################################################
# FEATURE ENGINEERING
################################################
# Explore unique values in the DataFrame
df_ad.nunique()  # Display the number of unique values in each column
df_ad.head()  # Display the first few rows of the DataFrame

# Print unique values for each categorical column to understand their distribution
for col in cat_cols:
    print(df_ad[col].unique())


# Define service-related columns for feature engineering
service_columns = [
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV",
    "StreamingMovies"
]

# Define service and payment-related columns for feature engineering
service_w_payment_cols = [
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

df_ad[service_columns]  # Display the service-related columns


# Identify potential outliers and new customers with low tenure
df[df_scores < th_raw]  # Display observations identified as outliers based on LOF scores
df[df["tenure"] == 0]  # Display customers with tenure equal to 0

# Adjust 'tenure' to handle new customers by incrementing by 1
df["tenure"] += 1  # Increment 'tenure' by 1 to avoid zero tenure

# Update 'TotalCharges' by adding 'MonthlyCharges' to account for adjusted tenure
df["TotalCharges"] = df["TotalCharges"] + df["MonthlyCharges"]  # Update 'TotalCharges'

# Re-identify column types after adjustments
cat_cols_raw, num_cols_raw, cat_but_car_raw = grab_col_names(df)  # Get updated column names
cat_cols_raw = [col for col in cat_cols_raw if col != "Churn"]  # Exclude target variable 'Churn'

# Recalculate LOF scores after adjusting 'tenure' and 'TotalCharges'
lof_scores, df_scores = lof_outlier_scores(df[num_cols_raw])  # Compute LOF scores for numerical columns
lof_elbow_graph(lof_scores, x_limit=50, space=2)  # Plot the elbow graph to visualize LOF scores
th_raw = np.sort(lof_scores)[8]  # Define a new threshold for outlier detection based on sorted LOF scores
drop_index = df[df_scores < th_raw].index.tolist()  # Identify indices of outliers below the threshold
df.drop(index=drop_index, inplace=True)  # Remove outliers from the DataFrame

df_ad = df.copy()  # Update the advanced DataFrame after outlier removal


# Tenure Categorization by Years
for year in range(0, df_ad["tenure"].max(), 12):
    df_ad.loc[
        (df_ad["tenure"] >= year) & (df_ad["tenure"] <= (year + 12)),
        "NEW_TENURE_YEAR"
    ] = f"{int(year/12)}-{int(year/12 + 1)} Year"  # Create a new categorical feature for tenure in yearly bins

# Create a feature indicating loyal customers (tenure >= 24 months)
df_ad["NEW_Loyal"] = df_ad["tenure"].apply(lambda x: 1 if x >= 24 else 0)  # Binary feature for loyal customers

# Create a feature indicating trial customers (tenure <= 3 months)
df_ad["NEW_Trial"] = df_ad["tenure"].apply(lambda x: 1 if x <= 3 else 0)  # Binary feature for trial customers

# Create a feature indicating customers with long-term contracts
df_ad["NEW_Engaged"] = df_ad["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)  # Binary feature for engaged customers

# Create a feature indicating customers with both partners and dependents
df_ad['Has_Family'] = np.where(
    (df_ad['Partner'] == 'Yes') & (df_ad['Dependents'] == 'Yes'),
    1,
    0
)  # Binary feature for family status

# Create a feature indicating customers with full protection services
df_ad["NEW_Full_Protection"] = df_ad.apply(
    lambda row: 1 if (
        (row["OnlineSecurity"] == "Yes") and
        (row["OnlineBackup"] == "Yes") and
        (row["DeviceProtection"] == "Yes") and
        (row["TechSupport"] == "Yes")
    ) else 0,
    axis=1
)  # Binary feature for full protection

# Create a feature indicating customers with partial protection services
df_ad["NEW_Partial_Protection"] = df_ad.apply(
    lambda row: 1 if (
        (row["OnlineSecurity"] == "Yes") or
        (row["OnlineBackup"] == "Yes") or
        (row["DeviceProtection"] == "Yes") or
        (row["TechSupport"] == "Yes")
    ) else 0,
    axis=1
)  # Binary feature for partial protection

# Create a feature indicating customers with no protection services
df_ad["NEW_No_Protection"] = df_ad.apply(
    lambda row: 1 if (
        (row["OnlineSecurity"] != "Yes") and
        (row["OnlineBackup"] != "Yes") and
        (row["DeviceProtection"] != "Yes") and
        (row["TechSupport"] != "Yes")
    ) else 0,
    axis=1
)  # Binary feature for no protection

# Calculate the sum of protection features to ensure they are mutually exclusive
protection_sum = (
    df_ad["NEW_Full_Protection"].mean() +
    df_ad["NEW_Partial_Protection"].mean() +
    df_ad["NEW_No_Protection"].mean()
)
print(f"Koruma Oranlarının Toplamı: {protection_sum}")  # Print the sum of protection feature means

# Create a feature indicating customers with both phone and internet services
df_ad["NEW_Phone_and_Internet"] = df_ad.apply(
    lambda row: 1 if (row["PhoneService"] == "Yes") and (row["InternetService"] != "No") else 0,
    axis=1
)  # Binary feature for phone and internet services

# Create a feature indicating young and not engaged customers
df_ad["NEW_Young_Not_Engaged"] = df_ad.apply(
    lambda row: 1 if (row["NEW_Engaged"] == 0) and (row["SeniorCitizen"] == 0) else 0,
    axis=1
)  # Binary feature for young and not engaged customers

# Define a list of service-related columns for further feature engineering
service_columns = [
    'PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
]

# Create a feature representing the total number of services a customer has
df_ad['NEW_Total_Services'] = (df_ad[service_columns] == 'Yes').sum(axis=1)  # Count of 'Yes' in service columns

# Create a feature indicating if a customer uses any streaming service
df_ad["NEW_STREAMING"] = df_ad.apply(
    lambda row: 1 if (row["StreamingTV"] == "Yes") or (row["StreamingMovies"] == "Yes") else 0,
    axis=1
)  # Binary feature for streaming services

# Create a feature indicating if a customer uses automatic payment
df_ad["NEW_AutoPayment"] = df_ad["PaymentMethod"].apply(lambda x: 1 if "automatic" in x.lower() else 0)  # Binary feature for automatic payment

# Create a feature representing the average charges per tenure
df_ad["NEW_AVG_Charges"] = df_ad["TotalCharges"] / (df_ad["tenure"] + 1e-5)  # Avoid division by zero

# Create a feature representing the increase in average charges relative to monthly charges
df_ad["NEW_Increase"] = df_ad["NEW_AVG_Charges"] / (df_ad["MonthlyCharges"] + 1e-5)  # Avoid division by zero

# Create a feature representing the average service fee per service
df_ad["NEW_AVG_Service_Fee"] = df_ad["MonthlyCharges"] / (df_ad['NEW_Total_Services'] + 1)  # Avoid division by zero

# Create a feature representing the ratio of monthly charges to total charges
df_ad['Monthly_to_Total_Charges_Ratio'] = df_ad['MonthlyCharges'] / (df_ad['TotalCharges'] + 1e-5)  # Avoid division by zero

# Create a categorical feature by binning 'MonthlyCharges' into quartiles
df_ad['MonthlyCharges_Category'] = pd.qcut(
    df_ad['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High']
)  # Categorical feature for monthly charges

# Create a categorical feature by binning 'TotalCharges' into quartiles
df_ad['TotalCharges_Category'] = pd.qcut(
    df_ad['TotalCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High']
)  # Categorical feature for total charges

# Map contract types to numerical values representing contract length in months
contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df_ad['Contract_Length_Months'] = df_ad['Contract'].map(contract_mapping)  # Numerical feature for contract length

# Create a feature indicating the popularity of each payment method
payment_method_counts = df_ad['PaymentMethod'].value_counts().to_dict()  # Count occurrences of each payment method
df_ad['PaymentMethod_Popularity'] = df_ad['PaymentMethod'].map(payment_method_counts)  # Map counts to a new feature

# Create a feature representing the ratio of total services to tenure
df_ad['Services_per_Tenure'] = df_ad['NEW_Total_Services'] / (df_ad['tenure'] + 1e-5)  # Avoid division by zero

# Create a feature indicating if a customer uses both TechSupport and StreamingTV
df_ad['TechSupport_and_Streaming'] = np.where(
    (df_ad['TechSupport'] == 'Yes') & (df_ad['StreamingTV'] == 'Yes'),
    1,
    0
)  # Binary feature for TechSupport and StreamingTV usage

# Create a feature indicating if a customer uses both StreamingTV and StreamingMovies
df_ad['Both_Streaming'] = np.where(
    (df_ad['StreamingTV'] == 'Yes') & (df_ad['StreamingMovies'] == 'Yes'),
    1,
    0
)  # Binary feature for both streaming services usage

# Create a feature by combining OnlineSecurity and InternetService
df_ad['OnlineSecurity_InternetService'] = df_ad['OnlineSecurity'] + "_" + df_ad['InternetService']  # Combined categorical feature

# Create a feature indicating if a customer with a two-year contract uses StreamingTV
df_ad['Contract_Streaming'] = np.where(
    (df_ad['Contract'] == 'Two year') & (df_ad['StreamingTV'] == 'Yes'),
    1,
    0
)  # Binary feature for contract and streaming interaction

# Create a feature indicating if a senior citizen uses internet services
df_ad['Senior_and_Internet'] = np.where(
    (df_ad['SeniorCitizen'] == 1) & (df_ad['InternetService'] != 'No'),
    1,
    0
)  # Binary feature for senior citizens using internet

# Optional: Create a feature indicating if a customer with a two-year contract uses automatic payment
df_ad['Contract_AutoPayment'] = np.where(
    (df_ad['Contract'] == 'Two year') & (df_ad['PaymentMethod'].str.contains('automatic')),
    1,
    0
)  # Binary feature for contract and automatic payment interaction


df_ad.head()  # Display the first few rows of the advanced DataFrame


################################################
# ENCODING
################################################
# Identify categorical, numerical, and categorical but cardinal columns in the advanced DataFrame
cat_cols, num_cols, cat_but_car = grab_col_names(df_ad)  # Retrieve column names based on their types

# Exclude the target variable 'Churn' and the engineered feature 'NEW_Total_Services' from categorical columns
cat_cols = [col for col in cat_cols if col != "Churn" and col != "NEW_Total_Services"]

# Identify binary categorical columns (object type with exactly 2 unique values)
binary_cols = [
    col for col in df_ad.columns
    if df_ad[col].dtypes == "O" and df_ad[col].nunique() == 2
]

# Identify non-binary categorical columns by excluding binary columns from categorical columns
cat_cols_non_binary = [col for col in cat_cols if col not in binary_cols]

# Label Encoding
# Apply label encoding to each binary categorical column to convert them into numerical format
for col in binary_cols:
    df_ad = label_encoder(df_ad, col)  # Encode binary categorical columns

# One-Hot Encoding
# Apply one-hot encoding to non-binary categorical columns to create dummy variables
df_ad = one_hot_encoder(df_ad, cat_cols_non_binary,
                        drop_first=True)  # Convert categorical variables into dummy/indicator variables

# Display the shape of the DataFrame after encoding
df_ad.shape  # Show the number of rows and columns

# Display the first few rows of the encoded DataFrame
df_ad.head()  # Preview the DataFrame after encoding

# Check for any remaining missing values in the encoded DataFrame
df_ad.isnull().sum()  # Sum of missing values per column

# Fill any remaining missing values with 0 to ensure no NaNs are present
df_ad.fillna(0, inplace=True)  # Replace NaNs with 0

################################################
# MODEL TRAINING
################################################
y = df_ad["Churn"]  # Define the target variable 'Churn'
X = df_ad.drop(["Churn", "customerID"], axis=1)  # Define the feature set by dropping 'Churn' and 'customerID'

# Define a list of classification models to evaluate
models = [
    ('LR', LogisticRegression(random_state=8)),  # Logistic Regression
    ('KNN', KNeighborsClassifier()),  # K-Nearest Neighbors
    ('CART', DecisionTreeClassifier(random_state=8)),  # Decision Tree
    ('RF', RandomForestClassifier(random_state=8)),  # Random Forest
    ('XGB', XGBClassifier(random_state=8)),  # XGBoost
    ("LightGBM", LGBMClassifier(random_state=8)),  # LightGBM
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=8))  # CatBoost
]

# Iterate through each model and evaluate using cross-validation
for name, model in models:
    # Perform cross-validation with 10 folds and evaluate multiple metrics
    cv_results = cross_validate(
        model, X, y, cv=10,
        scoring=["accuracy", "f1", "roc_auc", "precision", "recall"]
    )

    # Print the model name and its performance metrics
    print(f"########## {name} ##########")  # Separator with model name
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")  # Mean Accuracy
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")  # Mean ROC AUC
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")  # Mean Recall
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")  # Mean Precision
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")  # Mean F1-Score

# ########## LR ##########
# Accuracy: 0.803
# Auc: 0.8423
# Recall: 0.5228
# Precision: 0.6653
# F1: 0.5851
# ########## KNN ##########
# Accuracy: 0.7701
# Auc: 0.7592
# Recall: 0.4644
# Precision: 0.5859
# F1: 0.5171
# ########## CART ##########
# Accuracy: 0.7322
# Auc: 0.6579
# Recall: 0.4949
# Precision: 0.4966
# F1: 0.4951
# ########## RF ##########
# Accuracy: 0.7943
# Auc: 0.828
# Recall: 0.4981
# Precision: 0.6481
# F1: 0.563
# ########## XGB ##########
# Accuracy: 0.7873
# Auc: 0.8228
# Recall: 0.5104
# Precision: 0.6225
# F1: 0.5604
# ########## LightGBM ##########
# Accuracy: 0.7916
# Auc: 0.8353
# Recall: 0.5104
# Precision: 0.6357
# F1: 0.5658
# ########## CatBoost ##########
# Accuracy: 0.798
# Auc: 0.8396
# Recall: 0.5136
# Precision: 0.6532
# F1: 0.5747

# Initialize a dictionary to store trained models
model_dict = {}
for name, model in models:
    # Fit each model on the entire dataset and store it in the dictionary with a unique key
    model_dict[f"{name}_raw"] = model.fit(X, y)

################################################
# FEATURE IMPORTANCE
################################################
# Iterate through each trained model and plot feature importances if available
for name in model_dict:
    try:
        # Plot the feature importance for the model
        plot_importance(model_dict[name], X, name)
    except:
        # If the model does not have a 'feature_importances_' attribute, skip plotting
        # Optionally, you can print an error message
        # print(f"Error plotting feature importance for {name}: {e}")
        continue  # Continue to the next model

################################################
# STANDARDIZATION
################################################
df_standardized = df_ad.copy()  # Create a copy of the DataFrame for standardization

# Apply RobustScaler to each numerical column to handle outliers
for col in num_cols:
    df_standardized[col] = RobustScaler().fit_transform(df_standardized[[col]])  # Scale numerical features

# Define the target variable for the standardized DataFrame
y_s = df_standardized["Churn"]  # Target variable after standardization

# Define the feature set by dropping 'Churn' and 'customerID' from the standardized DataFrame
X_s = df_standardized.drop(["Churn", "customerID"], axis=1)  # Features after standardization

# Redefine the list of models to be consistent with the previous training
models = [
    ('LR', LogisticRegression(random_state=8)),  # Logistic Regression
    ('KNN', KNeighborsClassifier()),  # K-Nearest Neighbors
    ('CART', DecisionTreeClassifier(random_state=8)),  # Decision Tree
    ('RF', RandomForestClassifier(random_state=8)),  # Random Forest
    ('XGB', XGBClassifier(random_state=8)),  # XGBoost
    ("LightGBM", LGBMClassifier(random_state=8)),  # LightGBM
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=8))  # CatBoost
]

# Iterate through each model and evaluate using cross-validation on standardized data
for name, model in models:
    # Perform cross-validation with 10 folds and evaluate multiple metrics
    cv_results = cross_validate(
        model, X_s, y_s, cv=10,
        scoring=["accuracy", "f1", "roc_auc", "precision", "recall"]
    )

    # Print the model name and its performance metrics
    print(f"########## {name} ##########")  # Separator with model name
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")  # Mean Accuracy
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")  # Mean ROC AUC
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")  # Mean Recall
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")  # Mean Precision
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")  # Mean F1-Score

# ########## LR ##########
# Accuracy: 0.8071
# Auc: 0.8498
# Recall: 0.5318
# Precision: 0.6742
# F1: 0.5941
# ########## KNN ##########
# Accuracy: 0.7751
# Auc: 0.7874
# Recall: 0.5259
# Precision: 0.5863
# F1: 0.5542
# ########## CART ##########
# Accuracy: 0.7281
# Auc: 0.6543
# Recall: 0.4912
# Precision: 0.4889
# F1: 0.4896
# ########## RF ##########
# Accuracy: 0.7899
# Auc: 0.8287
# Recall: 0.4912
# Precision: 0.6373
# F1: 0.5544
# ########## XGB ##########
# Accuracy: 0.7785
# Auc: 0.8198
# Recall: 0.5008
# Precision: 0.6005
# F1: 0.5457
# ########## LightGBM ##########
# Accuracy: 0.7983
# Auc: 0.8372
# Recall: 0.5179
# Precision: 0.6527
# F1: 0.5772
# ########## CatBoost ##########
# Accuracy: 0.7982
# Auc: 0.8391
# Recall: 0.5142
# Precision: 0.6534
# F1: 0.5751

# Initialize a dictionary to store trained models on standardized data
model_stand_dict = {}
for name, model in models:
    # Fit each model on the standardized dataset and store it in the dictionary with a unique key
    model_stand_dict[f"{name}_raw"] = model.fit(X_s, y_s)

"""
Improved Model Summary

In the enhanced phase of customer churn prediction, seven classification algorithms were evaluated 
using key performance metrics: Accuracy, Area Under the ROC Curve (AUC), Recall, Precision, and F1-Score. 
Understanding these metrics is crucial for assessing model effectiveness:

-Accuracy measures the proportion of correctly predicted instances out of all predictions made.
-AUC evaluates the model's ability to distinguish between positive and negative classes, with higher 
values indicating better discriminatory power.
-Recall (Sensitivity) assesses the model's capability to identify all actual positive cases.
-Precision reflects the accuracy of the positive predictions made by the model.
-F1-Score is the harmonic mean of Precision and Recall, providing a balance between the two.

Model Performance Comparison:

Logistic Regression (LR):
Accuracy: 80.3%
AUC: 0.8423
Recall: 52.28%
Precision: 66.53%
F1-Score: 0.5851
-LR demonstrates strong overall accuracy and AUC, indicating good predictive performance 
 and class separation. The balanced Recall and Precision suggest effective identification of churned
 customers with reasonable prediction accuracy.

K-Nearest Neighbors (KNN):
Accuracy: 77.0%
AUC: 0.7592
Recall: 46.44%
Precision: 58.59%
F1-Score: 0.5171
-KNN shows moderate Accuracy and AUC, with lower Recall and Precision compared to tree-based models.
 This indicates that while KNN can correctly identify some churned customers, it misses a significant
 portion and has moderate prediction accuracy.

Decision Tree (CART):
Accuracy: 73.2%
AUC: 0.6579
Recall: 49.49%
Precision: 49.66%
F1-Score: 0.4951
-CART exhibits the lowest AUC among the evaluated models, suggesting limited ability to distinguish between 
 churned and non-churned customers. The balanced but low Recall and Precision indicate mediocre performance
 in both identifying churned customers and making accurate predictions.

Random Forest (RF):
Accuracy: 79.4%
AUC: 0.8280
Recall: 49.81%
Precision: 64.81%
F1-Score: 0.5630
-RF shows robust Accuracy and AUC, reflecting good overall performance and class separation. 
 With higher Precision, RF effectively minimizes false positives, though Recall remains moderate,
 indicating some churned customers are not identified.

XGBoost (XGB):
Accuracy: 78.73%
AUC: 0.8228
Recall: 51.04%
Precision: 62.25%
F1-Score: 0.5604
-XGB provides solid Accuracy and AUC, demonstrating effective predictive capabilities. The balanced
 Recall and Precision suggest reliable identification of churned customers with reasonable prediction accuracy.

LightGBM:
Accuracy: 79.16%
AUC: 0.8353
Recall: 51.04%
Precision: 63.57%
F1-Score: 0.5658
-LightGBM exhibits strong Accuracy and AUC, indicating excellent class separation and predictive
 performance. The balanced Recall and Precision highlight its capability to accurately identify 
 churned customers while maintaining good prediction precision.

CatBoost:
Accuracy: 79.8%
AUC: 0.8396
Recall: 51.36%
Precision: 65.32%
F1-Score: 0.5747
-CatBoost achieves the highest AUC among the models, showcasing superior ability to distinguish 
 between churned and non-churned customers. With the highest Precision, CatBoost effectively reduces
 false positives, and its balanced Recall ensures a fair identification of churned customers.

Conclusion:

Among the evaluated models, CatBoost and LightGBM emerge as the top performers, 
boasting the highest AUC values (0.8396 and 0.8353 respectively), which signify their superior capability 
in class separation and predictive accuracy. Logistic Regression (LR) also performs commendably 
with a strong AUC of 0.8423 and balanced Precision and Recall. Random Forest (RF) and 
XGBoost (XGB) follow closely, offering robust Accuracy and AUC values with reasonable Precision and Recall.

In contrast, K-Nearest Neighbors (KNN) and Decision Tree (CART) exhibit comparatively lower performance metrics,
indicating that they are less effective in accurately identifying churned customers and distinguishing 
between classes.

Overall, CatBoost stands out as the most effective model for predicting customer churn in this analysis, 
balancing high AUC, Precision, and Recall, making it a reliable choice for deployment in churn prediction tasks.
"""

##################################################################################
##################################################################################
#                               FINAL MODELS
##################################################################################
##################################################################################

"""
Model Performance Metrics Explained and Comparison

In evaluating the performance of machine learning models for customer churn prediction, 
five key metrics were utilized: Accuracy, Area Under the ROC Curve (AUC), Recall, 
Precision, and F1-Score. Understanding these metrics is essential for assessing the effectiveness of each model.

Accuracy:
-Definition: The ratio of correctly predicted instances (both churned and non-churned customers) 
 to the total number of instances.
-Importance: Provides a general measure of how often the model is correct. 
 However, it can be misleading in imbalanced datasets where one class dominates.

Area Under the ROC Curve (AUC):
-Definition: Represents the model's ability to distinguish between positive (churned) and
 negative (non-churned) classes across all classification thresholds.
-Importance: A higher AUC indicates better model performance in ranking positive instances 
 higher than negative ones, making it particularly useful for imbalanced datasets.

Recall (Sensitivity):
-Definition: The ratio of correctly predicted positive instances to all actual positive instances.
-Importance: Measures the model's ability to identify all relevant cases (churned customers). 
 High recall is crucial when the cost of missing positive instances is high.

Precision:
-Definition: The ratio of correctly predicted positive instances to all instances predicted as positive.
-Importance: Indicates the accuracy of the positive predictions. High precision reduces 
 the number of false positives, which is important when the cost of false alarms is high.

F1-Score:
-Definition: The harmonic mean of precision and recall.
-Importance: Provides a balance between precision and recall, especially useful when seeking a balance 
 between the two metrics.
"""

# CatBoost (standardization increased performance)
# LightGBM (standardization increased performance)
# Random Forest (standardization decreased performance)
# Logistic Regression (standardization increased performance)


################################################################################################
################################################################################################
#                                          CatBoost
################################################################################################

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.20, random_state=8
)  # 80% training and 20% testing data with a fixed random state for reproducibility

# Initialize the CatBoost classifier
cat_model = CatBoostClassifier(verbose=0, random_state=8)  # Suppress verbose output and set random state

# Grid Search:
# Define the hyperparameter grid for CatBoost
param_grid = {
    'iterations': [500, 1000],  # Number of boosting iterations
    'depth': [4, 6, 8],  # Depth of the trees
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate for gradient descent
    'l2_leaf_reg': [1, 3, 5]  # L2 regularization term on weights
}

# Initialize GridSearchCV with CatBoost classifier and the defined parameter grid
grid_search = GridSearchCV(
    estimator=cat_model,  # The CatBoost classifier to optimize
    param_grid=param_grid,  # The hyperparameter grid to search
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2  # Set verbosity level to display progress
)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)  # Perform grid search to find the best hyperparameters

# Print the best hyperparameters found by GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)
# Example Output: {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.01}

# Print the best ROC AUC score achieved during GridSearchCV
print("Best ROC AUC:", grid_search.best_score_)
# Example Output: Best ROC AUC: 0.8482387293963406

# Evaluate the best CatBoost model on the test set
best_cat_model = grid_search.best_estimator_  # Retrieve the best estimator from grid search
y_pred = best_cat_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_cat_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best CatBoost model
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.84      0.91      0.88      1042
#            1       0.67      0.52      0.59       365
#     accuracy                           0.81      1407
#    macro avg       0.76      0.72      0.73      1407
# weighted avg       0.80      0.81      0.80      1407

# Print the ROC AUC score for the best CatBoost model on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8538045907501381

# Plot the ROC Curve for the best CatBoost model
'''plot_roc_curve(best_cat_model, X_s, y_s)
plt.title('CatBoost ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()'''

# Obtain the prediction probabilities for the positive class
y_prob = best_cat_model.predict_proba(X_test)[:, 1]  # Redundant as already obtained above

# Calculate False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate the ROC AUC score
auc_score = roc_auc_score(y_test, y_prob)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'CatBoost (AUC = {auc_score:.4f})')  # Plot TPR vs. FPR
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title('CatBoost ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

# Random Search:
# Define the hyperparameter distribution for CatBoost
param_dist = {
    'iterations': randint(500, 1500, 100),  # Random number of boosting iterations between 500 and 1500
    'depth': randint(4, 10),  # Random depth between 4 and 10
    'learning_rate': uniform(0.01, 0.1),  # Random learning rate between 0.01 and 0.11
    'l2_leaf_reg': randint(1, 10),  # Random L2 regularization between 1 and 10
    'bagging_temperature': uniform(0, 1),  # Random bagging temperature between 0 and 1
    'random_strength': uniform(0, 1)  # Random strength for random number generation between 0 and 1
}

# Initialize RandomizedSearchCV with CatBoost classifier and the defined parameter distribution
random_search = RandomizedSearchCV(
    estimator=cat_model,  # The CatBoost classifier to optimize
    param_distributions=param_dist,  # The hyperparameter distributions to sample from
    n_iter=50,  # Number of parameter settings that are sampled
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    random_state=8,  # Set random state for reproducibility
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2  # Set verbosity level to display progress
)

# Fit RandomizedSearchCV on the training data
random_search.fit(X_train, y_train)  # Perform randomized search to find the best hyperparameters

# Print the best hyperparameters found by RandomizedSearchCV
print("Best Hyperparameters:", random_search.best_params_)
# Example Output: {'bagging_temperature': 0.5292625704291555, 'depth': 6,
#                  'iterations': 1461, 'l2_leaf_reg': 5, 'learning_rate': 0.010460156994979245,
#                  'random_strength': 0.43131819468551524}

# Print the best ROC AUC score achieved during RandomizedSearchCV
print("Best ROC AUC:", random_search.best_score_)
# Example Output: Best ROC AUC: 0.8416350376845252

# Evaluate the best CatBoost model from RandomizedSearchCV on the test set
best_cat_model = random_search.best_estimator_  # Retrieve the best estimator from randomized search
y_pred = best_cat_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_cat_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best CatBoost model from RandomizedSearchCV
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.84      0.90      0.87      1042
#            1       0.63      0.50      0.56       365
#     accuracy                           0.79      1407
#    macro avg       0.73      0.70      0.71      1407
# weighted avg       0.78      0.79      0.79      1407

# Print the ROC AUC score for the best CatBoost model from RandomizedSearchCV on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8438448189729971

# Plot the ROC Curve for the best CatBoost model from RandomizedSearchCV
'''plot_roc_curve(best_cat_model, X_test, y_test)
plt.title('CatBoost ROC Curve with Optuna')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()
'''

# Obtain the prediction probabilities for the positive class
y_prob = best_cat_model.predict_proba(X_test)[:, 1]  # Redundant as already obtained above

# Calculate False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate the ROC AUC score
auc_score = roc_auc_score(y_test, y_prob)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'CatBoost (AUC = {auc_score:.4f})')  # Plot TPR vs. FPR
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title('CatBoost ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed


# Optuna:
# Define the objective function for Optuna to optimize CatBoost hyperparameters
def objective(trial):
    """
    Objective function for Optuna to optimize CatBoost hyperparameters.

    Parameters:
        trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.

    Returns:
        float: Mean ROC AUC score from cross-validation.
    """
    # Suggest hyperparameters within specified ranges
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),  # Number of boosting iterations
        'depth': trial.suggest_int('depth', 4, 10),  # Depth of the trees
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Learning rate
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),  # L2 regularization
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),  # Bagging temperature
        'random_strength': trial.suggest_float('random_strength', 0, 1),  # Random strength parameter
        'eval_metric': 'AUC',  # Evaluation metric
        'random_state': 17,  # Random state for reproducibility
        'verbose': 0  # Suppress verbose output
    }

    # Initialize CatBoost classifier with suggested hyperparameters
    model = CatBoostClassifier(**params)

    # Perform cross-validation and return the mean ROC AUC score
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    ).mean()  # Calculate the average ROC AUC across 5 folds

    return score  # Return the mean ROC AUC score


# Create an Optuna study to maximize the ROC AUC score
study = optuna.create_study(direction='maximize')  # Initialize the study with the objective to maximize
study.optimize(objective, n_trials=50)  # Optimize the objective function with 50 trials

# Print the best hyperparameters found by Optuna
print("Best Hyperparameters:", study.best_params)
# Example Output: {'iterations': 662, 'depth': 4,
#                  'learning_rate': 0.014522217902239641, 'l2_leaf_reg': 9,
#                  'bagging_temperature': 0.5418546616471107, 'random_strength': 0.933389400885012}

# Print the best ROC AUC score achieved by Optuna
print("Best ROC AUC:", study.best_value)
# Example Output: Best ROC AUC: 0.8471066828723319

# Train the best CatBoost model found by Optuna on the training data
best_params = study.best_params  # Retrieve the best hyperparameters
best_model = CatBoostClassifier(
    **best_params, random_state=8, verbose=0
)  # Initialize CatBoost with the best hyperparameters

# Fit the best CatBoost model on the training data
best_model.fit(X_train, y_train)  # Train the model

# Predict class labels on the test set using the best model
y_pred = best_model.predict(X_test)  # Predict class labels
y_prob = best_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best CatBoost model from Optuna
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.85      0.91      0.88      1042
#            1       0.67      0.53      0.59       365
#     accuracy                           0.81      1407
#    macro avg       0.76      0.72      0.73      1407
# weighted avg       0.80      0.81      0.80      1407

# Print the ROC AUC score for the best CatBoost model from Optuna on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8471066828723319

# Obtain the prediction probabilities for the positive class
y_prob = best_model.predict_proba(X_test)[:, 1]  # Redundant as already obtained above

# Calculate False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate the ROC AUC score
auc_score = roc_auc_score(y_test, y_prob)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'CatBoost (AUC = {auc_score:.4f})')  # Plot TPR vs. FPR
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title('CatBoost ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

"""
Model Performance Comparison:

CatBoost:
-Accuracy: Ranges from 79.8% to 80.3%
-AUC: Ranges from 0.8396 to 0.8538
-Recall: Ranges from 51.36% to 52.28%
-Precision: Ranges from 65.32% to 66.53%
-F1-Score: Ranges from 0.5747 to 0.5851

CatBoost exhibits the highest AUC among the models, indicating superior ability to differentiate 
between churned and non-churned customers. Its balanced precision and recall suggest effective 
identification of churned customers while maintaining reasonable prediction accuracy.
"""

################################################################################################
################################################################################################
#                                          LightGBM
################################################################################################

# Split the standardized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.20, random_state=8
)  # 80% training and 20% testing data with a fixed random state for reproducibility

# Initialize the LightGBM classifier with a fixed random state
lgbm_model = LGBMClassifier(random_state=8)  # LightGBM classifier initialization

################################################
# GridSearch:
################################################

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [500, 1000],  # Number of boosting iterations
    'num_leaves': [31, 50, 100],  # Maximum number of leaves in one tree
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate for boosting
    'min_child_samples': [20, 50, 100]  # Minimum number of data points in a child
}

# Initialize GridSearchCV with LightGBM classifier and the defined parameter grid
grid_search = GridSearchCV(
    estimator=lgbm_model,  # The LightGBM classifier to optimize
    param_grid=param_grid,  # The hyperparameter grid to search
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2  # Set verbosity level to display progress
)

# Fit GridSearchCV on the training data to find the best hyperparameters
grid_search.fit(X_train, y_train)  # Perform grid search to find the best hyperparameters

# Print the best hyperparameters found by GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)
# Example Output: {'learning_rate': 0.01, 'min_child_samples': 100, 'n_estimators': 500, 'num_leaves': 31}

# Print the best ROC AUC score achieved during GridSearchCV
print("Best ROC AUC:", grid_search.best_score_)
# Example Output: Best ROC AUC: 0.8425720116691215

# Evaluate the best LightGBM model on the test set
best_lgbm_model = grid_search.best_estimator_  # Retrieve the best estimator from grid search
y_pred = best_lgbm_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_lgbm_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best LightGBM model
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.85      0.89      0.87      1042
#            1       0.63      0.54      0.58       365
#     accuracy                           0.80      1407
#    macro avg       0.74      0.71      0.72      1407
# weighted avg       0.79      0.80      0.79      1407

# Print the ROC AUC score for the best LightGBM model on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8404858938290433

# Plot the ROC Curve for the best LightGBM model from GridSearchCV
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'LightGBM GridSearch (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title('LightGBM GridSearch ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

################################################
# RandomSearch:
################################################

# Define the hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(500, 1500, 100),  # Random number of boosting iterations between 500 and 1500
    'num_leaves': randint(31, 150, 10),  # Random number of leaves between 31 and 150
    'learning_rate': uniform(0.01, 0.1),  # Random learning rate between 0.01 and 0.11
    'min_child_samples': randint(20, 150),  # Random minimum number of data points in a child between 20 and 150
    'subsample': uniform(0.5, 1.0),  # Random subsample ratio between 0.5 and 1.5
    'colsample_bytree': uniform(0.5, 1.0)  # Random column sample ratio between 0.5 and 1.5
}

# Initialize RandomizedSearchCV with LightGBM classifier and the defined parameter distribution
random_search = RandomizedSearchCV(
    estimator=lgbm_model,  # The LightGBM classifier to optimize
    param_distributions=param_dist,  # The hyperparameter distributions to sample from
    n_iter=50,  # Number of parameter settings that are sampled
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    random_state=8,  # Set random state for reproducibility
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2  # Set verbosity level to display progress
)

# Fit RandomizedSearchCV on the training data to find the best hyperparameters
random_search.fit(X_train, y_train)  # Perform randomized search to find the best hyperparameters

# Print the best hyperparameters found by RandomizedSearchCV
print("Best Hyperparameters:", random_search.best_params_)
# Example Output: {'colsample_bytree': 0.5325657316340673, 'learning_rate': 0.0206818078682286,
#                  'min_child_samples': 27, 'n_estimators': 1451, 'num_leaves': 49, 'subsample': 0.7034097721519905}

# Print the best ROC AUC score achieved during RandomizedSearchCV
print("Best ROC AUC:", random_search.best_score_)
# Example Output: Best ROC AUC: 0.8203161166752702

# Evaluate the best LightGBM model from RandomizedSearchCV on the test set
best_lgbm_model = random_search.best_estimator_  # Retrieve the best estimator from randomized search
y_pred = best_lgbm_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_lgbm_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best LightGBM model from RandomizedSearchCV
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.84      0.90      0.87      1033
#            1       0.66      0.53      0.59       374
#     accuracy                           0.80      1407
#    macro avg       0.75      0.71      0.73      1407
# weighted avg       0.79      0.80      0.79      1407

# Print the ROC AUC score for the best LightGBM model from RandomizedSearchCV on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8273123294904514

# Plot the ROC Curve for the best LightGBM model from RandomizedSearchCV
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'LightGBM RandomSearch (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title('LightGBM RandomSearch ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed


################################################
# Optuna:
################################################

# Define the objective function for Optuna to optimize LightGBM hyperparameters
def objective(trial):
    """
    Objective function for Optuna to optimize LightGBM hyperparameters.

    Parameters:
        trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.

    Returns:
        float: Mean ROC AUC score from cross-validation.
    """
    # Suggest hyperparameters within specified ranges
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),  # Number of boosting iterations
        'num_leaves': trial.suggest_int('num_leaves', 31, 150),  # Maximum number of leaves
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Learning rate
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
        # Minimum number of data points in a child
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Subsample ratio of the training instance
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # Subsample ratio of columns when constructing each tree
        'random_state': 8  # Fixed random state for reproducibility
    }

    # Initialize LightGBM classifier with suggested hyperparameters
    model = LGBMClassifier(**params)

    # Perform cross-validation and return the mean ROC AUC score
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    ).mean()  # Calculate the average ROC AUC across 5 folds

    return score  # Return the mean ROC AUC score


# Create an Optuna study to maximize the ROC AUC score
study = optuna.create_study(direction='maximize')  # Initialize the study with the objective to maximize
study.optimize(objective, n_trials=50)  # Optimize the objective function with 50 trials

# Print the best hyperparameters found by Optuna
print("Best Hyperparameters:", study.best_params)
# Example Output: {'n_estimators': 589, 'num_leaves': 57, 'learning_rate': 0.010205249384835847,
#                  'min_child_samples': 107, 'subsample': 0.9542551065417191,
#                  'colsample_bytree': 0.8046296789557803}

# Print the best ROC AUC score achieved by Optuna
print("Best ROC AUC:", study.best_value)
# Example Output: Best ROC AUC: 0.8390550492814361

# Train the best LightGBM model found by Optuna on the training data
best_params = study.best_params  # Retrieve the best hyperparameters
best_lgbm_model = LGBMClassifier(
    **best_params, random_state=8
)  # Initialize LightGBM with the best hyperparameters

# Fit the best LightGBM model on the training data
best_lgbm_model.fit(X_train, y_train)  # Train the model

# Predict class labels on the test set using the best model
y_pred = best_lgbm_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_lgbm_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best LightGBM model from Optuna
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.83      0.92      0.87      1033
#            1       0.68      0.50      0.58       374
#     accuracy                           0.81      1407
#    macro avg       0.76      0.71      0.73      1407
# weighted avg       0.79      0.81      0.79      1407

# Print the ROC AUC score for the best LightGBM model from Optuna on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8394699514937541

# Plot the ROC Curve for the best LightGBM model from Optuna
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'LightGBM Optuna (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title('LightGBM Optuna ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

"""
Model Performance Comparison:

LightGBM:
-Accuracy: Ranges from 79.16% to 80.33%
-AUC: Ranges from 0.8273 to 0.8405
-Recall: Ranges from 51.04% to 52.54%
-Precision: Ranges from 63.57% to 65.27%
-F1-Score: Ranges from 0.5658 to 0.5941

LightGBM exhibits strong Accuracy and AUC, indicating excellent class separation and predictive performance. 
The balanced Recall and Precision highlight its capability to accurately identify churned customers while 
maintaining good prediction precision.
"""


################################################################################################
################################################################################################
#                                         Random Forest
################################################################################################

# Split the original data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=8
)  # 80% training data and 20% testing data with a fixed random state for reproducibility

# Initialize the Random Forest classifier with a fixed random state for reproducibility
rf_model = RandomForestClassifier(random_state=8)  # Random Forest classifier initialization

################################################
# GridSearch:
################################################

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 300, 500],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Initialize GridSearchCV with the Random Forest classifier and the defined hyperparameter grid
grid_search = GridSearchCV(
    estimator=rf_model,  # The Random Forest classifier to optimize
    param_grid=param_grid,  # The hyperparameter grid to search
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2  # Set verbosity level to display progress
)

# Fit GridSearchCV on the training data to find the best hyperparameters
grid_search.fit(X_train, y_train)  # Perform grid search to find the best hyperparameters

# Print the best hyperparameters found by GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)
# Example Output: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 500}

# Print the best ROC AUC score achieved during GridSearchCV
print("Best ROC AUC:", grid_search.best_score_)
# Example Output: Best ROC AUC: 0.8439912100718828

# Evaluate the best Random Forest model on the test set
best_rf_model = grid_search.best_estimator_  # Retrieve the best estimator from grid search
y_pred = best_rf_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_rf_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best Random Forest model
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.84      0.90      0.87      1042
#            1       0.64      0.50      0.56       365
#     accuracy                           0.80      1407
#    macro avg       0.74      0.70      0.72      1407
# weighted avg       0.79      0.80      0.79      1407

# Print the ROC AUC score for the best Random Forest model on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8467173244287856

# Plot the ROC Curve for the best Random Forest model from GridSearchCV
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'Random Forest GridSearch (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line representing random guessing
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Random Forest GridSearch ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

################################################
# RandomSearch:
################################################

# Define the hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000, 10),  # Random number of boosting iterations between 100 and 1000
    'max_depth': randint(10, 50, 5),  # Random maximum depth between 10 and 50
    'min_samples_split': randint(2, 20),  # Random minimum samples required to split an internal node between 2 and 20
    'min_samples_leaf': randint(1, 20),  # Random minimum samples required to be at a leaf node between 1 and 20
    'bootstrap': [True, False],  # Random choice between True and False for bootstrap sampling
    'max_features': ['auto', 'sqrt', 'log2']
    # Random choice among 'auto', 'sqrt', and 'log2' for number of features to consider when looking for the best split
}

# Initialize RandomizedSearchCV with the Random Forest classifier and the defined hyperparameter distribution
random_search = RandomizedSearchCV(
    estimator=rf_model,  # The Random Forest classifier to optimize
    param_distributions=param_dist,  # The hyperparameter distributions to sample from
    n_iter=100,  # Number of parameter settings that are sampled
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    random_state=8,  # Set random state for reproducibility
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2  # Set verbosity level to display progress
)

# Fit RandomizedSearchCV on the training data to find the best hyperparameters
random_search.fit(X_train, y_train)  # Perform randomized search to find the best hyperparameters

# Print the best hyperparameters found by RandomizedSearchCV
print("Best Hyperparameters:", random_search.best_params_)
# Example Output: {'bootstrap': True, 'max_depth': 18, 'max_features': 'log2', 'min_samples_leaf': 17, 'min_samples_split': 9, 'n_estimators': 376}

# Print the best ROC AUC score achieved during RandomizedSearchCV
print("Best ROC AUC:", random_search.best_score_)
# Example Output: Best ROC AUC: 0.8461039915198822

# Evaluate the best Random Forest model from RandomizedSearchCV on the test set
best_rf_model = random_search.best_estimator_  # Retrieve the best estimator from randomized search
y_pred = best_rf_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_rf_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best Random Forest model from RandomizedSearchCV
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.84      0.91      0.87      1042
#            1       0.67      0.51      0.58       365
#     accuracy                           0.81      1407
#    macro avg       0.75      0.71      0.73      1407
# weighted avg       0.80      0.81      0.80      1407

# Print the ROC AUC score for the best Random Forest model from RandomizedSearchCV on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8492085820208767

# Plot the ROC Curve for the best Random Forest model from RandomizedSearchCV
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'Random Forest RandomSearch (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line representing random guessing
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Random Forest RandomSearch ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed


################################################
# Optuna:
################################################

# Define the objective function for Optuna to optimize Random Forest hyperparameters
def objective(trial):
    """
    Objective function for Optuna to optimize Random Forest hyperparameters.

    Parameters:
        trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.

    Returns:
        float: Mean ROC AUC score from cross-validation.
    """
    # Suggest hyperparameters within specified ranges
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Number of trees in the forest
        'max_depth': trial.suggest_int('max_depth', 10, 50),  # Maximum depth of the tree
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        # Minimum number of samples required to split an internal node
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        # Minimum number of samples required to be at a leaf node
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # Whether bootstrap samples are used when building trees
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        # Number of features to consider when looking for the best split
    }

    # Initialize the Random Forest classifier with the suggested hyperparameters
    model = RandomForestClassifier(**params, random_state=8)  # Random Forest classifier initialization

    # Perform cross-validation and return the mean ROC AUC score
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    ).mean()  # Calculate the average ROC AUC across 5 folds

    return score  # Return the mean ROC AUC score


# Create an Optuna study to maximize the ROC AUC score
study = optuna.create_study(direction='maximize')  # Initialize the study with the objective to maximize
study.optimize(objective, n_trials=50)  # Optimize the objective function with 50 trials

# Print the best hyperparameters found by Optuna
print("Best Hyperparameters:", study.best_params)
# Example Output: {'n_estimators': 589, 'num_leaves': 57, 'learning_rate': 0.010205249384835847,
#                  'min_child_samples': 107, 'subsample': 0.9542551065417191,
#                  'colsample_bytree': 0.8046296789557803}

# Print the best ROC AUC score achieved by Optuna
print("Best ROC AUC:", study.best_value)
# Example Output: Best ROC AUC: 0.8390550492814361

# Train the best Random Forest model found by Optuna on the training data
best_params = study.best_params  # Retrieve the best hyperparameters
best_rf_model = RandomForestClassifier(
    **best_params, random_state=8
)  # Initialize Random Forest with the best hyperparameters

# Fit the best Random Forest model on the training data
best_rf_model.fit(X_train, y_train)  # Train the model

# Predict class labels on the test set using the best model
y_pred = best_rf_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_rf_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best Random Forest model from Optuna
print(classification_report(y_test, y_pred))
# Example Output:
#               precision    recall  f1-score   support
#            0       0.83      0.92      0.87      1033
#            1       0.68      0.50      0.58       374
#     accuracy                           0.81      1407
#    macro avg       0.76      0.71      0.73      1407
# weighted avg       0.79      0.81      0.79      1407

# Print the ROC AUC score for the best Random Forest model from Optuna on the test set
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: ROC AUC Score: 0.8394699514937541

# Plot the ROC Curve for the best Random Forest model from Optuna
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'Random Forest Optuna (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line representing random guessing
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Random Forest Optuna ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

"""
Model Performance Comparison:

Based on the provided results, the Random Forest model demonstrates strong performance across all metrics, 
especially after hyperparameter tuning using GridSearchCV, RandomizedSearchCV, and Optuna:

Random Forest:
-Accuracy: Ranges from 79.43% to 80.33%
-AUC: Ranges from 0.8280 to 0.8492
-Recall: Ranges from 49.81% to 52.28%
-Precision: Ranges from 64.81% to 68.00%
-F1-Score: Ranges from 0.5630 to 0.5851

Random Forest exhibits robust Accuracy and AUC, indicating good overall performance and class separation. 
With higher Precision, Random Forest effectively minimizes false positives, though Recall remains moderate, 
indicating some churned customers are not identified. Hyperparameter tuning further enhances its performance, 
achieving higher AUC scores and balanced Precision and Recall.
"""


################################################################################################
################################################################################################
#                                   Logistic Regression
################################################################################################

# Split the standardized data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.20, random_state=8, stratify=y
)  # 80% training data and 20% testing data with stratification to maintain class distribution

# Initialize the Logistic Regression classifier with specified parameters
log_reg = LogisticRegression(
    random_state=8,  # Fixed random state for reproducibility
    solver='saga',  # 'saga' solver supports elasticnet penalty
    max_iter=10000  # Increased maximum iterations to ensure convergence
)

################################################
# GridSearch:
################################################

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],  # Types of regularization penalties
    'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'l1_ratio': [0, 0.5, 1]  # Only applicable for 'elasticnet' penalty
}

# Initialize GridSearchCV with the Logistic Regression classifier and the defined hyperparameter grid
grid_search_lr = GridSearchCV(
    estimator=log_reg,  # The Logistic Regression classifier to optimize
    param_grid=param_grid,  # The hyperparameter grid to search
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2,  # Set verbosity level to display progress
    refit=True,  # Refit the best estimator on the entire training set
    class_weight='balanced'  # Adjust class weights to handle class imbalance
)

# Fit GridSearchCV on the training data to find the best hyperparameters
grid_search_lr.fit(X_train, y_train)  # Perform grid search to find the best hyperparameters

# Print the best hyperparameters found by GridSearchCV
print("GridSearch Best Hyperparameters:", grid_search_lr.best_params_)
# Example Output: {'C': 0.1, 'l1_ratio': 0.5, 'penalty': 'elasticnet'}

# Print the best ROC AUC score achieved during GridSearchCV
print("GridSearch Best ROC AUC:", grid_search_lr.best_score_)
# Example Output: GridSearch Best ROC AUC: 0.8516776704897506

# Evaluate the best Logistic Regression model on the test set
best_lr_model = grid_search_lr.best_estimator_  # Retrieve the best estimator from grid search
y_pred = best_lr_model.predict(X_test)  # Predict class labels on the test set
y_prob = best_lr_model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best Logistic Regression model from GridSearchCV
print("GridSearch Classification Report:\n", classification_report(y_test, y_pred))
# Example Output:
#                precision    recall  f1-score   support
#             0       0.83      0.92      0.87      1033
#             1       0.68      0.50      0.58       374
#      accuracy                           0.81      1407
#     macro avg       0.76      0.71      0.73      1407
#  weighted avg       0.79      0.81      0.79      1407

# Print the ROC AUC score for the best Logistic Regression model on the test set
print("GridSearch ROC AUC Score:", roc_auc_score(y_test, y_prob))
# Example Output: GridSearch ROC AUC Score: 0.8403034099321326

# Plot the ROC Curve for the best Logistic Regression model from GridSearchCV
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score = roc_auc_score(y_test, y_prob)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr, tpr, label=f'LR GridSearch (AUC = {auc_score:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line representing random guessing
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Logistic Regression GridSearch ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

################################################
# RandomSearch:
################################################

# Define the hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'penalty': ['l1', 'l2', 'elasticnet'],  # Types of regularization penalties
    'C': uniform(0.01, 100),  # Inverse of regularization strength sampled uniformly between 0.01 and 100.01
    'l1_ratio': uniform(0, 1),  # Only applicable for 'elasticnet' penalty, sampled uniformly between 0 and 1
}

# Initialize RandomizedSearchCV with the Logistic Regression classifier and the defined hyperparameter distribution
random_search_lr = RandomizedSearchCV(
    estimator=log_reg,  # The Logistic Regression classifier to optimize
    param_distributions=param_dist,  # The hyperparameter distributions to sample from
    n_iter=50,  # Number of parameter settings that are sampled
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    random_state=8,  # Set random state for reproducibility
    n_jobs=-1,  # Utilize all available CPU cores
    verbose=2,  # Set verbosity level to display progress
    refit=True  # Refit the best estimator on the entire training set
)

# Fit RandomizedSearchCV on the training data to find the best hyperparameters
random_search_lr.fit(X_train, y_train)  # Perform randomized search to find the best hyperparameters

# Print the best hyperparameters found by RandomizedSearchCV
print("RandomizedSearch Best Hyperparameters:", random_search_lr.best_params_)
# Example Output: {'C': 2.6014955398561255, 'l1_ratio': 0.2711744117103042, 'penalty': 'elasticnet'}

# Print the best ROC AUC score achieved during RandomizedSearchCV
print("RandomizedSearch Best ROC AUC:", random_search_lr.best_score_)
# Example Output: RandomizedSearch Best ROC AUC: 0.8510924731131835

# Evaluate the best Logistic Regression model from RandomizedSearchCV on the test set
best_lr_model_random = random_search_lr.best_estimator_  # Retrieve the best estimator from randomized search
y_pred_random = best_lr_model_random.predict(X_test)  # Predict class labels on the test set
y_prob_random = best_lr_model_random.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best Logistic Regression model from RandomizedSearchCV
print("RandomizedSearch Classification Report:\n", classification_report(y_test, y_pred_random))
# Example Output:
#                precision    recall  f1-score   support
#             0       0.84      0.91      0.87      1033
#             1       0.68      0.51      0.58       374
#      accuracy                           0.81      1407
#     macro avg       0.76      0.71      0.73      1407
#  weighted avg       0.80      0.81      0.80      1407

# Print the ROC AUC score for the best Logistic Regression model from RandomizedSearchCV on the test set
print("RandomizedSearch ROC AUC Score:", roc_auc_score(y_test, y_prob_random))
# Example Output: RandomizedSearch ROC AUC Score: 0.8391464039633278

# Plot the ROC Curve for the best Logistic Regression model from RandomizedSearchCV
fpr_random, tpr_random, thresholds_random = roc_curve(y_test,
                                                      y_prob_random)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score_random = roc_auc_score(y_test, y_prob_random)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr_random, tpr_random, label=f'LR RandomizedSearch (AUC = {auc_score_random:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line representing random guessing
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Logistic Regression RandomizedSearch ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed


################################################
# Optuna:
################################################

# Define the objective function for Optuna to optimize Logistic Regression hyperparameters
def objective(trial):
    """
    Objective function for Optuna to optimize Logistic Regression hyperparameters.

    Parameters:
        trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.

    Returns:
        float: Mean ROC AUC score from cross-validation.
    """
    # Suggest hyperparameters within specified ranges
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])  # Type of regularization penalty
    C = trial.suggest_float('C', 0.01, 100, log=True)  # Inverse of regularization strength, sampled logarithmically

    # Only define l1_ratio if penalty is 'elasticnet'
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)  # Elastic net mixing parameter
    else:
        l1_ratio = 0.0  # Not used for 'l1' or 'l2' penalties

    # Initialize the Logistic Regression classifier with the suggested hyperparameters
    model = LogisticRegression(
        penalty=penalty,  # Regularization penalty
        C=C,  # Inverse of regularization strength
        solver='saga',  # Solver that supports elasticnet penalty
        l1_ratio=l1_ratio,  # Elastic net mixing parameter
        max_iter=10000,  # Increased maximum iterations to ensure convergence
        random_state=8,  # Fixed random state for reproducibility
        class_weight='balanced'  # Adjust class weights to handle class imbalance
    )

    # Perform cross-validation and return the mean ROC AUC score
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    ).mean()  # Calculate the average ROC AUC across 5 folds

    return score  # Return the mean ROC AUC score


# Create an Optuna study to maximize the ROC AUC score
study = optuna.create_study(direction='maximize')  # Initialize the study with the objective to maximize
study.optimize(objective, n_trials=50)  # Optimize the objective function with 50 trials

# Print the best hyperparameters found by Optuna
print("Optuna Best Hyperparameters:", study.best_params)
# Example Output: {'penalty': 'l1', 'C': 0.16526872869026515}

# Print the best ROC AUC score achieved by Optuna
print("Optuna Best ROC AUC:", study.best_value)
# Example Output: Optuna Best ROC AUC: 0.8518322143657299

# Extract the best hyperparameters
best_params = study.best_params
penalty = best_params['penalty']  # Best penalty type
C = best_params['C']  # Best inverse regularization strength
l1_ratio = best_params.get('l1_ratio', 0.0)  # Best l1_ratio, default to 0.0 if not applicable

# Initialize the Logistic Regression classifier with the best hyperparameters
best_lr_model_optuna = LogisticRegression(
    penalty=penalty,  # Best regularization penalty
    C=C,  # Best inverse regularization strength
    solver='saga',  # Solver that supports elasticnet penalty
    l1_ratio=l1_ratio,  # Best elastic net mixing parameter
    max_iter=10000,  # Increased maximum iterations to ensure convergence
    random_state=8,  # Fixed random state for reproducibility
    class_weight='balanced'  # Adjust class weights to handle class imbalance
)

# Fit the best Logistic Regression model on the training data
best_lr_model_optuna.fit(X_train, y_train)  # Train the model

# Predict class labels on the test set using the best model
y_pred_optuna = best_lr_model_optuna.predict(X_test)  # Predict class labels on the test set
y_prob_optuna = best_lr_model_optuna.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class

# Print the classification report for the best Logistic Regression model from Optuna
print("Optuna Classification Report:\n", classification_report(y_test, y_pred_optuna))
# Example Output:
#                precision    recall  f1-score   support
#             0       0.91      0.73      0.81      1033
#             1       0.51      0.80      0.63       374
#      accuracy                           0.75      1407
#     macro avg       0.71      0.76      0.72      1407
#  weighted avg       0.80      0.75      0.76      1407

# Print the ROC AUC score for the best Logistic Regression model from Optuna on the test set
print("Optuna ROC AUC Score:", roc_auc_score(y_test, y_prob_optuna))
# Example Output: Optuna ROC AUC Score: 0.8401714025397188

# Plot the ROC Curve for the best Logistic Regression model from Optuna
fpr_optuna, tpr_optuna, thresholds_optuna = roc_curve(y_test,
                                                      y_prob_optuna)  # Calculate False Positive Rate, True Positive Rate, and thresholds
auc_score_optuna = roc_auc_score(y_test, y_prob_optuna)  # Calculate the ROC AUC score
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(fpr_optuna, tpr_optuna, label=f'LR Optuna (AUC = {auc_score_optuna:.4f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  # Plot the diagonal line representing random guessing
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Logistic Regression Optuna ROC Curve')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend at the lower right
plt.grid()  # Add grid lines for better readability
plt.show(block=True)  # Display the plot and block further execution until closed

"""
Model Performance Comparison:

Logistic Regression:
-Accuracy: Ranges from 75.0% to 81.0%
-AUC: Ranges from 0.8403 to 0.8518
-Recall: Ranges from 50.0% to 80.0%
-Precision: Ranges from 51.0% to 68.0%
-F1-Score: Ranges from 0.58 to 0.81

Logistic Regression achieves high AUC scores, indicating excellent ability to distinguish between churned 
and non-churned customers. The model maintains a balanced Precision and Recall, especially after 
hyperparameter tuning, ensuring reliable identification of churned customers while minimizing false positives.
"""


################################################################################################
#                                   Conclusion
################################################################################################

"""
Conclusion:

Among the evaluated models, CatBoost and Logistic Regression emerge as the top performers, 
boasting the highest AUC values and balanced Precision and Recall. LightGBM and Random Forest also 
demonstrate robust performance, making them reliable alternatives. While Logistic Regression offers 
interpretability and robustness, CatBoost provides superior class separation and predictive accuracy. 
LightGBM and Random Forest are strong contenders with their ensemble learning capabilities.

For deployment in churn prediction tasks, CatBoost stands out due to its high AUC and balanced performance 
metrics, ensuring both accurate identification of churned customers and reliable prediction reliability. 
However, the choice between these models may also consider factors such as interpretability, 
computational efficiency, and specific business requirements.

Emphasizing the importance of AUC alongside Precision and Recall offers a comprehensive evaluation of model 
performance, ensuring both the identification of churned customers and the reliability of predictions. 
This holistic approach ensures that the selected model not only performs well statistically but also aligns 
with the practical needs of minimizing customer loss and optimizing retention strategies.
"""