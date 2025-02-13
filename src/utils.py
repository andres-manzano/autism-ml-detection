import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

sns.set_theme(context='notebook', style=plt.style.use('dark_background'))

def classify_variables(data: pd.DataFrame) -> tuple:
    """
        Classifies the variables in a DataFrame into different types based on their characteristics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame containing the data to be classified.
        
        Returns:
        --------
        tuple of lists
            A tuple containing five lists of variable names, in the following order:
            1. Continuous variables
            2. Discrete variables
            3. Categorical variables
            4. Temporal variables
            5. Possible mixed variables
    """
    numericals = list(data.select_dtypes(include = ['number']).columns)
    # numericals = list(data.select_dtypes(include = [np.int32, np.int64, np.float32, np.float64]).columns)
    discretes = [col for col in data[numericals] if len(data[numericals][col].unique()) < 20]
    continuous = [col for col in data[numericals] if col not in discretes]
    categoricals = list(data.select_dtypes(include = ['category', 'object', 'bool']).columns)
    temporary = list(data.select_dtypes(include = ['datetime', 'timedelta']).columns)
    possible_mixed = [col for col in data[categoricals] if data[col].apply(lambda x: re.search(r'[a-zA-Z]', str(x)) and re.search(r'\d', str(x))).any()]
    
    print('\tType of Variables')
    print(f'There are {len(continuous)} continous variables')
    print(f'There are {len(discretes)} discrete variables')
    print(f'There are {len(categoricals)} categorical variables')
    print(f'There are {len(temporary)} temporary variables')
    print(f'There are {len(possible_mixed)} possible mixed variables')
        
    return tuple((continuous, discretes, categoricals, temporary, possible_mixed))

def nan_values(data: pd.DataFrame, variable_types: list, variables = list):
    """
    Prints the percentage of missing (NaN) values for each variable in a DataFrame, grouped by variable type.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data to be analyzed.
    variable_types : list
        A list of strings representing different categories or types of variables (e.g., 'Continuous', 'Categorical').
    variables : list of lists
        A list of lists, where each sublist contains the names of variables corresponding to the respective category
        in `variable_types`.
        
    Returns:
    --------
        The function prints out the percentage of NaN values for each variable in each category, if any NaN values are present.
    """
    for var_type, var_list in zip(variable_types, variables):
        print(f'\t{var_type} Variables')
        for var in var_list:
            if data[var].isnull().sum() > 0:
                print(f'{var}: {data[var].isnull().mean()*100:.2f}%')
        print('\n')

def plot_missing_value_distribution(data: pd.DataFrame, variable_types: list, variables = list) -> any:
    """
    Plots the distribution of missing values (NaN) for different types of variables in a DataFrame.
    
    Parameters:
    data : pd.DataFrame
        The DataFrame containing the data to be analyzed.
    variable_types : list
        A list of strings representing different categories or types of variables (e.g., 'Continuous', 'Categorical').
    variables : list of lists
        A list of lists, where each sublist contains the names of variables corresponding to the respective category
        in `variable_types`.
    
    Returns:
    --------
        Bar plots showing the percentage of missing values for variables with NaN values, grouped by type.
    """
    for var_type, var_list in zip(variable_types, variables):
        vars_with_nan = [var for var in var_list if data[var].isnull().sum() > 0]
        if len(vars_with_nan) > 0:
            plt.figure(figsize=(14, 7))
            data[vars_with_nan].isnull().mean().sort_values(ascending=False).plot.bar(color='blue', width=0.5, edgecolor='ghostwhite', lw=0.8)
            plt.title(f'{var_type} Variables with NaN values')
            plt.xlabel(f'{var_type} Variables', fontsize=12)
            plt.xticks(fontsize=10, rotation=35)
            plt.axhline(0.05, color='crimson', ls='dashed', lw=1.5, label='5% Missing Values')
            plt.ylabel(f'Percentage of Missing Values', fontsize=12)
            plt.yticks(fontsize=10)
            plt.ylim(0, 1)
            plt.grid(color='white', linestyle='-', linewidth=0.25)
            plt.legend()
            plt.tight_layout()
        else:
            print(f'There are no NaN values in the {var_type} variables')

def cramers_v(x, y):
    # Create a contingency table
    contingency_table = pd.crosstab(x, y)
    chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    phi2 = chi2_statistic / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    k_corr = k - (k - 1) * (k - 2) / (n - 1)
    r_corr = r - (r - 1) * (r - 2) / (n - 1)
    v = np.sqrt(phi2corr / min(k_corr - 1, r_corr - 1))
    
    return v

def cnf_matrix(y_true: pd.DataFrame, y_pred: pd.Series, threshold=0.5):
    sns.set_theme(context='notebook', style=plt.style.use('dark_background'))
    y_true = y_true.to_numpy().reshape(-1)
    y_pred = np.where(y_pred > threshold, 1, 0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(font_scale=1.2)
    sns.heatmap([[tp, fp], [fn, tn]], annot=True, cmap='magma', fmt='g', square=True, linewidths=1,
                yticklabels=['True Positive', ''], 
                xticklabels=['False Negative', 'True Negative'], ax=ax)
    
    ax.set_title(f'Confusion Matrix\n', fontsize=14)
    plt.tight_layout()