import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def summary_statistics(df):
    '''
    Args:
        df: Pandas dataframe
        
    Returns:
        summary: Pandas DataFrame with summary statistics
    '''
    df.replace('', np.nan, inplace=True)
    summary = pd.DataFrame({
        'Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique(),
        'Min': df.min(numeric_only=True),
        'Max': df.max(numeric_only=True),
        'Mean': df.mean(numeric_only=True),
        'Median': df.median(numeric_only=True),
        'Std Dev': df.std(numeric_only=True)
    })
    
    return summary


def yearly_bar_plot(yearly_counts, x_label, y_label, title, figsize):
    '''
    Args:
        yearly_counts: frequency of each unique year present
        x_label: x axis label
        y_labe: y axis label
        title: title of the plot
        
    Returns:
        barplot
    '''
    plt.figure(figsize=figsize)
    sns.barplot(x=yearly_counts.index, y=yearly_counts.values, palette = 'Spectral')
    plt.xlabel(x_label,  fontsize = 12)
    plt.ylabel(y_label,  fontsize = 12)
    plt.title(title, fontsize = 18)
    plt.xticks(rotation=45)
    plt.show()


def line_plot(yearly_counts, x_label, y_label, title):
    '''
    Args:
        yearly_counts: frequency of each unique year present
        x_label: x axis label
        y_labe: y axis label
        title: title of the plot
        
    Returns:
        lineplot
    '''
    plt.figure(figsize=(8,4))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, linewidth=2.5, color = 'blue')
    plt.xlabel(x_label,  fontsize = 12)
    plt.ylabel(y_label,  fontsize = 12)
    plt.title(title, fontsize = 15)
    plt.show()


def dollar_format(total_amount):
    '''
    Args:
        total_amount: pandas DataFrame with total amount of dollar for each country
        
    Returns:
        formatted_amounts: total amount of dollar for each country in million/billion
    '''
    formatted_amounts = {
        country: f"{amount / 1000000:.2f} million" if amount < 1000000000 else f"{amount / 1000000000:.2f} billion"
    for country, amount in total_amount.items()
    }
    return formatted_amounts
