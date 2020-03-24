# Collection of functions useful for analysing TeXas InPateint Dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

import seaborn as sns
sns.set_style("darkgrid")

from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)  

import glob, os

DEBUG = False
SEED = 42

def eda_categorical(df, feature, target, max_categories=20, labels=None, header=True, brief=False):
    
    print("\n")
    if header: display(Markdown("#### %s" % feature))
    
    if df[feature].nunique()>max_categories:
        print("Warning: number of columns (%s) in feature (%s) is too large (>%s)") % (df[feature].nunique(), feature, max_categories)
        return
        
    # 1. Distribution table
    display(Markdown("**Distribution**"))
    
    if labels: 
        display(df[feature].map(labels).value_counts(dropna=False))
    else: 
        display(df[feature].value_counts(dropna=False))
    
    # 2. Count plot
    display(Markdown("**Count Plots**"))
    
    fig, ax = plt.subplots(figsize=(9,4), nrows=1, ncols=2)
    
    # left plot - freq of each category
    df_countplot = df.groupby(feature).size().sort_values()
    df_countplot.plot(kind='barh', ax=ax[0])

    # right plot - target breakdown within each category
    df_ft_countplot = pd.crosstab(df[feature], df[target], normalize='index')
    df_ft_countplot["total"] = df_countplot
    df_ft_countplot.sort_values("total", inplace=True)
    df_ft_countplot.drop(columns="total", inplace=True)
    df_ft_countplot.plot(kind='barh', stacked=True, ax=ax[1])

    ax[0].set_title=("Count plot of %s" % feature)
    ax[1].set_title=("Breakdown of %s" % target)
    fig.suptitle("Impact of feature '%s' on target '%s'" % (feature, target), fontsize="large")
    plt.show()
    
    # 3. Goodness of fit
    display(Markdown("**Chi-Sq Goodness of Fit**"))
    df_ft_countplot = pd.crosstab(df[feature], df[target])
    result = stats.chi2_contingency(df_ft_countplot)
    print('Chi-Square statistic %.4e (p=%.4e, dof=%d)' % result[0:3])
    
def encode_labels(data):
    return {line[0]: "(%s) %s" % (line[0], line[1:].strip()) for line in data.split("\n") if len(line) > 0}