# Collection of functions useful for analysing TeXas InPateint Dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from zipfile import ZipFile

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


def clean_categories(df):
    
    # TYPE_OF_ADMISSION 
    feature = "TYPE_OF_ADMISSION"
    df[feature].fillna("9", inplace=True)
    df.loc[df[feature] == "`", feature] = "9"
    print("Feture %s -> %s" % (feature, df[feature].unique()))

    # SOURCE_OF_ADMISSION
    feature = "SOURCE_OF_ADMISSION"
    df[feature].fillna("9", inplace=True)
    df.loc[df[feature].isin(["`", "3", "NaN"]), feature] = "9"
    print("Feture %s -> %s" % (feature, df[feature].unique()))

    # PAT_STATE
    feature = "PAT_STATE"
    df[feature].fillna("XX", inplace=True)
    df.loc[df[feature].isin(["`", "FC", "AR", "OK", "LA", "NM"]), feature] = "ZZ"
    df.loc[df[feature].isin(["`", "FC"]), feature] = "XX"
    print("Feture %s -> %s" % (feature, df[feature].unique()))

    # SEX_CODE
    feature = "SEX_CODE"
    df[feature].fillna("U", inplace=True)
    df.loc[df[feature].isin(["NaN"]), feature] = "U"
    print("Feture %s -> %s" % (feature, df[feature].unique()))

    # RACE
    feature = "RACE"
    df[feature].fillna("5", inplace=True)
    df.loc[df[feature].isin(["NaN", "`"]), feature] = "5"
    print("Feture %s -> %s" % (feature, df[feature].unique()))

    # ETHNICITY
    feature = "ETHNICITY"
    df[feature].fillna("3", inplace=True)
    df.loc[df[feature].isin(["NaN", "`"]), feature] = "3"
    print("Feture %s -> %s" % (feature, df[feature].unique()))


def make_assignment(files=[], archive="my_assignment.zip"):
    default_files = ["01-Import.ipynb", "02-EDA.ipynb", "03-Model.ipynb", "my_lib.py", "df_grading_pred.csv"]
    print(f"Creating archive: {archive}")
    with ZipFile(archive,"w") as zip:
        for f in files+default_files:
            if os.path.isfile(f):
                print(f"\t{f} - OK")
                zip.write(f)
            else:
                print(f"\t{f} - Skipped")