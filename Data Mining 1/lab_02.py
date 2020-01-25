import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


def main():
    # --- STEP 1 --- #

    csv_file = pd.read_csv('BSCY4_Lab_2.csv')

    # --- STEP 2 --- #

    # check MPG normality
    normality_check(csv_file, ['mpg'])

    # check other numerical data normality
    normality_check(csv_file, ['displacement', 'weight', 'horsepower', 'acceleration'])

    # --- STEP 3 --- #

    normalised_data = normalise_data(csv_file, ['displacement', 'weight', 'horsepower'])

    # --- STEP 4 & 5 --- #

    single_predictor_regression_model(normalised_data)

    # --- STEP 6 --- #
    multiple_predictor_regression_model(normalised_data)

    # --- STEP 7 --- #
    mediation_analysis(normalised_data)

    # --- STEP 8 --- #
    categorical_model(normalised_data, ['cylinders', 'model year', 'origin'])

    # --- STEP 9 --- #
    categorical_mediation_analysis(normalised_data)


def normality_check(data, series):
    # check skew and kurtosis for each
    print('\nKURTOSIS:\n', data[series].kurtosis(axis=0, skipna=True))
    print('\nSKEW:\n', data[series].skew(axis=0, skipna=True))

    # all numerical data including MPG is close to 0 for both skew & kurtosis
    # the largest is displacement, skew = 0.963273 and kurtosis = 0.583570

    for col in series:
        sns.distplot(data[col], kde=True, rug=True)
        plt.title('{0}: Distribution'.format(col))
        plt.show()

        # distribution not bell-shapped on displacement and horsepower

        # Shapiro walk to check p-values
        print('\nSHAPIRO TEST\n')
        print(col, stats.shapiro(data[col]))

        # p value is below 0.05 for displacement, weight and horsepower

        # q-q plot to check for linear graph
        stats.probplot(data[col], dist='norm', plot=plt)
        plt.title('{0}: Q-Q Plot'.format(col))
        plt.show()

        # displacement and horsepower are not linear

        # -- MPG assessment -- #
        # bell-shaped distribution, p-value > .05, linear Q-Q plot
        # normal


def normalise_data(data, series):
    print("\nNORMALISING DATA\n")
    for col in series:
        normalised = data[col].apply(np.log)
        print("{0} - normalised: {1}".format(col, stats.shapiro(normalised)))

    # shows only weight becomes normalised with np.log
    # weight value after --> 0.20337

    # only weight is normalised, make change
    data['weight'] = data['weight'].apply(np.log)

    # attempt ratio between two other variables to normalise others
    data['disp_hpwr_ratio'] = data['displacement'] / data['horsepower']
    data['disp_hpwr_ratio'] = data['disp_hpwr_ratio'].apply(np.exp)
    normality_check(data, ['disp_hpwr_ratio'])

    # kurtosis=0.93, skew=-0.13, p-value=0.073, displacement=bell curve, Q-Q Plot=linear
    # normalised

    # return normalised data
    return data


def single_predictor_regression_model(data):
    # two normal predictors, acceleration and weight.
    correlation = data['weight'].corr(data['acceleration'])
    print("\nCORRELATION: {0}\n".format(correlation))

    # 1. correlation is within .95 and no multicollinearity warnings -> predictor independance

    # --- BUILD MODEL WITH WEIGHT --- #

    model = sm.OLS(data['mpg'], data['weight'])
    fitted_model = model.fit()
    print(fitted_model.summary())

    # 2. normality of predictors is already established from earlier

    # 3. check homoscedasticity of residuals
    plt.figure()
    plt.title('Weight Residual Values')
    plt.scatter(data['mpg'], fitted_model.resid)
    plt.show()

    # no homoscedasticity as there is little data clump around 0

    # --- BUILD MODEL WITH ACCELERATION --- #

    model = sm.OLS(data['mpg'], data['acceleration'])
    fitted_model = model.fit()
    print(fitted_model.summary())

    plt.figure()
    plt.title('Acceleration Residual Values')
    plt.scatter(data['mpg'], fitted_model.resid)
    plt.show()

    # here there is homoscedasticity as there is more of a data clump at 0


def multiple_predictor_regression_model(data):
    predictors_stacked = np.column_stack((data['weight'], data['acceleration']))
    predictors_stacked = sm.add_constant(predictors_stacked)
    model = sm.OLS(data['mpg'], predictors_stacked)
    fitted_model = model.fit()
    print(fitted_model.summary())

    # this does not meet the regression model assumptions as there is now a multicollinearity warning

    plt.figure()
    plt.title('Acceleration w/ Weight Residual Values')
    plt.scatter(data['mpg'], fitted_model.resid)
    plt.show()

    # lesser clump of data around 0, still possibly homoscedasticity


def mediation_analysis(data):
    # assessing least significant predictor by creating model of acceleration and weight

    print("\nPREDICTOR AND MEDIATOR MODEL\n")
    weight_constant = sm.add_constant(data["weight"])
    model = sm.OLS(data['acceleration'], weight_constant)
    fitted_model = model.fit()
    print(fitted_model.summary())

    # comparing R squared to the previous model, it has dropped from 0.330 to 0.313
    # weight is the least significant predictor


def categorical_model(data, series):
    # introduce each categorical variable and assess R squared value

    for col in series:
        print("\nMODEL WITH: {0}\n".format(col))
        dummy = pd.get_dummies(pd.Series(data[col]))
        stacked = np.column_stack((data['acceleration'], dummy))
        stacked = sm.add_constant(stacked)
        model = sm.OLS(data["mpg"], stacked)
        fitted_model = model.fit()
        print(fitted_model.summary())

    # origin R value        --> 0.161
    # model year R value    --> 0.633
    # cylinders R value     --> 0.281

    # as shown, origin and cylinders are least significant - likely due to little variance
    # model year has greater variance leading to them being more significant


def categorical_mediation_analysis(data):

    # compare R squared value when modelling most significant categorical variable
    dummy = pd.get_dummies(pd.Series(data["model year"]), drop_first=True)
    constant = sm.add_constant(dummy)
    model = sm.OLS(data["mpg"], constant)
    results = model.fit()

    print("\nMODEL WITH CATEGORICAL VARIABLE\n")
    print(results.summary())

    # R squared is 0.482, compared to 0.633 --> therefore there is a mediation effect


if __name__ == '__main__':
    main()
