# Lab 03
# Luke Fox - 20076173

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # --- STEP 1 --- #
    iris = datasets.load_iris()

    # --- STEP 2 --- #
    subset_array = split_data(iris)

    # --- STEP 3 --- #
    validate_subset(iris, subset_array)

    # --- STEP 4--- #
    linear_classifier = first_classifier(subset_array)

    # --- STEP 5 --- #
    logistic_classifier = second_classifier(subset_array)

    # --- STEP 6 --- #
    decision_tree_classifier = third_classifier(subset_array)

    # --- STEP 7 --- #
    find_best_model(linear_classifier, logistic_classifier, decision_tree_classifier, subset_array)

    # --- STEP 8 --- #
    find_future_performance(linear_classifier, logistic_classifier, decision_tree_classifier, subset_array)


def split_data(iris_data):
    x = iris_data.data
    y = iris_data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=3000)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=3000)

    return {'x_train': x_train, 'x_test': x_test, 'x_validation': x_validation,
            'y_train': y_train, 'y_test': y_test, 'y_validation': y_validation}


def validate_subset(data, subset_array):
    value, counts = np.unique(data.target, return_counts=True)
    print("Base Data:\nValue: {0}, Count:{1}\n".format(value, counts))
    # --> Value: [0 1 2], Count:[50 50 50]

    value, counts = np.unique(subset_array.get('y_train'), return_counts=True)
    print("Y_Train Data:\nValue: {0}, Count:{1}\n".format(value, counts))
    # --> Value: [0 1 2], Count:[29 23 23]

    value, counts = np.unique(subset_array.get('y_test'), return_counts=True)
    print("Y_Test Data:\nValue: {0}, Count:{1}\n".format(value, counts))
    # --> Value: [0 1 2], Count:[10 13 14]

    value, counts = np.unique(subset_array.get('y_validation'), return_counts=True)
    print("Y_Validation Data:\nValue: {0}, Count:{1}".format(value, counts))
    # --> Value: [0 1 2], Count:[11 14 13]

    # as seen above, the unique count of each class differs to some degree within each subset
    # these implies these subsets are independent and representative of the original data-set


def first_classifier(subsets):
    # parameters introduced to remove warnings
    linear_classifier = svm.LinearSVC(max_iter=1500, tol=0.04)
    linear_classifier.fit(subsets.get('x_train'), subsets.get('y_train'))
    return linear_classifier


def second_classifier(subsets):
    # newton-cg selected as this deals with multi-class problems
    logistic_classifier = LogisticRegression(solver='newton-cg', multi_class='auto')
    logistic_classifier.fit(subsets.get('x_train'), subsets.get('y_train'))
    return logistic_classifier


def third_classifier(subsets):
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(subsets.get('x_train'), subsets.get('y_train'))
    return decision_tree_classifier


def find_best_model(linear_classifier, logistic_classifier, decision_tree_classifier, subsets):
    linear_prediction = linear_classifier.predict(subsets.get('x_test'))
    logistic_prediction = logistic_classifier.predict(subsets.get('x_test'))
    tree_prediction = decision_tree_classifier.predict(subsets.get('x_test'))

    print("\nAccuracy Scores:\n")
    print("Linear Classifier: {0}".format(accuracy_score(subsets.get('y_test'), linear_prediction)))
    print("Logistic Classifier: {0}".format(accuracy_score(subsets.get('y_test'), logistic_prediction)))
    print("Decision Tree Classifier: {0}".format(accuracy_score(subsets.get('y_test'), tree_prediction)))

    # Linear Classifier: 0.918918918918919
    # Logistic Classifier: 0.9459459459459459
    # Decision Tree Classifier: 0.8648648648648649

    # as seen above, testing the accuracy score of the predictions of each model shows that
    # the logistic classifier is the best fit for this data set with an accuracy of 0.95


def find_future_performance(linear_classifier, logistic_classifier, decision_tree_classifier, subsets):
    linear_prediction = linear_classifier.predict(subsets.get('x_validation'))
    logistic_prediction = logistic_classifier.predict(subsets.get('x_validation'))
    tree_prediction = decision_tree_classifier.predict(subsets.get('x_validation'))

    print("\nFuture Accuracy Scores:\n")
    print("Linear Classifier: {0}".format(accuracy_score(subsets.get('y_validation'), linear_prediction)))
    print("Logistic Classifier: {0}".format(accuracy_score(subsets.get('y_validation'), logistic_prediction)))
    print("Decision Tree Classifier: {0}".format(accuracy_score(subsets.get('y_validation'), tree_prediction)))

    # Linear Classifier: 0.9473684210526315
    # Logistic Classifier: 0.9473684210526315
    # Decision Tree Classifier: 0.8947368421052632

    # to analyse future performance, I used the validation set against each of the models once again
    # previously, the logistic classifier was the best fit - after testing the validation set it can be seen that
    # this can still be considered the best fit, but that the linear classifier will have a similar future performance


if __name__ == '__main__':
    main()
