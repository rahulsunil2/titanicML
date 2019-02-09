# Variable : Definition	               Key

# survival : Survival 	               0 = No, 1 = Yes
# pclass   : Ticket class 	           1 = 1st, 2 = 2nd, 3 = 3rd
# sex      : Sex
# Age      : Age in years
# sibsp    : No: of siblings / spouses aboard the Titanic
# parch    : No: of parents / children aboard the Titanic
# ticket   : Ticket number
# fare     : Passenger fare
# cabin    : Cabin number
# embarked : Port of Embarkation       C = Cherbourg, Q = Queenstown, S = Southampton

# import Numpy, Pandas and Tree

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot Class


class MatPlot:
    def plotg(self, forest, X):
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()

    def plotg1(self, tree, X):
        importances = tree.feature_importances_
        plt.plot([importances, X, 'ro'])
        # plt.axis([0, 6, 0, 20])
        plt.show()


# Read the Data

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Create and Assign a Column

train["Child"] = float("NaN")
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

# Test

test_one = test
test["Survived"] = 0
test["Survived"][test["Sex"] == "female"] = 1

# Embarked

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Age

train["Age"] = train["Age"].fillna(train.Age.median())

# Creating first decision tree

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Test Predictions

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Age"] = test["Age"].fillna(test.Age.median())
test["Fare"] = test["Fare"].fillna(test.Fare.median())

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
my_prediction = my_tree_one.predict(test_features)

# Predictions Printing

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Save to CSV

# my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

# Creating second decision tree

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_two = my_tree_two.fit(features_two, target)

# Creating third decision tree

train_two = train.copy()
train_two["family_size"] = train_two["Parch"] + train_two["SibSp"]

features_three = train_two[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "family_size"]].values

my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Creating Random Forest analysis

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)
# print my_tree_one.score(features_one, target)
# print my_tree_two.score(features_two, target)
# print my_tree_three.score(features_three, target)
print "Forest"
print my_forest.score(features_forest, target)

# plot

Plot1 = MatPlot()
Plot1.plotg1(my_tree_one, ["Pclass", "Sex", "Age", "Fare"])
