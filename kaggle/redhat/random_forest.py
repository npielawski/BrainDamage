import pandas as pd
import numpy as np
import sklearn.ensemble as skl
from sklearn.metrics import classification_report, accuracy_score
# Grid search
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter, attrgetter

# Author: Nicolas Pielawski
# Creation date: August 2nd 2016

# Loading the dataset
print("Loading dataset...", end="")
df = pd.read_csv("dataset/people_activity.csv")
print("OK")

# Let's split the columns into x and y
y = df["outcome"]
x = df.drop("outcome", 1)

# Let's create a training and a testing dataset
sep = np.random.rand(len(df)) < 0.9
train = x[sep]
train_lbl = y[sep]
test = x[~sep]
test_lbl = y[~sep]

# Creating the forest for the search
n_estimators = 50
rf = skl.RandomForestClassifier(n_estimators)

# Randomized search
param_dist = {
    "max_depth": [ 3, None ],
    "max_features": sp_randint(1, len(df.columns)),
    "min_samples_split": sp_randint(1, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [ True, False ],
    "criterion": [ "gini", "entropy" ]
}

n_iter = 5
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter)
random_search.fit(train, train_lbl)

ntop = 3
top = sorted(random_search.grid_scores_, key=itemgetter(1), reverse=True)[:ntop]
for i, score in enumerate(top):
    print("Top #{}".format(i+1))
    print("Mean score:", score.mean_validation_score)
    print("Std dev:", np.std(score.cv_validation_scores))
    print("Params:", score.parameters)
    print()

# Best params
args = top[0].parameters
# Training of the forest
rf = skl.RandomForestClassifier(n_estimators, **args)
rf.fit(train, train_lbl)

# Let's see the score
test_pred = rf.predict(test)
print("Preview of the accuracy")
print(classification_report(test_pred, test_lbl))
accuracy = accuracy_score(test_pred, test_lbl)

print("Random Forest accuracy: {:.3f}".format(accuracy))
print()
# Approximately 75-80% accuracy
