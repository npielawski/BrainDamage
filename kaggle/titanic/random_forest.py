import pandas as pd
import numpy as np
import sklearn.ensemble as skl
from sklearn.metrics import classification_report, accuracy_score
# Grid search
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter, attrgetter

# Author: Nicolas Pielawski
# Creation date: July 27 2016

# Loading the dataset
df = pd.read_csv("train.csv")
# Removing useless columns
df = df.drop(["PassengerId", "Name", "Ticket" ], 1)
#print(df.isnull().sum())

# We fill the gaps
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# We map the string data with ordinals
df["Cabin"] = df["Cabin"].fillna("N").apply(lambda s: s[0])
map_cabin = { "N": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8 }
df["Cabin"] = df["Cabin"].map(map_cabin)

map_sex = { "female": 0, "male": 1 }
df["Sex"] = df["Sex"].map(map_sex)

map_embarked = { "S": 0, "C": 1, "Q": 2 }
df["Embarked"] = df["Embarked"].fillna("S").map(map_embarked)

# Let's see  how the data looks like
print("Preview of the dataset")
print(df.head())
print()

# Let's split the columns into x and y
y = df["Survived"]
x = df[df.columns[1:]]

# Let's create a training and a testing dataset
sep = np.random.rand(len(df)) < 0.9
train = x[sep]
train_lbl = y[sep]
test = x[~sep]
test_lbl = y[~sep]

# Creating the forest for the search
n_estimators = 500
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

n_iter = 50
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

# Generation of the test set predictions
# Loading the new dataset
df = pd.read_csv("test.csv")
# Let's save the PassengerId and save them in the file at the end
id = df["PassengerId"]
# Removing useless columns as before
df = df.drop(["PassengerId", "Name", "Ticket" ], 1)

# We fill the gaps
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# We map the string data with ordinals
df["Cabin"] = df["Cabin"].fillna("N").apply(lambda s: s[0])
df["Cabin"] = df["Cabin"].map(map_cabin)
df["Sex"] = df["Sex"].fillna("N").map(map_sex)
df["Embarked"] = df["Embarked"].fillna("S").map(map_embarked)

# Let's see  how the data looks like
print("Preview of the test dataset")
print(df.head())

# Let's predict!
pred = pd.DataFrame({
    "PassengerId": id,
    "Survived": rf.predict(df)
})

# And we save the predicted results
pred.to_csv("predict.csv", index=False)
# Approximately 75% accuracy