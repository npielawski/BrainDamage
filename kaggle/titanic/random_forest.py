import pandas as pd
import numpy as np
import sklearn.ensemble as skl
from sklearn.metrics import classification_report, accuracy_score

# Author: Nicolas Pielawski
# Creation date: July 27 2016

# Loading the dataset
df = pd.read_csv("train.csv")
# Removing useless columns
df = df.drop(["PassengerId", "Name", "Ticket" ], 1)

# We fill the gaps
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# We map the string data with ordinals
df["Cabin"] = df["Cabin"].fillna("N").apply(lambda s: s[0])
map_cabin = { "N": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8 }
df["Cabin"] = df["Cabin"].map(map_cabin)

map_sex = { "N": 0, "female": 1, "male": 2 }
df["Sex"] = df["Sex"].fillna("N").map(map_sex)

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

# Random Forests
args = {
    "n_estimators": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1
}
rf = skl.RandomForestClassifier(**args)

# Training of the forest
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