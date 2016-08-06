import pandas as pd
import numpy as np
import sklearn.ensemble as skl
from sklearn import metrics
import xgboost as xgb

# Author: Nicolas Pielawski
# Creation date: August 3rd 2016

# Loading the dataset
print("Loading dataset...", end="")
df = pd.read_csv("dataset/people_activity_na.csv")
print("OK")

# Let's split the columns into x and y
y = df["outcome"]
x = df.drop("outcome", 1)

# Let's reduce the size of the dataset for now...
sep = np.random.rand(len(x)) > 0.9
x, y = x[sep], y[sep]

# Let's convert the dataset to xgboost type
dtrain = xgb.DMatrix(x, label=y, missing=np.nan)

# Creating the boosted gradient
# Parameters doc: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
param = {
    "n_estimators": 200,
    "learning_rate": 0.2,
    "max_depth": 20,
    "min_child_weight": 3,
    "gamma": 2,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "objective": "binary:logistic"
}

nrounds = 100
res = xgb.cv(param, dtrain, nrounds, 10, metrics=["auc"])
print(res[-3:])
'''
# Now we create and train our beloved bst!
bst = xgb.XGBClassifier(**param)
bst.fit(train, train_lbl)

# Let's see how we do on the train set
train_pred = bst.predict(train)
print("Accuracy of the train set")
print(metrics.classification_report(train_pred, train_lbl))
accuracy = metrics.accuracy_score(train_pred, train_lbl)

print("XGBoost accuracy: {:.3f}".format(accuracy))
print()

# Let's see the score on the test set
test_pred = bst.predict(test)
print("Preview of the accuracy")
print(metrics.classification_report(test_pred, test_lbl))
accuracy = metrics.accuracy_score(test_pred, test_lbl)

print("XGBoost accuracy: {:.3f}".format(accuracy))
print()

# Let's get the AUC
fpr, tpr, thresholds = metrics.roc_curve(test_lbl, test_pred)
auc = metrics.auc(fpr, tpr)
print("Final AUC: {:.04f}".format(auc))'''