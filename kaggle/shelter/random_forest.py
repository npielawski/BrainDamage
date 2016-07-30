import pandas as pd
import numpy as np
import sklearn.ensemble as skl
from sklearn.metrics import classification_report, accuracy_score
# Grid search
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter, attrgetter

from datetime import datetime
import re

# Author: Nicolas Pielawski
# Creation date: July 30 2016

def clean_dataset(df, breeds, colors, outcome=True):
    # We get the information: does the animal have a name?
    df["HasName"] = ~df["Name"].isnull()

    # We save the date as months since 2000
    def date_to_months(strdate):
        date = datetime.strptime(strdate, "%Y-%m-%d %H:%M:%S")
        return (date.year - 2000) * 12 + date.month
    df["Date"] = df["DateTime"].apply(date_to_months)
    # We save the hour (Possible correlation with day time?) as minutes
    def date_to_time(strdate):
        date = datetime.strptime(strdate, "%Y-%m-%d %H:%M:%S")
        return date.hour * 60 + date.minute
    df["Hour"] = df["DateTime"].apply(date_to_time)

    # We map the categories
    map_outcome = {
        "Adoption": 0,
        "Died": 1,
        "Euthanasia": 2,
        "Return_to_owner": 3, 
        "Transfer": 4
    }
    if outcome:
        df["Outcome"] = df["OutcomeType"].map(map_outcome)
    map_type = { "Dog": 0, "Cat": 1 }
    df["Animal"] = df["AnimalType"].map(map_type)

    # We change the age as weeks
    def age_to_weeks(age):
        nbr = float(age.split(" ")[0])
        if "year" in age: return nbr * 52.
        if "month" in age: return nbr * 52. / 12.
        if "week" in age: return nbr
        return nbr
    df["Age"] = df["AgeuponOutcome"].dropna().apply(age_to_weeks)

    # Let's get to the sex + castration
    def sex_to_gender(sex):
        gender = 0
        if "Male" in sex: gender = 1
        return gender
    df["Gender"] = df["SexuponOutcome"].dropna().apply(sex_to_gender)
    def sex_to_castrated(sex):
        castrated = 0
        if "Neutered" in sex or "Spayed" in sex: castrated = 1
        return castrated
    df["Castrated"] = df["SexuponOutcome"].dropna().apply(sex_to_castrated)

    # Let's create the new labels
    breeds_lbl = []
    for breed in breeds:
        breed_lbl = "Is" + re.sub("[^A-Za-z]+", "", breed)
        breeds_lbl.append(breed_lbl)
        df[breed_lbl] = df["Breed"].str.contains(breed)

    colors_lbl = []
    for color in colors:
        color_lbl = "Is" + re.sub("[^A-Za-z]+", "", color)
        colors_lbl.append(color_lbl)
        df[color_lbl] = df["Color"].str.contains(color)
        
    # Removing useless columns
    df = df.drop([ "Name", "DateTime", "AnimalType", "SexuponOutcome", "AgeuponOutcome", "Breed", "Color" ], 1).fillna(0.)
    if outcome:
        df = df.drop([ "AnimalID", "OutcomeType", "OutcomeSubtype" ], 1)
    return df

# Loading the dataset
df = pd.read_csv("train.csv")

# Most common breeds and colors
bctop = 10
breeds = df["Breed"].value_counts()[:bctop].axes[0]
colors = df["Color"].value_counts()[:bctop].axes[0]
    
df = clean_dataset(df, breeds, colors)
# Let's see  how the data looks like
print("Preview of the dataset")
# Let's not see the colors/breeds
print(df[df.columns[:-2*bctop]].head())
print()

# Let's split the columns into x and y
y = df["Outcome"]
x = df.drop("Outcome", 1)

# Let's create a training and a testing dataset
sep = np.random.rand(len(df)) < 0.9
train = x[sep]
train_lbl = y[sep]
test = x[~sep]
test_lbl = y[~sep]

# Creating the forest for the search
n_estimators = 200
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

# Generation of the test set predictions
# Loading the new dataset
df = pd.read_csv("test.csv")
# Let's clean that one
df = clean_dataset(df, breeds, colors, outcome=False)

# We extract the IDs
id = df["ID"]
df = df.drop("ID", 1)

# Let's see  how the data looks like
print("Preview of the test dataset")
print(df[df.columns[:-2*bctop]].head())

# We predict
predictions = rf.predict(df)

# We get the predictions
# We have to give an uncertain decision (cross entropy validation on kaggle)
def unsure_prediction(pred, value = 0.9):
    pred = pred.astype(float)
    pred[pred == True] = value
    pred[pred == False] = (1.0 - value) * 0.25
    return pred

pred = pd.DataFrame({
    "ID": id,
    "Adoption": unsure_prediction(predictions == 0),
    "Died": unsure_prediction(predictions == 1),
    "Euthanasia": unsure_prediction(predictions == 2),
    "Return_to_owner": unsure_prediction(predictions == 3),
    "Transfer": unsure_prediction(predictions == 4)
})
# We order the columns
pred = pred[["ID", "Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"]]

# And we save the predicted results
pred.to_csv("predict.csv", index=False, float_format="%.3f")
# Approximately 75% accuracy