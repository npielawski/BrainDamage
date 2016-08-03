import pandas as pd

# Author: Nicolas Pielawski
# Creation date: August 2nd 2016

# Loading the activity dataset
print("Loading training dataset")
activities = pd.read_csv("dataset/act_train.csv")
# Let's split the date
activities["year"] = activities["date"].apply(lambda d: d.split("-")[0])
activities["month"] = activities["date"].apply(lambda d: d.split("-")[1])
activities["day"] = activities["date"].apply(lambda d: d.split("-")[2])
# Let's transform the string values to integers'''
def str_to_cat(c):
    if pd.isnull(c): return 0
    return c.split(" ")[1]
activities["activity"] = activities["activity_category"].apply(str_to_cat) 
for i in range(1, 11):
    activities["char_" + str(i)] = activities["char_" + str(i)].apply(str_to_cat) 
# We remove the redundant columns
activities = activities.drop(["date", "activity_category", "activity_id"], 1)

# Loading the people dataset
print("Loading people dataset")
people = pd.read_csv("dataset/people.csv")
# We convert the string categories to integers
for i in range(1, 10):
    people["ppl_char_" + str(i)] = people["char_" + str(i)].apply(str_to_cat) 
# We change the booleans to numbers
for i in range(10, 38):
    people["ppl_char_" + str(i)] = people["char_" + str(i)] * 1.0
# We rename the last column
people["ppl_char_38"] = people["char_38"]
# We remove the useless cells
char_lst = [ "group_1", "date" ]
for i in range(1, 39):
    char_lst.append("char_" + str(i))
people = people.drop(char_lst, 1)

# Merging
print("Merging...")
final_set = activities.merge(people)

# Let's discard the people id
final_set = final_set.drop("people_id", 1)

# Save to new database
final_set.to_csv("dataset/people_activity.csv")