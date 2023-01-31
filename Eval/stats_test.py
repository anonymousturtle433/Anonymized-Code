import json
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.stats import kstest
import matplotlib.pyplot as plt

with open("model_scores.json") as json_file:
    model_dict = json.load(json_file)

with open("user_study_scores.json") as json_file:
    study_dict = json.load(json_file)

with open("model_scores_goals.json") as json_file:
    model_dict_goals = json.load(json_file)

with open("user_study_scores_goals.json") as json_file:
    study_dict_goals = json.load(json_file)

# model_constraints_df = pd.read_csv('Output_Data/all_scores_constraints.csv')
# model_goals_df = pd.read_csv('Output_Data/all_scores_goals.csv')
#
# human_constraints_df = pd.read_csv('Output_Data/human_scores.csv')
# human_goals_df = pd.read_csv('Output_Data/human_scores_goals.csv')
#
# print(model_goals_df['Score'][-30:])
# print(model_constraints_df['Score'][-30:])
#
# print(np.var(np.array(model_goals_df['Score'][-30:])))
# print(np.var(np.array(human_goals_df['Score'][-30:])))
# print(np.var(np.array(model_constraints_df['Score'][-30:])))
# print(np.var(np.array(human_constraints_df['Score'][-30:])))
#
# print("Goals Levene's test results")
# print(stats.levene(model_goals_df['Score'][-30:], human_goals_df['Score']))
#
# print("Constraints Levene's test results")
# print(stats.levene(model_constraints_df['Score'][-30:], human_constraints_df['Score']))


modified_study_dict = {}
modified_study_dict_goals = {}

for key in study_dict:
    modified_study_dict[key] = float(sum(study_dict[key])) / len(study_dict[key])
    modified_study_dict_goals[key] = float(sum(study_dict_goals[key])) / len(study_dict_goals[key])
    plt.hist(study_dict[key], bins=[0,1,2,3,4,5,6,7,8,9])
    plt.xlabel("Score")
    plt.ylabel("Number of humans")
    plt.title(f"Histogram for map {key}")
    plt.savefig(f"Plots/map_{key}_histogram.png")
    plt.clf()
    # modified_study_dict[key] = np.median(np.array(study_dict[key]))

plt.hist(list(modified_study_dict.values()), bins=[0,1,2,3,4,5,6,7,8,9])

plt.xlabel("Score")
plt.ylabel("Number of data points")
plt.title(f"Human Data Histogram")
plt.savefig(f"Plots/Human_Histogram.png")
plt.clf()

plt.hist(list(model_dict.values()), bins=[0,1,2,3,4,5,6,7,8,9])

plt.xlabel("Score")
plt.ylabel("Number of data points")
plt.title(f"Model Data Histogram")
plt.savefig(f"Plots/Model_Histogram.png")
plt.clf()

plt.hist(list(model_dict_goals.values()), bins=[0,1,2,3,4,5,6,7,8,9])

plt.xlabel("Score")
plt.ylabel("Number of data points")
plt.title(f"Model Data Histogram")
plt.savefig(f"Plots/Model_Histogram_Goals.png")
plt.clf()

print("Model scores")
print(model_dict.values())
print("Human scores")
print(modified_study_dict.values())

model_values = list(model_dict.values())
study_values = list(modified_study_dict.values())

model_values_goals = list(model_dict_goals.values())
study_values_goals = list(modified_study_dict_goals.values())


print("Model Mean")
print(np.mean(np.array(model_values)))
print("Human Mean")
print(np.mean(np.array(study_values)))

print("Model Variance")
print(np.var(np.array(model_values)))
print("Human Variance")
print(np.var(np.array(study_values)))
print("Levene's test results")
print(stats.levene(model_values, study_values))


print("Shapiro - Model")
print(stats.shapiro(model_values))
print("Shapiro - Study")
print(stats.shapiro(study_values))

print("-----------GOALS RESULTS----------")
print("Model Variance")
print(np.var(np.array(model_values_goals)))
print("Human Variance")
print(np.var(np.array(study_values_goals)))
print("Levene's test results")
print(stats.levene(model_values_goals, study_values_goals))


print("Shapiro - Model")
print(stats.shapiro(model_values_goals))
print("Shapiro - Study")
print(stats.shapiro(study_values_goals))

print("Two-Sample T-Test")
print(stats.ttest_ind(model_values, study_values, equal_var = True))