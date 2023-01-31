import math

import pandas as pd
import json
import csv
import numpy as np
from eval_helpers import *





constraint_classes = {1: "I must have troops on",
                      2: "I must not have troops on",
                      3: "I must be able to access",
                      4: "I need to protect the borders of",
                      5: "I need a total of at least",
                      6: "I must have at least",
                      7: "I must have troops on at least",
                      8: "I must place at least",
                      9: "I must have troops on at most"}

constraint_second_half = {1: "",
                          2: "",
                          3: " with one move",
                          4: "",
                          5: " troops to defend a continent",
                          6: " countries",
                          7: " continents",
                          8: " troops to effectively defend a country",
                          9: " continents"
                          }

df = pd.read_excel('../NL_constraints_data_prep/Output_Data/user_study_data_points.xlsx')

study_data_points = {}
df = df.fillna('NA')
for i in range(df.shape[0]):
	# study_data_points['map'] = df['map'][i]
	study_data_points[df['map'][i]] = {'Goals': real_to_buckets(list(df.loc[i][2:8])), 'Constraints': list(df.loc[i][15:23])}

# for i in range(1,31):
# 	print(study_data_points[i]['Constraints'])
# sdfsdf

print(study_data_points[11])
model_df = pd.read_csv('Data/model_predictions_best_acc_model_constraints.csv')
goals_df = pd.read_csv('Data/model_predictions_best_acc_model_goals.csv')
# print(user_df['map_1'])
f = open('Output_Data/all_scores_constraints.csv', 'a')
writer = csv.writer(f)
g = open('Output_Data/all_scores_goals.csv', 'a')
g_writer = csv.writer(g)
num_same = 0
total = 0
num_same_constraints = 0
total_constraints = 0
model_df = model_df.fillna('NA')
model_df_goals = goals_df.fillna('NA')
map_scores = {}
map_scores_goals = {}
for i in range(1,model_df.shape[0], 2):
	# print(model_df.loc[[i]])
	map = model_df['Map'][i]
	print(map)
	print(i)
	print(model_df['Input Text'][i])
	if map == 'NA':
		continue
	answer = study_data_points[int(map)]
	constraints = []
	for k in range(1,9):
		constraint = model_df['constraint_' + str(k)][i].replace("[", "").replace("]", "").replace("- ", "").replace("<NA><NA>", "NA")
		split_constraint = constraint.split()
		# print(split_constraint)
		for key in constraint_classes:
			if constraint_classes[key] == " ".join(split_constraint[:-1]):
				constraint = constraint + constraint_second_half[key]
				break
		constraints.append(constraint)
	# sjdhjsdf
	con_equals, con_total = compute_accuracy(constraints, answer['Constraints'])
	print(constraints)
	# print(goals)
	print(answer['Constraints'])
	print(con_equals)
	map_scores[int(map)] = con_equals
	num_same_constraints += con_equals
	total_constraints += 8
	print("-------------")
	writer.writerow(['--', con_equals, map, 'M', 1])

for i in range(model_df_goals.shape[0]):
	# print(model_df.loc[[i]])
	map = model_df_goals['Map'][i]
	print(map)
	print(model_df_goals['Map'][i])
	if map == 'NA':
		continue
	answer = study_data_points[int(map)]
	goals = []
	for k in range(1,7):
		goals.append(int(model_df_goals['G' + str(k) + '_1'][i]))
	# goals = real_to_buckets(goals)
	print(goals)
	print(answer['Goals'])
	goals_same = num_equals(goals, answer['Goals'])
	map_scores_goals[int(map)] = goals_same
	num_same += goals_same
	print(goals_same)
	total += 6
	print("-------------")
	g_writer.writerow(['--', goals_same, map, 'M', 1])

# print("Goals accuracy")
# print(float(num_same) / total)
with open("model_scores.json", "w") as json_file:
	json.dump(map_scores, json_file, indent = 2)
with open("model_scores_goals.json", "w") as json_file:
	json.dump(map_scores_goals, json_file, indent = 2)
print(num_same)
print(total)
print("Mean Score - Constraints")
print(np.mean(list(map_scores.values())))
print("Std Score - Constraints")
print(np.std(list(map_scores.values())))
print("CI Score - Constraints")
print(1.96 * np.std(list(map_scores.values())) / math.sqrt(len(list(map_scores.values()))))
print("Mean Score - Goals")
print(np.mean(list(map_scores_goals.values())))
print("Std Score - Goals")
print(np.std(list(map_scores_goals.values())))
print("CI Score - Goals")
print(1.96 * np.std(list(map_scores_goals.values())) / math.sqrt(len(list(map_scores_goals.values()))))
print("Goals accuracy")
print(float(num_same) / total)
print(num_same_constraints)
print(total_constraints)
print("Constraints accuracy")
print(float(num_same_constraints) / total_constraints)

