import json

import pandas as pd
import numpy as np
from eval_helpers import *
import csv
import itertools

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
user_df = pd.read_excel('Data/Risk_Human_Eval_Dataset.xlsx')
f = open('Output_Data/human_scores.csv', 'w')
writer = csv.writer(f)
writer.writerow(['Response_ID', 'Score', 'Map', 'Participant_ID', 'human_v_model'])
g = open('Output_Data/human_scores_goals.csv', 'w')
g_writer = csv.writer(g)	
g_writer.writerow(['Response_ID', 'Score', 'Map', 'Participant_ID', 'human_v_model'])
# print(user_df['map_1'])
num_same = 0
total = 0
num_same_constraints = 0
total_constraints = 0
user_df = user_df.fillna('NA')
map_scores = defaultdict(list)
map_scores_goals = defaultdict(list)
participant_id = 0
response = 0
for i in range(user_df.shape[0]-2):
	for j in range(1,4):
		map = user_df['map_' + str(j)][i+2]
		if map == 'NA':
			continue
		answer = study_data_points[int(map)]
		goals = []
		constraints = []
		for k in range(1,7):
			goals.append(int(user_df['Goals_' + str(j) + '_' + str(k)][i + 2]))
		# print(goals)
		goals = real_to_buckets(goals)
		for k in range(1,9):
			constraints.append(user_df[str(j) + '_constraint_' + str(k)][i+2])

		con_equals, con_total = compute_accuracy(constraints, answer['Constraints'])
		print(map)
		print(constraints)
		print(answer['Constraints'])
		num = num_equals(goals, answer['Goals'])
		print(con_equals)
		map_scores[int(map)].append(con_equals)
		map_scores_goals[int(map)].append(num)
		num_same += num
		total += 6
		num_same_constraints += con_equals
		total_constraints += 8
		writer.writerow([response, con_equals, map, participant_id, 0])
		g_writer.writerow(([response, num, map, participant_id, 0]))
		response += 1
	participant_id += 1
	print("-------------")
f.close()
print("Mean Score - Constraints")
print(np.mean(list(itertools.chain(*list(map_scores.values())))))
print("Std Score - Constraints")
print(np.std(list(itertools.chain(*list(map_scores.values())))))
print("CI Score - Constraints")
print(1.96 * np.std(list(itertools.chain(*list(map_scores.values())))) / math.sqrt(len(list(itertools.chain(*list(map_scores.values()))))))
print("Mean Score - Goals")
print(np.mean(list(itertools.chain(*list(map_scores_goals.values())))))
print("Std Score - Goals")
print(np.std(list(itertools.chain(*list(map_scores_goals.values())))))
print("CI Score - Goals")
print(1.96 * np.std(list(itertools.chain(*list(map_scores_goals.values())))) / math.sqrt(len(list(itertools.chain(*list(map_scores_goals.values()))))))
print(num_same)
print(total)
print("Goals accuracy")
print(float(num_same) / total)
print(num_same_constraints)
print(total_constraints)
print("Constraints accuracy")
print(float(num_same_constraints) / total_constraints)

with open("user_study_scores.json", "w") as json_file:
	json.dump(map_scores, json_file, indent = 2)

with open("user_study_scores_goals.json", "w") as json_file:
	json.dump(map_scores_goals, json_file, indent = 2)