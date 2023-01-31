from generate_utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_strategies", type=int, help="The number of strategies to generate", default=10000)
args = parser.parse_args()

with open("../Data/Synthetic_Strategies.csv", mode='w', encoding='utf8') as output_file:
    # output_file = csv.open("../Data/Generated_Strategy.csv")
    writer = csv.writer(output_file)
    writer.writerow(
        ["Instructions", "G1_1", "G2_1", "G3_1", "G4_1", "G5_1", "G6_1", "constraint_1", "constraint_2", "constraint_3", "constraint_4",
         "constraint_5", "constraint_6", "constraint_7", "constraint_8"])

GC_temp = GC_Templates()
GC_temp.load_templates()


num_strategies = args.num_strategies
for n in range(num_strategies):
    goal_assignments = []
    for g in goals.keys():
        g_assignment = random.randint(-100,100)
        goal_assignments.append((g, g_assignment))

    constraint_assignments = []
    stop = False
    count = 0
    prob_threshold = 0.8
    while not stop:
        if count < 3:
            constraint = choose_random_constraint()
            if constraint not in constraint_assignments:
                constraint_assignments.append(constraint)
                count += 1
        else:
            if count == 8:
                stop = True
                break
            constraint = choose_random_constraint()
            if constraint not in constraint_assignments:
                constraint_assignments.append(constraint)
                count += 1

    print(goal_assignments)
    print(constraint_assignments)

    goals_lang = GC_temp.generate_language_per_goal(goal_assignments)
    constraint_lang = GC_temp.generate_language_per_constraint(constraint_assignments)
    # goal_assignments = [(1,1), (2,33)]
    # constraint_assignments = [(1,1), (2,3)]

    save_strategy(goal_assignments, constraint_assignments, goals_lang, constraint_lang)