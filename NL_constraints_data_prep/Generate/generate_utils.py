import random
import csv
import xlrd
from collections import defaultdict


from NLG import get_paraphrase, get_summary
from augmentation_utils import *


goals = {1: "Surround enemy territories",
         2: "Maximize number of countries occupied",
         3: "Keep your troops close together",
         4: "Maximize battles throughout the game",
         5: "Fortify borders for the continents you control",
         6: "Battle opposing players one at a time"}

constraint_classes = {1: "I must have troops on ",
                      2: "I must not have troops on ",
                      3: "I must be able to access ",
                      4: "I need to protect the borders of ",
                      5: "I need a total of at least ",
                      6: "I must have at least ",
                      7: "I must have troops on at least ",
                      8: "I must place at least ",
                      9: "I must have troops on at most "}

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

constraint_values = {
    1: ["Red", "Purple", "Blue", "Green", "Yellow"],
    2: ["Red", "Purple", "Blue", "Green", "Yellow"],
    3: ["Red", "Purple", "Blue", "Green", "Yellow"],
    4: ["Red", "Purple", "Blue", "Green", "Yellow"],
    5: [4,5,6,7,8,9,10,11,12,13,14],
    6: [2,3,4,5,6,7],
    7: [2,3,4,5],
    8: [4,5,6,7,8,9,10,11,12,13,14],
    9: [1,2,3,4]
}


class GC_Templates:
    def __init__(self):
        '''
        Create a class consisting of dictionaries of goal and constraint templates
        Goal Template: List of dictionary of lists
            You have 6 goals total. Each goal has 5 buckets. The self.goal_template variable contains all possible template descriptions for each bucket
        Constraint Template: Dictionary of lists
            There are 9 constraints in total. The self.constraint_template variable contains all possible template decriptions for each constraint type.
        '''
        self.goal_template = [defaultdict(list) for i in range(6)]
        self.constraint_template = defaultdict(list)
        self.generic_sentences = []

    def create_templates(self, goal_data, constraint_data):
        '''
        Create object consisting of goal and constraint templates.
        :param goal_data:
        :param constraint_data:
        :return:
        '''
        for data in goal_data:
            for bucket, text in enumerate(data[2:]):
                if not text == "":
                    self.goal_template[int(data[0] - 1)][bucket].append(text)
        for data in constraint_data:
            self.constraint_template[int(data[0]) - 1].append(data[3])

    def load_templates(self):
        '''
        Load goal and constraint templates from xlsx files
        :return:
        '''
        goal_data = []
        constraint_data = []
        excel = xlrd.open_workbook('AugmentedGoals.xlsx')
        sheet_goals = excel.sheet_by_index(0)

        prev_row = [None for i in range(sheet_goals.ncols)]
        for row_index in range(sheet_goals.nrows):
            if row_index == 0:
                # Skip headers
                continue
            row = []
            for col_index in range(sheet_goals.ncols):
                value = sheet_goals.cell(rowx=row_index, colx=col_index).value
                if value == "" and col_index == 0:
                    value = prev_row[col_index]
                row.append(value)
            prev_row = row
            goal_data.append(row)

        sheet_constraints = excel.sheet_by_index(1)
        prev_row = [None for i in range(sheet_constraints.ncols)]
        for row_index in range(sheet_constraints.nrows):
            if row_index == 0:
                # Skip headers
                continue
            row = []
            for col_index in range(sheet_constraints.ncols):
                value = sheet_constraints.cell(rowx=row_index, colx=col_index).value
                if value == "" and col_index == 0:
                    value = prev_row[col_index]
                row.append(value)
            prev_row = row
            constraint_data.append(row)


        sheet_generics = excel.sheet_by_index(2)
        for row_index in range(sheet_generics.nrows):
            if row_index == 0:
                # Skip headers
                continue
            self.generic_sentences.append(sheet_generics.cell(rowx=row_index, colx=0).value)
        self.create_templates(goal_data, constraint_data)

    def generate_generics(self, generic_prob, max_generics=3):
        p = random.uniform(0, 1)
        generics = []
        current_generic_prob = generic_prob
        for _ in range(max_generics): #ensures that no more than max generics are added
            if p < current_generic_prob:
                random.shuffle(self.generic_sentences)
                if self.generic_sentences[0] not in generics:
                    generics.append(self.generic_sentences[0])
                current_generic_prob /= 2
            else:
                break

        return generics


    def generate_language_per_goal(self, goal_assignments):
        goal_sentences = []
        orig_goal_sentences = []
        for goal in goal_assignments:
            bucket = bucket_assignment(goal[1])
            lang = random.choice(self.goal_template[goal[0]-1][bucket])
            if bucket == 2:
                p = random.uniform(0, 1)
                if p < 0.5:
                    #Skip middle bucket 50% of the time (I don't care/Didn't consider)
                    continue
            if bucket == 1 or bucket == 3:
                p = random.uniform(0, 1)
                if p < 0.25:
                    # #Skip medium bucket 25% of the time (slightly important)
                    continue
            if bucket == 0 or bucket == 4:
                p = random.uniform(0, 1)
                if p < 0.05:
                    # #Skip extreme bucket 5% of the time (Must have)
                    continue
            # orig_goal_sentences.append(right_strip(lang))
            # lang = get_paraphrase(lang)
            goal_sentences.append(right_strip(lang))
        return goal_sentences

    def generate_must_have_variant(self, constraint):
        """
        Creates variant of must have troops on constraint
        Must have troops on Green -> I placed troops on Green_A, Green_D, etc.
        Args:
            constraint: Tuple -> (con_class, con_value)

        Returns:

        """
        assert constraint[0] == 1, "This functionality is only applicable for the must have troops on constraint"

        num_countries = random.randint(1,3)
        invalid = {"Yellow":['E'], "Red": ['D','E'], "Blue": ['E'], "Green": [], "Purple": []}
        countries = ['A', 'B', 'C', 'D', 'E']
        final_countries = []
        i = 0
        while i < num_countries:
            country = random.sample(countries, 1)[0]
            if country in invalid[constraint[1]]:
                if country in countries:
                    countries.remove(country)
                continue
            i += 1
            if country in countries:
                countries.remove(country)
            final_countries.append(constraint[1] + '_' + country)
        # print(final_countries)
        langs = ["I placed troops on " + ', '.join(final_countries), "I have troops on " + ', '.join(final_countries), "I put some troops on " + ', '.join(final_countries) ]
        return random.choice(langs)

    def check_must_have_at_least(self, constraints):
        '''
        Function to check whether the constraints have the same number of "must have" and "at least ___ troops on continent" constraints
        Args:
            constraints: list[(tuple)]

        Returns: bool

        '''
        num1 = 0
        num2 = 0
        for const in constraints:
            if const[0] == 1:
                num1 += len(const[1])
            elif const[0] == 5:
                num2 += 1
        if num1 == num2 and num1 == 1:
            return True
        return False

    def generate_num_continents_variant(self, constraint, constraint_assignments):
        '''
        NOT CURRENTLY BEING USED
        This function converts "I must have troops on at least 2 continents" to "I placed troops on Red, Blue"
        The number of continents is equal to the number in the constraint. The continents themselves are chosen randomly
        Args:
            constraint: tuple

        Returns: str

        '''
        assert constraint[0] == 7, 'This functionality is only applicable for I must have troops on at least N continents'

        continents = ['Red', 'Purple', 'Blue', 'Yellow', 'Green']
        not_continents = []
        have_continents = []
        # print(continents)
        for c in constraint_assignments:
            # print(c)
            if c[0] == 2:
                not_continents.extend(c[1])
            if c[0] == 1 or c[0] == 4:
                for a in c[1]:
                    if a not in have_continents:
                        have_continents.append(a)
                for a in c[1]:
                    if a in continents:
                        continents.remove(a)
        num_continents = constraint[1][0]
        final_continents = have_continents
        for c in not_continents:
            if c in continents:
                continents.remove(c)
            if c in final_continents:
                final_continents.remove(c)
        # if len(final_continents) == int(num_continents):
        #     conts = random.sample(final_continents, int(num_continents))
        #     return random.choice(self.constraint_template[constraint[0] - 1]).split('at least')[0] + ', '.join(conts), True
        # else:
        #     return random.choice(self.constraint_template[constraint[0] - 1]).replace("[]", ", ".join(constraint[1])), False
        if len(final_continents) > int(num_continents):
            conts = random.sample(final_continents, int(num_continents))
            return random.choice(self.constraint_template[constraint[0] - 1]).split('at least')[0] + ', '.join(conts)
        if int(num_continents) - len(final_continents) > len(continents):
            return random.choice(self.constraint_template[constraint[0]-1]).replace("[]", ", ".join(constraint[1]))
        conts = random.sample(continents, int(num_continents) - len(final_continents))
        conts.extend(final_continents)
        return random.choice(self.constraint_template[constraint[0] - 1]).split('at least')[0] + ', '.join(conts)

    def get_at_least_pairs(self, constraints):
        '''
        Create list of tuples combining the number of troops in the at least constraint with the must have constraint
        Must have troops on Yellow + I need at least 12 troops to effectively defend a continent = I placed 12 troops on the Yellow continent
        Args:
            constraints:

        Returns:

        '''
        must_haves = []
        at_leasts = []
        for const in constraints:
            if const[0] == 1:
                must_haves.append(const[1])
            if const[0] == 5:
                at_leasts.append(const[1])
        return list(zip(must_haves, at_leasts))

    def combine_common_constraint_classes(self, constraints):
        '''
        Constraint to combine constraints with the same class,
        (I must not have troops on Blue, I must not have troops on Green, I must not have troops on Purple) -> I must not have troops on Blue, Green, Purple
        Args:
            constraints: list[(tuple)], list of constraints

        Returns: final_assignments: list[(tuple)]

        '''
        assignments = defaultdict(list)
        final_assignments = []
        common_classes = [1,2,3,4]
        for const in constraints:
            if const[0] in common_classes:
                assignments[const[0]].append(const[1])
            else:
                final_assignments.append((const[0], [str(const[1])]))
        for key in assignments.keys():
            final_assignments.append((key, assignments[key]))
        # print(final_assignments)
        return final_assignments

    def get_sentence(self, const, constraint_assignments):

        lang = random.choice(self.constraint_template[const[0] - 1])
        constraint_lang = lang.replace("[]", ", ".join(const[1]))
        must_have_prob = random.uniform(0, 1)
        # continent_prob = random.uniform(0, 1)
        # if check_continent_constraint(constraint_assignments) and continent_prob < 1:
        #     if const[0] == 1:
        #         return ''
        if const[0] == 1 and must_have_prob < 0.2:
            for country in const[1]:
                constraint_lang = self.generate_must_have_variant((const[0], country))
        # elif const[0] == 7 and continent_prob < 0.2:
        #     # replace must have at least X troops on a continent with continent names
        #     constraint_lang = self.generate_num_continents_variant(const, constraint_assignments)


        return constraint_lang.rstrip()

    def generate_language_per_constraint(self, constraint_assignments):
        """
        Args:
            constraint_assignments: Randomly generated constraints

        Returns: constraint_sentences: List of randomly assigned sentences for each constraint

        """
        constraint_sentences = []
        # constraint_assignments = [(7, 3), (5, 11), (2, 'Yellow')]
        if random.uniform(0,1) < 0.4:
            constraint_assignments = self.combine_common_constraint_classes(constraint_assignments)
        else:
            for i, const in enumerate(constraint_assignments):
                constraint_assignments[i] = (const[0], [str(const[1])])
        # print(constraint_assignments)
        if self.check_must_have_at_least(constraint_assignments) and random.uniform(0,1) < 0.8:
            pairs = self.get_at_least_pairs(constraint_assignments)
            for pair in pairs:
                langs = [f"I placed {pair[1][0]} troops on the {pair[0][0]} continent", f"I needed to place {pair[1][0]} troops to control {pair[0][0]}", f"I needed to have {pair[1][0]} troops on {pair[0][0]}", f"I needed to have {pair[1][0]} troops on {pair[0][0]} to defend it"]
                lang = random.choice(langs)
                constraint_sentences.append(lang.rstrip())
            for const in constraint_assignments:
                if const[0] == 1 or const[0] == 5:
                    continue
                constraint_lang = self.get_sentence(const, constraint_assignments)
                if constraint_lang == '':
                    continue
                constraint_sentences.append(constraint_lang)
        else:
            for const in constraint_assignments:
                constraint_lang = self.get_sentence(const, constraint_assignments)
                if constraint_lang == '':
                    continue
                constraint_sentences.append(constraint_lang)
        # print(constraint_sentences)
        # sdfsdf
        return constraint_sentences, constraint_assignments

unique_dict = defaultdict(bool)

def right_strip(sentence):
    '''
    Remove spaces and periods from the right end of a string
    '''
    sentence = sentence.replace(".", "")
    sentence = sentence.rstrip()
    return sentence

def check_continent_constraint(constraint_assignments):
    '''
    Check if constraint number 7 is present within the set of constraint assignments
    Args:
        constraint_assignments:

    Returns:

    '''
    for c in constraint_assignments:
        if int(c[0]) == 7:
            return True
    return False

def check_constraint_class(constraint, constraint_assignments):
    '''
    Make sure numbered constraints aren't repeated more than once within the same set of constraint assignments
    Args:
        constraint_assignments:
    Returns:

    '''
    for c in constraint_assignments:
        if c[0] == constraint[0]:
            return True
    return False

def bucket_assignment(value):
    '''
    Convert value between -100 to 100 to the appropriate bucket
    :param value: int
    :return: bucket: int
    '''
    if value < -60:
        return 0
    elif value < -20:
        return 1
    elif value < 20:
        return 2
    elif value < 60:
        return 3
    else:
        return 4



def choose_random_constraint():
    '''
    Generate a random constraint assignment
    :return: tuple(type, val)
    '''
    con_type = random.choice(list(constraint_classes.keys()))
    con_val = random.choice(constraint_values[con_type])
    return (con_type, con_val)

def consolidate_constraint(constraint):
    '''
    Concatenate tuple in full constraint as a sentence
    :param constraint: tuple(type, val)
    :return: final_constraint: str
    '''
    return constraint_classes[constraint[0]] + str(constraint[1]) + constraint_second_half[constraint[0]]

def save_strategy(goal_assignment,
                  constraint_assignment,
                  goals_lang,
                  constraint_lang,
                  generics_lang,
                  comparison_on=False,
                  save_path="../Data/Synthetic_Strategies.csv"):
    '''
    Save goal and constraint assignments to csv file
    :return: None
    '''
    with open(save_path, mode ='a', encoding='utf-8') as output_file:
        # output_file = csv.open("../Data/Generated_Strategy.csv")
        writer = csv.writer(output_file)
        constraints = []
        # print(constraint_assignment)
        for assignment in constraint_assignment:
            for val in assignment[1]:
                constraint = consolidate_constraint((assignment[0], val))
                constraints.append(constraint)
        # print(constraint_lang)
        language_arr = goals_lang + constraint_lang + generics_lang
        if not comparison_on:
            random.shuffle(language_arr)
        final_language = ". ".join(language_arr)
        final_strategy = [final_language] + [goal[1] for goal in goal_assignment] + constraints
        bucketed_strategy = [bucket_assignment(goal[1]) for goal in goal_assignment] + constraints
        if not unique_dict[tuple(bucketed_strategy)]:
            writer.writerow(final_strategy)
            unique_dict[tuple(bucketed_strategy)] = True

def save_augmented_strategy(goal_assignment,
                            constraint_assignment,
                            goals_lang,
                            constraint_lang,
                            generics_lang,
                            model,
                            tokenizer,
                            saved_aug_info,
                            comparison_on=False,
                            augs_per_sentence=3,
                            augs_per_strategy=1,
                            aug_sentence_percent=0.5,
                            min_edit_percent=0.1,
                            torch_device='cpu',
                            save_path="../Data/Augmented_Synthetic_Strategies.csv"):
    '''
    Save goal and constraint assignments to csv file
    :return: None
    '''
    with open(save_path, mode ='a', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        constraints = []
        constraints = []
        for assignment in constraint_assignment:
            for val in assignment[1]:
                constraint = consolidate_constraint((assignment[0], val))
                constraints.append(constraint)
        language_arr = goals_lang + constraint_lang + generics_lang
        if not comparison_on:
            random.shuffle(language_arr)
        final_language = ". ".join(language_arr) + '.'
        augmetation_dict, save_aug_info = run_augmentation_on_strategy_list([final_language],
                                                             augs_per_sentence,
                                                             augs_per_strategy,
                                                             aug_sentence_percent,
                                                             min_edit_percent,
                                                             model,
                                                             tokenizer,
                                                             torch_device,
                                                             saved_aug_info)
        final_augmented_language = augmetation_dict[0][0]

        final_strategy = [final_augmented_language] + [goal[1] for goal in goal_assignment] + constraints

        bucketed_strategy = [bucket_assignment(goal[1]) for goal in goal_assignment] + constraints
        if not unique_dict[tuple(bucketed_strategy)]:
            writer.writerow(final_strategy)
            unique_dict[tuple(bucketed_strategy)] = True
        elif comparison_on:
            writer.writerow(final_strategy)

        return save_aug_info

