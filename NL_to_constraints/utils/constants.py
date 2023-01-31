CONSTRAINT_TYPES = {"<NA>": 0, "I must have troops on ": 1, "I must not have troops on ": 2, "I must be able to access ": 3, "I need to protect the borders of ": 4, "I need a total of at least ": 5, "I must have at least ": 6, "I must have troops on at least ": 7, "I must place at least ": 8, "I must have troops on at most ": 9}

VALUE_TYPES = {'<NA>': 0, 'Blue': 1, 'Green': 2, 'Yellow': 3, 'Red': 4, 'Purple': 5, '1': 6, '2': 7, '3':8, '4':9, '5':10, '6':11, '7':12, '8':13, '9':14, '10':15, '11':16, '12':17, '13': 18, '14':19 }



INV_CON = {v: k for k, v in CONSTRAINT_TYPES.items()}
# INV_CON[len(CONSTRAINT_TYPES)] = '<B>'
INV_VAL = {v: k for k, v in VALUE_TYPES.items()}

COUNTRY_MAP = {"<pad>": 0, "<sos>": 1, "<eos>": 2,  "Yellow_A": 3, "Yellow_B": 4, "Yellow_C": 5, "Yellow_D": 6, "Blue_A": 7, "Blue_B": 8, "Blue_C": 9, "Blue_D": 10, "Green_A": 11, "Green_B": 12, "Green_C": 13, "Green_D": 14, "Green_E": 15, "Purple_A": 16, "Purple_B": 17, "Purple_C": 18, "Purple_D": 19, "Purple_E": 20, "Red_A": 21, "Red_B": 22, "Red_C": 23}

INVALID_VALUES = {0: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 1: [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 2: [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 3: [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 4: [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 5: [i for i in range(6)], 6: [i for i in range(6)], 7: [i for i in range(6)], 8: [i for i in range(6)], 9: [i for i in range(6)], 10:[i for i in range(20)]}
# INVALID_VALUES = {0: [6,7,8,9,10,11,12,13,14,15,16,17,18,19], 1: [6,7,8,9,10,11,12,13,14,15,16,17,18,19], 2: [6,7,8,9,10,11,12,13,14,15,16,17,18,19], 3: [6,7,8,9,10,11,12,13,14,15,16,17,18,19], 4: [1,2,3,4,5], 5: [i for i in range(5)], 6: [i for i in range(5)], 7: [i for i in range(5)], 8: [i for i in range(5)], 9:[i for i in range(20)]}

BLANK_INDEX = len(CONSTRAINT_TYPES) * len(VALUE_TYPES)
