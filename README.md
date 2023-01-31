# NL_Constraints
Code for learning goals and constraints from language project

Start with installing the setup file to ensure code used in this repository is imported correctly
```commandline
pip install -e .
```

If installing directly gives you import issues, please run the following command instead
```commandline
python setup.py develop
```

### Install Requirements
```commandline
conda create --name nl_constraints python=3.8.5
conda activate nl_constraints
pip install -r requirements.txt
```

###Data 

Keep all the json files containing training data in the folder `Risk_NL_Commanders_Intent/NL_constraints_data_prep/Output_Data`

The file names should be kept as - 

```
nl_goals_constraints_{train/val/test}_{suffix}.json
```

The suffix depends on the type of data being used. It should be - 

```
human = Human Data
syn = Synthetic Data 
aug = Augemeted Data
human_aug = Human Augmented Data 
synthetic_sug = Synthetic Augmented Data
```

