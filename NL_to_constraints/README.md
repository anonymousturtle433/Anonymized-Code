# NL -> Constraints Codebase
This folder contains all the code to train NL-> constraints model

# Dataset
The data used in this codebase is not currently available in this repository. You will need to get the original dataset from the authors of the paper. You will need the following three JSON files to run this code. 
1. nl_goals_constraints_train_human.json - Training Data (85% of the entire dataset)
2. nl_goals_constraints_val_human.json - Validation Data (15% of the entire dataset)
3. nl_goals_constraints_test_human.json - Testing Data (30 examples used for human-subjects study)

These three datasets should be saved to ```../NL_constraints_data_prep/Output_Data/```

If you are running k-fold cross validation, you will need to have the following data file instead 

```nl_goals_constraints_kfold.json```

This file contains the entire train + validation dataset which will be randomly split into folds by the code. 

# Train model
To train the model on the human dataset, run the following code
```commandline
python run_gat.py --datasets v1 v2 v3
```

To train the model on the synthetic dataset, run
```commandline
python run_gat.py --datasets syn
```

To train the model on the synthetic_augmented dataset, run
```commandline
python run_gat.py --datasets syn aug
```

To train the model on the human_augmented dataset, run
```commandline
python run_gat.py --datasets human_augmented
```

To run k-fold analysis, run the following code
```commandline
python run_gat_kfold.py --overwrite_cache --datasets syn v1 v2 v3 [etc] --use_saved_model 
```

### Arguments
1. ```--overwrite_cache``` : Re-create the JSON files for the training, validation and testing set from the original csv files. If you exclude this argument the code uses the existing JSON files. (You should never have to include this argument if you have the json files)
2. ```--datasets``` : This argument takes in a list of datasets from which to build the JSON file from. Unless you are training on the synthetic models, use ```v1 v2 v3```. 
3. ```--use_saved_model``` : This argument specifies whether or not you want to initialize the model with a pretrained model trained on the synthetic or synthetic augmented dataset 
4. ```--restart_from_conf```: This argument restarts the grid search from a specific configuration if you are running a grid search
5. ```--warmup_steps```: Number of steps for linear learning rate warmup

### Parameters
Here are some important parameters within the run_gat.py file along with the plausible values for those parameters

| Parameter        | Description                                                | Possible Values                                                                                                                                                                                 |
|------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_type       | The type of model to use for training                      | "_roberta-con-tokens_" - model with classification tokens <br> "_roberta-base_" - standard roberta-base model with not modifications <br> "_simple_" - standard RNN model with no modifications | 
| pretrained_model | The name of the pretrained model to load prior to training | "synthetic" <br> "synthetic_augmented" <br> None                                                                                                                                                |
| selections       | Method of incorporating selections to the model            | "bert_no_map" - Train with the addition of selections as Text appended to the end of the paragraph. <br> None - no selections                                                                   |


# Pretrained Models
This code can load models that were pretrained on the synthetic or synthetic-augmented dataset. For using these models, you need to save the pretrained models to the following location

```/saved_models/pretrained_model_name.pt```