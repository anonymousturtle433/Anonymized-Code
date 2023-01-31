#Evaluate Goal Model

The evaluate_goal_model script in this folder will automatically evaluate a model on the current human test set.

To use this script, follow these steps:
1. Make sure that the correct human test dataset is located in the Output_Data folder
2. Copy the desired model to this folder. Note that only goal classification models are currently supported.
3. Run the evaluate_goal_model script

After running the script, there should be the following new files:
```
eval_results_{model_name}.json - json file containing the validation and test set accuracies for this model
map_to_score_{model_name}.json - json file containing the score for the model on each map in the test set
map_to_predictions_{model_name}.csv - csv file containing the models predictions for each map and goal
```