# Eval Scripts
This directory contains relevant scripts to run scripts to generate eval files from the output of the model and human predictions

## Data 
Input data should be located in the /Data folder. 
```text
Goals - model_predictions_best_acc_model_goals.csv
Constraints - model_predictions_best_acc_model_constraints.csv
Human - Risk_Human_Eval_Dataset.xlsx
```
Note: Before running any of the following scripts, you will need to make a few changes to ```model_predictions_best_acc_model_constraints.csv``` if you are using a fresh file output by the model. The data in the first two columns are swapped. The first column contains the Map IDs despite "Map" being the second column. Therefore you will need to swap these two columns. You also need to remove the line of "-- -- -- --" in the middle of the csv file. The code will break if you leave it in. 

## Generate csv files for R analysis 
Run the following scripts to compute the csv files for running statistical tests in R
```commandline
python user_study_accuracy.py
cp human_scores.csv all_scores_constraints.csv
cp human_scores_goals.csv all_scores_goals.csv
python model_accuracy.py
```



These scripts should generate two files which are 
```text
all_scores_constraints.csv - CSV file for the constraints results
all_scores_goals.csv - CSV file for the goals results
```