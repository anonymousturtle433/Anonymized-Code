This folder contains code for training different language models for predicting natural goal value. To set up and run the code on the servers, do the following:

1. Set up an Anaconda environment (i.e. risk_goal) using the **.txt file corresponding to your server (landru/spocks brain) as the environment 
2. Edit run_model_to_goal.py file 
   1. Select the types of architectures you want to run (rnn, bert, transformer)
   2. For each architecture, choose the set of hyperparameters you want to search over 
   3. Save the file
3. Check that there are GPU resources available by running nvidia-smi. If training a bert or transformer model, make sure that no one else is using the GPU that you want to use. If running an RNN model, if GPU utilization is low (<40%) you should be fine to run on the same gpu. You can use `watch -n 0.5 nvidia-smi` to monitor GPU utilization
4. Run train_models.sh 
   1. If running train_models.sh for the first time, check to make sure that the file is activating the correct anaconda environment 
   2. If you want/need to use a GPU other than 0, you will need to edit the .sh file to include the --gpu_core argument and the desired gpu number. 
   3. There are options that can be changed using other arguments - for full documentation, see run_model_to_goal.py 
   4. If you want to be able to disconnect from the server without interupting the training, make sure you run the .sh file with nohup and specifiy the full file path to the.sh file
5. Monitor the training scripts progress! 
   1. There are lots of error that can occur when training models (too large for gpu, not enough memory on server to save models, bug in code, etc), and there's nothing worse than checking back in on training after a day expecting results, and only getting an error message
6. Once training has completed, review results 
   1. Results by default, results are saved in the model_results folder. If you change the same of this folder, make sure that you add the new folder name to the gitignore so that models aren't uploaded to git (the models will be too large)
   2. Within model results, there will be a folder for each architecture, a folder called train_metrics, a folder called val_metrics, and some saved models. 
      1. Within each architecture folder, the tensorboard logs for each trained model are saved. These can be viewed as normal tensorboard loag files 
      2. The train_metrics will include the final performance of each model on each metric on the training dataset as a pickle file. These can be exported into excel to get a better sense of how different models are performing (tensorboard has nice visuals, but struggles to compare many models at once). The export script is currently incomplete, but should be finished soon. 
      3. The val_metrics folder contains the same type of information as the train_metrics folder, but evaluated on the validation dataset 
      4. The saved models correspond to the models for each architecture and each different loss criterion that had the best validation loss. We have to save a subset of models as they can be quite large (~2 gb per model for some bert/transformer models). @@Important@@ Currently new models are saved during each run of the training script. If you run the training script in many small batches, there will be many saved models taking up space. Make sure that you are managing these models and not hogging space on the server.
7. After getting the best configuration you can run kfold analysis on your model. Use `train_model_kfold.sh` for the same. Currently it is setup to do the 10 fold analysis 

Other notes:
	-The code assumes that the training/validation .json files for any datasets being used are in the ..\Risk_NL_Commanders_Intent\NL_constraints_data_prep\Output_Data\* folder. If this is not the case, you will run into an error
	-The models folder contains code for defining the current set of pytorch models. It is not currently well commmented
	- Try not to run the same hyperparamater combination multiple times - this will mess up the tensorboard log file and overwrite the training/validation saved information. If you need to do this for some reason, it's best to change the root save folder
	- The training script defaults to evaluating the models on 5 classification buckets. This can be changed in the input paramters.
	- For more information on the evaluation metrics or the training code, see the files in the Utils folder
	- All folders for model output should be generated automatically - if any do not, that is a bug


















