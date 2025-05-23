Conversation opened. 1 read message.

Skip to content
Using Gmail with screen readers
score 

LATETS CODE
Inbox

SAMEERA K M <km4sameera@gmail.com>
Attachments
Mon, Sep 23, 2024, 2:51 PM
to me


 7 Attachments
  •  Scanned by Gmail
from aggregator import Aggregator
from torch.utils.data import DataLoader
from dataset import *
from score import *
from model import Net
import os
import warnings
import torch
import torch.nn.functional as F
import math
import csv
warnings.filterwarnings("ignore")

def execute_validator_tests(global_epoch,trainloaders, num_clients, base_folder, DEVICE):
    """
    Executes models for all validators on a single batch from each partition,
    storing outputs in a nested dictionary.

    Args:
    - trainloaders: List of DataLoader objects for each dataset partition.
    - num_clients: The number of client models per partition.
    - base_folder: Base folder path where model weights are stored.
    - DEVICE: The DEVICE to run the model on ('cpu' or 'cuda').

    Returns:
    - A nested dictionary with model outputs for each batch and client.
    """
    all_outputs = {}

    for validator_id, data_loader in enumerate(trainloaders, start=0):
        validator_key = f"Validator {validator_id}"
        # Fetch the first batch from the DataLoader
        inputs, _ = next(iter(data_loader))
        inputs = inputs.to(DEVICE)
        
        validator_results = {}

        for client_id in range(num_clients):
            model_key = f"Model {client_id}"
            weights_path = os.path.join(base_folder, f"global_round_{global_epoch}", f"parameter_client_id_{client_id}.pt")

            # Load the model and its weights
            model = Net().to(DEVICE)
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()

            with torch.no_grad():
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)

            # Store the outputs
            validator_results[model_key] = probabilities.cpu().numpy()

        # Store the results for this validator
        all_outputs[validator_key] = validator_results
        #print("VALIDATOR SET OUTPUT RESULT LENGHT",len(all_outputs[validator_key]))

    return all_outputs

def compute_centroid(outputs):
    centroid=np.mean(outputs, axis=0)
    return centroid

def compute_average_error(outputs, centroid,model_key,BATCH_SIZE):
    diff = outputs - centroid
    euclidean_norm = np.linalg.norm(diff, axis=1)
    error_model = np.sum(euclidean_norm) / BATCH_SIZE  # Divide by the number of samples
    return error_model


def trust_calculation(all_outputs,BATCH_SIZE, NUM_VALIDATORS): 
    #Equation 2
    P_model_validator = {} # Initialize empty dictionary to store P_model_validator
    # Iterate over each validator
    for validator_key, validator_results in all_outputs.items():
        validator_centroid = compute_centroid(np.array(list(validator_results.values()))) # Compute centroid for the validator  
        validator_errors = {} # Initialize empty dictionary to store errors for each model

        # Iterate over each model's outputs
        for model_key, outputs in validator_results.items():          
            average_error = compute_average_error(outputs, validator_centroid,model_key,BATCH_SIZE) # Compute average error for the model
            validator_errors[model_key] = average_error
      
        P_model_validator[validator_key] = validator_errors # Store errors for the validator

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Equation 3  
    P_model = {} # Initialize dictionary to store average trust scores for each model

    # Iterate over models
    for model_key in all_outputs['Validator 0'].keys():
        model_scores_sum = 0

        # Iterate over validators
        for validator_key in all_outputs.keys():
            model_scores_sum += P_model_validator[validator_key][model_key] # Accumulate trust score for the current model and validator

        P_model[model_key] = model_scores_sum / NUM_VALIDATORS # Calculate average trust score for the model

     #------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Equation 4
    
    trust_values = {} # Initialize dictionary to store trust values for each model

    # Calculate trust value for each model
    for model_key, average_error  in P_model.items():
        trust_value = 1 - average_error 
        trust_values[model_key] = trust_value
    return trust_values

# Define function to save reputation models to CSV
def save_to_csv(models, global_epoch, filename, column_names):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for model_name, model_details in models.items():
            writer.writerow([global_epoch, model_name, model_details])
    csvfile.close()  # Close the file after writing

def calculate_dynamic_alpha(round_number):
    return 1 / math.exp(round_number - 1)
score.py
Displaying score.py.in:sent georgiana. Press tab to insert.