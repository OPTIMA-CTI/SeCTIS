from client import ctiSharingClient
from aggregator import Aggregator
from torch.utils.data import DataLoader
from dataset import *
from score import *
from model import Net
import os
import warnings
import time
import torch
import copy
warnings.filterwarnings("ignore")

# Configuration 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
DATASET_PATH = 'data/Darknet.csv'

NUM_LEARNERS = 6
BATCH_SIZE = 32
VALIDATOR_BATCH_SIZE=256
LEARNING_RATE = 1e-4
FL_ROUND = 50
LOCAL_EPOCH = 5


IS_TRUST_ON = True
TRUST_THRESHOLD = 0.8
REPUTATION_THRESHOLD = 0.8

ATTACK__ON = False
FLIPPED_LABEL=1
NEW_LABEL=3
FLIP_PCT=20
MALICIOUS_CLIENT_LIST= [2,5]

attack_name = f"No_Attack_TRUST_{IS_TRUST_ON}ATTACK_{ATTACK__ON}{FLIPPED_LABEL}_to_{NEW_LABEL}_of_{FLIP_PCT}_percentage"
PARAMETERS_FOLDER = f"{attack_name}_parameter/"

# ZK RELATED CONFIGURATIONS
CONTRACT_ADDRESS = "contract_address_here"
IPFS_GATEWAY = "ipfs_gateway_url_here"
WEB3_PROVIDER = "web3_provider_url_here"
ACCOUNT_PRIVATE_KEY = "private_key_here"
SMART_CONTRACT_ABI = []  

def main():
    start_total_time = time.time()
    print("Beginning...............")
    print("---------------------------------------------")
    
    if IS_TRUST_ON == True:
        all_rounds_trust = {}
        all_rounds_reputation = {}
        black_list = []   
        MODE = "SWARM"
    else:
        MODE = "FEDERATED"
    
    if ATTACK__ON == True:
        print("LABEL FLIPPENING ATTACK IS ON")
        trainloaders, testloader = prepare_dataset_attack(DATASET_PATH, NUM_LEARNERS, BATCH_SIZE, malicious_clients= MALICIOUS_CLIENT_LIST, flip_label=FLIPPED_LABEL, new_label=NEW_LABEL, flip_percentage=FLIP_PCT)

    else:
        print("LABEL FLIPPENING ATTACK IS OFF")
        trainloaders, testloader = prepare_dataset_no_attack(DATASET_PATH, NUM_LEARNERS, BATCH_SIZE)

    trainloaders_validators, _ = prepare_dataset_validators(DATASET_PATH, NUM_LEARNERS, BATCH_SIZE)
    print("---------------------------------------------")
    print(f"{MODE} LEARNING STARTS")
    print("---------------------------------------------")
    test=[1,2,3]
    print(test) 
    global_model = Net().to(DEVICE)  # Initializing the global model.
    # Main loop
    for global_epoch in range(0, FL_ROUND):
        print(global_epoch)
        start_time_training_global_epoch = time.time()
        print(f"Global Training Round: {global_epoch} ")
        # Train, export for each learner.
        for learner in range(0, NUM_LEARNERS):
            # Create a deep copy of the global model for each client
            client_model = copy.deepcopy(global_model).to(DEVICE)
            # Initialize the client
            client = ctiSharingClient(
                contract_address=CONTRACT_ADDRESS,
                ipfs_gateway=IPFS_GATEWAY,
                web3_provider=WEB3_PROVIDER,
                account_private_key=ACCOUNT_PRIVATE_KEY,
            )
            
            print(f"-------- Client {learner} is training ------- ")
            #print(f"Client: {learner}")
            print(trainloaders[learner])
            test=[1,2,3]
            print(test) 
            #print(f"id: {learner}, global epoch: {global_epoch}")
            client.train_model(client_model, trainloaders[learner], testloader, learner, LEARNING_RATE, LOCAL_EPOCH, global_epoch,attack_name,test)
            client.convert_model_to_onnx(learner,global_epoch,testloader, attack_name)
            print(test) 
            print("-----------------------------------------------------------")
        
        end_time_training_global_epoch = time.time()
        print("------------------------------------------------------------------------------------------------------------------------------------")
        print(f"Total Training for {NUM_LEARNERS} clients with Batch Size {BATCH_SIZE} time: {end_time_training_global_epoch - start_time_training_global_epoch} seconds")
        print("------------------------------------------------------------------------------------------------------------------------------------")
           
        # Aggregator
        round_trust = {}
        aggregator= Aggregator()
        if IS_TRUST_ON == True:
            # Validators
            print("-----------------------------------------------------------")
            print(f"Validators loading the models, and perform tests for the Global Training Round: {global_epoch}")
            print("-----------------------------------------------------------")
            
            all_outputs = execute_validator_tests(global_epoch, trainloaders_validators, NUM_LEARNERS, PARAMETERS_FOLDER, DEVICE)
            
            #print(all_outputs['Validator 1']) 
            print("-----------------------------------------------------------")
            print(f"Trust Score calculation for the Global Training Round: {global_epoch}")
            print("-----------------------------------------------------------")
            for id in range(0, NUM_LEARNERS):
                validator_str = f'Validator {id + 1}'
                round_trust[validator_str] = calculate_trust(id,all_outputs[validator_str], BATCH_SIZE)
                        
            all_rounds_trust[global_epoch] =  average_trust(round_trust) 
        
            # calculate the reputation scores.
            if global_epoch == 0:
                all_rounds_reputation[global_epoch] = all_rounds_trust[0].copy()
                print("-----------------------------------------------------------")
                print("GLOBAL EPOCH 0 REPUTATION SCORES:", all_rounds_reputation[global_epoch])
                print("-----------------------------------------------------------")
            else:
                alpha =  0.7 #calculate_dynamic_alpha(global_epoch, FL_ROUND)
                #print("I'M IN ELSE",all_rounds_trust[global_epoch])
                print("All rounds reputation before the update", all_rounds_reputation)
                for i, trust_score in all_rounds_trust[global_epoch].items():
                    if global_epoch not in all_rounds_reputation:
                        all_rounds_reputation[global_epoch] = {}  # Initialize with an empty dict or a default value
                    all_rounds_reputation[global_epoch][i] = alpha * all_rounds_reputation[global_epoch - 1][i] + (1 - alpha) * trust_score
                print(f"All rounds reputation after the updat. Epoch {global_epoch}",all_rounds_reputation[global_epoch] )
        

        if IS_TRUST_ON == True:
            # Filter models based on reputation score
            filtered_models = {}
            for model_id, reputation_score in all_rounds_reputation[global_epoch].items():
                if reputation_score >= REPUTATION_THRESHOLD and model_id not in black_list:
                    filtered_models[model_id] = reputation_score
                if global_epoch > 5 and reputation_score < REPUTATION_THRESHOLD :
                    black_list.append(model_id)
                    print(f"MODEL {model_id} is blacklisted.")
            print("-------------------------------------------")
            print(F"FILTERED MODELS GLOBAL EPOCH {global_epoch}")
            print("In this round the models above threshold are: ", filtered_models)
            print("-------------------------------------------")

            # Aggregate filtered models
            if filtered_models: # Check if there are any models above the threshold
                parameter_path = os.path.join(f"{attack_name}_parameter", f"global_round_{global_epoch}")
                print("PARAMETER PATH WHICH IS PASSED INTO THE AGGREGATE MODELS FUNCTION: ",parameter_path)
                aggreagate_parameter = aggregator.aggregate_selected_models(parameter_path, filtered_models)
                new_global_model = aggregator.update_new_global_model(global_model, aggreagate_parameter, global_epoch,attack_name)
                global_model = new_global_model
                aggregator.prediction_global_model(global_model, testloader, global_epoch,attack_name)
            else:
                print(f"No models with reputation score above {REPUTATION_THRESHOLD} in global round {global_epoch}. Skipping aggregation.")

        else:
            # Define the folder path for weights
            parameter_path = os.path.join(f"{attack_name}_parameter", f"global_round_{global_epoch}")
            print(parameter_path)
            aggreagate_parameter= aggregator.aggregate_models(parameter_path)
            print(aggreagate_parameter)
            new_global_model = aggregator.update_new_global_model(global_model, aggreagate_parameter,global_epoch,attack_name)
            global_model=new_global_model
            aggregator.prediction_global_model(global_model,testloader,global_epoch,attack_name)
            print("------------------------------------------------------------------------------------------------------------------------------------")
    end_total_time = time.time()
    print("------------------------------------------------------------------------------------------------------------------------------------")
    print(f"Total Training for {NUM_LEARNERS} clients with Batch Size {BATCH_SIZE} time: {end_total_time - start_total_time} seconds")
    print("------------------------------------------------------------------------------------------------------------------------------------")
           
    
if __name__ == "__main__":
    main()