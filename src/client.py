## Query a smart contract to see if the next round is ready.
## If the next round is ready, get the IPFS hash of the latest global model and download it from IPFS.
## Once download, train the model with your own data.
## Once you train the model do the followings:
#   -  Query the blockchain and get the SRS string.
#   -  Convert pytorch model to ONNX.
#   -  Generate the circuit using EZKL library.
#   -  Deploy the circuit to ethereum blockchain using EZKL Libary and foundry.
#   -  Call the main smart contract on blockchain and upload the contract address of the verifier smart contract for your ONNX model. 
#   -  Upload the gradients hash, model hash to the main smart contract on the blockchain and the files to the IPFS.
# Now you can start listening the blockchain for an even called, next round is ready. Once it's ready, do the process again.


import torch
from model import Net
from collections import OrderedDict
from typing import Dict
import os
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
import warnings
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR

# Disable all warnings
warnings.filterwarnings("ignore")
# Placeholder imports for EZKL and Foundry
# from ezkl import generate_circuit, deploy_circuit
# Assume Foundry functionalities are wrapped in Python

class ctiSharingClient:
    def __init__(self, contract_address, ipfs_gateway, web3_provider, account_private_key):
        #self.web3 = Web3(Web3.HTTPProvider(web3_provider))
        #self.contract = self.web3.eth.contract(address=contract_address, abi=your_contract_abi_here)
        #self.ipfs = ipfs_connect(ipfs_gateway)
        #self.account = self.web3.eth.account.privateKeyToAccount(account_private_key)
        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def query_next_round_ready(self):
        return self.contract.functions.isRoundReady().call()

    def download_onnx_model_from_ipfs(self, ipfs_hash):
        # Download and load the ONNX Model.
        pass

    def train_model(self, model, train_loaders, test_loader, learner, lr, local_epochs, global_epoch,attack_name,test):

        # Evaluate the model on the test set
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate evaluation metrics
        test=set(test)
        test.add(4)
        print(test)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        print("Client Weighted F1 Score before training:", f1_weighted)

        id = learner
        # A very standard looking optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        
        # Set the model to training mode
        model.train()
        for epoch in range(local_epochs):
            print("Client {}, Local epoch {}".format(id, epoch))
            for data in train_loaders:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{local_epochs}], Loss: {loss.item():.4f}')


        # Convert model parameters to a dictionary
        state_dict = model.state_dict()
        # Convert parameters to a list of tensors
        #parameters = [value.cpu().detach().numpy() for value in state_dict.values()]

        folder_path_parameters = os.path.join(f"{attack_name}_parameter", f"global_round_{global_epoch}")
        os.makedirs(folder_path_parameters, exist_ok=True)  # Create folder if it doesn't exist
        name = f"parameter_client_id_{id}"  
        weight_file_name = os.path.join(folder_path_parameters, f"{name}.pt")
        torch.save(state_dict, weight_file_name)
        print(" - Weights saved successfully.")

        # Define the folder path for the model
        folder_path_model = os.path.join(f"{attack_name}_local_model", f"global_round_{global_epoch}")
        os.makedirs(folder_path_model, exist_ok=True)  # Create folder if it doesn't exist
        
        # Save the entire model to a file
        model_file_path = os.path.join(folder_path_model, f"local_model_client_id_{id}.pt")
        torch.save(model.state_dict(), model_file_path)

        #print(" - Model saved successfully.")
        # Share parameters to blockchain after training for two epochs
        #self.update_smart_contract(verifier_contract_address, model_hash, gradients_hash)
        # Evaluate the model on the test set
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate evaluation metrics
        cm = confusion_matrix(all_labels, all_predictions)
        classification_rep = classification_report(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        # Print the evaluation metrics
        #print("Classification Report:\n", classification_rep)
        print("Client Weighted F1 Score:", f1_weighted)
        #print("Weighted Precision:", precision_weighted)
        #print("Weighted Recall:", recall_weighted)
        #print("Macro F1 Score:", f1_macro)
        #print("Macro Precision:", precision_macro)
        #print("Macro Recall:", recall_macro)
        # Save evaluation metrics to a file
        metrics_file_path = os.path.join(folder_path_model, f"before_aggregation_evaluation_metrics_client_id_{id}.txt")
        with open(metrics_file_path, 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report:\n")
            f.write(classification_rep)
            f.write(f"\n\nWeighted F1 Score: {f1_weighted}\n")
            f.write(f"Precision: {precision_weighted}, Recall: {recall_weighted}\n")
            f.write(f"\n\nMacro F1 Score: {f1_macro}\n")
            f.write(f"Precision: {precision_macro}, Recall: {recall_macro}\n")

        print(" - Evaluation metrics saved successfully.")
        print("-----------------------------------------------------------------------------------------")

    def convert_model_to_onnx(self,id,global_epoch,testloader,attack_name):
        # DINCY <CONVERT THE MODEL TO ONNX>
        # Get a batch of data from the test loader
        inputs, _ = next(iter(testloader))
        inputs = inputs.to(self.device)  # Move inputs to the GPU
        # Define the folder path where the model is saved
        folder_path_model = os.path.join(f"{attack_name}_local_model",f"global_round_{global_epoch}")

        # Load the model from the saved state dictionary
        model_file_path = os.path.join(folder_path_model, f"local_model_client_id_{id}.pt")
        
        model_state_dict = torch.load(model_file_path)
        model = Net().to(self.device)  # Move model to the GPU
        model.load_state_dict(model_state_dict)
        model.eval()

        folder_path_onnx = os.path.join(f"{attack_name}_onnx",f"global_round_{global_epoch}")
        os.makedirs(folder_path_onnx, exist_ok=True)  # Create folder if it doesn't exist

        # Define the output path for the ONNX model
        output_path = os.path.join(folder_path_onnx, f"local_model_client_id_{id}.onnx")

        # Export the model to ONNX format
        torch.onnx.export(
            model,                     # PyTorch model to convert
            inputs,                      # Dummy input data (can be None)
            output_path,               # Path to save the ONNX model
            export_params=True,        # Export model parameters
            opset_version=11,          # ONNX opset version
            do_constant_folding=True,  # Fold constant operations
            input_names=["input"],     # Input names
            output_names=["output"],   # Output names
            dynamic_axes={             # Dynamic axes configuration
                "input": {0: "batch_size"},  # Variable batch size axis
                "output": {0: "batch_size"}
            }
        )

        print(" - Model converted to ONNX format and saved at:", output_path)

    def prepare_input_json(id):
        # Define the folder path for ONNX models
        onnx_folder_path = os.path.join("python", "onnx")
        os.makedirs(onnx_folder_path, exist_ok=True)  # Ensure the folder exists

        # Construct the path to the ONNX model file based on the ID
        onnx_model_path = os.path.join(onnx_folder_path, f"model_id_{id}.onnx")

        # Load the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        print("ONNX model loaded successfully.")

        # Define the folder path for JSON files
        json_folder_path = os.path.join("python", "json")
        os.makedirs(json_folder_path, exist_ok=True)  # Ensure the folder exists

        # Define the output JSON file path
        output_json_path = os.path.join(json_folder_path, f"model_id_{id}.json")

        # Convert the ONNX model to a JSON string
        onnx_model_json = onnx_model.SerializeToString()

        # Write the JSON string to a file
        with open(output_json_path, "wb") as json_file:
            json_file.write(onnx_model_json)

        print("JSON representation of the ONNX model saved at:", output_json_path)
    
    def run_command(self, command):
        """Run a shell command and return its output."""
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        return process.stdout
    
    def generateCircuit(self, model_path="network.onnx", input_json="input.json"):
        """Generate the circuit using the EZKL library."""
        # Copy the model and input files to the current directory
        subprocess.run(f"cp {model_path} ./", shell=True)
        subprocess.run(f"cp {input_json} ./", shell=True)
        
        # Generate settings, calibrate, get SRS, compile circuit, and setup
        self.run_command("ezkl gen-settings -M network.onnx")
        self.run_command("ezkl calibrate-settings -M network.onnx -D input.json --target resources")
        settings_json = json.loads(self.run_command("ezkl get-srs -S settings.json"))
        self.run_command("ezkl compile-circuit -M network.onnx -S settings.json --compiled-circuit network.ezkl")
        self.run_command("ezkl setup -M network.ezkl --srs-path=kzg.srs")
        
        return settings_json
    
    def deployVerifier(self, rpc_url="http://127.0.0.1:3030"):
        """Deploy the EVM verifier to an Ethereum blockchain."""
        # Generate the EVM verifier Solidity code
        self.run_command("ezkl create-evm-verifier --srs-path=kzg.srs --vk-path vk.key --sol-code-path verif.sol --settings-path=settings.json")
        
        # Generate witness and prove
        self.run_command("ezkl gen-witness -D input.json -M network.ezkl")
        self.run_command("ezkl prove --witness witness.json -M network.ezkl --proof-path model.pf --pk-path pk.key --srs-path=kzg.srs")
        
        # Deploy the verifier smart contract
        deploy_output = self.run_command(f"ezkl deploy-evm-verifier --addr-path=addr.txt --rpc-url={rpc_url} --sol-code-path verif.sol")
        
        return deploy_output

    def upload_to_ipfs(self, file_path):
        result = self.ipfs.add(file_path)
        return result['Hash']

    def update_smart_contract(self, verifier_contract_address, model_hash, gradients_hash):
        # Assuming you have a function in your contract for this
        nonce = self.web3.eth.getTransactionCount(self.account.address)
        txn_dict = self.contract.functions.updateModelInfo(verifier_contract_address, model_hash, gradients_hash).buildTransaction({
            'chainId': 1,  # Adjust according to your network
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_txn = self.web3.eth.account.signTransaction(txn_dict, private_key=self.account.privateKey)
        txn_receipt = self.web3.eth.sendRawTransaction(signed_txn.rawTransaction)
        return txn_receipt

    def listen_for_next_round_event(self):
        pass

    # Additional methods for SRS string querying, etc.