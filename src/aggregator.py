
import torch
import os
import copy
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support,confusion_matrix


class Aggregator:
    
    def aggregate_models(self, parameter_path):
        """
        Aggregate model parameters from multiple clients to update the global model.

        Args:
            parameter_path (str): Path to the folder containing model parameters from different clients.

        Returns:
            aggregated_parameters (dict): Aggregated model parameters representing the updated global model.
        """
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # List all files in the folder
        model_files = os.listdir(parameter_path)
        print(model_files)
        # Check if there are any parameters to aggregate
        if not model_files:
            raise ValueError("No parameters to aggregate.")

        # Initialize aggregated parameters with parameters from the first client
        first_model_path = os.path.join(parameter_path, model_files[0])
        print(first_model_path)
        aggregated_parameters = torch.load(first_model_path, map_location=DEVICE)
        print(aggregated_parameters)
        # Aggregate parameters from other clients
        for model_file in model_files[1:]:
            model_path = os.path.join(parameter_path, model_file)
            client_params = torch.load(model_path, map_location=DEVICE)
            print(client_params)

            # Add the parameters from each client to the aggregated parameters
            for key in aggregated_parameters.keys():
                # Aggregate the parameter values by averaging
                aggregated_parameters[key] += client_params[key]

        # Normalize the aggregated parameters by dividing by the number of clients
        num_clients = len(model_files)
        for key in aggregated_parameters.keys():
            aggregated_parameters[key] /= num_clients

        print("Successfully aggregated")
        print(aggregated_parameters)
        return aggregated_parameters

    def update_new_global_model(self, global_model, model_parameters, global_epoch,attack_name):
        """
        Update the global model with the aggregated parameters.

        Args:
            global_model (torch.nn.Module): The global model to be updated.
            model_parameters (dict): Aggregated model parameters obtained from the aggregation step.

        Returns:
            new_model (torch.nn.Module): Updated global model.
        """
       # Load the state_dict of the global model
        state_dict = global_model.state_dict()
        #print("------- state_dict ------- \n",state_dict)
        # Update the state_dict with the aggregated parameters
        for key in state_dict.keys():
            if key in model_parameters:
                state_dict[key] = model_parameters[key]
        

        # Load the updated state_dict back to the global model
        global_model.load_state_dict(state_dict)

        # Define the folder path for the global model
        folder_path = os.path.join(f"{attack_name}_global_model")

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Save the global model
        model_path = os.path.join(folder_path, f"model_round_{global_epoch}.pth")
        torch.save(global_model.state_dict(), model_path)

        print(f"Global model updated and saved in {model_path}")
        return global_model
    
    def prediction_global_model(self, global_model, test_loader, global_epoch,attack_name):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Calculate evaluation metrics
        global_model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = global_model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate evaluation metrics
        cm = confusion_matrix(all_labels, all_predictions)
        # Calculate evaluation metrics
        classification_rep = classification_report(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

        # Verbose print statement to indicate the start of evaluation metrics section
        print("Global ModelEvaluation Metrics:")

        # Print a line of dashes for visual separation
        print("-----------------------------------------------------------------------------------------")

        # Print the classification report with a descriptive label
        print("Global Model Classification Report:\n", classification_rep)
        
        # Verbose print statements for each evaluation metric
        print("Weighted F1 Score:", f1_weighted)
        print("Weighted Precision:", precision_weighted)
        print("Weighted Recall:", recall_weighted)
        print("Macro F1 Score:", f1_macro)
        print("Macro Precision:", precision_macro)
        print("Macro Recall:", recall_macro)

        # Define the folder path for the global model
        folder_path = os.path.join(f"{attack_name}_global_model")

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        metrics_file_path = os.path.join(folder_path, f"evaluation_metrics_round_{global_epoch}.txt")
        with open(metrics_file_path, 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("Classification Report:\n")
            f.write(classification_rep)
            f.write("\n\nWeighted F1 Score: {}\n".format(f1_weighted))
            f.write("Weighted Precision: {}\n".format(precision_weighted))
            f.write("Weighted Recall: {}\n".format(recall_weighted))
            f.write("Macro F1 Score: {}\n".format(f1_macro))
            f.write("Macro Precision: {}\n".format(precision_macro))
            f.write("Macro Recall: {}\n".format(recall_macro))

        print("Global model Evaluation metrics saved successfully.")

    def match_model_names(self, threshold_dict, model_files):
        """
        Match model names in the dictionary with the corresponding file names.

        Args:
            threshold_dict (dict): Dictionary containing model names above the threshold.
            model_files (list): List of file names.

        Returns:
            matched_models (list): List of model names matched with the file names.
        """
        matched_models = []

        # Extract model IDs from file names and match with dictionary keys
        for model_file in model_files:
            model_id = model_file.split('_')[-1][:-3]  # Extract model ID from file name
            model_name = f"Model {model_id}"
            if model_name in threshold_dict:
                matched_models.append(model_file)

        return matched_models
           
    def aggregate_selected_models(self, parameter_path, threshold_dict):
        """
        Aggregate model parameters from specific clients to update the global model.

        Args:
            parameter_path (str): Path to the folder containing model parameters from different clients.
            threshold_dict (dict): Dictionary containing models above the threshold.

        Returns:
            aggregated_parameters (dict): Aggregated model parameters representing the updated global model.
        """
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Filter the array of model files based on models above the threshold
        model_files = os.listdir(parameter_path)
        model_files_above_threshold = self.match_model_names(threshold_dict, model_files)

        # Check if model_files_above_threshold is not empty
        if not model_files_above_threshold:
            raise ValueError("No model files found above the threshold.")

        # Initialize aggregated parameters
        first_model_path = os.path.join(parameter_path, model_files_above_threshold[0])
        aggregated_parameters = torch.load(first_model_path, map_location=DEVICE)

        # Aggregate parameters from specified model files
        for model_file in model_files_above_threshold[1:]:
            model_path = os.path.join(parameter_path, model_file)
            client_params = torch.load(model_path, map_location=DEVICE)

            # Add the parameters from each client to the aggregated parameters
            for key in aggregated_parameters.keys():
                # Aggregate the parameter values by averaging
                aggregated_parameters[key] += client_params[key]

        # Normalize the aggregated parameters by dividing by the number of clients
        num_clients = len(model_files_above_threshold)
        for key in aggregated_parameters.keys():
            aggregated_parameters[key] /= num_clients

        print("Successfully aggregated")
        return aggregated_parameters

        
