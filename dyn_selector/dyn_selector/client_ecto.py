"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict
from dataset import load_dataset
from flwr.common import Config, Scalar
from models import create_CNN_model, create_MLP_model
from omegaconf import DictConfig
import tensorflow as tf
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, config) -> None:
        super().__init__()
        self.model = model
        self.config = config
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = self.config["local_epochs"]

        batch_size = self.config["batch_size"]
            
        fraction_samples = self.config["fraction_samples"]

        x_train_selected = self.x_train
        y_train_selected = self.y_train

        # Randomly sample num_samples from the training set
        if fraction_samples is not None:
            num_samples = round(len(self.x_train) * fraction_samples)
            idx = np.random.choice(len(self.x_train), num_samples, replace=False)
            x_train_selected = self.x_train[idx]
            y_train_selected = self.y_train[idx]

        self.model.set_weights(parameters)
        history = self.model.fit(x_train_selected, y_train_selected, batch_size=batch_size, epochs=epochs, verbose=2)
        return self.model.get_weights(), len(self.x_train), {"training_loss": history.history["loss"][-1]}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def gen_client_fn(ips_mean: int, ips_var: int, num_clients: int, varying_config: Dict[str, float], default_config: Dict[str, float], comp_time: int, samples_per_client: int, is_cnn: bool = False) -> Callable[[str], fl.client.Client]:

    # Generate num_clients random ips from uniform distribution
    ips_min = ips_mean - ips_var
    ips_max = ips_mean + ips_var
    ips_dict = {}
    for i in range(0, num_clients):
        ips_dict.update({str(i): np.random.uniform(ips_min, ips_max)})

    # Compute local iterations for each client
    local_iterations = {}
    for cid, ips in ips_dict.items():
        local_iterations[cid] = int(comp_time * ips)

    config_dict = {}

    # if local epochs is the parameter varying:
    if varying_config["local_epochs"]:
        batch_size = default_config["batch_size"]
        fraction_samples = default_config["fraction_samples"]
        num_samples = round(samples_per_client * fraction_samples)
        for cid, local_iteration in local_iterations.items():
            local_epochs = max(1, int((local_iteration * batch_size) / num_samples))
            config = {
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "fraction_samples": fraction_samples,
            }
            config_dict[cid] = config
    elif varying_config["batch_size"]:
        local_epochs = default_config["local_epochs"]
        fraction_samples = default_config["fraction_samples"]
        num_samples = round(samples_per_client * fraction_samples)
        for cid, local_iteration in local_iterations.items():
            batch_size = max(1, int((num_samples * local_epochs) / local_iteration))
            config = {
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "fraction_samples": fraction_samples,
            }
            config_dict[cid] = config
    elif varying_config["fraction_samples"]:
        local_epochs = default_config["local_epochs"]
        batch_size = default_config["batch_size"]
        for cid, local_iteration in local_iterations.items():
            num_samples = max(1, int((local_iteration * batch_size) / local_epochs))
            fraction_samples = round(num_samples / samples_per_client, 2)
            fraction_samples = min(1.0, fraction_samples)
            config = {
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "fraction_samples": fraction_samples,
            }
            config_dict[cid] = config

    def client_fn(cid: str) -> fl.client.Client:

        # Load model
        if(is_cnn):
            model = create_CNN_model()
        else:
            model = create_MLP_model()
        
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        config = config_dict[cid]

        print(f"Client {cid} - Batch size {config['batch_size']} - Local epochs {config['local_epochs']}")

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        # Create and return client
        return FlwrClient(model, x_train_cid, y_train_cid, config)
    
    return client_fn