"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict
from models import create_CNN_model
from models import create_MLP_model
from dataset import load_dataset
from flwr.common import Config, Scalar
from omegaconf import DictConfig
import tensorflow as tf
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, ips, config) -> None:
        super().__init__()
        self.model = model
        self.ips = ips
        self.config = config
        split_idx = math.floor(len(x_train) * 0.95)  # Use 5% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # Return the client's dataset size
        x_entire = np.concatenate((self.x_train, self.x_val))

        properties = {
            "data_size": len(x_entire),
        }

        return properties

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = self.config["local_epochs"]

        batch_size = self.config["batch_size"]
            
        fraction_samples = self.config["fraction_samples"]

        learning_rate = config["learning_rate"]

        x_train_selected = self.x_train
        y_train_selected = self.y_train

        num_samples = len(self.x_train)

        # Randomly sample num_samples from the training set
        if fraction_samples is not None:
            num_samples = round(len(self.x_train) * fraction_samples)
            idx = np.random.choice(len(self.x_train), num_samples, replace=False)
            x_train_selected = self.x_train[idx]
            y_train_selected = self.y_train[idx]

        # Compute time metrics
        local_iterations = (epochs * num_samples) // batch_size
        estimated_time = local_iterations / self.ips

        print(f"""Client training on {len(x_train_selected)} samples, {epochs} epochs, 
              batch size {batch_size}, fraction samples {fraction_samples}, estimated time {estimated_time}""")


        # During training, update the learning rate as needed
        tf.keras.backend.set_value(self.model.optimizer.lr, learning_rate)

        self.model.set_weights(parameters)

        history = self.model.fit(x_train_selected, y_train_selected, batch_size=batch_size, epochs=epochs, verbose=2)
        return self.model.get_weights(), len(self.x_train), {"training_loss": history.history['loss'][-1], "estimated_time": estimated_time}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        if "first_phase" in config and config["first_phase"]:
            x_entire = np.concatenate((self.x_train, self.x_val))
            y_entire = np.concatenate((self.y_train, self.y_val))
            if config["is_cpow"] == False:
                # In the base variant, during the first phase we evaluate on the entire dataset
                loss, acc = self.model.evaluate(x_entire, y_entire, verbose=2)
            else:
                # In the cpow variant, during the first phase we evaluate on a mini-batch of b samples
                b = config["b"]
                idx = np.random.choice(len(x_entire), b, replace=False)
                x_entire_selected = x_entire[idx]
                y_entire_selected = y_entire[idx]
                loss, acc = self.model.evaluate(x_entire_selected, y_entire_selected, verbose=2)
        else:
            # In the normal evaluation phase, we evaluate on the validation set
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
        (x_train_cid, _) = load_dataset(cid, is_cnn)
        samples_per_client = len(x_train_cid)
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
        (x_train_cid, _) = load_dataset(cid, is_cnn)
        samples_per_client = len(x_train_cid)
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
            (x_train_cid, _) = load_dataset(cid, is_cnn)
            samples_per_client = len(x_train_cid)
            num_samples = max(1, int((local_iteration * batch_size) / local_epochs))
            if samples_per_client == 0:
                fraction_samples = 0
            else:
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
        
        model.compile("sgd", "sparse_categorical_crossentropy", metrics=["accuracy"])

        config = config_dict[cid]

        print(f"Client {cid} - Batch size {config['batch_size']} - Local epochs {config['local_epochs']}")

        ips = ips_dict[cid]

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        # Create and return client
        return FlwrClient(model, x_train_cid, y_train_cid, ips, config)
    
    return client_fn
