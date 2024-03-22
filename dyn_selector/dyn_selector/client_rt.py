"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict
from dataset import load_dataset
from flwr.common import Config, Scalar
from models import create_CNN_model, create_MLP_model
import tensorflow as tf
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, ips) -> None:
        super().__init__()
        self.model = model
        self.ips = ips
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # Return the client's ips property.
        properties = {
            "ips": self.ips
        }

        return properties

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = config["local_epochs"]

        if epochs is None:
            epochs = 2

        batch_size = config["batch_size"]

        if batch_size is None:
            batch_size = 32
            
        fraction_samples = config["fraction_samples"]

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


def gen_client_fn(ips_mean, ips_var, num_clients, is_cnn: bool = False) -> Callable[[str], fl.client.Client]:

    # Generate num_clients random ips from uniform distribution
    ips_min = ips_mean - ips_var
    ips_max = ips_mean + ips_var
    ips_dict = {}
    for i in range(0, num_clients):
        ips_dict.update({str(i): np.random.uniform(ips_min, ips_max)})

    def client_fn(cid: str) -> fl.client.Client:
        # Load model
        if(is_cnn):
            model = create_CNN_model()
        else:
            model = create_MLP_model()
            
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        ips = ips_dict[cid]

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        # Create and return client
        return FlwrClient(model, x_train_cid, y_train_cid, ips)
    
    return client_fn