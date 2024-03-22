"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable
from models import create_CNN_model, create_MLP_model
from dataset import load_dataset
from omegaconf import DictConfig
import tensorflow as tf
import flwr as fl
import numpy as np
import math

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = config["local_epochs"]

        if epochs is None:
            epochs = 2

        batch_size = config["batch_size"]

        if batch_size is None:
            batch_size = 32
            
        num_samples = config["num_samples"]

        x_train_selected = self.x_train
        y_train_selected = self.y_train

        # Randomly sample num_samples from the training set
        if num_samples is not None:
            idx = np.random.choice(len(self.x_train), num_samples, replace=False)
            x_train_selected = self.x_train[idx]
            y_train_selected = self.y_train[idx]

        print(f"Client training on {len(x_train_selected)} samples, {epochs} epochs, batch size {batch_size}")

        self.model.set_weights(parameters)
        history = self.model.fit(x_train_selected, y_train_selected, batch_size=batch_size, epochs=epochs, verbose=2)
        return self.model.get_weights(), len(self.x_train), {"training_loss": history.history["loss"][-1]}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def gen_client_fn(is_cnn: bool = False) -> Callable[[str], fl.client.Client]:

    def client_fn(cid: str) -> fl.client.Client:
        # Load model
        if(is_cnn):
            model = create_CNN_model()
        else:
            model = create_MLP_model()
        
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        # Create and return client
        return FlwrClient(model, x_train_cid, y_train_cid)
    
    return client_fn