"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict, Tuple
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
    def __init__(self, model, x_train, y_train, ips, epochs, fraction_samples, batch_size) -> None:
        super().__init__()
        self.model = model
        self.ips = ips
        self.epochs = epochs
        self.fraction_samples = fraction_samples
        self.batch_size = batch_size 
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
        epochs = self.epochs     
        fraction_samples = self.fraction_samples
        batch_size = self.batch_size  

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


def gen_client_fn(ips_mean: int, ips_var: int, epochs: Tuple[int, int], fraction_samples: Tuple[int, int], batch_size: Tuple[int, int], num_clients: int, is_cnn: bool = False) -> Callable[[str], fl.client.Client]:

    # Generate num_clients random ips from uniform distribution
    ips_min = ips_mean - ips_var
    ips_max = ips_mean + ips_var
    ips_dict = {}
    for i in range(0, num_clients):
        ips_dict.update({str(i): np.random.uniform(ips_min, ips_max)})


    # Generate epochs, fraction_samples, and batch_size for each client from a uniform distribution
    epochs_min, epochs_max = epochs
    fraction_samples_min, fraction_samples_max = fraction_samples
    batch_size_min, batch_size_max = batch_size

    epochs_dict = {}
    fraction_samples_dict = {}
    batch_size_dict = {}

    for i in range(0, num_clients):
        # Generate random values
        epochs_value = np.random.uniform(epochs_min, epochs_max)
        fraction_samples_value = np.random.uniform(fraction_samples_min, fraction_samples_max)
        batch_size_value = np.random.uniform(batch_size_min, batch_size_max)

        # Round values
        epochs_rounded = int(round(epochs_value))
        fraction_samples_rounded = round(fraction_samples_value, 2)
        batch_size_rounded = int(round(batch_size_value))

        # Store the rounded values in dictionaries
        epochs_dict[str(i)] = epochs_rounded
        fraction_samples_dict[str(i)] = fraction_samples_rounded
        batch_size_dict[str(i)] = batch_size_rounded

        print(f"Client {i} epochs: {epochs_dict[str(i)]}, fraction_samples: {fraction_samples_dict[str(i)]}, batch_size: {batch_size_dict[str(i)]}")

    def client_fn(cid: str) -> fl.client.Client:
        # Load model
        if(is_cnn):
            model = create_CNN_model()
        else:
            model = create_MLP_model()
        
        model.compile("sgd", "sparse_categorical_crossentropy", metrics=["accuracy"])

        ips = ips_dict[cid]

        epochs = epochs_dict[cid]
        fraction_samples = fraction_samples_dict[cid]
        batch_size = batch_size_dict[cid]

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        (x_train_cid, y_train_cid) = load_dataset(cid, is_cnn)

        # Create and return client
        return FlwrClient(model, x_train_cid, y_train_cid, ips, epochs, fraction_samples, batch_size)
    
    return client_fn
