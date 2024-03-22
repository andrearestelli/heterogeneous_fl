"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.

This script reproduces results using dynamic selector and static optimizer techniques
"""

import os
from typing import Dict, List, Optional, Tuple
from models import create_MLP_model, create_CNN_model
from flwr.common.typing import Metrics
from utils import save_results_as_pickle
from client import gen_client_fn
import hydra
import numpy as np
import flwr as fl
from logging import DEBUG, INFO
from flwr.common.logger import log
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    client_fn = gen_client_fn(cfg.is_cnn)

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    # Initialize ray_init_args
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config(server_round: int):
            """Return training configuration dict for each round.

            Take batch size, local epochs and number of samples of each client from the server config
            """

            config = {
                "batch_size": 32,
                "local_epochs": 1 if server_round < 2 else 2,
                "num_samples": None,
            }

            config["batch_size"] = cfg.batch_size
            config["local_epochs"] = cfg.local_epochs
            config["num_samples"] = cfg.num_samples

            log(INFO, f"Round {server_round} training config: batch_size={config['batch_size']}, local_epochs={config['local_epochs']}, num_samples={config['num_samples']}")

            return config
        
        return fit_config
    
    def get_fit_metrics_aggregation_fn():
        def fit_metrics_aggregation_fn(results: List[Tuple[int, Metrics]]) -> Metrics:
            # Initialize lists to store training losses
            training_losses = []

            # Extract training losses and client counts from results
            for _, metrics in results:
                if "training_loss" in metrics:
                    training_loss = metrics["training_loss"]
                    training_losses.append(training_loss)

            # Calculate the variance and average of training loss
            variance_training_loss = np.var(training_losses)
            average_training_loss = np.mean(training_losses)

            # Create the aggregated metrics dictionary
            aggregated_metrics = {
                "variance_training_loss": variance_training_loss,
                "average_training_loss": average_training_loss,
            }

            return aggregated_metrics

        return fit_metrics_aggregation_fn
    
    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""

        test_folder = cfg.dataset.dataset

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        x_test = np.load(os.path.join(test_folder, "x_test.npy"))
        y_test = np.load(os.path.join(test_folder, "y_test.npy"))

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
            return loss, {"accuracy": accuracy}

        return evaluate
    
    if cfg.is_cnn:
        server_model = create_CNN_model()
    else:
        server_model = create_MLP_model()
    
    server_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(
        cfg.strategy,
        on_fit_config_fn=get_on_fit_config(),
        evaluate_fn=get_evaluate_fn(server_model),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
    )

    # 5. Start Simulation
    # history = fl.simulation.start_simulation(<arguments for simulation>)

    print("Starting simulation")

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

    # 6. Save your results
    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

if __name__ == "__main__":
    main()
