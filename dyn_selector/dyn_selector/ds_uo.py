"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import os
from typing import Dict, List, Optional, Tuple
from flwr.common.typing import Metrics
from utils import save_results_as_pickle
import numpy as np
from client_uo import gen_client_fn
from models import create_MLP_model, create_CNN_model
import hydra
from hydra.utils import instantiate
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


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

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    client_fn = gen_client_fn((cfg.epochs_min, cfg.epochs_max), 
                              (cfg.fraction_samples_min, cfg.fraction_samples_max), 
                              (cfg.batch_size_min, cfg.batch_size_max), 
                              cfg.num_clients, 
                              cfg.is_cnn)
    
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

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # Create strategy
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=get_evaluate_fn(server_model),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
    )

    # Initialize ray_init_args
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 1},
        config=fl.server.ServerConfig(cfg.num_rounds),
        strategy= strategy,
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