"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import os
from typing import Dict, List, Optional, Tuple
from utils import save_results_as_pickle
from models import create_CNN_model, create_MLP_model
from client_rt import gen_client_fn
import hydra
import numpy as np
import flwr as fl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from flwr.common.typing import Metrics
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

    client_fn = gen_client_fn(cfg.client.mean_ips, cfg.client.var_ips, cfg.num_clients, cfg.is_cnn)

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    # Initialize ray_init_args
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }

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
        evaluate_fn=get_evaluate_fn(server_model),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
        max_local_epochs=cfg.epochs_max,
        batch_size = cfg.batch_size_default,
        fraction_samples = cfg.fraction_samples_default,
        use_RT=True 
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
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
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