"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import os

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
from logging import INFO
from typing import Dict, List, Optional, Tuple

import flwr as fl
import hydra
import numpy as np
from flwr.common.logger import log
from flwr.common.typing import Metrics
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from strategy import RTFedAvg

from power_of_choice.client_rt import gen_client_fn
from power_of_choice.models import create_CNN_model, create_MLP_model
from power_of_choice.server import PowerOfChoiceCommAndCompVariant, PowerOfChoiceServer
from power_of_choice.utils import save_results_as_pickle

enable_tf_gpu_growth()


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
            estimated_times = []

            # Extract training losses and client counts from results
            for _, metrics in results:
                if "training_loss" in metrics:
                    training_loss = metrics["training_loss"]
                    training_losses.append(training_loss)
                if 'estimated_time' in metrics:
                    estimated_time = metrics['estimated_time']
                    estimated_times.append(estimated_time)

            # Calculate the variance and average of training loss
            variance_training_loss = np.var(training_losses)
            average_training_loss = np.mean(training_losses)

            # Create the aggregated metrics dictionary
            aggregated_metrics = {
                "variance_training_loss": variance_training_loss,
                "average_training_loss": average_training_loss,
                'estimated_times': estimated_times
            }

            return aggregated_metrics

        return fit_metrics_aggregation_fn

    def get_on_evaluate_config(is_cpow: bool, b: Optional[int] = None):
        def evaluate_config(server_round: int):
            """Return evaluation configuration dict for each round.

            In case we are using cpow variant, we set b to the value specified in the
            configuration file.
            """
            config = {
                "is_cpow": False,
            }

            if is_cpow:
                config["is_cpow"] = True
                config["b"] = b

            return config

        return evaluate_config

    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""
        print(f"Current folder is {os.getcwd()}")

        test_folder = "mnist"
        if cfg.is_cnn:
            test_folder = "cifar10"

        # Load data and model here to avoid the overhead of doing it in `evaluate`
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

    server_model.compile(
        "adam", "sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    is_cpow = False
    is_rpow = False
    is_rand = False
    if cfg.variant == "cpow":
        is_cpow = True
    elif cfg.variant == "rpow":
        is_rpow = True
    elif cfg.variant == "rand":
        is_rand = True

    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    if is_rpow:
        # Build Atmp dictionary with num_of_clients items
        # with key=client_id and value=inf
        atmp = {}
        for i in range(cfg.num_clients):
            atmp[str(i)] = float("inf")

        # Instantiate strategy
        strategy = instantiate(
            cfg.strategy,
            variant="rpow",
            atmp=atmp,
            evaluate_fn=get_evaluate_fn(server_model),
            on_evaluate_config_fn=get_on_evaluate_config(is_cpow),
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
            max_local_epochs=cfg.epochs_max,
            batch_size = cfg.batch_size_default,
            fraction_samples = cfg.fraction_samples_default,
            use_RT=True
        )
    elif is_rand:
        # Instantiate FedAvg strategy
        strategy = RTFedAvg(
            fraction_fit=round(cfg.strategy.ck / cfg.num_clients, 2),
            fraction_evaluate=round(cfg.strategy.ck / cfg.num_clients, 2),
            min_fit_clients=round(cfg.strategy.ck / cfg.num_clients),
            min_evaluate_clients=round(cfg.strategy.ck / cfg.num_clients),
            evaluate_fn=get_evaluate_fn(server_model),
            on_evaluate_config_fn=get_on_evaluate_config(is_cpow, cfg.b),
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
            max_local_epochs=cfg.epochs_max,
            batch_size = cfg.batch_size_default,
            fraction_samples = cfg.fraction_samples_default,
            use_RT=True
        )
    else:
        # Instantiate strategy with base config
        strategy = instantiate(
            cfg.strategy,
            evaluate_fn=get_evaluate_fn(server_model),
            on_evaluate_config_fn=get_on_evaluate_config(is_cpow, cfg.b),
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
            max_local_epochs=cfg.epochs_max,
            batch_size = cfg.batch_size_default,
            fraction_samples = cfg.fraction_samples_default,
            use_RT=True
        )

    client_manager = SimpleClientManager()

    if is_rpow:
        # Instantiate rpow server with strategy and client manager
        server = PowerOfChoiceCommAndCompVariant(
            strategy=strategy, client_manager=client_manager
        )
    elif is_rand:
        log(INFO, "Using FedAvg strategy")
        server = Server(strategy=strategy, client_manager=client_manager)
    else:
        # Instantiate base server with strategy and client manager
        server = PowerOfChoiceServer(strategy=strategy, client_manager=client_manager)

    # 5. Start Simulation

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
        server=server,
        ray_init_args=ray_init_args,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth
            # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
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

    # # plot results and include them in the readme
    # strategy_name = strategy.__class__.__name__
    # file_suffix: str = (
    #     f"_{strategy_name}"
    #     f"_C={cfg.num_clients}"
    #     f"_B={cfg.batch_size}"
    #     f"_E={cfg.local_epochs}"
    #     f"_R={cfg.num_rounds}"
    #     f"_d={cfg.strategy.d}"
    #     f"_CK={cfg.strategy.ck}"
    # )

    # plot_metric_from_history(
    #     history,
    #     save_path,
    #     (file_suffix),
    # )


if __name__ == "__main__":
    main()
