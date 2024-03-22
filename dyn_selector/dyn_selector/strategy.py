"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common.typing import EvaluateIns, FitIns, MetricsAggregationFn, NDArrays, Parameters, Scalar, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW, FedAvg
from flwr.common.logger import log
import math


class FedAvgWithDynamicSelection(FedAvg):
    "Custom FedAvg which selects a dynamic number of clients per round."

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        decay_coefficient: float,
        initial_sampling_rate: int,
        max_local_epochs: int = 5,
        batch_size: int = 32,
        fraction_samples: float = 1.0,
        use_RT: bool = False
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        decay_coefficient : float
            Decay coefficient for the sampling rate (beta).
        initial_sampling_rate : int
            Initial sampling rate C.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.decay_coefficient = decay_coefficient
        self.initial_sampling_rate = initial_sampling_rate
        self.max_local_epochs = max_local_epochs
        self.batch_size = batch_size
        self.fraction_samples = fraction_samples
        self.use_RT = use_RT

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Compute sample rate c
        sample_rate = self.initial_sampling_rate / math.exp(self.decay_coefficient * server_round)

        num_available_clients = client_manager.num_available()

        # Compute number of clients to sample
        sample_size = max(round(sample_rate * num_available_clients), self.min_fit_clients)
        
        # Sample clients
        clients = client_manager.sample(
            num_clients=sample_size
        )

        # Create fit instructions for each client
        fit_ins = []

        if not self.use_RT:
            # use standard on fit_config
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)
            fit_ins = [(client, FitIns(parameters, config)) for client in clients]
        else:
            # use RT optimizer to dynamically configure hyperparameters

            ips_clients = []

            # Retrieve IPS of sampled clients
            for client in clients:
                config = {}
                propertiesRes = client.get_properties(GetPropertiesIns(config), None)
                ips = propertiesRes.properties["ips"]
                ips_clients.append((client, ips))

            # Find the maximum IPS among those of the selected clients
            max_ips = max(ips_clients, key=lambda x: x[1])[1]

            for client, ips in ips_clients:
                # Compute scaling factor
                scale_factor = ips / max_ips

                if(ips == max_ips):
                    local_epochs = self.max_local_epochs
                else:
                    local_epochs = max(1, int(self.max_local_epochs * scale_factor))

                config = {
                    "local_epochs": local_epochs,
                    "batch_size": self.batch_size,
                    "fraction_samples": self.fraction_samples,
                }

                print(f"Client {client.cid} - IPS {ips} - Local epochs {local_epochs} - Fraction samples {self.fraction_samples}")

                # Create fit instruction
                fit_ins.append((client, FitIns(parameters, config)))

        # Return client/config pairs
        return fit_ins
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Compute sample rate c
        sample_rate = self.initial_sampling_rate / math.exp(self.decay_coefficient * server_round)

        num_available_clients = client_manager.num_available()

        # Compute number of clients to sample
        sample_size = round(max(sample_rate * num_available_clients, self.min_fit_clients))

        print(f"Selected {sample_size} clients for round {server_round} evaluation.")
        
        # Sample clients
        clients = client_manager.sample(
            num_clients=sample_size
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]