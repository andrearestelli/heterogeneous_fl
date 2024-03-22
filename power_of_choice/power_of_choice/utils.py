"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""


import pickle
from pathlib import Path
from secrets import token_hex
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from flwr.server.history import History


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = None,
    default_filename: Optional[str] = "results.pkl",
) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name (so it doesn't overwrite.

        the file).
        """
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_dloss_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History LOSS ONLY, TO BE REMOVED LATER.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # Extract required metrics
    losses_distributed = hist.losses_distributed

    # Unpack rounds and loss values
    rounds_distributed, loss_values_distributed = zip(*losses_distributed)

    # Create a figure
    plt.figure()

    # Plot losses_distributed
    plt.plot(rounds_distributed, loss_values_distributed, label="Distributed Loss")
    plt.xlabel("Communication round")
    plt.ylabel("Training loss")
    plt.legend()

    plt.savefig(Path(save_plot_path) / Path(f"Plot_loss{suffix}.eps"), format='eps', bbox_inches='tight')
    plt.close()


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # Extract required metrics
    training_losses_distributed = hist.metrics_distributed_fit["average_training_loss"]
    accuracy_centralized = hist.metrics_centralized["accuracy"]

    # Unpack rounds and loss values
    rounds_distributed, loss_values_distributed = zip(*training_losses_distributed)
    rounds_centralized, accuracy_values_centralized = zip(*accuracy_centralized)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Plot accuracy_centralized on the left subplot
    ax1.plot(
        rounds_centralized,
        accuracy_values_centralized,
        label="Centralized Accuracy",
        color="orange",
    )
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Test accuracy")
    ax1.legend()

    # Plot losses_distributed on the right subplot
    ax2.plot(rounds_distributed, loss_values_distributed, label="Distributed Loss")
    ax2.set_xlabel("Communication round")
    ax2.set_ylabel("Training loss")
    ax2.legend()

    # Adjust layout to avoid overlapping labels and titles
    plt.tight_layout()

    plt.savefig(Path(save_plot_path) / Path(f"Plot_metrics{suffix}.eps"), format='eps', bbox_inches='tight')
    plt.close()


def plot_variance_training_loss_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # Extract required metrics
    variance_training_losses_distributed = hist.metrics_distributed_fit[
        "variance_training_loss"
    ]

    # Unpack rounds and loss values
    rounds_distributed, variance_training_loss_values_distributed = zip(
        *variance_training_losses_distributed
    )

    # Create a figure
    plt.figure()

    # Plot losses_distributed
    plt.plot(
        rounds_distributed,
        variance_training_loss_values_distributed,
        label="Distributed Loss Variance",
    )
    plt.xlabel("Communication round")
    plt.ylabel("Training loss variance")
    plt.legend()

    plt.savefig(Path(save_plot_path) / Path(f"Plot_loss_variance{suffix}.eps"), format='eps', bbox_inches='tight')
    plt.close()


def plot_metrics_from_histories(
    title_and_histories: List[Tuple[str, History]],
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot metrics from Flower server Histories.

    Parameters
    ----------
    title_and_histories : List[Tuple[str, History]]
        List of tuples where each tuple contains a title and a History object.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # Adjust the figure size as needed
    fig_size = (17, 14)  

    # Font and line sizes
    font_size = 30  # Increase font size
    line_width = 3  # Increase line width

    # Apply moving average smoothing
    window_size = 12

    model = "CNN"
    dataset = "CIFAR10"
    alpha = 0.6

    # Plot Test Accuracy
    fig1, ax1 = plt.subplots(figsize=fig_size)

    for title, hist in title_and_histories:
        # Extract required metrics
        training_losses_distributed = hist.metrics_distributed_fit[
            "average_training_loss"
        ]
        accuracy_centralized = hist.metrics_centralized["accuracy"]

        # Unpack rounds and loss values
        rounds_distributed, loss_values_distributed = zip(*training_losses_distributed)
        rounds_centralized, accuracy_values_centralized = zip(*accuracy_centralized)

        
        smoothed_loss = []
        smoothed_accuracy = []

        for i in range(len(loss_values_distributed)):
            actual_window_size = min(i + 1, window_size)
            window = np.ones(actual_window_size) / actual_window_size
            value = np.convolve(loss_values_distributed[max(0, i-actual_window_size+1):i+1], window, 'valid')[0]
            smoothed_loss.append(value)
            if i == 99 or i == 199:
                print(f"Round {i} loss: {value}")
        
        for i in range(len(accuracy_values_centralized)):
            actual_window_size = min(i + 1, window_size)
            window = np.ones(actual_window_size) / actual_window_size
            value = np.convolve(accuracy_values_centralized[max(0, i-actual_window_size+1):i+1], window, 'valid')[0]
            smoothed_accuracy.append(value)
            if i == 99 or i == 199:
                print(f"Round {i} accuracy: {value}")

        ax1.plot(rounds_centralized, smoothed_accuracy, label=f"{title}", linewidth=line_width)

    # Configure accuracy plot
    ax1.set_xlabel("Communication round", fontsize=font_size)
    ax1.set_ylabel("Test accuracy", fontsize=font_size)
    ax1.set_ylim(0.1, 0.6)
    ax1.legend(fontsize=font_size+4)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    plt.title(f"{model} on {dataset} with alpha={alpha} | Test Accuracy", fontsize=font_size+12)
    plt.tight_layout()
    plt.savefig(Path(save_plot_path) / Path(f"W{window_size}_{model}_a{alpha}_Test_accuracy{suffix}.eps"), format='eps', bbox_inches='tight')
    plt.close()

    # Plot Training Loss
    fig2, ax2 = plt.subplots(figsize=fig_size)
    for title, hist in title_and_histories:
        # Extract required metrics
        training_losses_distributed = hist.metrics_distributed_fit[
            "average_training_loss"
        ]
        accuracy_centralized = hist.metrics_centralized["accuracy"]

        # Unpack rounds and loss values
        rounds_distributed, loss_values_distributed = zip(*training_losses_distributed)
        rounds_centralized, accuracy_values_centralized = zip(*accuracy_centralized)

        # Apply moving average smoothing
        smoothed_loss = []
        smoothed_accuracy = []

        for i in range(len(loss_values_distributed)):
            actual_window_size = min(i + 1, window_size)
            window = np.ones(actual_window_size) / actual_window_size
            value = np.convolve(loss_values_distributed[max(0, i-actual_window_size+1):i+1], window, 'valid')[0]
            smoothed_loss.append(value)
            if i == 99 or i == 199:
                print(f"Round {i} loss: {value}")
        
        for i in range(len(accuracy_values_centralized)):
            actual_window_size = min(i + 1, window_size)
            window = np.ones(actual_window_size) / actual_window_size
            value = np.convolve(accuracy_values_centralized[max(0, i-actual_window_size+1):i+1], window, 'valid')[0]
            smoothed_accuracy.append(value)
            if i == 99 or i == 199:
                print(f"Round {i} accuracy: {value}")

        ax2.plot(rounds_distributed, smoothed_loss, label=f"{title}", linewidth=line_width)

    # Configure loss plot
    ax2.set_xlabel("Communication round", fontsize=font_size)
    ax2.set_ylabel("Training loss", fontsize=font_size)
    ax2.legend(fontsize=font_size+4)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)
    plt.title(f"{model} on {dataset} with alpha={alpha} | Training Loss", fontsize=font_size+12)
    plt.tight_layout()
    plt.savefig(Path(save_plot_path) / Path(f"W{window_size}_{model}_a{alpha}_Training_loss{suffix}.eps"), format='eps', bbox_inches='tight')
    plt.close()


def plot_variances_training_loss_from_history(
    title_and_histories: List[Tuple[str, History]],
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot variances from Flower server History.

    Parameters
    ----------
    hist : History
        List containing histories and corresponding titles.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # Adjust the figure size as needed
    fig_size = (30, 20)

    # Font and line sizes
    font_size = 30  # Increase font size
    line_width = 3  # Increase line width

    # Plot Test Accuracy
    fig1, ax1 = plt.subplots(figsize=fig_size)

    for title, hist in title_and_histories:
        # Extract required metrics
        variance_training_losses_distributed = hist.metrics_distributed_fit[
            "variance_training_loss"
        ]

        # Unpack rounds and loss values
        rounds_distributed, variance_training_loss_values_distributed = zip(
            *variance_training_losses_distributed
        )

        # Plot losses_distributed
        ax1.plot(
            rounds_distributed,
            variance_training_loss_values_distributed,
            label=f"{title}",
            linewidth=line_width
        )

    # Configure plot
    ax1.set_xlabel("Communication round", fontsize=font_size)
    ax1.set_ylabel("Training loss variance", fontsize=font_size)
    ax1.legend(fontsize=font_size+4)

    # Add gridlines
    ax1.grid(True)

    # Increase text size
    ax1.tick_params(axis='both', which='major', labelsize=font_size)

    # Add a title to the figure
    plt.title("MLP on MNIST with alpha=0.6 | Training Loss Variance", fontsize=font_size+12)
    plt.tight_layout()

    plt.savefig(Path(save_plot_path) / Path(f"Plot_loss_variance{suffix}.eps"), format='eps', bbox_inches='tight')
    plt.close()

def log_time_metrics(
    title_and_histories: List[Tuple[str, History]],
    log_file_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Computes time metrics from Flower server History and logs them to a file.

    Parameters
    ----------
    hist : History
        List containing histories and corresponding titles.
    log_file_path : Path
        Folder to save the log file to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # Create a log file
    with open(Path(log_file_path) / Path(f"log_time_metrics{suffix}.txt"), "w") as log_file:
        for title, hist in title_and_histories:
            # Extract required metrics
            estimated_times = hist.metrics_distributed_fit[
                "estimated_times"
            ]

            # Unpack rounds and times values
            rounds_distributed, estimated_times = zip(
                *estimated_times
            )

            # compute the max estimated time for each round
            max_estimated_times = [max(times) for times in estimated_times]
            # compute the average of the max estimated times
            avg_max_estimated_times = np.mean(max_estimated_times)
            # compute the variance among the estimated times for each round
            variance_estimated_times = [np.var(times) for times in estimated_times]
            # compute the average of the variance estimated times
            avg_variance_estimated_times = np.mean(variance_estimated_times)

            log_file.write(f"{title}\n")
            log_file.write(f"Max estimated time: {max_estimated_times}\n")
            log_file.write(f"Average max estimated time: {avg_max_estimated_times}\n")
            log_file.write(f"Variance estimated time: {variance_estimated_times}\n")
            log_file.write(f"Average variance estimated time: {avg_variance_estimated_times}\n")
            log_file.write("\n") 
            
    log_file.close()