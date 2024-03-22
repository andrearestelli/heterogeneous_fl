import argparse
import os
import pickle
import pandas as pd  # Added import for pandas
from logging import INFO

from flwr.common.logger import log

from power_of_choice.utils import (
    plot_metrics_from_histories,
    plot_variances_training_loss_from_history,
)

def extract_data_from_history(history, round_numbers):
    """Extract data from History at specified round numbers."""
    extracted_data = []
    for round_number in round_numbers:
        if round_number in history:
            data_at_round = {
                "Round": round_number,
                "Training Loss": history[round_number]["train_loss"],
                "Validation Loss": history[round_number]["validation_loss"],
                # Add more fields as needed
            }
            extracted_data.append(data_at_round)
    return extracted_data

def create_table(histories, round_numbers):
    """Create a table with data from History objects at specified round numbers."""
    table_data = []
    for title, history in histories:
        extracted_data = extract_data_from_history(history, round_numbers)
        table_data.extend(
            {
                "Title": title,
                "Round": data["Round"],
                "Training Loss": data["Training Loss"],
                "Validation Loss": data["Validation Loss"],
                # Add more fields as needed
            }
            for data in extracted_data
        )

    df = pd.DataFrame(table_data)
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot Distributed Losses from History")
    parser.add_argument(
        "--metrics-type",
        type=str,
        choices=["paper_metrics", "variance"],
        help="Type of metrics to plot",
    )
    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Paths to the pickle files containing history data",
    )
    args = parser.parse_args()

    round_numbers_of_interest = [100, 200]
    
    # Load and display the table
    for path in args.paths:
        with open(path, "rb") as pkl_file:
            history_data = pickle.load(pkl_file)

        log(INFO, f"Loaded history data {history_data}")

        title = input(f"Enter title for the table: ")
        history = history_data["history"]

        table = create_table([(title, history)], round_numbers_of_interest)
        print(table)

if __name__ == "__main__":
    main()
