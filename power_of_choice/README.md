# Heterogeneity Management for Edge-Based Federated Learning

**Authors:** Luciano Baresi, Tommaso Dolci and Andrea Restelli

**Abstract:** The rise of machine learning—fueled by open datasets, affordable processing, and cost-effective storage—has accelerated model development across various applications, from computer vision to natural language processing. Federated learning (FL) is a technique to train multiple individual models that are progressively aggregated by a central server. This approach maintains data privacy, while deriving collective knowledge from the distributed user data.
However, FL introduces complexities such as statistical and system heterogeneity, requiring innovative algorithms for convergence in non-IID datasets and performance optimization of the FL system. Despite the presence of different solutions to address statistical and system heterogeneity in FL, they often lack a proper evaluation and
comparison. This work proposes an evaluation methodology for analysis and comparison of dynamic client selection and resource-aware workload allocation techniques, two promising directions to address the problem of heterogeneity in FL. We include an experimental phase to assess the performance and impact of these algorithms compared to baseline techniques, by considering different datasets, model architectures, and degrees of heterogeneity in the training data. The results showcase the competitiveness of the proposed strategies, particularly in heterogeneous settings, providing insights on their effectiveness, convergence speed, and stability. Finally, we discuss and highlight the importance of strategic client selection and workload distribution for effective and stable model training in FL environments. By implementing our solution in Flower, a flexible user-friendly FL framework, we prioritize reproducibility and extensibility of the experiments, two crucial properties for advancing research in FL.


## About the experiments

****Datasets:**** MNIST, CIFAR10 from Keras

****Hardware Setup:**** These experiments were run on a desktop machine with 10 CPU threads. Any machine with 4 CPU cores or more would be able to run it in a reasonable amount of time. Note: the entire experiment runs on CPU-only mode.


## Experimental Setup

****Model:**** This directory implements two models:
* A Multi Layer Perceptron (MLP) used in the paper on MNIST. 
This is the model used by default.
* A CNN used in the paper on CIFAR10 dataset. To use this model you have to set is_cnn=True in the configuration file base.yaml.

****Dataset:**** The experiments include two datasets: MINST (MNIST) and CIFAR10. Both are partitioned by default among 100 clients, creating imbalanced non-iid partitions using Latent Dirichlet Allocation (LDA) without resampling. All the clients have the same number of samples. Parameter `alpha` of the LDA can be set in the `base.yaml` or passed as argument, by default it is set to 2.

| Dataset | #classes | #partitions | partitioning method | partition settings |
| :------ | :---: | :---: | :---: | :---: |
| MNIST | 10 | 100 | Latent Dirichlet Allocation | All clients with same number of samples |
| CIFAR10 | 10 | 100 | Latent Dirichlet Allocation | All clients with same number of samples |


****Training Hyperparameters:**** 
| Hyperparameter | Description | Default Value |
| ---- | ----------- | ----- |
| `num_clients` | Number of total clients | 100 |
| `batch_size` | Batch size | 32 |
| `local_epochs` | Number of epochs during training | 4 |
| `fraction_samples` | Fraction of local samples to be used by clients | 1.0|
| `b` | Number of samples in the mini-batch of *rpow* | 64 |
| `d` | Number of clients selected in the *first phase* | 6 |
| `ck` | Number of clients selected at each round | 3 |


## Environment Setup
By default, Poetry will use the Python version in your system. 
In some settings, you might want to specify a particular version of Python 
to use inside your Poetry environment. You can do so with `pyenv`. 
Check the documentation for the different ways of installing `pyenv`,
but one easy way is using the automatic installer:

```bash
curl https://pyenv.run | bash
```
You can then install any Python version with `pyenv install <python-version>`
(e.g. `pyenv install 3.10.6`) and set that version as the one to be used. 
```bash
# cd to your power_of_choice directory (i.e. where the `pyproject.toml` is)
pyenv install 3.10.6

pyenv local 3.10.6

# set that version for poetry
poetry env use 3.10.6
```
To build the Python environment as specified in the `pyproject.toml`, use the following commands:
```bash
# cd to your power_of_choice directory (i.e. where the `pyproject.toml` is)

# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Running the Experiments

First ensure you have activated your Poetry environment (execute `poetry shell` from this directory).

### Generate Clients' Dataset
First (and just the first time), the data partitions of clients must be generated.

To generate the partitions for the MNIST dataset, run the following command:

```bash
# this will generate the datasets using the default settings in the `conf/base.yaml`
python -m power_of_choice.dataset_preparation
```

The generated datasets will be saved in the `mnist` folder.

If you want to modify the `alpha` parameter used to create the LDA partitions, you can override the parameter:

```bash
python -m power_of_choice.dataset_preparation alpha=<alpha>
```

To generate partitions of the CIFAR10 dataset (used in Figure 6 of the paper), you can override the parameter:

```bash
python -m power_of_choice.dataset_preparation dataset.dataset="cifar10"
```

In this case the generated datasets will be saved in the `cifar10` folder.

### Running simulations and reproducing results
If you have not done it yet, [generate the clients' dataset](#generate-clients-dataset).


#### MLP on MNIST 

The default configuration for `power_of_choice.main` uses the base version Power of Choice strategy with MLP on MNIST dataset. It can be run with the following:

```bash
python -m power_of_choice.main # this will run using the default settings in the `conf/config.yaml`
```

You can override settings directly from the command line in this way:

```bash
python -m power_of_choice.main num_rounds=100 # will set the number of rounds to 100
```

To run using FedAvg:
```bash
# This will use FedAvg as strategy
python -m power_of_choice.main variant="rand" 
```

## Expected Results

This directory can reproduce the results for 4 experiments presented in the paper: FedAvg, pow-d, cpow-d, rpow-d. Moreover, it can reproduce the results for the experiments using resource-aware workload allocators.

### FedAvg selector

```bash
# This will run the experiment using FedAvg strategy
python -m power_of_choice.main variant="rand"
```

### Power-of-choice selectors

```bash
# This will run the experiment using pow-d with d=20 and CK=9
python -m power_of_choice.main variant="base" strategy.d=20 strategy.ck=9

# This will run the experiment using cpow-d with d=20 and CK=9
python -m power_of_choice.main variant="cpow" strategy.d=20 strategy.ck=9

# This will run the experiment using rpow-d with d=60 and CK=9
python -m power_of_choice.main variant="rpow" strategy.d=60 strategy.ck=9
```

### Resource-aware workload allocators

```bash
# This will run the experiment using static optimizer
python -m power_of_choice.pow_so

# This will run the experiment using uniform optimizer
python -m power_of_choice.pow_uo optimizer=uo

# This will run the experiment using rt optimizer
python -m power_of_choice.pow_rt optimizer=rt

# This will run the experiment using ecto optimizer
python -m power_of_choice.pow_ecto optimizer=ecto
```

### Plotting the results

The above commands would generate results by creating a directory under the following path `outputs/<date>/<hour-minutes-seconds>/<dataset_name>_${variant}_d${strategy.d}_CK${strategy.ck}`, containing a `results.pkl` file that you can plot by using the following command:

```bash
# This will plot a set of results in the same figure. 
python -m power_of_choice.plot_from_pickle --metrics-type="paper_metrics" <paths_to_results>
```

To plot the time metrics, that can be plot only when using resource-aware workload allocators.

```bash
# This will plot a set of results in the same figure. 
python -m power_of_choice.plot_from_pickle --metrics-type="time" <paths_to_results>
```
