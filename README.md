# Heterogeneity Management for Edge-Based Federated Learning

**Authors:** Luciano Baresi, Tommaso Dolci, and Andrea Restelli

## Abstract
The rise of machine learning—fueled by open datasets, affordable processing, and cost-effective storage—has accelerated model development across various applications, from computer vision to natural language processing. Federated learning (FL) is a technique to train multiple individual models that are progressively aggregated by a central server. This approach maintains data privacy, while deriving collective knowledge from the distributed user data. However, FL introduces complexities such as statistical and system heterogeneity, requiring innovative algorithms for convergence in non-IID datasets and performance optimization of the FL system. Despite the presence of different solutions to address statistical and system heterogeneity in FL, they often lack a proper evaluation and comparison. This work proposes an evaluation methodology for analysis and comparison of dynamic client selection and resource-aware workload allocation techniques, two promising directions to address the problem of heterogeneity in FL. We include an experimental phase to assess the performance and impact of these algorithms compared to baseline techniques, by considering different datasets, model architectures, and degrees of heterogeneity in the training data. The results showcase the competitiveness of the proposed strategies, particularly in heterogeneous settings, providing insights on their effectiveness, convergence speed, and stability. Finally, we discuss and highlight the importance of strategic client selection and workload distribution for effective and stable model training in FL environments. By implementing our solution in Flower, a flexible user-friendly FL framework, we prioritize reproducibility and extensibility of the experiments, two crucial properties for advancing research in FL.

## Directory Structure
- `dyn_selector` directory: Contains the code for the experiments using dynamic selector from paper [Ji et al. 2021](https://arxiv.org/abs/2003.09603).
- `power_of_choice` directory: Contains the code for the experiments using Power of Choice family of selectors, based on paper by [YJ Cho et al. 2022](https://proceedings.mlr.press/v151/jee-cho22a.html).

In each subdirectory, there is a `README.md` file explaining how to setup and replicate the experiments.

