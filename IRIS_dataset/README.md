# Classifying the IRIS dataset using ssGMM
The purpose of this directory is providing an a very simple dataset to test the ssGMM code to ensure it, and most the other packages, are performing properly through the use of a Jupyter-Notebook (`IRIS_notebook.ipynb`). The dataset is the [IRIS dataset](https://archive.ics.uci.edu/ml/datasets/iris) and it has the following characteristics:
* Number of data points: 150
* Number of dimensions/attributes: 4
* Number of classes: 3 (equally distributed, i.e. 50 points/class)

To simulate a semi-supervised scenario, we randomly split the dataset into 10% for training and 90% for testing. Two figures are included in this directory that depict the full dataset and the 10% used for training (`IRIS_dataset.png` and `IRIS_dataset_training.png`, respectively). For a simple comparison, we apply Gaussian Naive Bayes (GNB), Support Vector Machines (SVM), and AdaBoost to this situation. If the notebook remains unchanged, the accuracies for these three supervised methods on the 90% unlabeled data should be:
* GNB: 94.07%
* SVM: 92.59%
* AdaBoost: 85.19%

The ssGMM code is located in the `ssGMM_code` directory and the notebook loads the code from this directory. Again, if the notebook remains unchanged, the accuracy of ssGMM on the 90% unlabeled data should be:
* ssGMM: 98.52%

We also include two figures that show the soft prediction matrix (GAMMA) on the unlabeled data (`IRIS_ssGMM_GAMMA_matrix.png`) and the objective function vs. the number of iterations (`IRIS_ssGMM_objective_function.png`). If the results and the figures can be reproduced, then proceed to the `Well_log_classification` directory.

