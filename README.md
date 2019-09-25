# Semi-supervised Gaussian mixture models (ssGMM)
This repository contains a semi-supervised Gaussian mixture models (ssGMM) code written in Python 3 that is necessary for reproducing the results in our paper:

``
Dunham et al., *in review*, Improved well log classification using semi-supervised Gaussian mixture models and a new model selection strategy: Computers and Geosciences, 11 figures, 4 tables.
``

This repository contains three directories:
* `ssGMM_code`:   Contains the .py code for ssGMM.
* `IRIS_dataset`: Contains a Jupyter-Notebook that applies ssGMM to the [IRIS dataset](https://archive.ics.uci.edu/ml/datasets/iris)
* `Well_log_classification`: Contains a three-part series of Jupyter-Notebooks of applying ssGMM and two supervised methods to a well log classification example where the original data can be found [here](https://github.com/seg/2016-ml-contest).

The ssGMM code works as a function with a series of inputs (i.e. training data, testing data, training labels, testing labels, hyper-parameters, etc.) and outputs (predicted labels for testing data, probability matrix for testing data, objective function values). Have a look through the comments in the code to obtain more details on how to operate it.

The IRIS dataset application is included to serve as a quick test to ensure the ssGMM code is working properly, and I would encourage running this Jupyter-Notebook first before running the well log classification example.

The `Well_log_classification` example contains all the necessary information for reproducing all the figures/results in our paper. See the README in that directory for more information.

## Requirements
The ssGMM code is written in Python 3 and the associated Jupyter notebooks use standard packages included with Anaconda or other distribution platforms. The versions of these packages that have been tested are given below and are up-to-date as of mid-2019. If you have different versions, everything will likely still run properly, but you **must** be using Python 3, the ssGMM code will not work with Python 2 and the syntax in the Jupyter-Notebooks is written according to Python 3.

* Python (tested with 3.6.6 - 3.7.3)
* Pandas:       0.24.2
* Numpy:        1.16.2
* Matplotlib:   3.0.3
* Seaborn:      0.9.0
* Sklearn:      0.20.3
* XGBoost:      0.82
* Joblib:       0.13.2

The only package in this list that is *external* to a python distribution platform is XGBoost, a gradient boosting classifier, and must be downloaded separately. However, this package is only needed for `Well_log_classification` and the README in that directory provides details on how to download this package.

## Cite

Please cite our paper if you use this code in your own work:

```
@Article{Dunham_2020_ssGMM,
  title={Improved well log classification using semi-supervised Gaussian mixture models and a new model selection strategy},
  author={Dunham, Michael W. and Malcolm, Alison and Welford, J. Kim},
  journal={Computers and Geosciences},
  year={in review}
}
```
