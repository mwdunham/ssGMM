## UPDATE - February 2022
An update is being made to the ssGMM code that will be available soon. The current version is written in 'function' form which is not user friendly. I am currently getting a version of the code written as a python Class. I am also adding functionality so the ssGMM can actually behave as a classifier (perform induction rather than just transduction). This update was delayed, but I anticipate this updated version will be made available in this repository by ~ May 2022. If you would like the 'beta' version sooner, send me an email (mwdunham@mun.ca). Happy learning! 

# Semi-supervised Gaussian mixture models (ssGMM)
This repository contains a semi-supervised Gaussian mixture models (ssGMM) code written in Python 3 that is necessary for reproducing the results in our paper:

``
Dunham et al., 2020, Improved well log classification using semi-supervised Gaussian mixture models and a new model selection strategy: Computers and Geosciences, 140, 1-12.
``

This repository contains three directories:
* `ssGMM_code`:   Contains the .py code for ssGMM.
* `IRIS_dataset`: Contains a Jupyter-Notebook that applies ssGMM to the [IRIS dataset](https://archive.ics.uci.edu/ml/datasets/iris) (this is a quick demo that is used to test the code)
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
@article{DUNHAM2020104501,
title = {Improved well log classification using semisupervised Gaussian mixture models and a new hyper-parameter selection strategy},
journal = {Computers & Geosciences},
volume = {140},
pages = {1-12},
year = {2020},
issn = {0098-3004},
doi = {https://doi.org/10.1016/j.cageo.2020.104501},
url = {https://www.sciencedirect.com/science/article/pii/S0098300419309951},
author = {Michael W. Dunham and Alison Malcolm and J. Kim Welford},
keywords = {Lithofacies, Well logs, Semisupervised, Classification, Hyper-parameter selection}}
```
