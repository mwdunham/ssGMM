# Well log classification using ssGMM and supervised methods 
The well log dataset we use for this study consists of 10 wells drilled in the Hugoton and Panoma fields of southwest Kansas and northwestern Oklahoma. This dataset was the same dataset used for an SEG machine learning competition held in 2016 [(Hall, 2016)](http://dx.doi.org/10.1190/tle35100906.1), but the data were ultimately made public by the Kansas Geological Survey. 

The objective of our paper is to apply ssGMM this well log dataset and compare its performance to supervised techniques, while also exploring a new model selection strategy. This directory contains the necessary information for reproducing the contents of our paper, and we do through a three-part Jupyter-Notebook series.

This directory contains three Jupyter-Notebooks that are needed to reprouce the contents of our paper, and the contents of each one are discussed below. For all of these notebooks, you should be able to simply run them from start-to-finish without editing the cells.

## Jupyter-Notebook PART I: Estimating missing log values
The original dataset provided for the 2016 machine learning competition (https://github.com/seg/2016-ml-contest/master/facies_vectors.csv) contains missing log values (photo-electric effect, PE) for two of the wells. In order to fully leverage both of these wells in the subsequent classification problem, we need to fill in these missing values. 

We set up a regression problem using a support vector regressor (SVR) to use all the information we do know to predict the missing PE values for the two wells. We output the predictions for these two wells (`KIMZEY_A_well_PE_log_values.csv` and `ALEXANDER_D_well_PE_log_values.csv`) and add them to a new datafile in this directory called `facies_vectors_complete.csv` that contains a complete dataset. 

We do not discuss these details in our paper for brevity purposes, but this 'complete dataset' is used to obtain all the subseqent results and we want to describe how we obtained this new dataset so our results can be reproduced.

## Jupyter-Notebook PART II: Initial test of ssGMM
Before we conduct a formal classification of the well log data using ssGMM, we first perform a convergence test to see if the code is working propery. Our goal is to simulate a semi-supervised scenario with this dataset, so we split the 10 wells such that 1 well represents the training data and 9 wells represent the testing data. ssGMM has two hyper-parameters, but we do not perform cross-validation on the training data to determine values for them, we simply fix them to default values. The results indicate that ssGMM is unable to converge and further investigation reveals that the non-marine/marine indicator variable (NM_M) is a binary variable that is violating the inherent Gaussian assumption for ssGMM. 

Our solution (as discussed in the paper) is to decompose the dataset into two pieces based on this NM_M variable this gives two datasets: a non-marine dataset that largely corresponds to Classes 1-3, and the marine dataset that largely corresponds to Classes 4-9. The convergence of ssGMM on the two split datasets behaves correctly and converges quickly, which indicates that the NM_M variable is the root cause of problem. We use 'split dataset' of non-marine and marine data for generating all subsequent results. 

## Jupyter-Notebook PART III: Facies classification on split data
This notebook focuses on applying ssGMM and two supervised methods, Gaussian Naive Bayes (GNB) and XGBoost, to the split well log dataset (i.e. non-marine and marine datasets). XGBoost was deemed the winning algorithm for the 2016 machine learning competition [(Hall and Hall, 2017)](https://doi.org/10.1190/tle36030267.1), and so this method appears to be the best supervised method to compare against. We train ssGMM and XGBoost using various model selection techniques, as outlined in our paper, and this Notebook reproduces many of the figures that appear in our publication (see the `Figures` subdirectory)
* In order for this Notebook to run, XGBoost does need to be installed on your machine. 
* Details on how to do so are given [here](https://xgboost.readthedocs.io/en/latest/build.html). 
* I was able to install XGBoost by simply typing `pip install xgboost`
