# Housing-Price-Predictor
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RaffaToSpace/House-Price-Predictor/master)
## Table of Contents
 - Installation and use
 - Motivation
 - The data
 - Files description
 - Conclusions

## Installation and use
The code is described as it unfolds within the notebooks.
You can clone the repository, download the single files or alternatively [use it in the Binder environment].

## Motivation
Based on the [Predict the Housing Price] Kaggle competition, in this project I go through all passages of a data science commercial project to analyse the data, prepare it, and feed it to ML models to predict the price of houses in Ames, Iowa, given a number of houses properties. The goal is to make a predictor for house prices and identify the key variables that influence the house price in Ames the most. The analysis is easily extendable to similar rural towns, and suburban town in which a centre or another location of interest is identifiable.  

See [my post on Medium] for more details on my work on this project.

The provided dataset can also be seen as a playground to practice and explore data science techniques. I have have structured my project as follows, with the purpose of showcasing recently acquired competencies and receive constructive feedback:
- Data exploration and preprocessing;
- Feature engineering techniques implementation and comparison;
- Implementation of a baseline linear regression model;
- Implementation and optimisation of a SGD regressor model;
- Prediction with the test data and conclusions.

## The data
The data can be retrieved from the link to the Kaggle dataset provided. It is composed of a 79 explanatory variables describing wide variety of properties, from the floor size to building materials and road access. Of the 79 explanatory variables, 36 are numerical (both categorical and qualtitative) and 43 are non-numerical. Many of the non numerical however, are equivalent to an ordinal numerical variable as they express an evaluation, or a score (e.g. variable with values "good","excellent", "bad" etc.). The training set sample size is 1021.
A test data set is provided, with a sample size of 439.

## Files Description
 - **pthp_data.ipynb**:
     In this notebook the data is imported and preprocessed. The neighbourhood information is used to calculate the distance from the town centre, the data is cleaned and non numerical variables are transformed either through encoding or the creation of dummy variables;
 - **pthp_data_exploration.ipynb**:
     In this notebook I take a look at some of the variables within the data set, especially categorical non-numeric variables. Some of them have a high number of possible values, and verifying if they correlate to other variables or the target can be helpful in choosing how to process them;
 - **pthp_feat_eng.ipynb**:
     In this notebook I show some possible feature engineering criteria  and techniques to reduce the dataset dimensionality (Pearson correlation between target and variables, removal of highly correlated variables, principal component analysis, backward elimination, recursive feature elimination, LassoCV), and produce corresponding reduced training sets;
 - **pthp_baseline_model.ipynb**:
     In this notebook I use a simple linear regression model to produce predictions with the preprocessed training sets;
 - **pthp_SGD.ipynb**:
     In this notebook I implement and optimise a Stochastic Gradient Descent Regressor to produce predictions with the preprocessed training sets;
  - **pthp_data_prep package**: 
     This package contains the all the necessary functions and models to preprocess the data, reduce the data dimensionality and produce the house price predictions. It has been uploaded to TestPyPi, and it can be installed by running `pip install -i https://test.pypi.org/simple/ pthp-data-prep` in terminal.
 
## Conclusions
The predictions made with the SGD regressor show good accuracy, with a mean absolute error on the test dataset house prices of $39824. The model is easily optimisable to obtain a lower MAE, through further use of grid searching and cross validation, and the project can be used as a playground for data science learners.


[use it in the Binder environment]: https://mybinder.org/v2/gh/RaffaToSpace/House-Price-Predictor/master
[Predict the Housing Price]: https://www.kaggle.com/c/predict-the-housing-price/
[my post on Medium]: https://medium.com/@raffaelenolli/how-to-make-a-simple-machine-learning-project-and-learn-in-the-process-c4d0dd605eb9
