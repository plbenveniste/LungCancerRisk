# LungCancerRisk

P. Benveniste $^1$, J. Alberge $^1$

$^1$ Ecole Normale Supérieure Paris-Saclay

In this project, we propose a machine learning (ML) tool in order to compute the likelihood of lung cancer  occurrence based on smoking habits and patient data from the Prostate, Lung, Colorectal and Ovarian (PLCO) Cancer Screening Trial and the National Lung Screening Trial (NLST). From this ML-based tool, we developed a freely-available web application which people can use to estimate their likelihood of developing lung cancer and sensitize them to lung cancer screening for early detection of lung cancer.

## Requirements

Python version needed: `3.7 - 3.9`

Install libraries

~~~
pip install -r requirements.txt
~~~

## Description 

After performing feature extraction using risk factors and importance classification, we performed hyperparameter optimisation of an XGBoost model. This XGBoost model was trained using the Prostate, Lung, Colorectal and Ovarian (PLCO) Cancer Screening Trial and tested on the National Lung Screening Trial (NLST).

A web application was then created using the Heroku platform. 

## Repository files

The repository contains the following files: 
- in terms of code:
    - `notebooks/XGBoost_training.ipynb` : the Jupyter Notebook explaining the process for xgboost training
    - `notebooks/Data_analysis.ipynb`: the Jupyter Notebook explaining the data analysis of the pre-processed datasets
    - `notebooks/USPSTF_recommendations.ipynb`: the Jupyter Notebook explaining the results of the USPSTF recommendations
- in terms of output:
    - `output/data_analysis.txt`: the text file resulting of the data analysis on the pre-processed datasets
    - `output/USPSTF_recommendations.txt`: the text file resulting of the result computation of the USPSTF recommendations
- the model:
    - `model/model_lung_cancer.pkl`: the pickle file containing the final xgboost model



