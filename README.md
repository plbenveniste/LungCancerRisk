# LungCancerRisk

Pierre-Louis Benveniste $^{1*}$, Julie Alberge $^{2*}$, Jean-Emmanuel Bibault $^3$

- 1 - Ecole Normale Supérieure de Paris Saclay
- 2 - École nationale des ponts et chaussées
- 3 - Radiation Oncology Department, Hôpital Européen Georges Pompidou, AP-HP, Université Paris Cité

*contributed equally

In this project, we propose a machine learning (ML) tool in order to compute the likelihood of lung cancer occurrence in the next five years, based on smoking habits and patient data from the Prostate, Lung, Colorectal and Ovarian (PLCO) Cancer Screening Trial and the National Lung Screening Trial (NLST). From this ML-based tool, we developed a freely-available web application which people can use to estimate their likelihood of developing lung cancer and sensitize them to lung cancer screening for early detection of lung cancer.

## Paper

ADD LINK TO PAPER 

## Web Application

A web application waas created using the Heroku platform.
You can use the web application [here](https://lung-cancer-risk-7f6ac1f97fd0.herokuapp.com/). 

## Abstract 

**Introduction:** Lung cancer is a significant cause of mortality worldwide, emphasizing the importance of early detection for improved survival rates. In this study, we propose a machine learning (ML) tool trained on data from the PLCO Cancer Screening Trial and validated on the NLST to estimate the likelihood of lung cancer occurrence within five years.

**Methods:** The study utilized two datasets, the PLCO (n=55,161) and NLST (n=48,595), consisting of comprehensive information on risk factors, clinical measurements, and outcomes related to lung cancer. Data pre-processing involved removing patients who were not current or former smokers and those who had died of causes unrelated to lung cancer. Additionally, a focus was placed on mitigating bias caused by censored data. Feature selection, hyperparameter optimization, and model calibration were performed using XGBoost, an ensemble learning algorithm that combines gradient boosting and decision trees.

**Results:** The final ML model was trained on the pre-processed PLCO dataset and tested on the NLST dataset. The model incorporated features such as age, gender, smoking history, medical diagnoses, and family history of lung cancer. The model was well-calibrated (Brier score=0.044). ROC-AUC was 82% on the PLCO dataset and 70% on the NLST dataset. PR-AUC was 29% and 11% respectively. When compared to the USPSTF guidelines for lung cancer screening, our model provided the same recall with a precision of 13.1% vs 9.3% on the PLCO dataset and 3.2% vs 3.1% on the NLST dataset.

**Conclusion:** The developed ML tool provides a freely-available web application for estimating the likelihood of developing lung cancer within five years. By utilizing risk factors and clinical data, individuals can assess their risk and make informed decisions regarding lung cancer screening. This research contributes to the efforts in early detection and prevention strategies, aiming to reduce lung cancer-related mortality rates.


## Requirements

Python version needed: `3.7 - 3.9`

Install libraries

~~~
pip install -r requirements.txt
~~~

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



