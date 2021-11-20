
  
# Project - Applications of Big Data  
  
The goal of this project is to apply some concepts & tools seen in the 3 parts of this course, this project is organized into 3 parts :   
- Part 1 : Building Classical ML projects with respect to basic ML Coding best practices   
- Part 2 : Integrate MLFlow to your project   
- Part 3 : Integrate ML Interpretability to your project  
  
## Requirements  
  
### DataSet (Finance use case)  
  
DataSet of Home Credit Risk Classification: https://www.kaggle.com/c/home-credit-default-risk/data :  
- application_train.csv  
- application_test.csv  
  
### Versions  
  
- Python 3.9  
  
### Prerequisite  
  
To start the project, it is important to set up the python environment with the yml file :  
  
 conda env create -f env_application_bigdata.yml  ***Note :*** If you want to set up your own environment, please take into consideration the **requirements.txt** file which contains the libraries that you need  
  
### Installation  
  
#### Environment   
To start the project, it is important to set up the python environment with the yml file :  
  
 conda env create -f env_application_bigdata.yml  ***Note :*** If you want to set up your own environment, please take into consideration the **requirements.txt** file which contains the libraries that you need  
  
 pip install -r requirements.txt   
  
## Start  
  
Run the main. You can change the main to set up models and interpretation that what you want 

![seaborn image](https://github.com/ClementBou/Application_BigData_Project/blob/main/images/main.PNG?raw=true)

 - models_training - first parameter :  is a list of models that you want to train (model available : **Xgboost**, **Random Forest** and **Gradient Boosting**
 - models_training - third parameter : is a list of interpretation that you want to do (interpretation available : **shap_values_all**, **shap_values_one** and **summary_plot**

## Details on parts of the project

### Part one

The first part was to prepare the data for the next features.
To prepare the data, we decided to :

 -   to retrieve the same number of targets at 1 and 0 (for train_data)
 -   reduce the number of lines (because requires too many resources to compile and not necessarily useful)
 - we also decided to harmonize the data
 -  drop duplicates
 
Now, to talk about the features that we have put in place for feature engineering :
 - calculate the correlation for each feature and keep only the one with a correlation greater than 2%

To do this, we decided to create a **Seaborn Correlation Heatmap**

![seaborn image](https://github.com/ClementBou/Application_BigData_Project/blob/main/images/seaborn.png?raw=true)

 - drop columns with only 0
 - columns that have more than 50% zero values

### Part two

### Part three
  
  In this third part, we wanted to interpret the predictions that we were able to make.
  To do this, we used shap and generated several figures :
  

 - shap values for one

![seaborn image](https://github.com/ClementBou/Application_BigData_Project/blob/main/images/plot_shapley_values_i.png?raw=true)

 - shap values for all

![seaborn image](https://github.com/ClementBou/Application_BigData_Project/blob/main/images/plot_shapley_values_all.PNG?raw=true)

 - summary plot

![seaborn image](https://github.com/ClementBou/Application_BigData_Project/blob/main/images/summary_plot.png?raw=true)
  
## Development tips  
  
### Update environment  
  
If you have installed a new library for the project, it's important to update the environment. To do this (with conda), we can simply generate a new yml file with the following command :  
  
 conda env export > env_application_bigdata.yml  ***Note :*** Don't forget to update the readme (library versions) and push the new yml  
  
## Software used  
  
_For the implementation of this project, we decided to use these different software:_  
* [Anaconda](https://www.anaconda.com/) - Python distribution platform  
* [PyCharm](https://www.jetbrains.com/fr-fr/pycharm/) - IDE  
  
## Library versions  
  
**python :** 3.9  
**pandas :** 1.3.4  
**sphinx :** 4.0.1  
**seaborn :** 0.11.1  
**sklearn :** 0.24.1  
**xgboost :** 1.3.3  
**mlflo:** 1.21.0
**shap:** 0.34.0
  
## Authors  
  
* **Enzo ALEIXO-CARVALHO** _alias_ [@ealeixoc-99](https://github.com/ealeixoc-99)  
* **Clément BOULANGER** _alias_ [@ClementBou](https://github.com/ClementBou)
