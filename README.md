
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

    conda env create -f env_application_bigdata.yml
    
***Note :*** If you want to set up your own environment, please take into consideration the **requirements.txt** file which contains the libraries that you need

### Installation

#### Environment 

To start the project, it is important to set up the python environment with the yml file :

    conda env create -f env_application_bigdata.yml
    
***Note :*** If you want to set up your own environment, please take into consideration the **requirements.txt** file which contains the libraries that you need

    pip install -r requirements.txt
    

## Start

Run the main

## Development tips

### Update environment

If you have installed a new library for the project, it's important to update the environment. To do this (with conda), we can simply generate a new yml file with the following command :

    conda env export > env_application_bigdata.yml
    
***Note :*** Don't forget to update the readme (library versions) and push the new yml

## Software used

_For the implementation of this project, we decided to use these different software:_
* [Anaconda](https://www.anaconda.com/) - Python distribution platform
* [PyCharm](https://www.jetbrains.com/fr-fr/pycharm/) - IDE

## Library versions

**Python :** 3.9
**pandas :** 1.3.4
**sphinx :** 4.0.1
**seaborn :** 0.11.1
**sklearn :** 0.24.1
**xgboost :** 1.3.3

## Authors

* **Enzo ALEIXO-CARVALHO** _alias_ [@ealeixoc-99](https://github.com/ealeixoc-99)
* **Clément BOULANGER** _alias_ [@ClementBou](https://github.com/ClementBou)
