# 2020 NSF Group 2 Mini Project: Covid 19

In this project we set out to clean, analyze, and train machine learning models to make predictions on a covid 19 dataset.

The data set had ~2 million rows in a ~500MB CSV file.

We extracted out symptoms and cleaned up the ages, then imputed missing gender values:

    Cleaning & Extraction - Dates, Outcomes, Symptoms.ipynb
    Cleaning - Ages and Genders.ipynb

Then explored the data to look for patterns and trends to model:

    Data Exploration - Symptoms and Travel.ipynb
    Exploratory Analysis - Case Counts.ipynb
    Modelling Travel.ipynb
    
And finally, we then explored two primary topics in our modelling and analysis.
    1. Predicting symptoms and hospitalizations.
    2. Establishing a relationship betyween population density and case counts.
    This analysis can be seen in this notebook,
    Predicting Hospitalizations & Deaths From Symptoms.ipynb
    
We found that a logistic regression model makes for a good classification model when we use a vector of each symptom as the dependent variable, though it does not do as well when applied only to the Pneumonia symptom.
