# <p style="font-family: 'Brush Script MT', cursive;">Hello!</p>

Hi! Thank you for reviewing my project!

My name is Tauras Aleksandravicius and this is my Data Science Module 3 Sprint Capstone project.
In this project I focused on  **Machine learning fundamentals and deploying models to GCP**.
For this purpose I worked with a Home Credit Default Risk dataset. [Link to the Dataset](https://www.kaggle.com/c/home-credit-default-risk/data)

**Requirements:**
Please run first:

```Python
pip install -r requirements.txt
```

My repository contains the following files:
- **1.Data-Merging-M3S4-TA.ipynb**: Notebook file for Merging data from different databases (actually its .csv files).
- **2.Data-Preprocessing+EDA+M3S4+TA.ipynb**: Notebook file for Data Preprocessing - Data Cleaning and etc;.
- **3.Data-Modelling+M3S4+TA.ipynb**: Notebook file for ML modelling, feature selection, model selection and tuning.
- **tools.py**: My resuable functions from Module 2 till out here;
- **/deployment**: This directory contains Dockerfile, lgbm.pipeline.pkl, main.py, test.csv, test.py, feature_engineering.py files.
- **tools.py**: My resuable functions from Module 2 till out here;
- **requirements.txt**: Requirements file;

**To check ML models deployment to GCP Cloud Run:**
Please go to deployment directory and in the Terminal Run:

```Python
python3.12 test.py
```

# <p style="font-family: 'Brush Script MT', cursive;">Project Notes/ Findings:</p>
* I extensively used Pandas Profiling Reports to take first glances at the data. Each weights a lot so they will not be included into the repo;
* Same goes with the datasets. Even the final dataset, that the model was trained on is more than 100MB and is not uploaded to repo.
* Final model achieved 0.77 AUC score. Kaggle competition winner, 1st place prize that won 35 000 USD had 0.80 AUC score.
* This project nicely tested out Big Data handling, cleaning and finally deployment capabilities.
* Most important features for the defaulting are count of late payments, gender, credit sum debt. All of them totally make sense;
* Feature Engineered features and features from other datasets helped vastly. Without them model would be a lot inferior;

# <p style="font-family: 'Brush Script MT', cursive;">POC Plan for the Startup:</p>

- First goal is to migrate backend data to unified, scalable data storage, sth like *Google Big Query*;
- Create data models that aggregate data into unified dataset for ML modelling; Models are ran via *dbt*, orchestrated via *Airflow* and training data is incrementally updated on every new application.
- Having unified data model - perform data cleaning on each user_id and trained model would be accessible via web-page UI.
- User on UI might have to authenticate himself, expose some of the data. Then banking service would aggregate all the rest fields from the application.csv file and aggregate data from other datasets. After it, aggregated data would be fed into a model and we would return predicted probability for defaulting. We could later develop bins for predicted probabilities.

Thank you for your attention!

Have a good day!