import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    
    # Data Cleaning Steps first;
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)

    # Creating new features
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AMT_CREDIT-AMT_GOODS_PRICE'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['AMT_CREDIT/AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['AMT_CREDIT/AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AMT_CREDIT/AMT_INCOME_TOTAL'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['WORKING_LIFE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_FAM'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    
    df['EXT_SOURCE_2 * EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2^2'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_3^2'] = df['EXT_SOURCE_3'] * df['EXT_SOURCE_3']
    df['ANNUITY*EXT_SOURCE_2'] = df['AMT_ANNUITY'] * df['EXT_SOURCE_2']
    df['ANNUITY*EXT_SOURCE_3'] = df['AMT_ANNUITY'] * df['EXT_SOURCE_3']
    return df

def label_encode(X):
    return X.apply(lambda col: LabelEncoder().fit_transform(col))

def label_encode_transform(X):
    return pd.DataFrame(label_encode(pd.DataFrame(X)))