import pandas as pd


def summary(df):
    """Generate a summary of a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to summarize.

    Returns:
        pandas.DataFrame: A summary DataFrame containing data types,
        uniqueness, missing values, and descriptive statistics.
    """
    print(f'Dataframe shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summ['# Unique'] = df.nunique().values
    summ['Missing'] = df.isna().sum()
    summ['Missing %'] = ((df.isna().sum() / len(df)) * 100).round(1)
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['Min'] = desc['min'].values
    summ['Max'] = desc['max'].values
    summ['Mean'] = desc['mean'].values
    summ['Standard Deviation'] = desc['std'].values

    return summ