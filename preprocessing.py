# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_features(df):
    df = df[~(df['AMT_INCOME_TOTAL'] > 1e8)]
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    
    # make age columns
    df['age'] = np.abs(df['DAYS_BIRTH'])  // 365
#     df = df.drop('DAYS_BIRTH', axis=1)
    df['era'] = df['age'].apply(lambda x: x // 10 * 10)
    
    return df


def fill_missing_value(df, imputer):
    if 'SK_ID_CURR' in df:
        df = df.drop(columns=['SK_ID_CURR'])
    else:
        df = df.copy()
    # print(X_train.shape)
    features = df.columns.tolist()

    return pd.DataFrame(imputer.fit_transform(df),columns=[features])


def fill_median(df, group, cols):
    tmp = pd.DataFrame()
    for col in cols:
        col_df = pd.DataFrame()
        for i in df[group].unique():
            median = df[df[group]== i][col].median()
            tmp_df = df[df[group]== i][['SK_ID_CURR', col]].fillna(median)

            if len(col_df) == 0:
                col_df = col_df.append(tmp_df)
            else:
                col_df = pd.concat([col_df, tmp_df])
                
        if len(tmp) == 0:
            tmp = tmp.append(col_df)
        else:
            tmp = tmp.merge(col_df, how='inner', on='SK_ID_CURR')

    return tmp.sort_values('SK_ID_CURR').reset_index(drop=True)


def missing_values_table(df):
    
    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns