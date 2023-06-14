import pandas as pd
import numpy as np
from env import get_db_url
import os


from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------
def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

    
# ----------------------------------------------------------------------------------
def get_zillow_data():
    """
    This function will:
        - from the connection made to the `zillow` DB
            - using the `get_db_url` from my wrangle module.
            
        - output a df with the zillow `parcelid` set as it's index
                - `parcelid` is the table's PK. 
                    This id is an attribute of the table but will not be used as a feature to investigate.
    """
    # How to import a database from MySQL
    url = get_db_url('zillow')
    query = '''
        select p.*, pred.logerror
        from properties_2017 p
        left join propertylandusetype using(propertylandusetypeid)
        left join predictions_2017 pred using(parcelid)
        left join airconditioningtype using( airconditioningtypeid)
        left join architecturalstyletype using(architecturalstyletypeid)
        left join buildingclasstype using(buildingclasstypeid)
        left join heatingorsystemtype using(heatingorsystemtypeid)
        left join storytype using(storytypeid)
        left join typeconstructiontype using(typeconstructiontypeid)
        where YEAR(transactiondate) = 2017;
        '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    return df

# ----------------------------------------------------------------------------------
def remove_columns(df, cols_to_remove):
    """
    This function will:
    - take in a df and list of columns (you need to create a list of columns that you would like to drop under the name 'cols_to_remove')
    - drop the listed columns
    - return the new df
    """
    df = df.drop(columns=cols_to_remove)
    
    return df


# ----------------------------------------------------------------------------------
def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df


# ----------------------------------------------------------------------------------
def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df
# ----------------------------------------------------------------------------------
def get_split(df):
    '''
    train=tr
    validate=val
    test=ts
    test size = .2 and .25
    random state = 123
    '''  
    # split your dataset
    # 80/20
    train_validate, ts = train_test_split(df, test_size=.2, random_state=123)
    # 75/25
    tr, val = train_test_split(train_validate, test_size=.25, random_state=123)
    
    return tr, val, ts


# ----------------------------------------------------------------------------------
def get_Xs_ys_to_scale_baseline(tr_m, val_m, ts_m, target):
    '''
    tr = train
    val = validate
    ts = test
    target = target value
    '''

    # Separate the features (X) and target variable (y) for the training set
    X_tr, y_tr = tr_m.drop(columns=[target,'gender']), tr_m[target]
    
    # Separate the features (X) and target variable (y) for the validation set
    X_val, y_val = val_m.drop(columns=[target,'gender']), val_m[target]
    
    # Separate the features (X) and target variable (y) for the test set
    X_ts, y_ts = ts_m.drop(columns=[target,'gender']), ts_m[target]
    
    # Get the list of columns to be scaled
    to_scale = X_tr.columns.tolist()
    
    # Calculate the baseline (mean) of the target variable in the training set
    baseline = y_tr.mean()
    
    # Return the separated features and target variables, columns to scale, and baseline
    return X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline
# ----------------------------------------------------------------------------------

def scale_data(X,Xv,Xts,to_scale):
    '''
    X = X_train
    Xv = X_validate
    Xts = X_test
    to_scale, is found in the get_Xs_ys_to_scale_baseline
    '''
    
    #make copies for scaling
    X_tr_sc = X.copy()
    X_val_sc = Xv.copy()
    X_ts_sc = Xts.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(X[to_scale])

    #use the thing
    X_tr_sc[to_scale] = scaler.transform(X[to_scale])
    X_val_sc[to_scale] = scaler.transform(Xv[to_scale])
    X_ts_sc[to_scale] = scaler.transform(Xts[to_scale])
    
    return X_tr_sc, X_val_sc, X_ts_sc
# ----------------------------------------------------------------------------------
# remove all outliers put each feature one at a time
def outlier(df, feature, m=1.5):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound

# upper_bound, lower_bound = outlier(df, 'bedroomcnt')

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
