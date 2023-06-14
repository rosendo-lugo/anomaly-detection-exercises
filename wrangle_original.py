import pandas as pd
import numpy as np
from env import get_db_url
import os


from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------
def check_file_exists(fn, query, url):
    """
    This function will:
    - check if file exists in my local directory, if not, pull from sql db
    - read the given `query`
    - return dataframe
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
def get_grocery_data():
    """
    This function will:
        - from the connection made to the `grocery` DB
            - using the `get_db_url` from my wrangle module.
    """
    # How to import a database from MySQL
    url = get_db_url('grocery_db')

    query = """
    select *
    from grocery_customers
    """

    filename = 'grocery_customers.csv'
    df = check_file_exists(filename, query, url)

    df = pd.read_sql(query, url, index_col="customer_id")
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
        SELECT prop.* ,
            pred.logerror,
            pred.transactiondate,
            air.airconditioningdesc,
            arch.architecturalstyledesc,
            build.buildingclassdesc,
            heat.heatingorsystemdesc,
            land.propertylandusedesc,
            story.storydesc,
            type.typeconstructiondesc
        from properties_2017 prop
        JOIN ( -- used to filter all properties with their last transaction date in 2017, w/o dups
                SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) trans using (parcelid)
        -- bringing in logerror & transaction_date cols
        JOIN predictions_2017 pred ON trans.parcelid = pred.parcelid
                          AND trans.max_transactiondate = pred.transactiondate
        -- bringing in all other fields related to the properties
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        -- exercise stipulations
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL;
        '''
    
    
    
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    return df


def prep_zillow_data(df):
    # # How to import a database from MySQL
    # url = get_db_url('zillow')
    # query = '''
    # select *
    # from properties_2017 p
    #     join predictions_2017 p2 using (parcelid)
    # where p.propertylandusetypeid = 261 and 279
    #         '''
    # filename = 'zillow.csv'
    # df = check_file_exists(filename, query, url)
    
#     # filter to just 2017 transactions
#     df = df[df['transactiondate'].str.startswith("2017", na=False)]
    
#     # split transaction date to year, month, and day
#     df_split = df['transactiondate'].str.split(pat='-', expand=True).add_prefix('transaction_')
#     df = pd.concat([df.iloc[:, :40], df_split, df.iloc[:, 40:]], axis=1)
    
    
    # rename columns
    # df.columns
    # df = df.rename(columns={'id':'customer_id','bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms',
    #                         'calculatedfinishedsquarefeet':'area','taxvaluedollarcnt':'property_value',
    #                         'fips':'county'})  

#     # Look at properties less than .5 and over 4.5 bathrooms (Outliers were removed)
#     df = df[~(df['bathrooms'] < .5) & ~(df['bathrooms'] > 4.5)]

#     # Look at properties less than 1906.5 and over 2022.5 years (Outliers were removed)
#     df = df[~(df['yearbuilt'] < 1906.5) & ~(df['yearbuilt'] > 2022.5)]

#     # Look at properties less than -289.0 and over 3863.0 area (Outliers were removed)
#     df = df[~(df['area'] < -289.0) & ~(df['area'] > 3863.0)]

#     # Look at properties less than -444576.5 and over 1257627.5 property value (Outliers were removed)
#     df = df[~(df['property_value'] < -444576.5) &  ~(df['property_value'] > 1257627.5)]
    
    # replace missing values with "0"
    df = df.fillna({'bedrooms':0,'bathrooms':0,'area':0,'property_value':0,'county':0})
    
    # drop any nulls in the dataset
    df = df.dropna()
    
    # drop all duplicates
    df = df.drop_duplicates(subset=['parcelid'])
    
    # change the dtype from float to int  
    df[['bedrooms','area','property_value','yearbuilt']] = df[['bedrooms','area','property_value','yearbuilt']].astype(int)
    
    # rename the county codes inside county
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # get dummies and concat to the dataframe
    dummy_tips = pd.get_dummies(df[['county']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_tips], axis=1)
    
    # dropping these columns for right now until I find a use for them
    # df = df.drop(columns =['parcelid','transactiondate','transaction_year','transaction_month','transaction_day'])
    
    # dropping these columns for right now until I find a use for them
    df = df.drop(columns =['parcelid','airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid','buildingqualitytypeid','decktypeid','finishedfloor1squarefeet',
 'finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','finishedsquarefeet6','heatingorsystemtypeid',
 'poolsizesum','pooltypeid10','pooltypeid2','pooltypeid7','roomcnt','storytypeid','typeconstructiontypeid','unitcnt','yardbuildingsqft17',
 'yardbuildingsqft26','fireplaceflag'])
    
    # expand the output display of a Pandas DataFrame by setting the option 'display.max_columns' in pandas.
    # pd.set_option('display.max_columns', None)
    
    # Define the desired column order
    # new_column_order = ['bedrooms','bathrooms','area','yearbuilt','county','county_Orange','county_Ventura','property_value']

    # Reindex the DataFrame with the new column order
    # df = df.reindex(columns=new_column_order)

    # write the results to a CSV file
    # df.to_csv('df_prep.csv', index=False)

    # read the CSV file into a Pandas dataframe
    # prep_df = pd.read_csv('df_prep.csv')
    
    return df.set_index('customer_id')
    # return df
# ----------------------------------------------------------------------------------
def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return cols_missing
 
# ----------------------------------------------------------------------------------
def nulls_by_row(df, index_id='customer_id'):
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    row_missing = num_missing.value_counts().sort_index()

    rows_missing = pd.DataFrame({
        'num_cols_missing': num_missing,
        'percent_cols_missing': pct_miss,
        'num_rows': row_missing
    }).reset_index()

    result_df = df.merge(rows_missing, left_index=True, right_on='index').drop('index', axis=1)[['num_cols_missing', 'percent_cols_missing', 'num_rows']]

    return result_df #[['num_cols_missing', 'percent_cols_missing', 'num_rows']]

# ----------------------------------------------------------------------------------
def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

# ----------------------------------------------------------------------------------
def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols


# ----------------------------------------------------------------------------------
def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    # distribution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
_______________________________________""")    


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
def get_upper_outliers(s, m=1.5):
    '''
    Given a series and a cutoff value, m, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + (m * iqr)
    
    return s.apply(lambda x: max([x - upper_bound, 0]))

# ----------------------------------------------------------------------------------
def add_upper_outlier_columns(df, m=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('number'):
        df[col + '_outliers_upper'] = get_upper_outliers(df[col], m)
    return df

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
    train_validate, ts = train_test_split(df, test_size=.2, random_state=123)
    tr, val = train_test_split(train_validate, test_size=.25, random_state=123)
    
    return tr, val, ts