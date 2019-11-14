import os
import cudf
import pandas as pd
import datetime
import argparse
import xgboost

from cuml.preprocessing.model_selection import train_test_split as cumlTTS
from cuml.ensemble import RandomForestRegressor as cumlRF
from cuml.metrics.regression import r2_score as cuml_r2

from azureml.core.run import Run

########## GLOBAL VARS ############
# list of column names that need to be re-mapped
remap = {}
remap['tpep_pickup_datetime'] = 'pickup_datetime'
remap['tpep_dropoff_datetime'] = 'dropoff_datetime'
remap['ratecodeid'] = 'rate_code'

#create a list of columns & dtypes the df must have
must_haves = {
    'pickup_datetime': 'datetime64[ms]',
    'dropoff_datetime': 'datetime64[ms]',
    'passenger_count': 'int32',
    'trip_distance': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'rate_code': 'int32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'fare_amount': 'float32'
}

query_frags = [
    'fare_amount > 0 and fare_amount < 500',
    'passenger_count > 0 and passenger_count < 6',
    'pickup_longitude > -75 and pickup_longitude < -73',
    'dropoff_longitude > -75 and dropoff_longitude < -73',
    'pickup_latitude > 40 and pickup_latitude < 42',
    'dropoff_latitude > 40 and dropoff_latitude < 42'
]

## UTILS
def print_time(t_curr, t_next, t_start):
    print('> Step time: {0}, elapsed time: {1}'.format(str(t_curr - t_next), str(t_curr - t_start)).rjust(64, '-'))

########## METHODS DECLARATIONS ############
# data cleaner and type converted function
def clean(df, remap, must_haves, gpu):    
    # some col-names include pre-pended spaces remove & lowercase column names
    tmp = {col:col.strip().lower() for col in list(df.columns)}
    
    ### rename columns
    if gpu:
        df = df.rename(tmp)
        df = df.rename(remap)
    else:
        df = df.rename(tmp, axis=1)
        df = df.rename(remap, axis=1)
    
    # iterate through columns in this df partition
    for col in df.columns:
        # drop anything not in our expected list
        if col not in must_haves:
            if gpu:
                df = df.drop(col)
            else:
                df = df.drop(col, axis=1)
            continue

        if df[col].dtype == 'object' and col in ['pickup_datetime', 'dropoff_datetime']:
            df[col] = df[col].astype('datetime64[ms]')
            continue
            
        # if column was read as a string, recast as float
        if df[col].dtype == 'object':
            df[col] = df[col].str.fillna('-1')
            df[col] = df[col].astype('float64')
        else:
            # downcast from 64bit to 32bit types
            # Tesla T4 are faster on 32bit ops
            if 'int' in str(df[col].dtype):
                df[col] = df[col].astype('int64')
            if 'float' in str(df[col].dtype):
                df[col] = df[col].astype('float64')
            df[col] = df[col].fillna(-1)
    
    return df

def add_features(df, gpu):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['diff'] = df['dropoff_datetime'].astype('int64') - df['pickup_datetime'].astype('int64')
    
    df['pickup_latitude_r'] = df['pickup_latitude'] // .01 * .01
    df['pickup_longitude_r'] = df['pickup_longitude'] // .01 * .01
    df['dropoff_latitude_r'] = df['dropoff_latitude'] // .01 * .01
    df['dropoff_longitude_r'] = df['dropoff_longitude'] // .01 * .01
    
    if gpu:
        df = df.drop('pickup_datetime')
        df = df.drop('dropoff_datetime')
    else:
        df = df.drop('pickup_datetime', axis=1)
        df = df.drop('dropoff_datetime', axis=1)
        
    return df

########## WORKFLOWS ############
def run_workflow(data_path):
    t_start = datetime.datetime.now()
    print(' LOADING DATA '.center(64, '#'))
    
    taxi_df = []
    taxi_df.append(cudf.read_csv(os.path.join(data_path, '2016/yellow_tripdata_2016-01.csv')))
    taxi_df.append(cudf.read_csv(os.path.join(data_path, '2016/yellow_tripdata_2016-02.csv')))
    taxi_df = cudf.concat(taxi_df)
    t_next = datetime.datetime.now()
    print_time(t_next, t_start, t_start)
    
    print()
    print('> NUMBER OF ROWS: {0:,} <'.format(len(taxi_df)).center(48, '-'))
    print()
    
    print(' CLEANING DATA '.center(64, '#'))
    taxi_df = clean(taxi_df, remap, must_haves, gpu=1)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print(' SUBSETTING DATA '.center(64, '#'))
    # apply a list of filter conditions to throw out records with missing or outlier values
    taxi_df = taxi_df.query(' and '.join(query_frags))
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print(' FEATURIZING DATA '.center(64, '#'))
    taxi_df = add_features(taxi_df, gpu=1)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    
    print(' SPLITTING INTO TRAIN AND TEST '.center(64, '#'))
    X_train, X_test, y_train, y_test = cumlTTS(taxi_df, 'fare_amount', train_size=0.75)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print()
    print('> Train size: {0:,} <'.format(len(X_train)).center(64,'-'))
    print('> Test  size: {0:,} <'.format(len(X_test)).center(64,'-'))
    print()
    
    print(' FITTING MODEL '.center(64, '#'))
    params = {
        'learning_rate'  : 0.3,
        'max_depth'      : 8,
        'objective'      : 'reg:squarederror',
        'subsample'      : 0.6,
        'gamma'          : 1,
        'silent'         : True,
        'verbose_eval'   : True,
        'tree_method'    :'gpu_hist'
    }

    xg_train = xgboost.DMatrix(X_train, label=y_train)
    trained_model = xgboost.train(params, xg_train, 20)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print()
    print(' PREDICTING '.center(64, '#'))
    xg_test = xgboost.DMatrix(X_test, label=y_test)
    pred = trained_model.predict(xg_test)
    y_hat = cudf.from_pandas(pd.DataFrame(pred, columns=['fare_amount']))
    print()
    print('> R^2 of the model: {0:.4f} <'.format(cuml_r2(y_test.astype('float32'), y_hat)).center(64, '-'))
    print()
    
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    
def main():
    parser = argparse.ArgumentParser("NYC Taxi GPU vs CPU XGBoost Regression comparison")
    parser.add_argument("--data_dir", type=str, help="Location of data")
    parser.add_argument('-f', type=str, default='') # added for notebook execution scenarios
    args = parser.parse_args()
    data_dir = args.data_dir
    
    run = Run.get_context()

    print("Running end-2-end XGBoost training on GPU...\n")
    t1 = datetime.datetime.now()
    data_path = os.path.join(data_dir, "data/nyctaxi")
    
    run_workflow(data_path)
        
    t2 = datetime.datetime.now()
    print("\nTotal ETL Time: {0}".format(str(t2-t1)))
    run.log("Total runtime", t2-t1)
    
if __name__ == '__main__':
    main()