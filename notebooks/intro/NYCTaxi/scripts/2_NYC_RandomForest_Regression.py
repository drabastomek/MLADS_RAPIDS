import numpy as np
import pandas as pd
import cudf

import os
from collections import OrderedDict
import argparse
import datetime

from sklearn.model_selection import train_test_split as skTTS
from cuml.preprocessing.model_selection import train_test_split as cumlTTS

from sklearn.ensemble import RandomForestRegressor as skRF
from cuml.ensemble import RandomForestRegressor as cumlRF

from sklearn.metrics import r2_score as sk_r2
from cuml.metrics.regression import r2_score as cuml_r2

from azureml.core.run import Run

## UTILS
def print_time(t_curr, t_next, t_start):
    print('> Step time: {0}, elapsed time: {1}'.format(str(t_curr - t_next), str(t_curr - t_start)).rjust(64, '-'))

## GPU execution
def gpu_load_data(fname, ncols):
    dtypes = ["float32" for i in range(ncols)]
    return cudf.read_csv(fname, delimiter=',', dtype=dtypes)

def run_gpu_workflow(fname, ncols):
    t_start = datetime.datetime.now()
    print(' LOADING DATA '.center(64, '#'))
    df = gpu_load_data(fname, ncols)
    t_next = datetime.datetime.now()
    print_time(t_next, t_start, t_start)
    
    print(' SPLITTING INTO TRAIN AND TEST '.center(64, '#'))
    X_train, X_test, y_train, y_test = cumlTTS(df, 'fare_amount', train_size=0.75)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print()
    print('> Train size: {0:,} <'.format(len(X_train)).center(64,'-'))
    print('> Test  size: {0:,} <'.format(len(X_test)).center(64,'-'))
    print()
    
    print(' FITTING MODEL '.center(64, '#'))
    model = cumlRF(
          max_features = 1.0
        , n_estimators = 40
        , split_algo = 1 # global_quantile
        , n_bins = 8
    )

    model.fit(X_train, y_train)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print()
    print(' PREDICTING '.center(64, '#'))
    y_hat = model.predict(X_test)
    print()
    print('> R^2 of the model: {0:.4f} <'.format(cuml_r2(y_test, y_hat)).center(64, '-'))
    print()
    
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)

def cpu_load_data(fname, ncols):
    return pd.read_csv(fname, delimiter=',', dtype=np.float32)
    
def run_cpu_workflow(fname, ncols):
    t_start = datetime.datetime.now()
    print(' LOADING DATA '.center(64, '#'))
    df = cpu_load_data(fname, ncols)
    t_next = datetime.datetime.now()
    print_time(t_next, t_start, t_start)
    
    print(' SPLITTING INTO TRAIN AND TEST '.center(64, '#'))
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    
    X_train, X_test, y_train, y_test = skTTS(X, y, train_size=0.75)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print()
    print('> Train size: {0:,} <'.format(len(X_train)).center(64,'-'))
    print('> Test  size: {0:,} <'.format(len(X_test)).center(64,'-'))
    print()
    
    print(' FITTING MODEL '.center(64, '#'))
    model = skRF(
          max_features = 1.0
        , n_estimators = 40
        , n_jobs = 4
    )

    model.fit(X_train, y_train)
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)
    t_next = t_curr
    
    print()
    print(' PREDICTING '.center(64, '#'))
    y_hat = model.predict(X_test)
    print()
    print('> R^2 of the model: {0:.4f} <'.format(sk_r2(y_test, y_hat)).center(64, '-'))
    print()
    
    t_curr = datetime.datetime.now()
    print_time(t_curr, t_next, t_start)


def main():
    parser = argparse.ArgumentParser("NYC Taxi GPU vs CPU Random Forest Regression comparison")
    parser.add_argument("--data_dir", type=str, help="Location of data")
    parser.add_argument("--gpu", type=int, help="Use GPU?", default=0)
    parser.add_argument('-f', type=str, default='') # added for notebook execution scenarios
    args = parser.parse_args()
    data_dir = args.data_dir
    gpu = args.gpu
    
    run = Run.get_context()
    run.log("Running on GPU?", gpu)

    print("Running Random Forest on {0}...\n".format('GPU' if gpu else 'CPU'))
    t1 = datetime.datetime.now()
    data_path = os.path.join(data_dir, "data/nyctaxi/2016/featurized_yellow_tripdata_2016-01.csv")
    
    if gpu:
        run_gpu_workflow(data_path, 17)
    else:
        run_cpu_workflow(data_path, 17)
        
    t2 = datetime.datetime.now()
    print("\nTotal Random Forest Time: {0}".format(str(t2-t1)))
    run.log("Total runtime", t2-t1)
    
if __name__ == '__main__':
    main()