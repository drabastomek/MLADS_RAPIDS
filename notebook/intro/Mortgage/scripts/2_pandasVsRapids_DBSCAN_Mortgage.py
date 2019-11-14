import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as skDBSCAN
from cuml import DBSCAN as cumlDBSCAN
import cudf
import os
from collections import OrderedDict
import argparse
import datetime
from azureml.core.run import Run

## GPU execution
def gpu_load_data(fname, ncols):
    dtypes = OrderedDict([
        ("feature_{0}".format(i), "float64") for i in range(ncols)])
    
    print(fname)
    
    return cudf.read_csv(fname, names=list(dtypes.keys()), delimiter=',', dtype=list(dtypes.values()), skiprows=1)

def run_gpu_workflow(fname, ncols, eps, min_samples):
    mortgage_cudf = gpu_load_data(fname, ncols)
    clustering_cuml = cumlDBSCAN(eps = eps,min_samples = min_samples)
    dbscan_gpu = clustering_cuml.fit(mortgage_cudf)

## CPU execution
def cpu_load_data(fname):
    print(fname)
    return pd.read_csv(fname, header=0)

def run_cpu_workflow(fname, eps, min_samples):
    mortgage_df = cpu_load_data(fname)
    clustering_sk = skDBSCAN(eps = eps, min_samples = min_samples)
    clustering_sk.fit(mortgage_df)

def main():
    parser = argparse.ArgumentParser("RAPIDS_DBSCAN")
    parser.add_argument("--data_dir", type=str, help="Location of data")
    parser.add_argument("--gpu", type=int, help="Use GPU?", default=0)
    parser.add_argument("--ncols", type=int, help="How many columns?", default=128)    
    parser.add_argument("--eps", type=int, help="How many columns?", default=3)
    parser.add_argument("--min_samples", type=int, help="How many columns?", default=2)
    parser.add_argument('-f', type=str, default='') # added for notebook execution scenarios
    
    args = parser.parse_args()
    data_dir = args.data_dir
    gpu = args.gpu
    ncols = args.ncols
    eps = args.eps
    min_samples = args.min_samples
    
    run = Run.get_context()
    run.log("Running on GPU?", gpu)
    run.log("ncols", ncols)
    run.log("eps", eps)
    run.log("min_samples", min_samples)

    print("Running DBSCAN on {0}...".format('GPU' if gpu else 'CPU'))
    t1 = datetime.datetime.now()
    
    fname = data_dir + "/mortgage.csv"

    if gpu: 
        run_gpu_workflow(fname, ncols, eps, min_samples)
    else:
        run_cpu_workflow(fname, eps, min_samples)
        
    t2 = datetime.datetime.now()
    print("Total DBSCAN Time on {0}: {1}".format('GPU' if gpu else 'CPU', str(t2-t1)))
    
    run.log("Total runtime", t2-t1)

if __name__ == '__main__':
    main()