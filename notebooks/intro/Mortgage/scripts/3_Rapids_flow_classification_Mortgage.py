import datetime
from scapy.all import *
import cudf as cd
import pandas as pd
import nvstrings
from collections import OrderedDict
import math
import numpy as np
import xgboost as xgb
from azureml.core.run import Run

# sklearn is used to binarize the labels as well as calculate ROC and AUC
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
from sklearn.preprocessing import label_binarize

# scipy is used for interpolating the ROC curves
from scipy import interp

# our old friend matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

# choose whatever style you want
plt.style.use('fivethirtyeight')

# cycle is used just to make different colors for the different ROC curves
from itertools import cycle

## Cyber Use Case Tutorial: Multiclass Classification on IoT Flow Data with XGBoost

# ### Goals:
# - Learn the basics of cyber network data with respect to consumer IoT devices
# - Load network data into a cuDF
# - Explore network data and features
# - Use XGBoost to build a classification model
# - Evaluate the model

def main():
    parser = argparse.ArgumentParser("RAPIDS_DBSCAN")
    parser.add_argument("--data_dir", type=str, help="Location of data")
    parser.add_argument('-f', type=str, default='') # added for notebook execution scenarios
    
    args = parser.parse_args()
    data_dir = args.data_dir
    
    run = Run.get_context()
    
    # specify the location of the data files
    DATA_PATH = data_dir

    # the sample PCAP file used for explanation 
    DATA_PCAP = DATA_PATH + "/small_sample.pcap"

    # the flow connection log (conn.log) file
    DATA_SOURCE = DATA_PATH + "/conn.log"

    # the data label file (matches IP addresses with MAC addresses)
    DATA_LABELS = DATA_PATH + "/lab_mac_labels_cats.csv"

    print("Running NETWORK FLOW on GPU...")
    t1 = datetime.now()
    
    # ### Background

    ##### Types of Network Data
    # The most detailed type of data that is typically collected on a network is full Packet CAPture (PCAP) data. This information is detailed and contains everything about the communication, including: source address, destination address, protocols used, bytes transferred, and even the raw data (e.g., image, audio file, executable). PCAP data is fine-grained, meaning that there is a record for each frame being transmitted. A typical communication is composed of many individual packets/frames.
    # 
    # If we aggregate PCAP data so that there is one row of data per communication session, we call that flow level data. A simplified example of this relationship is shown in the figure below.
    # 
    # ![PCAP_flow_relationship](images/pcap_vs_flow.png "PCAP vs FLOW")
    # 
    # For this tutorial, we use data from the University of New South Wales. In a lab environment, they [collected nearly three weeks of IoT data from 21 IoT devices](http://149.171.189.1). They also kept a detailed [list of devices by MAC address](http://149.171.189.1/resources/List_Of_Devices.txt), so we have ground-truth with respect to each IoT device's behavior on the network.
    # 
    # **Our goal is to utilize the behavior exhibited in the network data to classify IoT devices.**

    ##### The Internet of Things and Data at a Massive Scale
    # Gartner estimates there are currently over 8.4 billion Internet of Things (IoT) devices. By 2020, that number is [estimated to surpass 20 billion](https://www.zdnet.com/article/iot-devices-will-outnumber-the-worlds-population-this-year-for-the-first-time/). These types of devices range from consumer devices (e.g., Amazon Echo, smart TVs, smart cameras, door bells) to commercial devices (e.g., building automation systems, keycard entry). All of these devices exhibit behavior on the Internet as they communicate back with their own clouds and user-specified integrations.

    ### Data Investigation

    # Let's first see some of the data. We'll load a PCAP file in using Scapy. If you don't want to or can't install Scapy, feel free to skip this section.
    cap = rdpcap(DATA_PCAP)

    # get the frames
    eth_frame = cap[3]
    ip_pkt = eth_frame.payload
    segment = ip_pkt.payload
    data = segment.payload
    
    print(eth_frame.show())

    # There's really a lot of features there. In addition to having multiple layers (which may differ between packets), there are a number of other issues with working directly with PCAP. Often the payload (the `Raw` section above) is encrypted, rendering it useless. The lack of aggregation also makes it difficult to differentiate between packets. What we really care about for this application is what a *session* looks like. In other words, how a Roku interacts with the network is likely quite different than how a Google Home interacts. 
    # 
    # To save time for the tutorial, all three weeks of PCAP data have already been transformed to flow data, and we can load that in to a typical Pandas dataframe. Due to how the data was created, we have a header row (with column names) as well as a footer row. We've already removed those rows, so nothing to do here.
    # 
    # For this application, we used [Zeek](https://www.zeek.org) (formerly known as Bro) to construct the flow data. To include MAC addresses in the conn log, we used the [mac-logging.zeek script](https://github.com/bro/bro/blob/master/scripts/policy/protocols/conn/mac-logging.zeek).
    # 
#     # If you've skipped installing Scapy, you can pick up here.
#     pdf = pd.read_csv(DATA_SOURCE, sep=\'\t')
#     print("==> pdf shape: ", pdf.shape)


#     # We can look at what this new aggregated data looks like, and get a better sense of the columns and their data types. Let's do this the way we're familiar with, using Pandas.
#     print(pdf.head())
#     pdf.dtypes


    # That's Pandas, and we could continue the analysis there if we wanted. But what about  [cuDF](https://github.com/rapidsai/cudf)? Let's pivot to that for the majority of this tutorial.
    # 
    # One thing cuDF neeeds is for us to specify the data types. We'll write a function to make this easier. As of version 0.6, [strings are supported in cuDF](https://rapidsai.github.io/projects/cudf/en/latest/10min.html?highlight=string#String-Methods). We'll make use of that here.
    def get_dtypes(fn, delim, floats, strings):
        with open(fn, errors='replace') as fp:
            header = fp.readline().strip()

        types = []
        for col in header.split(delim):
            if 'date' in col:
                types.append((col, 'date'))
            elif col in floats:
                types.append((col, 'float64'))
            elif col in strings:
                types.append((col, 'str'))
            else:
                types.append((col, 'int64'))

        return OrderedDict(types)

    dtypes_data_processed = get_dtypes(DATA_SOURCE, '\t', floats=['ts','duration'],
                                 strings=['uid','id.orig_h','id.resp_h','proto','service',
                                          'conn_state','local_orig','local_resp',
                                          'history','tunnel_parents','orig_l2_addr',
                                          'resp_l2_addr'])

    raw_cdf = cd.io.csv.read_csv(DATA_SOURCE, delimiter='\t', names=list(dtypes_data_processed), 
                                 dtype=list(dtypes_data_processed.values()), skiprows=1)

    # Those data types seem right. Let's see what this data looks like now that it's in cuDF.
    # ### Adding ground truth labels back to the data

    # We'll need some labels for our classification task, so we've already prepared a file with those labels.
    dtypes_labels_processed = get_dtypes(DATA_LABELS, ',', floats=[],
                                 strings=['device','mac','connection','category'])

    labels_cdf = cd.io.csv.read_csv(DATA_LABELS, delimiter=',', names=list(dtypes_labels_processed), 
                                           dtype=list(dtypes_labels_processed.values()), skiprows=1)

    print('Labels...')
    print(labels_cdf.head())

    # We now perform a series of merges to add the ground truth data (device name, connection, category, and categoryID) back to the dataset. Since each row of netflow has two participants, we'll have to do this twice - once for the originator (source) and once for the responder (destination).
    labels_cdf.columns = ['orig_device','orig_l2_addr','orig_connection','orig_category','orig_category_id']
    merged_cdf = cd.merge(raw_cdf, labels_cdf, how='left', on='orig_l2_addr')
    labels_cdf.columns = ['resp_device','resp_l2_addr','resp_connection','resp_category','resp_category_id']
    merged_cdf = cd.merge(merged_cdf, labels_cdf, how='left')
    labels_cdf.columns = ['device','mac','connection','category','category_id']


    # Let's just look at our new dataset to make sure everything's okay.
    print('Merged...')
    print(merged_cdf.head())

    # ### Exploding the Netflow Data into Originator and Responder Rows

    # We now have netflow that has one row per (sessionized) communication between an originator and responder. However, in order to classify an individual device, we need to explode data. Instead of one row that contains both originator and responder, we'll explode to one row for originator information (orig_bytes, orig_pkts, orig_ip_bytes) and one for responder information (resp_bytes, resp_pkts, resp_ip_bytes).
    # 
    # The easiest way to do this is to create two new dataframes, rename all of the columns, then `concat` them back together. Just for sanity, we'll also check the new shape of our exploded data frame.
    orig_comms_cdf = merged_cdf[['ts','id.orig_h','id.orig_p','proto','service','duration',
                                 'orig_bytes','orig_pkts','orig_ip_bytes','orig_device',
                                 'orig_l2_addr','orig_category','orig_category_id']]
    orig_comms_cdf.columns = ['ts','ip','port','proto','service','duration','bytes','pkts',
                              'ip_bytes','device','mac','category','category_id']

    resp_comms_cdf = merged_cdf[['ts','id.resp_h','id.resp_p','proto','service','duration',
                                 'resp_bytes','resp_pkts','resp_ip_bytes','resp_device',
                                 'resp_l2_addr','resp_category','resp_category_id']]
    resp_comms_cdf.columns = ['ts','ip','port','proto','service','duration','bytes','pkts',
                              'ip_bytes','device','mac','category','category_id']

    exploded_cdf = cd.multi.concat([orig_comms_cdf, resp_comms_cdf])
    print("==> shape (original) =", merged_cdf.shape)
    print("==> shape =", exploded_cdf.shape)

    num_categories = labels_cdf['category_id'].unique().shape[0]
    print("==> number of IoT categories =", num_categories)


    # We currently need to remove null values before we proceed. Although `dropna` doesn't exist in cuDF yet, we can use a workaround to get us there. Also, due to what's available currently, we can't have any nulls in any place in the DF.
    print('Check if any missing...')
    for col in exploded_cdf.columns:
        print(col, exploded_cdf[col].null_count)
        
    exploded_cdf['category_id'] = exploded_cdf['category_id'].fillna(-999)
    exploded_cdf['device'] = exploded_cdf['device'].str.fillna("none")
    exploded_cdf['category'] = exploded_cdf['category'].str.fillna("none")

    print('After missing observations imputation...')
    for col in exploded_cdf.columns:
        print(col, exploded_cdf[col].null_count)


    # Looks like all the null values are gone, so now we can proceed. If an IP doesn't have a category ID, we can't use it. So we'll filter those out.
    exploded_cdf = exploded_cdf[exploded_cdf['category_id'] != -999]

    # ### Binning the Data and Aggregating the Features
    # 

    # But wait, there's still more data wrangling to be done! While we've exploded the flows into rows for orig/resp, we may want to bin the data further by time. The rationale is that any single communication may not be an accurate representation of how a device typically reacts in its environment. Imagine the simple case of how a streaming camera typically operates (most of its data will be uploaded from the device to a destination) versus how it operates during a firmware update (most of the data will be pushed down to the device, after which a brief disruption in connectivity will occur).
    # 
    # There's a lof ot different time binning we could do. It also would be useful to investigate what the average duration of connection is relative to how many connections per time across various time granularities. With that said, we'll just choose a time bin of 1 hour to begin with. In order to bin, we'll use the following formula:
    # 
    # $$\text{hour_time_bin}=\left\lfloor{\frac{ts}{60*60}}\right\rfloor$$
    exploded_cdf['hour_time_bin'] = exploded_cdf['ts'].applymap(lambda x: math.floor(x/(60*60))).astype(int)


    # We also have to make a choice about how we'll aggregate the binned data. One of the simplest ways is to sum the bytes and packets. There are really two choices for bytes, `bytes` and `ip_bytes`. With Bro, `bytes` is taken from the TCP sequence numbers and is potentially inaccurate, so we select `ip_bytes` instead for both originator and responder. We'll also use the sum of the number of packets.
    one_hour_time_bin_cdf = (exploded_cdf[['bytes','pkts','ip_bytes',
                                      'mac','category_id',
                                      'hour_time_bin']]
                            .groupby(['mac','category_id','hour_time_bin'])
                            .agg({'category_id': 'min',
                                'bytes':'sum',
                                  'pkts':'sum',
                                  'ip_bytes':'sum'})
                         [['min_category_id', 'sum_bytes', 'sum_pkts', 'sum_ip_bytes']]
                        )

    one_hour_time_bin_cdf.columns = ['category_id',
                                 'bytes', 'pkts', 'ip_bytes']


    # ### Creating the Training and Testing Datasets

    # We'll take a traditional 70/30 train/test split, and we'll randomly sample into a train and test data frame.
    cdf_msk = np.random.rand(len(one_hour_time_bin_cdf)) < 0.7
    train_cdf = one_hour_time_bin_cdf[cdf_msk]
    test_cdf = one_hour_time_bin_cdf[~cdf_msk]

    print("==> train length =",len(train_cdf))
    print("==> test length =",len(test_cdf))
    
    run.log('Train length', len(train_cdf))
    run.log('Test length', len(test_cdf))

    # Prepare the training input (`train_X`), training target (`train_Y`), test input (`test_X`) and test target (`test_Y`) datasets.
    train_X = train_cdf[['pkts','ip_bytes']]
    train_Y = train_cdf[['category_id']]

    test_X = test_cdf[['pkts','ip_bytes']]
    test_Y = test_cdf[['category_id']]

    # ### Configure XGBoost

    # We choose a classification algorithm that utilizes the GPU - [XGBoost](https://xgboost.readthedocs.io/en/latest/). The package provides support for gradient boosted trees and can leverage distributed GPU compute environments.

    # Getting data into a format for XGBoost is really easy. Just make a `DMatrix` for both training and testin.
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)


    # Like any good ML package, there's quite a few parameters to set. We're going to start with the softmax objective function. This will let us get a predicted category out of our model. We'll also set other parameters like the maximum depth and number of threads. You can read more about the parameters [here](https://xgboost.readthedocs.io/en/latest/parameter.html). Experiment with them!

    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.1
    param['max_depth'] = 8
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = num_categories
    param['max_features'] = 'auto'
    param['n_gpus'] = 1
    param['tree_method'] = 'gpu_hist'

    # XGBoost allows us to define a watchlist so what we can keep track of performance as the algorithm trains. We'll configure a simple watchlist that is watching `xg_train` and `xg_gest` error rates.
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 20


    # ### Training our First XGBoost Model

    # Now it's time to train
    bst = xgb.train(param, xg_train, num_round, watchlist)


    # Prediction is also easy (and fast).
    pred = bst.predict(xg_test)


    # We might want to get a sense of how our model is by calculating the error rate.
    pred_cdf = cd.from_pandas(pd.DataFrame(pred, columns=['pred']))
    pred_cdf.add_column('category_id',test_Y['category_id'])
    error_rate = (pred_cdf[pred_cdf['pred'] != pred_cdf['category_id']]['pred'].count()) / test_Y.shape[0]
    run.log('Error rate', error_rate)
    t2 = datetime.now()
    
    run.log('Runtime', t2-t1)

    # ### Conclusions

    # As we've shown, it's possible to get fairly decent multiclass classification results for IoT data using only basic features (bytes and packets) when aggregated. This isn't surprising, based on the fact that we used expert knowledge to assign category labels. In addition, the majority of the time, IoT devices are in a "steady state" (idle), and are not heavily influenced by human interaction. This lets us take larger samples (e.g., aggregate to longer time bins) while still maintaining decent classification performance. It should also be noted that this is a very clean dataset. The traffic is mainly IoT traffic (e.g., little traditional compute traffic), and there are no intentional abnormal activities injected (e.g., red teaming).
 
    # ### References

    # 1. Nadji, Y., "Passive DNS-based Device Identification", *NANOG 67*, https://www.nanog.org/sites/default/files/Nadji.pdf.
    # 1. Shams, R., "Micro- and Macro-average of Precision, Recall, and F-Score", http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html.
    # 1. Sivanathan, A. et al., "Characterizing and Classifying IoT Traffic in Smart Cities and Campuses", *2017 IEEE Conference on Computer Communications Workshops*, May 2017, http://www2.eet.unsw.edu.au/~vijay/pubs/conf/17infocom.pdf.
    # 1. University of New South Wales Internet of Things Network Traffic Data Collection, http://149.171.189.1

if __name__ == '__main__':
    main()