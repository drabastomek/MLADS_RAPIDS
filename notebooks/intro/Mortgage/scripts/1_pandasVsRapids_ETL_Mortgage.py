import datetime
import numpy as np
import pandas as pd
import cudf
from cudf.dataframe import DataFrame
from collections import OrderedDict
import argparse
from azureml.core.run import Run

#### GPU ETL
def gpu_load_performance_csv(performance_path, **kwargs):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
    """
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("monthly_reporting_period", "date"),
        ("servicer", "category"),
        ("interest_rate", "float64"),
        ("current_actual_upb", "float64"),
        ("loan_age", "float64"),
        ("remaining_months_to_legal_maturity", "float64"),
        ("adj_remaining_months_to_maturity", "float64"),
        ("maturity_date", "date"),
        ("msa", "float64"),
        ("current_loan_delinquency_status", "int32"),
        ("mod_flag", "category"),
        ("zero_balance_code", "category"),
        ("zero_balance_effective_date", "date"),
        ("last_paid_installment_date", "date"),
        ("foreclosed_after", "date"),
        ("disposition_date", "date"),
        ("foreclosure_costs", "float64"),
        ("prop_preservation_and_repair_costs", "float64"),
        ("asset_recovery_costs", "float64"),
        ("misc_holding_expenses", "float64"),
        ("holding_taxes", "float64"),
        ("net_sale_proceeds", "float64"),
        ("credit_enhancement_proceeds", "float64"),
        ("repurchase_make_whole_proceeds", "float64"),
        ("other_foreclosure_proceeds", "float64"),
        ("non_interest_bearing_upb", "float64"),
        ("principal_forgiveness_upb", "float64"),
        ("repurchase_make_whole_proceeds_flag", "category"),
        ("foreclosure_principal_write_off_amount", "float64"),
        ("servicing_activity_indicator", "category")
    ])

    print(performance_path)
    
    return cudf.read_csv(performance_path, names=list(dtypes.keys()), delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def gpu_load_acquisition_csv(acquisition_path, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    GPU DataFrame
    """
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("orig_channel", "category"),
        ("seller_name", "category"),
        ("orig_interest_rate", "float64"),
        ("orig_upb", "int64"),
        ("orig_loan_term", "int64"),
        ("orig_date", "date"),
        ("first_pay_date", "date"),
        ("orig_ltv", "float64"),
        ("orig_cltv", "float64"),
        ("num_borrowers", "float64"),
        ("dti", "float64"),
        ("borrower_credit_score", "float64"),
        ("first_home_buyer", "category"),
        ("loan_purpose", "category"),
        ("property_type", "category"),
        ("num_units", "int64"),
        ("occupancy_status", "category"),
        ("property_state", "category"),
        ("zip", "int64"),
        ("mortgage_insurance_percent", "float64"),
        ("product_type", "category"),
        ("coborrow_credit_score", "float64"),
        ("mortgage_insurance_type", "float64"),
        ("relocation_mortgage_indicator", "category")
    ])
    
    print(acquisition_path)
    
    return cudf.read_csv(acquisition_path, names=list(dtypes.keys()), delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def gpu_load_names(col_names_path, **kwargs):
    """ Loads names used for renaming the banks
    
    Returns
    -------
    GPU DataFrame
    """
    
    dtypes = OrderedDict([
        ("seller_name", "category"),
        ("new", "category"),
    ])

    return cudf.read_csv(col_names_path, names=list(dtypes.keys()), delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def create_ever_features(gdf, **kwargs):
    print('Creating ever delinquent statuses...')
    everdf = gdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', method='hash', as_index=False).max()
    del(gdf)
    everdf['ever_30'] = (everdf['current_loan_delinquency_status'] >= 1).astype('int8')
    everdf['ever_90'] = (everdf['current_loan_delinquency_status'] >= 3).astype('int8')
    everdf['ever_180'] = (everdf['current_loan_delinquency_status'] >= 6).astype('int8')
    everdf.drop_column('current_loan_delinquency_status')
    return everdf

def create_delinq_features(gdf, **kwargs):
    print('Creating delinquency statuses...')

    delinq_gdf = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    del(gdf)
    delinq_30 = delinq_gdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash', as_index=False).min()
    delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    delinq_30.drop_column('monthly_reporting_period')
    delinq_90 = delinq_gdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash', as_index=False).min()
    delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    delinq_90.drop_column('monthly_reporting_period')
    delinq_180 = delinq_gdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash', as_index=False).min()
    delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    delinq_180.drop_column('monthly_reporting_period')
    del(delinq_gdf)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'], type='hash')
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'], type='hash')
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    del(delinq_30)
    del(delinq_90)
    del(delinq_180)
    return delinq_merge

def join_ever_delinq_features(everdf_tmp, delinq_merge, **kwargs):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left', type='hash')
    del(everdf_tmp)
    del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    return everdf

def create_12_mon_features(joined_df, **kwargs):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        print('\tProcessing month: {0}'.format(str(y)))
        tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        tmpdf['josh_mody_n'] = ((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12).floor()
        tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], method='hash', as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
        tmpdf['delinquency_12'] = (tmpdf['max_delinquency_12']>3).astype('int32')
        tmpdf['delinquency_12'] +=(tmpdf['min_upb_12']==0).astype('int32')
        tmpdf.drop_column('max_delinquency_12')
        tmpdf['upb_12'] = tmpdf['min_upb_12']
        tmpdf.drop_column('min_upb_12')
        tmpdf['timestamp_year'] = (((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).floor().astype('int16')
        tmpdf['timestamp_month'] = np.int8(y)
        tmpdf.drop_column('josh_mody_n')
        testdfs.append(tmpdf)
        del(tmpdf)
    del(joined_df)

    return cudf.concat(testdfs)

def create_joined_df(gdf, everdf, **kwargs):
    test = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    del(gdf)
    test['timestamp'] = test['monthly_reporting_period']
    test.drop_column('monthly_reporting_period')
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test.drop_column('current_loan_delinquency_status')
    test['upb_12'] = test['current_actual_upb']
    test.drop_column('current_actual_upb')
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)
    
    joined_df = test.merge(everdf, how='left', on=['loan_id'], type='hash')
    del(everdf)
    del(test)
    
    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)
    
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')
    
    return joined_df

def combine_joined_12_mon(joined_df, testdf, **kwargs):
    joined_df.drop_column('delinquency_12')
    joined_df.drop_column('upb_12')
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')

def final_performance_delinquency(gdf, joined_df, **kwargs):
    merged = null_workaround(gdf)
    joined_df = null_workaround(joined_df)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')
    merged.drop_column('timestamp_year')
    merged.drop_column('timestamp_month')
    return merged

def join_perf_acq_gdfs(perf, acq, **kwargs):
    perf = null_workaround(perf)
    acq = null_workaround(acq)
    return perf.merge(acq, how='left', on=['loan_id'], type='hash')

def last_mile_cleaning(df, **kwargs):
    drop_list = [
        'loan_id', 'orig_date', 'first_pay_date', 'seller_name',
        'monthly_reporting_period', 'last_paid_installment_date', 'maturity_date', 'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180', 'upb_12',
        'zero_balance_effective_date','foreclosed_after', 'disposition_date','timestamp'
    ]
    for column in drop_list:
        df.drop_column(column)
    for col, dtype in df.dtypes.iteritems():
        if str(dtype)=='category':
            df[col] = df[col].cat.codes
        df[col] = df[col].astype('float32')
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
    for column in df.columns:
        df[column] = df[column].fillna(np.dtype(str(df[column].dtype)).type(-1))
    return df#.to_arrow(preserve_index=False)

def null_workaround(df, **kwargs):
    for column, data_type in df.dtypes.items():
        if str(data_type) == "category":
            df[column] = df[column].astype('int32').fillna(-1)
        if str(data_type) in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            df[column] = df[column].fillna(np.dtype(data_type).type(-1))
    return df

def run_gpu_workflow(col_path, acq_path, quarter=1, year=2000, perf_file="", **kwargs):
    names = gpu_load_names(col_names_path=col_path)
    acq_gdf = gpu_load_acquisition_csv(acquisition_path= acq_path + "/Acquisition_"
                                      + str(year) + "Q" + str(quarter) + ".txt")
    
    acq_gdf = acq_gdf.merge(names, how='left', on=['seller_name'])
    acq_gdf.drop_column('seller_name')
    acq_gdf['seller_name'] = acq_gdf['new']
    acq_gdf.drop_column('new')

    perf_df_tmp = gpu_load_performance_csv(perf_file)
    gdf = perf_df_tmp
    
    everdf = create_ever_features(gdf)
    delinq_merge = create_delinq_features(gdf)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    del(delinq_merge)
    
    joined_df = create_joined_df(gdf, everdf)
    testdf = create_12_mon_features(joined_df)
    joined_df = combine_joined_12_mon(joined_df, testdf)
    del(testdf)
    
    perf_df = final_performance_delinquency(gdf, joined_df)
    del(gdf, joined_df)
    
    final_gdf = join_perf_acq_gdfs(perf_df, acq_gdf)
    del(perf_df)
    del(acq_gdf)
    
    final_gdf = last_mile_cleaning(final_gdf)
    
    return final_gdf

#### CPU ETL
def pandas_load_performance_csv(performance_path, **kwargs):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
    """
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("monthly_reporting_period", "date"),
        ("servicer", "category"),
        ("interest_rate", "float64"),
        ("current_actual_upb", "float64"),
        ("loan_age", "float64"),
        ("remaining_months_to_legal_maturity", "float64"),
        ("adj_remaining_months_to_maturity", "float64"),
        ("maturity_date", "date"),
        ("msa", "float64"),
        ("current_loan_delinquency_status", "int32"),
        ("mod_flag", "category"),
        ("zero_balance_code", "category"),
        ("zero_balance_effective_date", "date"),
        ("last_paid_installment_date", "date"),
        ("foreclosed_after", "date"),
        ("disposition_date", "date"),
        ("foreclosure_costs", "float64"),
        ("prop_preservation_and_repair_costs", "float64"),
        ("asset_recovery_costs", "float64"),
        ("misc_holding_expenses", "float64"),
        ("holding_taxes", "float64"),
        ("net_sale_proceeds", "float64"),
        ("credit_enhancement_proceeds", "float64"),
        ("repurchase_make_whole_proceeds", "float64"),
        ("other_foreclosure_proceeds", "float64"),
        ("non_interest_bearing_upb", "float64"),
        ("principal_forgiveness_upb", "float64"),
        ("repurchase_make_whole_proceeds_flag", "category"),
        ("foreclosure_principal_write_off_amount", "float64"),
        ("servicing_activity_indicator", "category")
    ])
    
    dates = [f for f in dtypes if dtypes[f] == "date"]

    print(performance_path)
    
    return pd.read_csv(performance_path, names=list(dtypes.keys()), delimiter='|', parse_dates=dates, infer_datetime_format=True, skiprows=1)

def pandas_load_acquisition_csv(acquisition_path, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    GPU DataFrame
    """
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("orig_channel", "category"),
        ("seller_name", "category"),
        ("orig_interest_rate", "float64"),
        ("orig_upb", "int64"),
        ("orig_loan_term", "int64"),
        ("orig_date", "date"),
        ("first_pay_date", "date"),
        ("orig_ltv", "float64"),
        ("orig_cltv", "float64"),
        ("num_borrowers", "float64"),
        ("dti", "float64"),
        ("borrower_credit_score", "float64"),
        ("first_home_buyer", "category"),
        ("loan_purpose", "category"),
        ("property_type", "category"),
        ("num_units", "int64"),
        ("occupancy_status", "category"),
        ("property_state", "category"),
        ("zip", "int64"),
        ("mortgage_insurance_percent", "float64"),
        ("product_type", "category"),
        ("coborrow_credit_score", "float64"),
        ("mortgage_insurance_type", "float64"),
        ("relocation_mortgage_indicator", "category"),
        ("date", str)
    ])
      
    print(acquisition_path)
    
    dates = [f for f in dtypes if dtypes[f] == "date"]

    
    return pd.read_csv(acquisition_path, names=list(dtypes.keys()), delimiter='|', parse_dates=dates, infer_datetime_format=True, skiprows=1).drop("date", axis=1)


def pandas_load_names(col_names_path, **kwargs):
    """ Loads names used for renaming the banks
    
    Returns
    -------
    pandas DataFrame
    """
    
    dtypes = OrderedDict([
        ("seller_name", object),
        ("new", object),
    ])

    return pd.read_csv(col_names_path, names=list(dtypes.keys()), delimiter='|', dtype=dtypes, skiprows=1)

def pandas_create_ever_features(gdf, **kwargs):
    print('Creating ever delinquent statuses...')
    everdf = gdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', as_index=False).max()
    del(gdf)
    everdf['ever_30'] = (everdf['current_loan_delinquency_status'] >= 1).astype('int8')
    everdf['ever_90'] = (everdf['current_loan_delinquency_status'] >= 3).astype('int8')
    everdf['ever_180'] = (everdf['current_loan_delinquency_status'] >= 6).astype('int8')
    everdf=everdf.drop('current_loan_delinquency_status', axis=1)
    return everdf

def pandas_create_delinq_features(gdf, **kwargs):
    print('Creating delinquency statuses...')

    delinq_gdf = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    del(gdf)
    delinq_30 = delinq_gdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    delinq_30=delinq_30.drop('monthly_reporting_period', axis=1)
    delinq_90 = delinq_gdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    delinq_90=delinq_90.drop('monthly_reporting_period', axis=1)
    delinq_180 = delinq_gdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    delinq_180=delinq_180.drop('monthly_reporting_period', axis=1)
    del(delinq_gdf)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    del(delinq_30)
    del(delinq_90)
    del(delinq_180)
    return delinq_merge

def pandas_join_ever_delinq_features(everdf_tmp, delinq_merge, **kwargs):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left')
    del(everdf_tmp)
    del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    return everdf

def pandas_create_12_mon_features(joined_df, **kwargs):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        print('\tProcessing month: {0}'.format(str(y)))
        tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        tmpdf['josh_mody_n'] = ((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12).apply(np.floor)
        tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
        tmpdf['delinquency_12'] = (tmpdf['delinquency_12']>3).astype('int32')
        tmpdf['delinquency_12'] +=(tmpdf['upb_12']==0).astype('int32')
        tmpdf['timestamp_year'] = (((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).apply(np.floor).astype('int16')
        tmpdf['timestamp_month'] = np.int8(y)
        tmpdf=tmpdf.drop('josh_mody_n', axis=1)
        testdfs.append(tmpdf)
        del(tmpdf)
    del(joined_df)

    return pd.concat(testdfs)

def pandas_create_joined_df(gdf, everdf, **kwargs):
    test = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    del(gdf)
    test['timestamp'] = test['monthly_reporting_period']
    test=test.drop('monthly_reporting_period', axis=1)
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test=test.drop('current_loan_delinquency_status', axis=1)
    test['upb_12'] = test['current_actual_upb']
    test=test.drop('current_actual_upb', axis=1)
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)
    
    joined_df = test.merge(everdf, how='left', on=['loan_id'])
    del(everdf)
    del(test)
    
    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)
    
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')
    
    return joined_df

def pandas_combine_joined_12_mon(joined_df, testdf, **kwargs):
    joined_df=joined_df.drop('delinquency_12', axis=1)
    joined_df=joined_df.drop('upb_12', axis=1)
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])

def pandas_final_performance_delinquency(gdf, joined_df, **kwargs):
    merged = pandas_null_workaround(gdf)
    joined_df = pandas_null_workaround(joined_df)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
    merged=merged.drop('timestamp_year', axis=1)
    merged=merged.drop('timestamp_month', axis=1)
    return merged

def pandas_join_perf_acq_gdfs(perf, acq, **kwargs):
    perf = pandas_null_workaround(perf)
    acq = pandas_null_workaround(acq)
    return perf.merge(acq, how='left', on=['loan_id'])

def pandas_last_mile_cleaning(df, **kwargs):
    drop_list = [
        'loan_id', 'orig_date', 'first_pay_date', 'seller_name',
        'monthly_reporting_period', 'last_paid_installment_date', 'maturity_date', 'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180', 'upb_12',
        'zero_balance_effective_date','foreclosed_after', 'disposition_date','timestamp'
    ]
    for column in drop_list:
        print('dropping: {0}'.format(column))
        df=df.drop(column, axis=1)
        
    for col, dtype in df.dtypes.iteritems():
        print(col, str(dtype))
        if str(dtype)=='object':
            df[col] = df[col].astype('category').cat.codes
            
        df[col] = df[col].astype('float32')
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
    for column in df.columns:
        df[column] = df[column].fillna(np.dtype(str(df[column].dtype)).type(-1))
    return df

def pandas_null_workaround(df, **kwargs):
    for column, data_type in df.dtypes.items():
        if str(data_type) == "object":
            df[column] = df[column].astype('str').fillna("NONE")
        if str(data_type) in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            df[column] = df[column].fillna(np.dtype(data_type).type(-1))
    return df

def run_cpu_workflow(col_path, acq_path, quarter=1, year=2000, perf_file="", **kwargs):
    names = pandas_load_names(col_names_path=col_path)
    acq_gdf = pandas_load_acquisition_csv(acquisition_path= acq_path + "/Acquisition_"
                                      + str(year) + "Q" + str(quarter) + ".txt")
    
    acq_gdf = acq_gdf.merge(names, how='left', on=['seller_name'])
    acq_gdf.drop('seller_name', axis=1)
    acq_gdf['seller_name'] = acq_gdf['new']
    acq_gdf.drop('new', axis=1)
    
    
    perf_df_tmp = pandas_load_performance_csv(perf_file)
    gdf = perf_df_tmp
    
    everdf = pandas_create_ever_features(gdf)
    delinq_merge = pandas_create_delinq_features(gdf)
    everdf = pandas_join_ever_delinq_features(everdf, delinq_merge)
    del(delinq_merge)
    
    joined_df = pandas_create_joined_df(gdf, everdf)
    testdf = pandas_create_12_mon_features(joined_df)
    joined_df = pandas_combine_joined_12_mon(joined_df, testdf)
    del(testdf)
    
    perf_df = pandas_final_performance_delinquency(gdf, joined_df)
    del(gdf, joined_df)
    
    final_gdf = pandas_join_perf_acq_gdfs(perf_df, acq_gdf)
    del(perf_df)
    del(acq_gdf)
    
    final_gdf = pandas_last_mile_cleaning(final_gdf)
    
    return final_gdf


def main():
    parser = argparse.ArgumentParser("RAPIDS_ETL")
    parser.add_argument("--data_dir", type=str, help="Location of data")
    parser.add_argument("--gpu", type=int, help="Use GPU?", default=0)
    parser.add_argument('-f', type=str, default='') # added for notebook execution scenarios
    args = parser.parse_args()
    data_dir = args.data_dir
    gpu = args.gpu
    
    run = Run.get_context()
    run.log("Running on GPU?", gpu)

    print("Running ETL...")
    t1 = datetime.datetime.now()
    
    acq_data_path = "{0}/acq".format(data_dir) #"/rapids/data/mortgage/acq"
    perf_data_path = "{0}/perf".format(data_dir) #"/rapids/data/mortgage/perf"
    col_names_path = "{0}/names.csv".format(data_dir) # "/rapids/data/mortgage/names.csv"    
    perf_file = perf_data_path + "/Performance_2000Q1.txt"

    if gpu: 
        run_gpu_workflow( col_path=col_names_path,
                          acq_path=acq_data_path,
                         perf_file=perf_file)
    else:
        run_cpu_workflow( col_path=col_names_path,
                          acq_path=acq_data_path,
                         perf_file=perf_file)
        
    t2 = datetime.datetime.now()
    print("Total ETL Time: {0}".format(str(t2-t1)))
    run.log("Total runtime", t2-t1)
    
if __name__ == '__main__':
    main()