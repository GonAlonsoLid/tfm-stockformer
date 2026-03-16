# Import required packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
from numpy import *

warnings.filterwarnings('ignore')  # Ignore matching warnings
matplotlib.rc("font", family='Kaiti')  # Set font for Chinese (or use default)
matplotlib.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly



def fun(factor, df_ltsz, df_indus):    

    # Store all factors
    all_factor = []

    # Get industry
    indus = df_indus
    # Get all stock codes
    codes = indus.index.to_list()

    factor_1 = factor.drop(columns=['LABEL0'])
    grouped_dfs = [group_df for _, group_df in factor_1.groupby('datetime')]

    # All factors per date
    for dfs in grouped_dfs:

        dfs = dfs.rename(columns={'instrument':'code'})
        dfs = dfs.set_index(['code'])
        # Get date
        date = dfs['datetime'][0]

        # Get market cap
        ltsz = np.log(df_ltsz.loc[date].to_frame())

        # Get factor names
        lst_f = dfs.columns[2:].tolist()

        # Record all factors for each date
        all_factor_temp = []
        for x in lst_f:

            # x: factor name, y: factor values for codes
            y = dfs[x].to_frame(name = date)

            # Check if all are NaN
            if (~y.isna().values).sum() == 0:
                
                break

            # Concatenate
            df = pd.concat([y,ltsz,indus], axis=1)
            df.columns = range(df.shape[1])
            # Regression
            model = sm.OLS(df.iloc[:,0].to_frame(), df.drop(0,axis=1),missing='drop')
            results = model.fit()
            # Residuals are the neutralized factor
            y = (y.loc[results.fittedvalues.index] - results.fittedvalues.to_frame().rename(columns={0:date}))
            y = y.reindex(y.index)
            y = y.rename(columns = {date:x})
            # Factor standardization
            y = y.replace(np.inf,np.nan).replace(-np.inf,np.nan)
            y = (y - y.mean()) / y.std()

            all_factor_temp.append(y)

        df_pro = pd.concat(all_factor_temp,axis=1).reindex(codes)
        df_pro['datetime'] = date
        all_factor.append(df_pro)

        # Add time column

    
    factor_final = pd.concat(all_factor)

    return factor_final


if __name__ == '__main__':
    ####################### Build parameter set
    # Backtest parameters
    param = {}
    param['start_date'] = '2018-03-15'  # Backtest start date
    param['end_date'] = '2024-02-29'  # Backtest end date
    param['market_value_neutral'] = True  # Market value neutralization

    # Import closing price
    df = pd.read_csv('../data/18-24标准数据.csv',index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df_close = df.pivot(index='date', columns='code', values='close')
    df_ltsz = df.pivot(index='date', columns='code', values='ltsz')

    # Import factors
    factor = pd.read_csv('../data/factor158_未中性化.csv')

    # Import industry
    indus = pd.read_csv('../data/沪深300成分股行业哑变量.csv',index_col=0)
    indus = indus.set_index('code')

    hh = fun(factor,df_ltsz,indus)
    hh.to_csv('../data/factor_158_中性化.csv')