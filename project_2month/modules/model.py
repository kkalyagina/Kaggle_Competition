import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.arima_model import ARIMA
import itertools
import statsmodels.api as sm

#Model0
def model_0(df):
    mae_cv = list()
    rmse_cv = list()
    tscv = TimeSeriesSplit(n_splits=17)
    for train_index, test_index in tscv.split(df.values):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = df.values[train_index], df.values[test_index]
        mae_cv.append(mae(np.squeeze(X_test, axis=0).astype(int), X_train[-1].astype(int)))
        rmse_cv.append(rmse(np.squeeze(X_test, axis=0).astype(int), X_train[-1].astype(int)))

#SimpleExpSmoothing
def ses(df):
    mae_cv = list()
    rmse_cv = list()
    tscv = TimeSeriesSplit(n_splits=17) 
    for train_index, test_index in tscv.split(df.values):
        for i in df.columns:
            model = SimpleExpSmoothing(np.asarray(df[i].iloc[train_index])).fit(smoothing_level = 0.8, optimized=False)
            forecast = pd.Series(model.forecast(len(test_index)))
            actual = df[i].iloc[test_index]
            mae_cv.append(mae(actual.values, forecast.values))
            rmse_cv.append(rmse(actual.values, forecast.values))

#SARIMA

#Generation of combinations of seasonal parameters p, q and q

def gen_par(df):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))] 
    tscv = TimeSeriesSplit(n_splits=17) 
    for train_index, test_index in tscv.split(df.values):
        for i in df.columns:
            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(df[i].iloc[train_index],
                                      order=param,
                                      seasonal_order=param_seasonal,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
                        results = mod.fit()
                        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                    except:
                        continue

def sarima(df, order, seasonal_order):
    mae_cv = list()
    rmse_cv = list()
    tscv = TimeSeriesSplit(n_splits=17) 
    for train_index, test_index in tscv.split(df.values):
        for i in df.columns:
            mod = sm.tsa.statespace.SARIMAX(df[i].iloc[train_index], order=order, 
                                    seasonal_order=seasonal_order, 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
            results = mod.fit()
            start = len(train_index)
            pred = results.get_prediction(start=start, dynamic=False)
            actual = df[i].iloc[test_index]
            forecast = pred.predicted_mean
            mae_cv.append(mae(actual.values, forecast))
            rmse_cv.append(rmse(actual.values, forecast))
            diagnostic=results.plot_diagnostics(figsize=(15, 12))
            plt.show()