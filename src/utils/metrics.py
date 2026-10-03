import numpy as np


def mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def wape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(actual - forecast)) / denom * 100


def mase(actual, forecast, train_series, seasonal_period=7):
    actual, forecast = np.array(actual), np.array(forecast)
    train_series = np.array(train_series)
    naive_errors = np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period])
    if len(naive_errors) == 0:
        return np.nan
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return np.mean(np.abs(actual - forecast)) / scale
