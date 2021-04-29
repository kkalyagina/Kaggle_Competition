import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def plot_missing_values_heatmap(df: pd.DataFrame) -> None:
  sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm', yticklabels=False)
  plt.title('Missing value in the dataset')

def get_mean_na_part(df: pd.DataFrame):
  return round((df.isna().sum() / df.shape[0]).mean(), 4)

def _error(actual: np.ndarray, predicted: np.ndarray):
  """ Simple error """
  return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
  """ Mean Squared Error """
  return np.mean(np.square(_error(actual, predicted)))

def mae(actual: np.ndarray, predicted: np.ndarray):
  """ Mean Absolute Error """
  return np.mean(np.abs(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
  """ Root Mean Squared Error """
  return np.sqrt(mse(actual, predicted))

def iterative(df):
  imp_mean = IterativeImputer(random_state=0)
  imp_mean.fit(df)
  IterativeImputer(random_state=0)
  a=imp_mean.transform(df)
  data_iterative=pd.DataFrame(a, columns=df.columns)
  data_iterative=round(data_iterative)
  return data_iterative