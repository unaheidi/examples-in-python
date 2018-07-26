"""
Chapter 3. K-Nearest Neighbors
"""

import random
import sys

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)


class Regression(object):
  """
  Performs kNN regression
  """

  def __init__(self):
    self.k = 5
    self.metric = np.mean
    self.kdtree = None
    self.houses = None
    self.values = None

  def set_data(self, houses, values):
    """
    Sets houses and values data
    :param houses: pandas.DataFrame with houses parameters
    :param values: pandas.Series with houses values
    """
    self.houses = houses
    self.values = values
    self.kdtree = KDTree(self.houses)
    print("######### set_data #############")
    print(self.houses)
    print(self.values)
    print("######## set_data end ##########")

  def regress(self, query_point):
    """
    Calculates predicted value for house with particular parameters
    :param query_point: pandas.Series with house parameters
    :return: house value
    """
    _, indexes = self.kdtree.query(query_point, self.k)
    print("\n********* regress index **********")
    print(query_point)
    print(indexes)
    
    value = self.metric(self.values.iloc[indexes])
    print("*** mean of indexes's values  ***")
    print(value)
    print("*** self's values ***")
    print(self.values)

    if np.isnan(value):
      raise Exception('Unexpected result')
    else:
      print(value)
      return value
    
    print("******** regress end **********\n")


class RegressionTest(object):
  """
  Take in King County housing data, calculate and plot the kNN regression error rate.
  """

  def __init__(self):
    self.houses = None
    self.values = None

  def load_csv_file(self, csv_file, limit=None):
    """
    Loads CSV file with houses data
    :param csv_file: CSV file name
    :param limit: number of rows of file to read
    """
    houses = pd.read_csv(csv_file, nrows=limit)
    self.values = houses['AppraisedValue']
    houses = houses.drop('AppraisedValue', 1)
    houses = (houses - houses.mean()) / (houses.max() - houses.min())
    self.houses = houses
    self.houses = self.houses[['lat', 'long', 'SqFtLot']]

  def plot_error_rates(self):
    """
    Plots MAE vs #folds
    """
    folds_range = range(2, 7)
    errors_df = pd.DataFrame({'max': 0, 'min': 0}, index=folds_range)
    for folds in folds_range:
      errors = self.tests(folds)
      errors_df['max'][folds] = max(errors)
      errors_df['min'][folds] = min(errors)
    
    errors_df.plot(title='Mean Absolute Error of KNN over different folds_range')
    plt.xlabel('#folds_range')
    plt.ylabel('MAE')
    plt.show()

  def tests(self, folds):
    """
    Calculates mean absolute errors for series of tests
    :param folds: how many times split the data
    :return: list of error values
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
    holdout = 1 / float(folds)
    errors = []
    print("@@ range(folds) @@")
    print(range(folds))

    for _ in range(folds):
      print("@@ for _ in range(folds) @@")
      print(_)
      print("@begin testing gression")
      values_regress, values_actual = self.test_regression(holdout)
      print("@finish testing gression")
      errors.append(mean_absolute_error(values_actual, values_regress))
      print('@values_regress')
      print(values_regress)
      print('@values_actual')
      print(values_actual)
      print("@for end\n")
    
    print('@errors')
    print(errors)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests over! ~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests over! ~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests over! ~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")
    return errors

  def test_regression(self, holdout):
    """
    Calculates regression for out-of-sample data
    :param holdout: part of the data for testing [0,1]
    :return: tuple(y_regression, values_actual)
    """
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ test regression ~~~~~~~~~~~~~~~~~~~")

    print("!!!! test_regression holdout !!!!")
    print(holdout)
    print("!!! holdout !!!")
    test_rows = random.sample(self.houses.index.tolist(),
                  int(round(len(self.houses) * holdout)))
    train_rows = set(range(len(self.houses))) - set(test_rows)
    df_test = self.houses.ix[test_rows]
    df_train = self.houses.drop(test_rows)

    train_values = self.values.ix[train_rows]
    regression = Regression()
    regression.set_data(houses=df_train, values=train_values)

    values_regr = []
    values_actual = []

    for idx, row in df_test.iterrows():
      values_regr.append(regression.regress(row))
      values_actual.append(self.values[idx])
    
    print("~~~~~~~~~~~~~~~~~~~~~~~ test regression end! ~~~~~~~~~~~~~~~~~~")
    return values_regr, values_actual


def main():
  regression_test = RegressionTest()
  regression_test.load_csv_file('king_county_data_geocoded.csv', 20)
  regression_test.plot_error_rates()


if __name__ == '__main__':
  main()
