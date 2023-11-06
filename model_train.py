#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("financials2.csv")
df

df.head()

# ### Data Cleaning and Preparation

columns = ['Symbol', 'Name', 'Sector', 'Price', 'Price/Earnings', 'Dividend Yield', 'Earnings/Share', '52 Week Low', '52 Week High', 'Market Cap', 'EBITDA', 'Price/Sales', 'Price/Book']

df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')
df


df.isnull().sum()

df.price_earnings.fillna(0, inplace=True)
df.price_book.fillna(0, inplace=True)

df.isnull().sum()

list(df.iloc[1, :10])


df.dtypes

categorical = ['symbol', 'name', 'sector']
numerical = ['price', 'price_earnings', 'dividend_yield', 'earnings_share', '52_week_low', '52_week_high', 'market_cap', 'ebitda', 'price_sales', 'price_book']


# ### Splitting the Data

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_test = df_test.price.values
y_val = df_val.price.values

del df_train['price']
del df_val['price']
del df_test['price']

y_train

# ### Mutual Info Score

from sklearn.metrics import mutual_info_score

def mutual_info(series):
    return mutual_info_score(series, df_full_train.price)
info = df_full_train[numerical].apply(mutual_info)
info.sort_values(ascending=False)


# ### Using Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_val, y_val))

y_pred = model.predict(X_val)
y_pred
print("Predicted price using Linear Regerssion:", y_pred)


# ### Using Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
print(regressor.score(X_val, y_val))

y_pred = regressor.predict(X_val)

print("Predicted price", y_pred)


# ### Visualizing the Decision Tree

from sklearn.tree import export_graphviz 
from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(regressor, out_file=None, 
                                feature_names =list(dv.get_feature_names_out()),  
                                class_names=True,
                                filled=True)

graph = graphviz.Source(dot_data, format="svg") 
graph
graph.render("decision_tree_graphviz")


# ### Accuracy of Model

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

price_logs = np.log1p(df.price)

sns.histplot(price_logs, bins=50)

sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_val, color='blue', alpha=0.5, bins=50)


sns.lmplot(x ="price", y ="market_cap", data = df_full_train, order = 2, ci = None)
plt.show()

rmse = mean_squared_error(y_val, y_pred, squared=False)
print("RMSE is:", rmse)


# ### Saving and Loading the Model

import pickle


output_file = 'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
