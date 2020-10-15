# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:42:53 2020

@author: HUAN
"""
#前置步驟
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz
import pydotplus

df = pd.read_csv("adult.data")
df.columns = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 
              'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country','class']

df.replace(" ?",np.nan,inplace=True)
df2 = df.dropna()


#onehot ecoding
pd.get_dummies(df2['Workclass'])
Workclass_encoding = pd.get_dummies(df2['Workclass'], prefix = 'Workclass')
df2 = df2.drop('Workclass', 1)

pd.get_dummies(df2['Education'])
Education_encoding = pd.get_dummies(df2['Education'], prefix = 'Education')
df2 = df2.drop('Education', 1)

pd.get_dummies(df2['Marital-status'])
marital_status_encoding = pd.get_dummies(df2['Marital-status'], prefix = 'Marital-status')
df2 = df2.drop('Marital-status', 1)

pd.get_dummies(df2['Occupation'])
occupation_encoding = pd.get_dummies(df2['Occupation'], prefix = 'Occupation')
df2 = df2.drop('Occupation', 1)

pd.get_dummies(df2['Relationship'])
Relationship_encoding = pd.get_dummies(df2['Relationship'], prefix = 'Relationship')
df2 = df2.drop('Relationship', 1)

pd.get_dummies(df2['Race'])
Race_encoding = pd.get_dummies(df2['Race'], prefix = 'Race')
df2 = df2.drop('Race', 1)

pd.get_dummies(df2['Sex'])
Sex_encoding = pd.get_dummies(df2['Sex'], prefix = 'Sex')
df2 = df2.drop('Sex', 1)

pd.get_dummies(df2['Native-country'])
Native_country_encoding = pd.get_dummies(df2['Native-country'], prefix = 'Native-country')
df2 = df2.drop('Native-country', 1)

class_mapping = {' >50K':1,' <=50K':0}
df2['class'] =  df2['class'].map(class_mapping)


#讀檔合併onehot encoding
df2 = pd.concat([Workclass_encoding,Education_encoding,marital_status_encoding,occupation_encoding,
           Relationship_encoding,Race_encoding,Sex_encoding,Native_country_encoding,df2],axis=1)
#資料正規化
# df3 = preprocessing.normalize(df2, norm='l2')
# scaler = MinMaxScaler()
# df4 = scaler.fit(df2)
# df4 = scaler.transform(df2)

#建立特徵X，與目標y
X = df2.drop('class',axis = 1)
X_norm = preprocessing.normalize(X, norm='l2')
y = df2['class']

# #將資料區分成訓練集與測試集，可自行設定區分的百分比
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2)

#初步調整
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=5)

#用建立好的模型來預測資料
df2_clf = clf.fit(X_train, y_train)

# 預測
test_y_predicted = df2_clf.predict(X_test)


# 結果
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print(accuracy)

dot_tree = tree.export_graphviz(df2_clf,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_tree)
graph.write_pdf("adult-gini.pdf")





