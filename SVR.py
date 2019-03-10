# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:58:15 2018

@author: hp
"""

#SUPPORT VECTOR REGRESSION
#doesnt apply feature scaling itself

import pandas as pd
import numpy as np

dataset=pd.read_csv("Position_Salaries.csv")
features=dataset.iloc[:,1:2].values
labels=dataset.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc_features=StandardScaler()
sc_labels=StandardScaler()
features=sc_features.fit_transform(features)
labels=sc_labels.fit_transform(labels)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(features,labels)

y_pred=sc_labels.inverse_transform(regressor.predict(sc_features.transform(np.array([[6.5]]))))