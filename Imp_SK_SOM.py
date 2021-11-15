# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:12:03 2021

@author: Myriam
"""

import pandas as pd
import numpy as np
from sklearn_som.som import SOM
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Importation des données
X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)
print(X)

# Construction du SOM
XSOM = SOM(m=8, n=8, dim=15)

# Prediction de cluster pour chaque données dans X
predictions = XSOM.predict(X)

# PCA pour la visualisation
pca = PCA(n_components=2)
ND = pca.fit_transform(X)

# Distance euclidienne de chaque observations initial avec chaque cluster center
Res= XSOM.fit_transform(X)

# concatenation des observations transformé avec PCA et leur clusters prédits
pred=predictions[:,np.newaxis]
print(pred.shape)
New_ND=np.concatenate((ND,pred),axis=1)
